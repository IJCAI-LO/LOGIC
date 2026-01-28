from clip.clip import load, tokenize, available_models
import torch
from dataset import *
from torchvision import transforms
import argparse
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm
import logging
import os

from dataset.mpdd import MPDDDataset
from dataset.mvtec3d_few_shot import MVTec3DDataset
from util.utils import eval_all_class
import copy

def extract_trainable_state(model):
    """把所有会影响最终结果的‘训练权重/关键参数’打包出来"""
    state = {}

    # 1) prompt (Parameter)
    if hasattr(model, "state_prompt_embedding"):
        state["state_prompt_embedding"] = model.state_prompt_embedding.detach().cpu()

    # 2) adaptor (Module)
    if hasattr(model, "adaptor") and model.adaptor is not None:
        state["adaptor_state_dict"] = model.adaptor.state_dict()

    # 3) fusion params (Parameters)
    if hasattr(model, "scale_logits_pix") and model.scale_logits_pix is not None:
        state["scale_logits_pix"] = model.scale_logits_pix.detach().cpu()

    if hasattr(model, "gamma_pix") and model.gamma_pix is not None:
        state["gamma_pix"] = model.gamma_pix.detach().cpu()

    # 4) 其它你可能后来加的（例如 gate）
    if hasattr(model, "seg_gate_alpha") and model.seg_gate_alpha is not None:
        state["seg_gate_alpha"] = model.seg_gate_alpha.detach().cpu()

    # 5) memorybank（非训练权重，但会影响结果；可选保存，默认保存）
    if hasattr(model, "memorybank") and model.memorybank is not None:
        # memorybank 是 list[tensor]，直接存 cpu
        state["memorybank"] = [mb.detach().cpu() for mb in model.memorybank]

    return state


def save_full_ckpt(model, args, epoch, optimizer=None, metrics=None):
    os.makedirs(args.save_dir, exist_ok=True)

    name = f"{args.dataset}_fs{args.fewshot}_seed{args.seed}_ep{epoch}"
    if args.save_tag:
        name += f"_{args.save_tag}"
    ckpt_path = os.path.join(args.save_dir, name + ".pt")

    ckpt = {
        "epoch": epoch,
        "seed": args.seed,
        "args": vars(args),               # 保存所有超参，复现非常关键
        "trainable_state": extract_trainable_state(model),
    }

    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()

    if metrics is not None:
        ckpt["metrics"] = metrics  # 你可以把 i_auroc/p_pro 等也存进去

    torch.save(ckpt, ckpt_path)
    return ckpt_path


def load_full_ckpt(model, ckpt_path, device="cuda", load_memorybank=True):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    st = ckpt["trainable_state"]

    # prompt
    if "state_prompt_embedding" in st and hasattr(model, "state_prompt_embedding"):
        with torch.no_grad():
            model.state_prompt_embedding.copy_(st["state_prompt_embedding"].to(device))

    # adaptor
    if "adaptor_state_dict" in st and hasattr(model, "adaptor"):
        model.adaptor.load_state_dict(st["adaptor_state_dict"])

    # fusion
    if "scale_logits_pix" in st and hasattr(model, "scale_logits_pix"):
        with torch.no_grad():
            model.scale_logits_pix.copy_(st["scale_logits_pix"].to(device))

    if "gamma_pix" in st and hasattr(model, "gamma_pix"):
        with torch.no_grad():
            model.gamma_pix.copy_(st["gamma_pix"].to(device))

    # optional gate
    if "seg_gate_alpha" in st and hasattr(model, "seg_gate_alpha"):
        with torch.no_grad():
            model.seg_gate_alpha.copy_(st["seg_gate_alpha"].to(device))

    # memorybank (optional)
    if load_memorybank and "memorybank" in st and hasattr(model, "memorybank"):
        model.memorybank = [mb.to(device) for mb in st["memorybank"]]

    return ckpt

def setup_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def l1_loss(inputs, targets, reduction="mean"):
    return F.l1_loss(inputs, targets, reduction=reduction)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')


def patch_alignment_loss(img_tokens, labels, gts):
    gts = gts.reshape(img_tokens[0].size(0), -1)
    labels = labels.reshape(labels.size(0), 1)
    # labels = torch.cat([labels, gts], dim=1)
    new_gts = copy.copy(gts)
    if (len(new_gts[new_gts == 0])) == 0:
        return 0
    new_gts[new_gts == 0] = -1
    b, l = new_gts.size()
    mask = torch.matmul(new_gts.reshape(b, l, 1), new_gts.reshape(b, 1, l))
    total_sim = 0
    for img_token in img_tokens:
        img_token = img_token[:, 1:, :]
        img_token = torch.nn.functional.normalize(img_token, dim=-1)
        sim = torch.matmul(img_token, img_token.permute(0, 2, 1))
        sim = sim[mask == -1].mean() - sim[mask == 1].mean()
        sim = sim if sim > 0 else 0
        total_sim = total_sim + sim
    return total_sim / len(img_tokens)

def build_optimizer(model, args, fewshot_mode: bool):
    # prompt / adaptor / fusion 分组
    param_groups = []

    # 1) prompt
    if hasattr(model, "state_prompt_embedding"):
        param_groups.append({
            "params": [model.state_prompt_embedding],
            "lr": args.lr_prompt,
            "weight_decay": 0.0
        })

    # 2) adaptor
    if hasattr(model, "adaptor") and model.adaptor is not None:
        param_groups.append({
            "params": list(model.adaptor.parameters()),
            "lr": args.lr_adaptor,
            "weight_decay": args.wd_adaptor
        })

    # 3) fusion（few-shot 不训练它）
    if (not fewshot_mode) and hasattr(model, "scale_logits_pix"):
        param_groups.append({
            "params": [model.scale_logits_pix, model.gamma_pix],
            "lr": args.lr_fusion,
            "weight_decay": 0.0
        })

    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    return optimizer

def train(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = get_logger(os.path.join(args.log_dir, '{}_{}_s{}.txt'.format(args.dataset, args.fewshot, args.seed)))
    print_args(logger, args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_transform = load(name=args.model, jit=(not args.model in available_models()), device=device,
                                      download_root=args.clip_download_dir)

    clip_transform.transforms[0] = transforms.Resize(size=(args.img_size, args.img_size),
                                                     interpolation=transforms.InterpolationMode.BICUBIC)
    clip_transform.transforms[1] = transforms.CenterCrop(size=(args.img_size, args.img_size))
    target_transform = transforms.Compose([
        transforms.Resize(size=clip_transform.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    clip_model.eval()

    for param in clip_model.parameters():
        param.requires_grad_(False)


    clip_model = clip_model.to(device)
    clip_model.insert(args=args, tokenizer=tokenize, device=device)

    test_dataset_mvtec = MVTecDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                      gt_target_transform=target_transform)
    test_dataset_isic = ISICDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                    gt_target_transform=target_transform)
    test_dataset_clinic = ClinicDBDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                          gt_target_transform=target_transform)
    test_dataset_colon = ColonDBDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                        gt_target_transform=target_transform)
    test_dataset_visa = VisaDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                    gt_target_transform=target_transform)
    test_dataset_btad = BTADDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                    gt_target_transform=target_transform)
    test_dataset_mpdd = MPDDDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                  gt_target_transform=target_transform)
    test_dataset_mvtec_3d = MVTec3DDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                    gt_target_transform=target_transform)
    test_dataset_brainmri = BrainMRIDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                            gt_target_transform=target_transform)
    test_dataset_br35h = Br35HDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                      gt_target_transform=target_transform)
    test_dataset_dagm = DAGMDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                    gt_target_transform=target_transform)
    test_dataset_kvasir = KvasirDataset(root=args.data_dir, train=False, category=None, transform=clip_transform,
                                        gt_target_transform=target_transform)

    all_test_dataset_dict = {
        "mvtec": test_dataset_mvtec,
        "visa": test_dataset_visa,
        "btad": test_dataset_btad,
        "mpdd": test_dataset_mpdd,
        "mvtec_3d": test_dataset_mvtec_3d,
        'dagm': test_dataset_dagm,
        "isic": test_dataset_isic,
        "clinic": test_dataset_clinic,
        "colon": test_dataset_colon,
        "brainmri": test_dataset_brainmri,
        "br35h": test_dataset_br35h,
        'kvasir': test_dataset_kvasir,
    }
    if len(args.test_dataset) < 1:
        test_dataset_dict = all_test_dataset_dict
    else:
        test_dataset_dict = {}
        for ds_name in args.test_dataset:
            test_dataset_dict[ds_name] = all_test_dataset_dict[ds_name]
    if args.dataset in test_dataset_dict:
        del test_dataset_dict[args.dataset]
    if args.dataset == 'mvtec':
        train_dataset = test_dataset_mvtec
    else:
        train_dataset = test_dataset_visa

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    if args.weight is not None:
        ckpt = load_full_ckpt(clip_model, args.weight, device=device, load_memorybank=True)
        logger.info(f"Loaded ckpt: {args.weight} (epoch={ckpt.get('epoch', 'NA')}, seed={ckpt.get('seed', 'NA')})")
    else:
        fewshot_mode = (args.fewshot > 0)  # fewshot=0 → 0-shot；>0 → few-shot
        optimizer = torch.optim.Adam(clip_model.get_trainable_parameters(fewshot=fewshot_mode), lr=args.lr, betas=(0.5, 0.999))


        for epoch in range(1, args.epochs + 1):
            total_loss = []
            for items in tqdm(train_dataloader):
                imgs, labels, gts = items[:3]
                imgs = imgs.to(device)
                labels = labels.to(device)
                gts = gts.to(device)

                predict_labels, predict_masks, img_tokens = clip_model.detect_forward_seg(imgs, args)
                gts = F.interpolate(gts, size=predict_masks.shape[-2:], mode='bilinear')
                gts = (gts > 0.5).float()

                # =========================
                # 1) 分类 loss
                # =========================
                cls_loss = focal_loss(predict_labels, labels)

                # =========================
                # 2) 分割 loss（主要监督 scale_logits）
                # =========================
                seg_loss = focal_loss(predict_masks, gts) + l1_loss(predict_masks, gts)

                # =========================
                # 3) patch 对齐 loss（弱化）
                # =========================
                align_loss = 0.05 * patch_alignment_loss(img_tokens, labels, gts)

                # =========================
                # 4) 多尺度权重的 L2（单次计算）
                # =========================
                scale_l2 = 0.0002 * torch.sum(clip_model.scale_logits_pix ** 2)

                # =========================
                # 5) gate 正则（方向正确！）
                # 惩罚 gate 巨大，而不是惩罚 gate 本身
                # -------------------------


                # =========================
                # 🎯 最终总损失（正确方式）
                # =========================
                loss = (1.2 * cls_loss) + 1 * seg_loss + 0.01 * align_loss + scale_l2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss.append(loss.item())

            logger.info("Epoch: {}/{}, Loss: {:.6f}".format(epoch, args.epochs, np.mean(total_loss)))
            # 每 save_every 个 epoch 保存一次
            if (epoch % args.save_every) == 0:
                ckpt_path = save_full_ckpt(clip_model, args, epoch, optimizer=optimizer,
                                           metrics={"loss": float(np.mean(total_loss))})
                logger.info(f"[CKPT] Saved: {ckpt_path}")

    final_metrics = None

    for dataset_name, test_ds in test_dataset_dict.items():
        logger.info("---------------------------{}------------------------------".format(dataset_name))
        m = eval_all_class(clip_model, dataset_name, test_ds, args, logger, device)
        logger.info("-------------------------------------------------------------")

        # 记录当前数据集的 average 结果（m 是一个 dict）
        # 如果你只测一个 test_dataset（如 visa），那最后就返回它
        # ✅ 必须真修改：Pixel 可能不存在（如 br35h 无像素 GT 或 eval 未计算 Pixel）
        if m is not None and final_metrics is None:
            # ✅ 必须真修改：固定返回第一个评估数据集，避免被后面覆盖
            final_metrics = m
            final_name = dataset_name

    # ✅ 把某个数据集的平均结果返回
    if final_metrics is None:
        return None

    # 统一成 run_multiseed 需要的键
    # ✅ 必须真修改：Pixel 可能不存在（如 br35h 无像素 GT 或 eval 未计算 Pixel）
    sample = final_metrics.get("Sample_CLS", {})
    pixel = final_metrics.get("Pixel", {})

    return {
        "i_auroc": sample.get("AUROC", float("nan")),
        "i_ap": sample.get("AP", float("nan")),
        "p_auroc": pixel.get("AUROC", float("nan")),
        "p_pro": pixel.get("PRO", float("nan")),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch implemention of AF-CLIP')

    parser.add_argument('--clip_download_dir', type=str, default='./download/clip/', help='training dataset')

    parser.add_argument('--data_dir', type=str, default='./data', help='training dataset')

    parser.add_argument('--dataset', type=str, default='mvtec', help='training dataset', choices=['mvtec', 'visa'])

    parser.add_argument('--model', type=str, default="ViT-L/14@336px", help='model')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning tate')

    parser.add_argument('--alpha', type=float, default=0.1, help='label combination')

    parser.add_argument('--epochs', type=int, default=2, help='training epoch')

    parser.add_argument('--prompt_len', type=int, default=12, help='prompt length')

    parser.add_argument('--category', type=str, default=None, help='normal class')

    parser.add_argument('--fewshot', type=int, default=0, help='few shot num')

    parser.add_argument('--seed', type=int, default=122, help='seed')

    parser.add_argument('--log_dir', type=str, default='./log/', help='log dir')

    parser.add_argument('--suffix', type=str, default='defect', help='prompt suffix')

    parser.add_argument('--img_size', type=int, default=518)

    parser.add_argument('--feature_layers', nargs='+', type=int, default=[6, 12, 18, 24],
                        help='choose vit layers to extract features')

    parser.add_argument('--test_dataset', nargs='+', type=str, default=[], help='choose vit layers to extract features')

    parser.add_argument('--weight', type=str, default=None, help='load weight path')

    parser.add_argument('--vis', type=int, default=0, help='visualization results')

    parser.add_argument('--vis_dir', type=str, default='./vis_results/', help='visualization results dir')

    parser.add_argument('--memory_layers', nargs='+', type=int, default=[6, 12, 18, 24],
                        help='choose resnet layers to store and compare features')

    parser.add_argument('--lambda1', type=float, default=1, help='lambda1 for loss')

    parser.add_argument('--lambda2', type=float, default=1, help='lambda2 for loss')

    parser.add_argument('--lambda_cls', type=float, default=1.5, help='loss weight for classification branch')
    parser.add_argument('--save_dir', type=str, default='./weights/zero', help='checkpoint save dir')
    parser.add_argument('--save_every', type=int, default=1, help='save checkpoint every N epochs')
    parser.add_argument('--save_tag', type=str, default='', help='extra tag in ckpt name')

    args = parser.parse_args()

    args.seed = setup_seed(args.seed)
    train(args)


