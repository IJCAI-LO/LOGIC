from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import math
from clip.adaptor import Adaptor


def gaussian_kernel(size, sigma=2.0):
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    y = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    x, y = torch.meshgrid(x, y, indexing='ij')

    kernel = torch.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        print("resblocks: ", len(self.resblocks))

    def forward(self, x: torch.Tensor, fearure_layers=None, visual_prompt=None):
        out = []
        prefix_len = len(visual_prompt) if visual_prompt is not None else 0
        for i in range(len(self.resblocks)):
            if i < prefix_len:
                x = torch.cat([visual_prompt[i:i + 1].repeat(x.size(0), 1, 1), x], dim=1)
            x = self.resblocks[i](x)
            if i < prefix_len:
                x = x[:, visual_prompt[i:i + 1].size(1):]
            if fearure_layers is not None and i + 1 in fearure_layers:
                out.append(x)
        if fearure_layers is None:
            return x
        else:
            return out


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        print(self.positional_embedding.size())

    def forward(self, x: torch.Tensor, feature_layers=[24], visual_prompt=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        # update the position embedding during inference for varied input size
        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(x.shape[-1], new_side * new_side).transpose(0, 1)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos], 0)

        x = x + self.positional_embedding.to(x.dtype)

        if visual_prompt is not None:
            x = torch.cat([x, visual_prompt[:1].repeat(x.size(0), 1, 1)], dim=1)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        out = self.transformer(x, feature_layers)
        for i, o in enumerate(out):
            out[i] = o.permute(1, 0, 2)
            if visual_prompt is not None:
                out[i] = out[i][:, :-visual_prompt.size(1), :]
        return out
class ScaleFusion(nn.Module):
    def __init__(self, d_model, num_layers, num_scales):
        super().__init__()
        self.weight_mlp = nn.Sequential(
            nn.Linear(d_model, d_model//4),
            nn.ReLU(),
            nn.Linear(d_model//4, 1)   # 输出每个层尺度的权重
        )
        self.num = num_layers * num_scales

    def forward(self, z_list):
        # z_list: 每个 (layer, scale) 的 tokens
        weights = []
        for tokens in z_list:
            cls = tokens[:, 0, :]            # 使用 CLS 统计尺度权重
            w = self.weight_mlp(cls)         # (B,1)
            weights.append(w)

        weights = torch.stack(weights, dim=1)  # (B, L*S, 1)
        weights = torch.softmax(weights, dim=1)
        return weights
class EdgeRefineHead(nn.Module):
    """
    输入:
        img_small: (B,3,H,W)  已缩放到与 seg 相同分辨率的图像
        seg:       (B,1,H,W)  当前的分割概率图 (0~1)
    输出:
        residual:  (B,1,H,W)  对 seg 的残差信号
    """
    def __init__(self, mid_channels=16):
        super().__init__()

        # Sobel 边缘检测（固定参数，不训练）
        self.sobel_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        sobel_x = torch.tensor([[[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]]], dtype=torch.float32)
        with torch.no_grad():
            self.sobel_x.weight.copy_(sobel_x.unsqueeze(0))
            self.sobel_y.weight.copy_(sobel_y.unsqueeze(0))
        for p in self.sobel_x.parameters():
            p.requires_grad = False
        for p in self.sobel_y.parameters():
            p.requires_grad = False

        # 小卷积网络：输入 [seg, image_edge]，输出 Δmask
        # in_channels = 2 (seg + edge_img)
        self.conv = nn.Sequential(
            nn.Conv2d(2, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1)
        )

    def forward(self, img_small, seg):
        """
        img_small: (B,3,H,W)
        seg:       (B,1,H,W)
        """
        # 图像转灰度
        gray = img_small.mean(dim=1, keepdim=True)  # (B,1,H,W)

        # 图像边缘
        edge_x = self.sobel_x(gray)
        edge_y = self.sobel_y(gray)
        edge_img = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-6)  # (B,1,H,W)

        # 拼接 [当前分割, 图像边缘]
        x = torch.cat([seg, edge_img], dim=1)  # (B,2,H,W)

        residual = self.conv(x)  # (B,1,H,W)
        return residual


# 最新 1/9 15：00
class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype
    #  2025.12.5
    def insert(self, args, tokenizer, device):
        # ========= 文本 prompt =========

        # ===== choose prompt mode =====
        use_simple_prompt =False  # True: without/with defect
        # False: multi-template photo prompts

        if use_simple_prompt:
            self.normal_templates = [
                "without defect."
            ]
            self.anomaly_templates = [
                "with defect."
            ]
        else:
            self.normal_templates = [
                "a photo of a normal object.",
                "a photo of an intact object.",
                "a photo without defects.",
                "a photo with no anomaly.",
                "a clean object."
            ]
            self.anomaly_templates = [
                "a photo of a defective object.",
                "a photo with defects.",
                "a photo with an anomaly.",
                "a damaged object.",
                "a photo with an irregularity."
            ]

        self.num_normal = len(self.normal_templates)
        self.num_anomaly = len(self.anomaly_templates)

        # tokenize all prompts (K = num_normal + num_anomaly)
        all_prompts = self.normal_templates + self.anomaly_templates
        self.state_prompt_tokens = tokenizer(all_prompts).to(device)

        self.tokenizer = tokenizer
        self.device = device
        self.prompt_len = args.prompt_len

        self.state_prompt_embedding = nn.Parameter(
            torch.empty(1, args.prompt_len, self.token_embedding.weight.shape[-1]).to(device)
        )
        nn.init.normal_(self.state_prompt_embedding, std=0.01)
        self.state_prompt_embedding.requires_grad_(True)

        # ========= 图像 adaptor =========
        self.adaptor = Adaptor(
            inplanes=self.visual.proj.shape[0],
            outplanes=self.visual.proj.shape[0]
        ).to(device)

        self.memorybank = None
        self.memory_backbone = None

        # ========= 多尺度高斯核（1,3,5,9；1 不需要 kernel） =========
        self.gaussian_kernel = {
            '3': gaussian_kernel(size=3, sigma=4).to(device),
            '5': gaussian_kernel(size=5, sigma=4).to(device),
            '9': gaussian_kernel(size=9, sigma=8).to(device)
        }

        # ========= 多尺度残差融合：A 档线性版 =========
        # 这里假设 feature_layers = [6,12,18,24]，num_scales = 4 → N=16 个分支
        num_scales = 4  # 对应 r=1,3,5,9
        num_layers = len(args.feature_layers)
        self.num_img_tokens = num_scales * num_layers

        # 多尺度权重 logits（softmax 后是 w_i）
        self.scale_logits_pix = nn.Parameter(
            torch.zeros(self.num_img_tokens, dtype=self.dtype, device=device)
        )
        # 残差强度 γ：初始化为 0.08（在 VisA 上你验证过是比较好的）
        self.gamma_pix = nn.Parameter(
            torch.tensor(0.08, dtype=self.dtype, device=device)
        )

    def aggerate_neighbors(self, img_tokens, scales=None):
        """
        scales: list[int], e.g. [1,3,5] or [1] or [1,3,5,9]
        if scales is None -> default [1,3,5,9] (backward compatible)
        """
        if scales is None:
            scales = [1, 3, 5, 9]

        img_token_list = []
        for img_token in img_tokens:
            for r in scales:
                new_img_token = self.aggerate_neighbor(img_token, int(r))
                img_token_list.append(new_img_token)
        return img_token_list

    def encode_state_prompt(self):
        state_x = self.token_embedding(self.state_prompt_tokens).type(self.dtype)
        state_x = torch.cat([self.state_prompt_embedding.repeat(state_x.size(0), 1, 1), state_x], dim=1)[:, :77, :]
        state_x = state_x + self.positional_embedding.type(self.dtype)
        state_x = state_x.permute(1, 0, 2)  # NLD -> LND
        state_x = self.transformer(state_x)
        state_x = state_x.permute(1, 0, 2)  # LND -> NLD
        state_x = self.ln_final(state_x).type(self.dtype)
        state_x = state_x[torch.arange(state_x.shape[0]), self.prompt_len + self.state_prompt_tokens.argmax(
            dim=-1)] @ self.text_projection
        return state_x

    def encode_state_prompt_2class(self):
        """
        将 multi-template prompts 编码后聚合成 2 个向量：
          - normal: 平均(或 logsumexp)所有 normal templates
          - anomaly: 平均(或 logsumexp)所有 anomaly templates
        返回: (2, embed_dim)
        """
        # 1) encode all prompts -> (K, embed_dim)
        x = self.token_embedding(self.state_prompt_tokens).type(self.dtype)
        x = torch.cat([self.state_prompt_embedding.repeat(x.size(0), 1, 1), x], dim=1)[:, :self.context_length, :]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # NLD
        x = self.ln_final(x).type(self.dtype)

        # 取每个 prompt 的 EOT 表示（CLIP 常用 argmax 定位 EOT）
        eot_pos = self.prompt_len + self.state_prompt_tokens.argmax(dim=-1)
        x = x[torch.arange(x.shape[0], device=x.device), eot_pos] @ self.text_projection  # (K, embed_dim)

        # 2) split normal/anomaly
        K = x.size(0)
        assert hasattr(self, "num_normal") and hasattr(self, "num_anomaly"), "call insert() before encoding prompts."
        assert self.num_normal + self.num_anomaly == K, f"K mismatch: {K} vs {self.num_normal}+{self.num_anomaly}"

        x_normal = x[:self.num_normal]  # (Kn, D)
        x_anom = x[self.num_normal:]  # (Ka, D)

        # 3) aggregate
        # 方式1：简单平均（稳定，推荐先用它对齐 baseline）
        # logsumexp aggregation (temperature tau controls sharpness)
        tau = 0.01  # 0.01~0.05 都可；越小越接近 max
        normal_feat = (tau * torch.logsumexp(x_normal / tau, dim=0, keepdim=True))
        anom_feat = (tau * torch.logsumexp(x_anom / tau, dim=0, keepdim=True))

        text_features = torch.cat([normal_feat, anom_feat], dim=0)  # (2,D)
        return text_features
    def get_trainable_parameters(self, fewshot: bool = False):
        """
        fewshot = False：0-shot / 全数据训练，训练所有模块（prompt + adaptor + fusion）
        fewshot = True ：few-shot（1-shot 等），只训练 prompt + adaptor，冻结多尺度残差参数
        """
        params = []

        # 1) 文本 prompt 始终可训练
        params.append(self.state_prompt_embedding)

        # 2) 图像 adaptor 始终可训练
        params += list(self.adaptor.parameters())
        # 3) 多尺度残差模块：
        #    - 0-shot / full data：允许学习
        #    - few-shot：不把它交给 optimizer（仍参与前向，但不被梯度更新）
        if not fewshot:
            params.append(self.scale_logits_pix)
            params.append(self.gamma_pix)

        return params

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_image(self, image, feature_layers=None):
        return self.visual(image.type(self.dtype), feature_layers)

    def aggerate_neighbor(self, x, patchsize, stride=1):
        if patchsize == 1:
            return x
        cls_token = x[:, :1, :]
        x = x[:, 1:, :]
        padding = patchsize // 2
        b, l, c = x.size()
        h = w = int(math.sqrt(l))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        x = torch.nn.functional.unfold(x, kernel_size=patchsize, padding=padding,
                                       stride=stride)  # b, (c * r * r), h * w
        x = x.permute(0, 2, 1).reshape(-1, c, patchsize * patchsize).permute(0, 2, 1)  # (b * h * w,  r * r, c)
        kernel = self.gaussian_kernel[str(patchsize)].reshape(1, patchsize * patchsize, 1)
        x = torch.sum(x * kernel, dim=1).reshape(b, l, c)
        x = torch.cat([cls_token, x], dim=1)
        return x



    def detect_encode_image(self, image, args, scales=None):
        """
        scales: list[int], e.g. [1,3,5], [1], [1,3,5,9]
        """
        img_tokens = self.encode_image(image, args.feature_layers)  # list of (B, 1+L, C)
        img_tokens = self.aggerate_neighbors(img_tokens, scales)
        img_tokens = [self.visual.ln_post(self.adaptor(tok)) @ self.visual.proj for tok in img_tokens]
        return img_tokens
    # 1/4 22：30
    def _fuse_scores(
            self,
            scores_list,
            fusion: str,
            gamma_max: float = 0.30,    # 原 0.30：残差注入上限减半
            clip_mode: str = "l2",  # "l2" or "tanh" or "none"
            clip_c: float = 2.0,  # 原 2.0：收紧残差幅度
            ref_mode: str = "mean",  # "mean" or "base"
            patch_only: bool = True,
            alpha_align: float = 0.3,   # 原 0.3：对齐更温和
            align_target: str = "margin"  # "margin" or "2ch"/"logits"
    ):
        """
        fusion:
          - "sum"   : direct sum (AF-CLIP baseline)
          - "anmsa" : aligned sum (alignment only)
          - "lclrf" : baseline + constrained residual (NO alignment)
          - "full"  : baseline + constrained residual (WITH alignment)

        Key robustness:
          - supports arbitrary channel C (C may be 2 or >2)
          - margin-alignment is ONLY applied when C==2, otherwise falls back to logits alignment

        Returns:
          scores, debug(dict)
        """
        import torch
        import torch.nn.functional as F

        N = len(scores_list)
        assert N > 0, "scores_list is empty."

        # -----------------------
        # baseline sum (semantic anchor)
        # -----------------------
        scores_base = scores_list[0]
        for s in scores_list[1:]:
            scores_base = scores_base + s  # (B,T,C)

        # -----------------------
        # weights + gamma
        # -----------------------
        scale_logits = self.scale_logits_pix[:N]
        scale_w = torch.softmax(scale_logits, dim=0)  # (N,)
        gamma = torch.clamp(self.gamma_pix, 0.0, gamma_max)  # scalar
        # gamma = torch.tensor(0.08, device=self.device, dtype=self.dtype)

        def weighted_sum(lst):
            out = 0.0
            for i, x in enumerate(lst):
                out = out + scale_w[i] * x
            return out

        # -----------------------
        # helpers
        # -----------------------
        def token_slice(x: torch.Tensor):
            # x: (B,T,C)
            return x[:, 1:, :] if patch_only else x

        def stats_over_tokens(x: torch.Tensor):
            # x: (B,T,C')  compute stats across token dim
            x2 = token_slice(x)
            mu = x2.mean(dim=1, keepdim=True)
            sig = x2.std(dim=1, keepdim=True) + 1e-6
            return mu, sig

        # -----------------------
        # reference distribution
        # -----------------------
        if ref_mode == "mean":
            ref = scores_list[0]
            for s in scores_list[1:]:
                ref = ref + s
            ref = ref / float(N)
        elif ref_mode == "base":
            ref = scores_base / float(N)
        else:
            raise ValueError(f"Unknown ref_mode: {ref_mode}")

        # -----------------------
        # determine channel count + effective alignment target
        # -----------------------
        C = int(scores_list[0].size(-1))
        # margin alignment only valid when C==2
        if align_target == "margin" and C != 2:
            align_target_eff = "logits"  # fallback
        else:
            align_target_eff = align_target

        # -----------------------
        # alignment function
        # -----------------------
        if align_target_eff == "margin":
            # margin stats on ref
            mref = ref[..., 1:2] - ref[..., 0:1]  # (B,T,1)
            mu_ref_m, sig_ref_m = stats_over_tokens(mref)

            def align_one(s: torch.Tensor):
                # s: (B,T,2)
                sn = s[..., 0:1]
                sa = s[..., 1:2]
                m = sa - sn

                mu_m, sig_m = stats_over_tokens(m)
                m_aligned = (m - mu_m) / sig_m * sig_ref_m + mu_ref_m

                sa_new = sn + m_aligned
                s_aligned = torch.cat([sn, sa_new], dim=-1)  # (B,T,2)

                # soft alignment
                return (1 - alpha_align) * s + alpha_align * s_aligned

        elif align_target_eff in ["2ch", "logits"]:
            mu_ref, sig_ref = stats_over_tokens(ref)

            def align_one(s: torch.Tensor):
                mu_s, sig_s = stats_over_tokens(s)
                s_aligned = (s - mu_s) / sig_s * sig_ref + mu_ref
                return (1 - alpha_align) * s + alpha_align * s_aligned
        else:
            raise ValueError(f"Unknown align_target: {align_target}")

        # -----------------------
        # residual constraint (approx Lipschitz control)
        # -----------------------
        def constrain_residual(r: torch.Tensor):
            if clip_mode == "none":
                return r
            if clip_mode == "tanh":
                return clip_c * torch.tanh(r / clip_c)
            if clip_mode == "l2":
                B = r.size(0)
                r_flat = r.view(B, -1)
                norm = torch.norm(r_flat, p=2, dim=1, keepdim=True) + 1e-6
                scale = torch.clamp(clip_c / norm, max=1.0)
                return (r_flat * scale).view_as(r)
            raise ValueError(f"Unknown clip_mode: {clip_mode}")

        # -----------------------
        # fusion modes
        # -----------------------
        aligned_list = None
        residual_raw = None
        residual = None

        if fusion == "sum":
            scores = scores_base

        elif fusion == "anmsa":
            aligned_list = [align_one(s) for s in scores_list]
            scores = aligned_list[0]
            for s in aligned_list[1:]:
                scores = scores + s
            # for debug (alignment-only)
            residual_raw = weighted_sum(aligned_list)
            residual = residual_raw

        elif fusion == "lclrf":
            raw_mix = weighted_sum(scores_list)
            raw_mean = scores_base / float(N)
            residual_raw = raw_mix - raw_mean
            residual = constrain_residual(residual_raw)
            scores = scores_base + gamma * residual

        elif fusion == "full":
            aligned_list = [align_one(s) for s in scores_list]
            aligned_mix = weighted_sum(aligned_list)

            aligned_mean = aligned_list[0]
            for s in aligned_list[1:]:
                aligned_mean = aligned_mean + s
            aligned_mean = aligned_mean / float(N)

            residual_raw = aligned_mix - aligned_mean
            residual = constrain_residual(residual_raw)
            scores = scores_base + gamma * residual

        else:
            raise ValueError(f"Unknown fusion: {fusion}")

        # -----------------------
        # margins (only meaningful for C>=2)
        # -----------------------
        def to_margin(x: torch.Tensor):
            # define margin only when at least 2 channels exist
            # For C>2, this is still a useful proxy: class1 - class0
            if x.size(-1) < 2:
                return None
            return x[..., 1:2] - x[..., 0:1]  # (B,T,1)

        margins_list = [to_margin(s) for s in scores_list]  # list[(B,T,1) or None]
        aligned_margins_list = None
        if aligned_list is not None:
            aligned_margins_list = [to_margin(s) for s in aligned_list]

        ref_margin = to_margin(ref)

        # -----------------------
        # debug package (paper-ready)
        # -----------------------
        debug = {
            # meta
            "fusion": fusion,
            "N": N,
            "C": C,

            # core logits
            "scores_list": scores_list,  # list of (B,T,C)
            "scores_base": scores_base,  # (B,T,C)
            "scores_final": scores,  # (B,T,C)

            # weights
            "scale_w": scale_w,  # (N,)
            "scale_logits": scale_logits,  # (N,)
            "gamma": gamma,  # scalar

            # alignment meta
            "ref_mode": ref_mode,
            "patch_only": patch_only,
            "alpha_align": alpha_align,
            "align_target": align_target,
            "align_target_eff": align_target_eff,

            # residuals
            "residual_raw": residual_raw,  # (B,T,C) or None
            "residual": residual,  # (B,T,C) or None

            # reference
            "ref": ref,  # (B,T,C)
            "ref_margin": ref_margin,  # (B,T,1) or None

            # aligned branches (for Raw vs Aligned matrix)
            "aligned_list": aligned_list,  # list of (B,T,C) or None
            "margins_list": margins_list,  # list of (B,T,1) or None
            "aligned_margins_list": aligned_margins_list,  # list of (B,T,1) or None

            # constraint meta
            "clip_mode": clip_mode,
            "clip_c": clip_c,
            "gamma_max": gamma_max,
        }

        return scores, debug

    def store_memory(self, image, args):
        img_tokens = self.encode_image(image, args.memory_layers)
        img_tokens = self.aggerate_neighbors(img_tokens)
        b, l, c = img_tokens[0].size()
        self.memorybank = [torch.nn.functional.normalize(img_token[:, 1:], dim=-1).reshape(-1, c) for img_token in
                           img_tokens]

    def detect_forward_seg(self, image, args, return_debug=False, ablation: str = "E"):
        """
        Ablations (as you finalized):
          A Base (AF-CLIP): scales=[1,3,5],   fusion="sum"
          B single-full   : scales=[1],       fusion="full"
          C ANMSA         : scales=[1,3,5,9], fusion="anmsa"
          D LCLRF         : scales=[1,3,5,9], fusion="lclrf"
          E Full          : scales=[1,3,5,9], fusion="full"
        """

        # -------- 0) choose (scales, fusion) --------
        if ablation == "A":
            scales = [1, 3, 5]
            fusion = "sum"
        elif ablation == "B":
            scales = [1]
            fusion = "full"
        elif ablation == "C":
            scales = [1, 3, 5, 9]
            fusion = "anmsa"
        elif ablation == "D":
            scales = [1, 3, 5, 9]
            fusion = "lclrf"
        elif ablation == "E":
            scales = [1, 3, 5, 9]
            fusion = "full"
        else:
            raise ValueError(f"Unknown ablation: {ablation}, choose from ['A','B','C','D','E'].")

        # -------- 1) text features --------
        text_features = self.encode_state_prompt()  # (2, C)
        text_features = F.normalize(text_features, dim=-1)  # (2, C)
        # # -------- 1) text features (2-class aggregated; critical fix) --------
        # text_features = self.encode_state_prompt_2class()  # (2, C): [normal, anomaly]
        # text_features = F.normalize(text_features, dim=-1)

        # -------- 2) image tokens with selected scales --------
        img_tokens = self.detect_encode_image(image, args, scales=scales)  # list of (B, 1+L, C)

        # -------- 3) logits per branch --------
        scores_list = []
        for tok in img_tokens:
            tok = F.normalize(tok, dim=-1)
            score = torch.matmul(tok, text_features.T) / 0.07  # (B, 1+L, 2)
            scores_list.append(score)

        # -------- 4) fuse --------
        scores, fuse_debug = self._fuse_scores(scores_list, fusion=fusion)

        # -------- 5) Softmax -> cls + seg --------
        prob = F.softmax(scores, dim=-1)  # (B, 1+L, 2)
        cls_label = prob[:, 0, 1].view(-1)

        predict_map = prob[:, 1:, 1]  # (B, L)
        b, l = predict_map.size()
        h = w = int(math.sqrt(l))
        predict_map = predict_map.reshape(b, 1, h, w)

        if return_debug:
            debug = {
                "ablation": ablation,
                "scales": scales,
                "fusion": fusion,
                "scores_list": [s.detach().float().cpu() for s in scores_list],
                "scores_base": fuse_debug["scores_base"].detach().float().cpu(),
                "scores_final": fuse_debug["scores_final"].detach().float().cpu(),
                "scale_w": fuse_debug["scale_w"].detach().float().cpu(),
                "scale_logits": fuse_debug["scale_logits"].detach().float().cpu(),
                "gamma": fuse_debug["gamma"].detach().float().cpu(),
            }

            # ✅ 关键修复：不仅要判断 key，还要判断 value 不是 None
            if ("residual" in fuse_debug) and (fuse_debug["residual"] is not None):
                debug["residual"] = fuse_debug["residual"].detach().float().cpu()

            # （可选但推荐）你新加的 residual_raw 也做同样处理
            if ("residual_raw" in fuse_debug) and (fuse_debug["residual_raw"] is not None):
                debug["residual_raw"] = fuse_debug["residual_raw"].detach().float().cpu()

            # （可选但强烈推荐）保存 aligned_list，后面才能画 Raw vs Aligned 矩阵
            if ("aligned_list" in fuse_debug) and (fuse_debug["aligned_list"] is not None):
                debug["aligned_list"] = [a.detach().float().cpu() for a in fuse_debug["aligned_list"]]

            # （可选）margin 也一起存，距离矩阵/对齐证据图会更硬
            if ("margins_list" in fuse_debug) and (fuse_debug["margins_list"] is not None):
                debug["margins_list"] = [m.detach().float().cpu() for m in fuse_debug["margins_list"]]
            if ("aligned_margins_list" in fuse_debug) and (fuse_debug["aligned_margins_list"] is not None):
                debug["aligned_margins_list"] = [m.detach().float().cpu() for m in fuse_debug["aligned_margins_list"]]

            return cls_label, predict_map, img_tokens, debug

        return cls_label, predict_map, img_tokens

    def detect_forward_memorybank(self, image, args):
        scores = 0
        img_tokens = self.encode_image(image, args.memory_layers)
        img_tokens = self.aggerate_neighbors(img_tokens)

        k_patch = 5  # ✅ patch 级 Top-K（3~7 都可以，5 是经验最稳）
        k_img = 20  # ✅ image 级 Top-K（patch 数的 2%~5%）

        for i, img_token in enumerate(img_tokens):
            img_token = F.normalize(img_token, dim=-1)  # (B, 1+L, C)
            sim = torch.matmul(img_token, self.memorybank[i].T)  # (B, 1+L, N_mem)
            dist = 1 - sim  # anomaly distance

            # ===== patch-level: Top-K mean instead of min =====
            topk_patch = torch.topk(dist, k=k_patch, dim=-1, largest=False)[0]
            score = topk_patch.mean(dim=-1) / 2  # (B, 1+L)

            scores += score[:, 1:]  # drop CLS

        scores = scores / len(img_tokens)  # (B, L)

        # ===== image-level: Top-K mean instead of max =====
        k_img_eff = min(k_img, scores.size(1))
        cls_label = torch.topk(scores, k=k_img_eff, dim=-1)[0].mean(dim=-1)

        b, l = scores.size()
        h = w = int(math.sqrt(l))
        predict_map = scores.reshape(b, 1, h, w)

        return cls_label, predict_map

    def detect_forward(self, image, args):
        cls_label, predict_map, _ = self.detect_forward_seg(image, args)
        if self.memorybank is not None:
            cls_label_memory, predict_map_memory = self.detect_forward_memorybank(image, args)
            predict_map = predict_map_memory + args.alpha * predict_map
            cls_label = cls_label_memory + args.alpha * cls_label
        return cls_label, predict_map

    def forward(self, image, text):
        image_features = self.encode_image(image)
        if isinstance(image_features, (list, tuple)):
            image_features = image_features[0]
        text_features = self.encode_text(text)
        if isinstance(text_features, (list, tuple)):
            text_features = text_features[0]

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()