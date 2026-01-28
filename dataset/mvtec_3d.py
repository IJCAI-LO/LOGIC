"""
  @Author: 王权
  @FileName: mvtec_3d.py
  @DateTime: 2026/1/18 21:40
  @SoftWare: PyCharm
"""
import os
import glob
import numpy as np
from torch.utils.data import Dataset
from PIL import Image


class MVTec3DDataset(Dataset):
    """
    MVTec-3D dataset loader with the SAME interface/style as your MVTecDataset:
      - preprocess() builds img_paths/gt_paths/labels dict for train/test
      - update(category) selects current subset
      - fewshot sampling on TRAIN split
      - __getitem__ returns: (img, label, gt, category, img_path)
    """

    def __init__(self, root, train=True, category=None, fewshot=0,
                 transform=None, gt_target_transform=None):
        super().__init__()

        # ✅ 你需要把这里的类别名改成你本地 mvtec-3d 的真实类别文件夹名
        # （不同 release 可能不一样；你可以先 ls root/mvtec_3d 看看）
        self.categories = [
            # 常见示例（请按你的实际目录修改）
            "bagel", "cable_gland", "carrot", "cookie",
            "dowel", "foam", "peach", "potato", "rope", "tire"
        ]

        self.train = train
        self.category = category
        self.fewshot = fewshot
        self.root = os.path.join(root, "mvtec_3d_anomaly_detection")  # ✅ 根目录约定
        self.transform = transform
        self.gt_target_transform = gt_target_transform

        self.preprocess()
        self.update(self.category)

        assert len(self.cur_img_paths) == len(self.cur_img_labels)
        assert len(self.cur_img_paths) == len(self.cur_img_categories)
        assert len(self.cur_img_paths) == len(self.cur_gt_paths)

        self.dataset_name = "mvtec_3d_anomaly_detection"

    @staticmethod
    def _list_images(folder):
        exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
        paths = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(folder, e)))
        paths.sort()
        return paths

    def _find_gt_dir(self, img_dir, defect_type):
        """
        Try common GT folder names in different MVTec-3D releases.
        Return the first existing path, else None.
        """
        candidates = [
            os.path.join(img_dir, "ground_truth", defect_type),
            os.path.join(img_dir, "gt", defect_type),
            os.path.join(img_dir, "mask", defect_type),
            os.path.join(img_dir, "masks", defect_type),
            os.path.join(img_dir, "ground_truth"),   # sometimes no defect subfolder
            os.path.join(img_dir, "gt"),
            os.path.join(img_dir, "mask"),
            os.path.join(img_dir, "masks"),
        ]
        for p in candidates:
            if os.path.isdir(p):
                return p
        return None

    def preprocess(self):
        self.img_paths = {"train": {c: [] for c in self.categories},
                          "test":  {c: [] for c in self.categories}}
        self.gt_paths = {"train": {c: [] for c in self.categories},
                         "test":  {c: [] for c in self.categories}}
        self.labels = {"train": {c: [] for c in self.categories},
                       "test":  {c: [] for c in self.categories}}

        for phase in ["train", "test"]:
            for category in self.categories:
                img_dir = os.path.join(self.root, category)

                phase_dir = os.path.join(img_dir, phase)
                if not os.path.isdir(phase_dir):
                    raise FileNotFoundError(f"[MVTec3D] Missing: {phase_dir}")

                defect_types = sorted(os.listdir(phase_dir))
                for defect_type in defect_types:
                    defect_dir = os.path.join(phase_dir, defect_type)
                    if not os.path.isdir(defect_dir):
                        continue

                    img_paths = self._list_images(defect_dir)

                    if defect_type == "good":
                        self.img_paths[phase][category].extend(img_paths)
                        self.gt_paths[phase][category].extend([None] * len(img_paths))
                        self.labels[phase][category].extend([0] * len(img_paths))
                    else:
                        # abnormal: try find gt folder
                        gt_dir = self._find_gt_dir(img_dir, defect_type)
                        if gt_dir is None:
                            # 没找到 GT，就用 None（不会崩，但你会拿不到像素监督）
                            gt_paths = [None] * len(img_paths)
                        else:
                            gt_paths = self._list_images(gt_dir)

                            # 若 GT 数量不匹配，尝试按文件名对齐（更稳）
                            if len(gt_paths) != len(img_paths):
                                gt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in gt_paths}
                                aligned = []
                                for ip in img_paths:
                                    key = os.path.splitext(os.path.basename(ip))[0]
                                    aligned.append(gt_map.get(key, None))
                                gt_paths = aligned

                        self.img_paths[phase][category].extend(img_paths)
                        self.gt_paths[phase][category].extend(gt_paths)
                        self.labels[phase][category].extend([1] * len(img_paths))

    def update(self, category=None):
        self.category = category
        self.cur_img_paths, self.cur_gt_paths = [], []
        self.cur_img_labels, self.cur_img_categories = [], []

        phase = "train" if self.train else "test"

        if self.category is not None:
            self.cur_img_paths = list(self.img_paths[phase][self.category])
            self.cur_gt_paths = list(self.gt_paths[phase][self.category])
            self.cur_img_labels = list(self.labels[phase][self.category])
            self.cur_img_categories = [self.category] * len(self.cur_img_paths)
        else:
            for c in self.categories:
                self.cur_img_paths.extend(self.img_paths[phase][c])
                self.cur_gt_paths.extend(self.gt_paths[phase][c])
                self.cur_img_labels.extend(self.labels[phase][c])
                self.cur_img_categories.extend([c] * len(self.img_paths[phase][c]))

        # ✅ few-shot：只对训练集做采样（与你的 MVTecDataset 一致）
        if self.train and self.fewshot and self.fewshot > 0:
            assert self.fewshot <= len(self.cur_img_paths), \
                f"fewshot={self.fewshot} > num_train={len(self.cur_img_paths)}"
            randidx = np.random.choice(len(self.cur_img_paths), size=self.fewshot, replace=False)
            self.cur_img_paths = [self.cur_img_paths[i] for i in randidx]
            self.cur_gt_paths = [self.cur_gt_paths[i] for i in randidx]
            self.cur_img_labels = [self.cur_img_labels[i] for i in randidx]
            self.cur_img_categories = [self.cur_img_categories[i] for i in randidx]

    def __len__(self):
        return len(self.cur_img_paths)

    def __getitem__(self, idx):
        category = self.cur_img_categories[idx]
        img_path = self.cur_img_paths[idx]
        label = self.cur_img_labels[idx]

        img = Image.open(img_path).convert("RGB")

        gt_path = self.cur_gt_paths[idx]
        if gt_path is not None and os.path.isfile(gt_path):
            gt = np.array(Image.open(gt_path))
            # 兼容 RGB mask / 单通道 mask
            if gt.ndim == 3:
                gt = gt[..., 0]
        else:
            gt = np.zeros((img.size[1], img.size[0]), dtype=np.uint8)

        gt = Image.fromarray(gt)

        if self.transform is not None:
            img = self.transform(img)
        if self.gt_target_transform is not None:
            gt = self.gt_target_transform(gt)

        return img, label, gt, category, img_path
