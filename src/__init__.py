import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2  # ★★★ 核心武器：OpenCV 用於 CLAHE

# ===== 請確認這裡的路徑正確 =====
DATA_ROOT = r"D:\AI\CTS_dataset"

# 類別 ID 對應
# 0 = 背景, 1 = MN, 2 = FT, 3 = CT
CLASS_IDS = {
    "MN": 1,
    "FT": 2,
    "CT": 3,
}

def _load_gray_normalized(path: str) -> np.ndarray:
    """
    讀取灰階圖 (T1/T2) 並做 CLAHE 增強 + 正規化。
    這是讓 MN/FT 分數暴漲的關鍵。
    """
    # 1. 讀取原始影像 (0-255)
    img = Image.open(path).convert("L")
    arr_u8 = np.array(img, dtype=np.uint8)
    
    # ★★★ 應用 CLAHE (對比度增強) ★★★
    # clipLimit=2.0 讓細節浮現，但不過度曝光
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr_enhanced = clahe.apply(arr_u8)
    
    # 3. 正規化到 0.0 ~ 1.0
    arr_float = arr_enhanced.astype(np.float32) / 255.0
    return arr_float

def _load_binary_mask(path: str) -> np.ndarray:
    """
    讀取標註圖 (MN/FT/CT)，轉為 0 或 1。
    """
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    mask = (arr > 127).astype(np.uint8) 
    return mask

def random_flip_rotate(img: np.ndarray, mask: np.ndarray):
    """
    資料增強：隨機翻轉與旋轉 (訓練時增加多樣性)
    """
    # 水平翻轉
    if random.random() < 0.5:
        img = img[:, :, ::-1]
        mask = mask[:, ::-1]

    # 垂直翻轉
    if random.random() < 0.5:
        img = img[:, ::-1, :]
        mask = mask[::-1, :]

    # 隨機旋轉 0/90/180/270
    k = random.randint(0, 3)
    if k > 0:
        img = np.rot90(img, k, axes=(1, 2))
        mask = np.rot90(mask, k, axes=(0, 1))

    return img, mask

class CTSDatasetV5(Dataset):
    def __init__(
        self,
        root: str = DATA_ROOT,
        case_ids=None,
        augment: bool = False,
        debug=False
    ):
        super().__init__()
        self.root = root
        self.augment = augment
        self.debug = debug

        # 掃描 case (0~9)
        if case_ids is None:
            case_ids = [
                d for d in os.listdir(root)
                if d.isdigit() and os.path.isdir(os.path.join(root, d))
            ]
            case_ids = sorted(case_ids, key=lambda x: int(x))
        self.case_ids = case_ids

        self.samples = []
        self._build_index()

    def _build_index(self):
        self.samples.clear()
        for cid in self.case_ids:
            case_dir = os.path.join(self.root, cid)
            t1_dir = os.path.join(case_dir, "T1")
            
            if not os.path.isdir(t1_dir):
                continue

            for fname in os.listdir(t1_dir):
                if not fname.lower().endswith((".jpg", ".png")):
                    continue
                
                try:
                    slice_idx = int(os.path.splitext(fname)[0])
                except ValueError:
                    slice_idx = -1

                t1_path = os.path.join(case_dir, "T1", fname)
                t2_path = os.path.join(case_dir, "T2", fname)
                mn_path = os.path.join(case_dir, "MN", fname)
                ft_path = os.path.join(case_dir, "FT", fname)
                ct_path = os.path.join(case_dir, "CT", fname)

                # 確保檔案都存在
                if (os.path.exists(t2_path) and os.path.exists(mn_path)
                        and os.path.exists(ft_path) and os.path.exists(ct_path)):
                    self.samples.append({
                        "case_id": cid,
                        "slice_idx": slice_idx,
                        "t1": t1_path,
                        "t2": t2_path,
                        "mn": mn_path,
                        "ft": ft_path,
                        "ct": ct_path,
                    })

        # 排序：這對 GUI 很重要，訓練時沒差但保持一致比較好
        self.samples.sort(key=lambda x: (int(x["case_id"]), x["slice_idx"]))

    def __len__(self):
        return len(self.samples)

    def _build_multiclass_label(self, mn_path, ft_path, ct_path):
        """
        組合三個 Mask 成為單一張多類別標籤 (0,1,2,3)
        重要順序：越小的組織越晚畫，避免被覆蓋
        """
        mn = _load_binary_mask(mn_path)
        ft = _load_binary_mask(ft_path)
        ct = _load_binary_mask(ct_path)

        h, w = mn.shape
        label = np.zeros((h, w), dtype=np.uint8)

        # 1. 先畫最大的 CT (ID=3)
        label[ct == 1] = CLASS_IDS["CT"]
        # 2. 再畫 FT (ID=2)
        label[ft == 1] = CLASS_IDS["FT"]
        # 3. 最後畫最小的 MN (ID=1) - 權重最高，必須確保它存在
        label[mn == 1] = CLASS_IDS["MN"]

        return label

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 1. 讀取影像 (Input) -> 包含 CLAHE
        t1 = _load_gray_normalized(s["t1"])
        t2 = _load_gray_normalized(s["t2"])
        
        img = np.stack([t1, t2], axis=0).astype(np.float32)

        # 2. 讀取標籤
        label = self._build_multiclass_label(s["mn"], s["ft"], s["ct"])

        # 3. 資料增強 (Train 才有，Test/Val 會關閉)
        if self.augment:
            img, label = random_flip_rotate(img, label)

        # 4. 轉 Tensor (確保記憶體連續，防止報錯)
        img = np.ascontiguousarray(img)
        label = np.ascontiguousarray(label)

        img_tensor = torch.from_numpy(img)
        label_tensor = torch.from_numpy(label).long()

        return img_tensor, label_tensor