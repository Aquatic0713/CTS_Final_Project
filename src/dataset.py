import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2  # ★★★ 核心武器：OpenCV 用於 CLAHE (對比度限制直方圖均衡化)

# ===== 設定資料路徑 =====
DATA_ROOT = r"D:\AI\CTS_dataset"  # 資料集的根目錄

# 類別 ID 對應表
# 0 = 背景 (Background), 1 = 正中神經 (MN), 2 = 屈肌腱 (FT), 3 = 腕隧道 (CT)
CLASS_IDS = {
    "MN": 1,
    "FT": 2,
    "CT": 3,
}

def _load_gray_normalized(path: str) -> np.ndarray:
    """
    讀取灰階圖 (T1/T2) 並做 CLAHE 增強 + 正規化。
    ★ 這是讓 MN/FT 在暗處也能被模型看清楚的關鍵技術。
    """
    # 1. 讀取圖片並轉為灰階 (L mode)
    img = Image.open(path).convert("L")
    arr_u8 = np.array(img, dtype=np.uint8)
    
    # ★★★ 應用 CLAHE (對比度增強) ★★★
    # clipLimit=2.0: 限制對比度增強的倍率，避免把雜訊也放大
    # tileGridSize=(8, 8): 將圖片切成 8x8 的小區塊分別做均衡化，讓細節更明顯
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr_enhanced = clahe.apply(arr_u8)
    
    # 3. 正規化到 0.0 ~ 1.0 (深度學習模型通常喜歡吃 0-1 之間的浮點數)
    arr_float = arr_enhanced.astype(np.float32) / 255.0
    return arr_float

def _load_binary_mask(path: str) -> np.ndarray:
    """
    讀取標註圖 (MN/FT/CT)，轉為 0 或 1 的二值遮罩。
    """
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    # 只要像素值 > 127 就視為該組織 (1)，否則為背景 (0)
    mask = (arr > 127).astype(np.uint8) 
    return mask

def random_flip_rotate(img: np.ndarray, mask: np.ndarray):
    """
    資料增強 (Data Augmentation)：隨機翻轉與旋轉。
    ★ 目的是讓訓練資料變多樣，防止模型「背答案」(Overfitting)。
    """
    # 50% 機率做水平翻轉
    if random.random() < 0.5:
        img = img[:, :, ::-1]   # 影像翻轉
        mask = mask[:, ::-1]    # 標籤也要跟著翻

    # 50% 機率做垂直翻轉
    if random.random() < 0.5:
        img = img[:, ::-1, :]
        mask = mask[::-1, :]

    # 隨機旋轉 0, 90, 180, 270 度
    k = random.randint(0, 3)
    if k > 0:
        # rot90 是 numpy 的旋轉函數，k 是旋轉次數
        img = np.rot90(img, k, axes=(1, 2))  # 影像維度是 (C, H, W)，所以在 (1,2) 平面旋轉
        mask = np.rot90(mask, k, axes=(0, 1)) # 標籤維度是 (H, W)，所以在 (0,1) 平面旋轉

    return img, mask

class CTSDatasetV5(Dataset):
    def __init__(
        self,
        root: str = DATA_ROOT,
        case_ids=None,
        augment: bool = False, # 只有訓練時設為 True
        debug=False
    ):
        super().__init__()
        self.root = root
        self.augment = augment
        self.debug = debug

        # 自動掃描資料夾 (0~9)，找出所有病例資料夾
        if case_ids is None:
            case_ids = [
                d for d in os.listdir(root)
                if d.isdigit() and os.path.isdir(os.path.join(root, d))
            ]
            case_ids = sorted(case_ids, key=lambda x: int(x)) # 按數字排序
        self.case_ids = case_ids

        self.samples = [] # 用來存放所有切片的路徑清單
        self._build_index()

    def _build_index(self):
        """建立索引：掃描所有資料夾，把 T1, T2, MN, FT, CT 的路徑配對好存起來"""
        self.samples.clear()
        for cid in self.case_ids:
            case_dir = os.path.join(self.root, cid)
            t1_dir = os.path.join(case_dir, "T1")
            
            if not os.path.isdir(t1_dir): continue

            # 遍歷 T1 資料夾中的每一張圖
            for fname in os.listdir(t1_dir):
                if not fname.lower().endswith((".jpg", ".png")): continue
                
                # 解析切片索引 (例如 1.jpg -> 1)
                try:
                    slice_idx = int(os.path.splitext(fname)[0])
                except ValueError:
                    slice_idx = -1

                # 組合各個組織的路徑
                t1_path = os.path.join(case_dir, "T1", fname)
                t2_path = os.path.join(case_dir, "T2", fname)
                mn_path = os.path.join(case_dir, "MN", fname)
                ft_path = os.path.join(case_dir, "FT", fname)
                ct_path = os.path.join(case_dir, "CT", fname)

                # 確保 5 個檔案都存在才加入清單 (防呆)
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

        # 再次排序，確保讀取順序是 0/1.jpg, 0/2.jpg... (這對 GUI 拉動條很重要)
        self.samples.sort(key=lambda x: (int(x["case_id"]), x["slice_idx"]))

    def __len__(self):
        return len(self.samples)

    def _build_multiclass_label(self, mn_path, ft_path, ct_path):
        """
        ★ 核心邏輯：將三個分開的黑白 Mask 疊加成一張多類別 Mask (0,1,2,3)
        重要順序：越小的組織 (MN) 要越晚畫，避免被大組織 (CT) 蓋掉
        """
        mn = _load_binary_mask(mn_path)
        ft = _load_binary_mask(ft_path)
        ct = _load_binary_mask(ct_path)

        h, w = mn.shape
        label = np.zeros((h, w), dtype=np.uint8) # 初始化全黑背景

        # 1. 先畫最大的 CT (ID=3) - 最底層
        label[ct == 1] = CLASS_IDS["CT"]
        # 2. 再畫 FT (ID=2) - 中間層
        label[ft == 1] = CLASS_IDS["FT"]
        # 3. 最後畫最小的 MN (ID=1) - 最上層，權重最高，必須確保它顯示出來
        label[mn == 1] = CLASS_IDS["MN"]

        return label

    def __getitem__(self, idx):
        """PyTorch DataLoader 每次抓資料時會呼叫這個函式"""
        s = self.samples[idx]

        # 1. 讀取影像 (Input) -> 已經做過 CLAHE 和 正規化
        t1 = _load_gray_normalized(s["t1"])
        t2 = _load_gray_normalized(s["t2"])
        
        # 堆疊 T1 和 T2 變成 (2, 512, 512) 的 Input
        img = np.stack([t1, t2], axis=0).astype(np.float32)

        # 2. 讀取並組合標籤
        label = self._build_multiclass_label(s["mn"], s["ft"], s["ct"])

        # 3. 如果是訓練模式，進行資料增強 (翻轉/旋轉)
        if self.augment:
            img, label = random_flip_rotate(img, label)

        # 4. 轉成記憶體連續的陣列 (避免 PyTorch 報錯)
        img = np.ascontiguousarray(img)
        label = np.ascontiguousarray(label)

        # 轉成 PyTorch Tensor
        img_tensor = torch.from_numpy(img)
        label_tensor = torch.from_numpy(label).long() # Label 必須是 Long 型態 (整數)

        return img_tensor, label_tensor
