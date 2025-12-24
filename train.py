import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp  # ★★★ 核心武器：使用現成的強大分割模型庫

from dataset_cts_v5 import CTSDatasetV5, DATA_ROOT

# ===== 1. 參數設定 =====
BATCH_SIZE = 1  # 醫學影像較大，顯卡記憶體有限時設為 1
EPOCHS = 100    # 訓練總回數
LEARNING_RATE = 1e-4  # ★ 使用預訓練模型時，學習率要調小 (1e-4)，避免破壞學好的特徵
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 權重策略：MN 極小，權重加重 (10.0)，強迫模型關注神經
# [背景, MN, FT, CT]
CLASS_WEIGHTS = torch.tensor([0.5, 10.0, 2.0, 2.0], device=DEVICE)

CHECKPOINT_DIR = "checkpoints_sota"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ===== 2. 定義 SOTA 模型 (使用 SMP) =====
class CTSModel(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        # 使用 U-Net++ 架構 (比 U-Net 更強，有更密集的跳接)
        # 搭配 EfficientNet-B3 骨幹 (特徵提取能力強)
        # encoder_weights="imagenet": 使用 ImageNet 預訓練權重 (轉移學習)
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",  
            encoder_weights="imagenet",      
            in_channels=2,                   # 輸入通道 = 2 (T1 + T2)
            classes=n_classes,               # 輸出類別 = 4
        )

    def forward(self, x):
        return self.model(x)

# ===== 3. 評估函式 (Dice Score) =====
def compute_dice_union(pred, target, cls_id):
    """計算 Dice Coefficient"""
    if cls_id == 3: # CT (腕隧道) 的特殊邏輯：Union
        # CT 包含了 MN, FT, CT 所有區域，所以只要預測是 1, 2, 或 3 都算在 CT 範圍內
        p = (pred == 1) | (pred == 2) | (pred == 3)
        t = (target == 1) | (target == 2) | (target == 3)
    else:
        # 其他組織 (MN, FT) 就正常計算
        p = (pred == cls_id)
        t = (target == cls_id)
    
    # 交集 (Intersection)
    inter = (p & t).sum()
    # 聯集 (Sum of areas)
    union = p.sum() + t.sum()
    
    if union == 0: return 1.0 # 兩邊都是空的，算滿分
    return 2.0 * inter / (union + 1e-6) # 加上 1e-6 避免除以零

# ===== 4. 訓練單一 Fold (一折) =====
def train_one_fold(fold_idx, train_ids, val_ids, test_ids):
    print(f"\n⚡ Fold {fold_idx+1}/5 | Train:{train_ids} | Val:{val_ids} | Test:{test_ids}")
    
    # 準備 Dataset
    train_ds = CTSDatasetV5(root=DATA_ROOT, case_ids=train_ids, augment=True) # 訓練集要開增強
    val_ds   = CTSDatasetV5(root=DATA_ROOT, case_ids=val_ids, augment=False)
    test_ds  = CTSDatasetV5(root=DATA_ROOT, case_ids=test_ids, augment=False)
    
    # 準備 DataLoader
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

    # 初始化模型、優化器、排程器
    model = CTSModel(n_classes=4).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # ReduceLROnPlateau: 當 loss 卡住不降時，自動把學習率減半
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    
    best_val_score = 0.0
    best_model_state = None
    patience = 20 # 早停機制：如果 20 個 epoch 都沒進步就停
    counter = 0

    # --- Training Loop (訓練迴圈) ---
    for epoch in range(1, EPOCHS + 1):
        model.train() # 切換到訓練模式
        epoch_loss = 0.0
        
        # tqdm 進度條
        with tqdm(total=len(train_loader), desc=f"Ep {epoch}/{EPOCHS}", unit="batch", leave=False) as pbar:
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).long()
                
                optimizer.zero_grad()           # 1. 梯度歸零
                logits = model(imgs)            # 2. 模型預測
                # 3. 計算 Loss (Cross Entropy)
                loss = F.cross_entropy(logits, labels, weight=CLASS_WEIGHTS)
                loss.backward()                 # 4. 反向傳播
                optimizer.step()                # 5. 更新權重
                
                epoch_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                pbar.update(1)
        
        # --- Validation (驗證階段) ---
        model.eval() # 切換到評估模式 (關閉 Dropout 等)
        val_dice_sum = 0.0; n_val = 0
        with torch.no_grad(): # 不計算梯度，省記憶體
            for imgs, labels in val_loader:
                imgs = imgs.to(DEVICE)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()[0] # 取出預測結果
                lbl = labels.numpy()[0]
                
                # 計算 MN 和 CT 的 Dice 分數
                d1 = compute_dice_union(preds, lbl, 1) # MN
                d3 = compute_dice_union(preds, lbl, 3) # CT
                val_dice_sum += (d1 + d3) / 2.0
                n_val += 1
        
        avg_val_score = val_dice_sum / n_val
        scheduler.step(avg_val_score) # 根據分數調整學習率
        
        print(f"   [Ep {epoch}] Loss: {epoch_loss/len(train_loader):.4f} | Val Score: {avg_val_score:.4f}", end="")
        
        # 儲存最佳模型
        if avg_val_score > best_val_score:
            best_val_score = avg_val_score
            best_model_state = model.state_dict()
            counter = 0
            print(" 🌟 New Best!") # 創新高
        else:
            counter += 1
            print("")
        
        # 早停檢查
        if counter >= patience:
            print("🛑 Early Stopping")
            break

    # --- 訓練結束，存檔並進行測試 ---
    save_path = os.path.join(CHECKPOINT_DIR, f"best_fold_{fold_idx+1}.pth")
    if best_model_state:
        torch.save(best_model_state, save_path)
        model.load_state_dict(best_model_state) # 載入最佳權重進行測試
    
    # 在 Test Set 上跑一次分數
    model.eval()
    mn_list, ft_list, ct_list = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()[0]
            lbl = labels.numpy()[0]
            mn_list.append(compute_dice_union(preds, lbl, 1))
            ft_list.append(compute_dice_union(preds, lbl, 2))
            ct_list.append(compute_dice_union(preds, lbl, 3))
    
    final_mn = np.mean(mn_list)
    final_ft = np.mean(ft_list)
    final_ct = np.mean(ct_list)
    print(f"✅ Fold {fold_idx+1} Result >> MN: {final_mn:.4f}, FT: {final_ft:.4f}, CT: {final_ct:.4f}")
    return final_mn, final_ft, final_ct

# ===== 5. 執行 5-Fold 交叉驗證 =====
def run_sota_training():
    # 定義 5 個 Fold 的分組 (依據 PPT 規則)
    pairs = [["8", "9"], ["6", "7"], ["4", "5"], ["2", "3"], ["0", "1"]]
    mn_all, ft_all, ct_all = [], [], []
    
    print(f"🚀 開始 SOTA 訓練 (U-Net++ with EfficientNet-B3)")
    
    for i in range(5):
        pair = pairs[i]
        val_ids = [pair[0]]; test_ids = [pair[1]] # 輪流當驗證和測試
        train_ids = []
        for p in pairs:
            if p != pair: train_ids.extend(p) # 其他都當訓練
        
        mn, ft, ct = train_one_fold(i, train_ids, val_ids, test_ids)
        mn_all.append(mn); ft_all.append(ft); ct_all.append(ct)

    # 輸出最終平均成績
    print("\n==================================")
    print(f"🏆 SOTA 5-Fold 最終平均分數")
    print(f"Mean MN: {np.mean(mn_all):.4f}")
    print(f"Mean FT: {np.mean(ft_all):.4f}")
    print(f"Mean CT: {np.mean(ct_all):.4f}")
    print("==================================")

if __name__ == "__main__":
    run_sota_training()
