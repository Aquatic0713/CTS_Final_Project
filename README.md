# 專案名稱：基於雙模態 U-Net 的腕隧道症候群 MRI 影像多類別分割
# (Multi-Class Carpal Tunnel Syndrome Segmentation via Dual-Modal U-Net)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-Medical_Segmentation-red)]()

## 📌 專案簡介 (Project Overview)

本專案旨在利用深度學習技術，自動從手腕 MRI 影像中精確分割出關鍵解剖結構，以輔助醫師診斷腕隧道症候群 (CTS)。

我們提出了一種 **雙模態融合 (Dual-Modal Fusion)** 的策略，同時利用 **T1-weighted** 與 **T2-weighted** 影像作為輸入，透過 U-Net 架構學習不同組織的特徵，並輸出四種類別的分割結果：

1.  **正中神經 (Median Nerve, MN)** - 黃色標示
2.  **屈肌腱 (Flexor Tendons, FT)** - 藍色標示
3.  **腕隧道 (Carpal Tunnel, CT)** - 紅色標示
4.  **背景 (Background)**

本系統結合了穩健的數據增強與混合損失函數優化，旨在達到高精度的 Dice Coefficient (DC) 指標。

---

## 🌟 核心技術與亮點 (Key Features)

* **模型架構 (Model Architecture)**：
    * 採用經典且高效的 **Multi-class U-Net**。
    * **雙通道輸入 (2-Channel Input)**：輸入層修改為接受 T1 與 T2 兩張影像堆疊，讓模型能同時學習解剖結構 (T1) 與病理特徵 (T2)。
    * **雙線性上採樣 (Bilinear Upsampling)**：在解碼器 (Decoder) 階段使用雙線性插值，減少棋盤效應並保持邊緣平滑。

* **優化策略 (Optimization Strategy)**：
    * **混合損失函數 (Hybrid Loss)**：結合 `CrossEntropy Loss` (0.5) 與 `Dice Loss` (0.5)。CE Loss 負責像素級分類準確度，而 Dice Loss 則直接優化評估指標，解決類別不平衡問題。
    * **動態學習率調整 (Adaptive Scheduler)**：使用 `ReduceLROnPlateau`，當驗證集 Mean Dice 停止進步時自動降低學習率，以進行更精細的權重微調。

* **資料前處理與增強 (Preprocessing & Augmentation)**：
    * **自動二值化 (Input Binarization)**：針對特定數據特性，提供輸入影像二值化選項，強化結構對比。
    * **幾何增強**：訓練過程中隨機應用水平/垂直翻轉 (Flip) 與 90度旋轉 (Rotation)，增加模型對不同手腕擺放角度的泛化能力。

---

## 📂 檔案結構說明 (File Structure)

| 檔案名稱 | 類型 | 功能說明 |
| :--- | :--- | :--- |
| **`train_binary_unet_v5.py`** | 訓練腳本 | 核心訓練程式。包含 `UNetMulti` 模型定義、訓練迴圈、驗證邏輯以及 Checkpoint 自動儲存機制。 |
| **`dataset_cts_v5.py`** | 資料處理 | 定義 `CTSDatasetV5`。負責讀取 T1/T2 影像、解析 MN/FT/CT 標註並整合成 Multi-class Label，以及執行資料增強。 |
| **`gui_cts.py`** | 工具腳本 | 用於快速檢查資料集讀取是否正常，統計各類別 (MN/FT/CT) 的像素分佈情形。 |
| **`checkpoints_multiclass_v5/`** | 權重儲存 | 訓練過程中會自動生成此資料夾，保存 `best.pth` (最佳模型) 與 `last.pth` (最新進度)。 |

---

## 🚀 訓練策略指南 (Training Strategy)

本專案採用單一階段但具備適應性的訓練流程：

* **Epochs**: 預設 200 (可依需求調整 `EPOCHS_PER_RUN`)。
* **Batch Size**: 1 (針對高解析度醫學影像優化顯存使用)。
* **Optimizer**: **AdamW** (Learning Rate = 1e-3, Weight Decay = 1e-5)。相較於傳統 SGD，AdamW 能更快適應稀疏梯度的參數更新。
* **Checkpointing**: 系統會自動監控驗證集 (Validation Set) 的 **Mean Dice**。
    * 若分數創新高，自動儲存為 `unet_multiclass_best.pth`。
    * 無論分數如何，每個 epoch 結束皆儲存 `unet_multiclass_last.pth` 以支援斷點續訓。

---

## 💻 安裝與執行 (Installation & Usage)

### 1. 環境需求

請確保已安裝 Python 3.8+ 與 PyTorch。

```bash
pip install torch torchvision numpy Pillow
