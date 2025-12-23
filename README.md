# 專案名稱：基於雙模態 U-Net 的腕隧道症候群 MRI 影像多類別分割
# (Multi-Class Carpal Tunnel Syndrome Segmentation via Dual-Modal U-Net)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active-green)]()

## 📌 專案簡介 (Project Overview)

本專案實作了一套基於深度學習的醫學影像分割系統，旨在自動從手腕 MRI 影像中精確分割出關鍵解剖結構，以輔助醫師診斷腕隧道症候群 (CTS)。

我們採用 **雙模態融合 (Dual-Modal Fusion)** 策略，將 **T1-weighted** 與 **T2-weighted** 影像進行堆疊輸入，利用 U-Net 架構學習組織特徵，目標是精確分割以下三個區域並達到高 Dice Score：

1.  **正中神經 (Median Nerve, MN)** - (Class 1)
2.  **屈肌腱 (Flexor Tendons, FT)** - (Class 2)
3.  **腕隧道 (Carpal Tunnel, CT)** - (Class 3)

本系統結合了穩健的數據增強 (Augmentation) 與混合損失函數 (Hybrid Loss) 優化策略，確保在醫學影像數據量有限的情況下仍能達到良好的泛化能力。

---

## 🌟 核心技術與亮點 (Key Features)

* **模型架構 (Model Architecture)**：
    * **Dual-Channel U-Net**：針對 CTS 任務客製化的 Multi-class U-Net。輸入層修改為接受 (T1, T2) 雙通道，讓模型能同時捕捉解剖結構與病理特徵。
    * **Bilinear Upsampling**：解碼器 (Decoder) 採用雙線性插值 (Bilinear Interpolation) 進行上採樣，有效減少棋盤效應並保持分割邊緣平滑。

* **訓練優化策略 (Optimization Strategy)**：
    * **混合損失函數 (Hybrid Loss)**：結合 `CrossEntropy Loss` (0.5) 與 `Dice Loss` (0.5)。CrossEntropy 負責優化像素級分類準確度，Dice Loss 則直接最大化區域重疊率。
    * **動態學習率調整 (Adaptive Scheduler)**：使用 `ReduceLROnPlateau` 監控驗證集 Dice 分數，當指標停滯時自動降低學習率。
    * **優化器**：使用 **AdamW** (Weight Decay 1e-5)，相較於傳統 SGD 能更快適應稀疏梯度的參數更新。

* **前處理與增強 (Preprocessing)**：
    * **Input Binarization**：支援影像二值化前處理，強化組織結構對比。
    * **幾何增強**：訓練過程中隨機應用水平翻轉、垂直翻轉與 90 度旋轉，增加模型魯棒性。

---

## 📂 檔案結構說明 (File Structure)

| 檔案名稱 | 類型 | 功能說明 |
| :--- | :--- | :--- |
| **`train.py`** | 核心程式 | 專案主訓練程式。整合了模型定義、訓練迴圈、驗證邏輯與自動存檔機制。 |
| **`app.py`** | 應用程式 | 圖形化介面 (GUI) 啟動腳本，用於載入模型並展示分割結果。 |
| **`src/dataset.py`** | 資料模組 | 封裝於 `src` 套件中。負責讀取 T1/T2 雙模態影像、解析標註並執行資料增強。 |
| **`checkpoints/`** | 權重儲存 | 訓練過程中自動生成的資料夾，保存最佳模型 (`.pth`) 與最新進度。 |
| **`requirements.txt`** | 環境設定 | 專案所需的 Python 套件清單。 |

---
## 🧠 技術名詞深度解析 (Technical Deep Dive)

為了讓大家更了解本專案的核心技術，這裡詳細說明我們使用的模型架構與評估指標。

### 1. 核心架構：從 U-Net 到 U-Net++

#### **什麼是 U-Net？**
U-Net 是醫學影像分割領域最經典的模型，因其架構圖形狀像一個 **"U"** 字而得名。它主要由兩部分組成：
* **左半邊 (Encoder/縮減路徑)**：負責「壓縮」圖片，提取特徵（例如：這是神經還是肌肉？），但會遺失位置資訊。
* **右半邊 (Decoder/擴張路徑)**：負責「還原」圖片尺寸，將特徵定位回原本的像素位置（例如：神經在圖片的左上角）。
* **跳躍連接 (Skip Connections)**：這是 U-Net 的精髓。它像一座橋，直接把左邊的淺層特徵（高解析度資訊）傳遞給右邊，幫助模型找回在壓縮過程中遺失的邊緣細節。

#### **為什麼選擇 U-Net++？ (與 U-Net 的差異)**
雖然 U-Net 很強，但它在結合「深層特徵」與「淺層特徵」時比較生硬。本專案採用的 **U-Net++** 是其改良版，主要改進了以下幾點：
* **巢狀跳躍路徑 (Nested Skip Pathways)**：U-Net 只有一條長長的橋連接左右；U-Net++ 則在中間建立了「網狀」的密集連接路徑。
* **降低語義落差**：透過中間層的卷積單元，讓左邊的特徵在傳過去右邊之前，先經過多次的消化與融合。
* **效果**：這讓模型在處理 **形狀不規則** 或 **邊界模糊** 的物體（如 MRI 中的細小神經）時，能切得更精準。

### 2. 評估指標：Dice 係數 (Dice Coefficient)

在醫學影像分割中，我們不用「準確率 (Accuracy)」來評分，因為背景通常是一大片黑色，猜對背景沒意義。我們使用的是 **Dice Coefficient (Dice Score)**，專門用來計算「兩個集合的重疊程度」。

#### **運算公式**
假設 **$X$** 是模型預測的區域 (Prediction)，**$Y$** 是醫生標註的標準答案 (Ground Truth)，公式如下：

$$
\text{Dice} = \frac{2 \times |X \cap Y|}{|X| + |Y|}
$$

* **分子 ($2 \times |X \cap Y|$)**：兩者「重疊區域」面積的兩倍。
* **分母 ($|X| + |Y|$)**：兩者總面積的相加。

#### **數值意義**
* **範圍**：0 到 1。
* **0.0**：完全沒有重疊（完全猜錯）。
* **1.0**：完全重疊（完美預測，跟醫生畫的一模一樣）。
* 通常在醫學分割任務中，Dice 超過 **0.85** 就算是非常優秀的結果。

### 3. 其他關鍵技術

* **骨幹網路 (Backbone - EfficientNet-B3)**：
    我們沒有從零訓練 U-Net 的編碼器，而是使用了預訓練的 EfficientNet-B3。它擁有強大的特徵提取能力，能用較少的參數達到更高的效能，加速訓練收斂。

* **CLAHE (限制對比度自適應直方圖均衡化)**：
    MRI 影像常有局部過暗或過亮的問題。CLAHE 不會對整張圖做一樣的調整，而是將圖片切成小塊，分別增強對比度，同時限制雜訊放大。這能顯著提升神經與周圍軟組織的邊界清晰度。
---

## 🚀 訓練策略指南 (Training Strategy)
本專案採用 **SOTA (State-of-the-Art) 轉移學習策略**，利用 EfficientNet 預訓練骨幹加速收斂，並設定了自動化的訓練停止機制以防止過擬合。

### 1. 關鍵參數配置 (Hyperparameters)
* **Epochs (訓練場數)**：設定上限為 **300**。系統結合了 **Early Stopping** 機制，若驗證集分數在連續 **20** 個 Epoch 內未創新高，訓練將自動停止，以節省運算資源並確保模型處於最佳泛化狀態。
* **Batch Size**：設定為 **1**，針對高解析度醫學影像優化顯存使用。
* **Learning Rate**：初始設為 **1e-4**，配合預訓練權重進行微調。

### 2. 優化策略 (Optimization)
* **類別權重 (Class Weights)**：針對 MN (正中神經) 極小目標設定高權重 `[0.5, 10.0, 2.0, 2.0]`，強迫模型專注於學習細微特徵。
* **動態排程 (Scheduler)**：使用 `ReduceLROnPlateau` 監控驗證集分數。當指標停滯超過 **10** 個 Epoch 時，自動將學習率減半 (Factor=0.5)，進行更精細的權重搜索。

---

## 💻 安裝與執行 (Installation & Usage)

本專案建議在乾淨的 Python 環境下執行，以下是從零開始的完整建置步驟。

### 1. 基礎環境設定 (Prerequisites)

若您的電腦尚未安裝 Python，請先至官網下載並安裝：
* **下載連結**：[Python 3.8+ (Python.org)](https://www.python.org/downloads/)
* **⚠️ 重要提示**：安裝時請務必勾選 **"Add Python to PATH"** 選項，以確保可以在終端機直接執行指令。
建議先建立虛擬環境，接著使用以下指令一次安裝所有套件：
⚠️ 注意 (Note regarding GPU)： 如果您需要使用 NVIDIA 顯卡進行加速 (CUDA)，建議先至 PyTorch 官網 查詢適合您顯卡版本的指令
例如：
```bash
# 例如：安裝支援 CUDA 11.8 的 PyTorch
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# 接著再安裝其餘套件
pip install -r requirements.txt
```
### 2\. 準備資料

資料下載：https://drive.google.com/drive/folders/1IYyMttKgM-yVCLweL9d5zqUlhRyDSsma?usp=sharing

請將資料集放置於 `CTS_dataset/` 資料夾中，結構如下：

```text
CTS_dataset/
├── 0/
│   ├── T1/
│   ├── T2/
│   ├── CT/ (Ground Truth)
│   ├── FT/ (Ground Truth)
│   └── MN/ (Ground Truth)
├── 1/
...
```
### 3\. 執行訓練
執行前請將train.py、dataset.py、app.py等三個檔案下載好，並移到自己的虛擬環境裡(同一個資料夾)。

```bash
# 資料檢查與確認 執行資料處理腳本，確認資料集路徑正確且能順利讀取（此步驟亦包含 Augmentation 邏輯的驗證）。
python dataset.py

# 啟動模型訓練 執行主訓練程式。系統將在訓練過程中即時 (On-the-fly) 進行資料增強（翻轉、旋轉），以擴增訓練樣本的多樣性。
python train.py
```
### 4\. 啟動 GUI 展示

```bash
python app.py
```
在介面中選擇 checkpoints_sota 資料夾下的模型 (如 best_fold_1.pth) 以及資料集路徑，即可開始瀏覽分割結果。

---
## 📊 評估指標 (Evaluation Metrics)

本專案採用醫學影像分割領域最常用的兩大指標來評估模型效能：**Dice Coefficient** 與 **Intersection over Union (IoU)**。

### 1. Dice Coefficient (Dice Score)
Dice 係數衡量預測區域與真實標註區域的相似度（值域 0~1，越高越好），對於醫學影像中常見的小目標（如神經）特別敏感。

$$Dice = \frac{2 |P \cap G|}{|P| + |G|}$$

### 2. Intersection over Union (IoU)
IoU 又稱為 Jaccard Index，計算的是「交集」除以「聯集」的比例。它是評估物件偵測與分割最直觀的幾何指標。

$$IoU = \frac{|P \cap G|}{|P \cup G|} = \frac{|P \cap G|}{|P| + |G| - |P \cap G|}$$

> 其中 $P$ 為 Prediction (預測結果)，$G$ 為 Ground Truth (真實標註)。

程式會在驗證階段輸出：
* ✅ **Mean MN**: 正中神經平均分數 (Dice/IoU)
* ✅ **Mean FT**: 屈肌腱平均分數 (Dice/IoU)
* ✅ **Mean CT**: 腕隧道平均分數 (Dice/IoU)
---
## 💻 GUI 介面顯示

程式介面將即時計算並顯示以下資訊，並支援 T1/T2 雙模態影像的自動讀取與預測：

* **Sequence Mean (Dice Score)**：該病例 (Case) 所有 MRI 切片的平均 Dice 分數。系統會在載入病例時自動執行全序列推論，用於評估模型在整體 3D 結構上的表現。
* **Current Slice (Dice Score)**：當前檢視切片的即時 Dice 分數，用於細部檢視模型在特定解剖結構上的表現。分數針對三類組織分別計算：
    * **🟡 正中神經 (Median Nerve)**
    * **🔵 屈肌腱 (Flexor Tendons)**
    * **🔴 腕隧道 (Carpal Tunnel)**
* **Visual Overlay (視覺化疊圖)**：介面同時展示「原始 T1 影像」、「真實標註 (GT)」與「AI 預測結果 (Prediction)」，並應用了 CLAHE 增強技術以提升組織對比度。
---
## 影片展示
 
https://drive.google.com/file/d/1rJGlH3nuB1Rxqtggut-kLJj-TKmlDBKP/view?usp=sharing

-----

### 📝 作者

[95079/吳閎申]
DLP 2025 Final Project

-----
