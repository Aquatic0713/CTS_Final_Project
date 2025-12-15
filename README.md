\# CTS MRI 影像分割系統 (DLP 2025 期末專案)



\[!\[Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)

\[!\[PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)

\[!\[GUI](https://img.shields.io/badge/GUI-PyQt6-green)]()



\## 📖 專案概述

本專案實作了一套深度學習方法，用於從磁振造影 (MRI) 影像中自動分割腕部組織，以輔助腕隧道症候群 (CTS) 的診斷。



系統可針對以下三個關鍵結構進行分割：

1\.  \*\*正中神經 (Median Nerve, MN)\*\* - 黃色標示

2\.  \*\*屈肌腱 (Flexor Tendons, FT)\*\* - 藍色標示

3\.  \*\*腕隧道 (Carpal Tunnel, CT)\*\* - 紅色標示



我們採用 \*\*U-Net++\*\* 架構搭配 \*\*EfficientNet-B3\*\* 作為骨幹網路，以實現強健的分割效果，並整合了基於 PyQt6 的圖形使用者介面 (GUI)，以便進行即時視覺化與醫學分析。



!\[展示截圖](demo.png)

\*(請確認你已將截圖命名為 demo.png 並放在專案根目錄，或是移除此行)\*



---



\## 🚀 功能特色

\* \*\*雙模態輸入 (Dual-Modality)\*\*：同時處理 T1 加權與 T2 加權 MRI 序列，獲取更豐富特徵。

\* \*\*進階前處理\*\*：實作 \*\*CLAHE\*\* (限制對比度自適應直方圖均衡化) 技術，顯著增強組織對比度。

\* \*\*SOTA 模型架構\*\*：使用 `segmentation-models-pytorch` (U-Net++) 並載入預訓練權重。

\* \*\*強健推論機制\*\*：包含 \*\*TTA (測試時增強)\*\* 與 \*\*LCC (最大連通分量)\*\* 後處理演算法，有效濾除雜訊。

\* \*\*互動式 GUI\*\*：友善的使用者介面，可切換模型權重、顯示疊圖結果，並即時計算 Dice 係數。



---



\## 📂 專案結構



```text

CTS\_Final\_Project/

├── checkpoints/             # 存放最佳模型權重 (best\_fold\_1.pth 等)

├── src/

│   ├── dataset.py           # 資料載入、CLAHE 前處理與資料增強邏輯

│   └── model.py             # (選用) 若不直接呼叫 SMP 套件，可在此定義模型

├── app.py                   # PyQt6 圖形使用者介面主程式

├── train.py                 # 包含 5-Fold 交叉驗證的訓練腳本

├── requirements.txt         # Python 套件需求清單

└── README.md                # 專案說明文件 (本檔案)

