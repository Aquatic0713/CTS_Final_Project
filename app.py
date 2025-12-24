import sys
import os
import cv2
import torch
import numpy as np
# 引入 PyQt6 介面庫，這是用來畫視窗、按鈕、下拉選單的工具
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QFrame, QGridLayout, QMessageBox, QComboBox, 
                             QSizePolicy, QProgressBar, QSplitter)
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor
from PyQt6.QtCore import Qt
import torch.nn as nn
import segmentation_models_pytorch as smp # 引入強大的分割模型庫

# ===== 全域設定區 =====
# 預設資料路徑 (程式啟動時會預設看這裡)
DEFAULT_DATA_ROOT = r"D:\AI\CTS_dataset"

# 判斷是否有顯卡 (GPU)，有的話跑起來會快很多，沒有就用 CPU 硬跑
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型權重存放的資料夾 (我們訓練好的 best_fold_x.pth 都在這)
CHECKPOINT_DIR = "checkpoints_sota"  
N_CLASSES = 4 # 我們要切 4 類：0=背景, 1=神經, 2=肌腱, 3=隧道

# 設定介面的漂亮外觀 (CSS 樣式表)
STYLE_SHEET = """
    QMainWindow { background-color: #f4f6f9; }
    QFrame#LeftPanel { background-color: white; border-right: 1px solid #d1d5db; }
    QLabel { font-family: "Microsoft JhengHei"; color: #333; }
    QComboBox { border: 1px solid #ced4da; border-radius: 5px; padding: 5px; background: white; font-family: "Microsoft JhengHei"; font-size: 13px; }
    QPushButton { background-color: #3498db; color: white; border-radius: 5px; padding: 8px; font-family: "Microsoft JhengHei"; font-weight: bold; }
    QPushButton:hover { background-color: #2980b9; }
    QProgressBar { border: none; background-color: #e9ecef; border-radius: 4px; height: 8px; }
    QProgressBar::chunk { background-color: #2ecc71; border-radius: 4px; }
"""

# ===== 1. 定義 AI 模型架構 (這是大腦) =====
class CTSModel(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        # 這裡定義了我們的模型結構：U-Net++
        # 使用 EfficientNet-B3 作為骨幹 (backbone)，負責從影像中提取特徵
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights=None, # 推論時不需要重新下載 ImageNet 權重，我們會載入自己訓練好的
            in_channels=2,        # 輸入有 2 個通道 (T1 MRI + T2 MRI)
            classes=n_classes,    # 輸出有 4 個類別
        )

    def forward(self, x):
        # 當圖片丟進來時，會經過這裡進行運算，吐出預測結果
        return self.model(x)

# ===== 2. 定義主視窗 (這是身體/介面) =====
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 設定視窗標題與大小
        self.setWindowTitle("DLP Final Project - SOTA Demo System (Catch-All Fixed)")
        self.resize(1450, 850)
        self.setStyleSheet(STYLE_SHEET)

        # 初始化變數，用來記住現在選了哪個資料夾、哪張圖
        self.root_dir = DEFAULT_DATA_ROOT
        self.t1_folder = ""; self.t2_folder = ""; self.gt_folder = ""
        self.mn_folder = ""; self.ft_folder = ""; self.ct_folder = ""
        self.use_combined_gt = False # 用來標記是否為考試用的 testData (GT 在同一張圖)
        self.image_list = []
        
        self.current_model = None    # 存放目前載入的 AI 模型
        self.available_folds = []    # 存放找到的所有模型檔案

        # 建構介面外觀
        self.init_ui()
        # 掃描資料夾看看有哪些模型可以用
        self.check_models()
        
        # 如果有找到模型，自動載入第一個，方便 demo
        if self.combo_models.count() > 0:
            self.on_model_changed()

        # 如果資料路徑存在，自動列出病例
        if os.path.exists(self.root_dir):
            self.populate_case_combo()

    # 一個小工具：產生包含中文標題與英文副標題的 HTML 文字
    def get_bilingual_text(self, cn, en, color="#333", size_cn=11):
        return f"<div style='color: {color}; line-height: 1.3;'><span style='font-size: {size_cn}pt; font-weight: bold;'>{cn}</span><br><span style='font-family: Arial; font-size: 9pt; color: #7f8c8d;'>{en}</span></div>"

    # ★★★ 初始化介面 (畫出所有按鈕、圖片框) ★★★
    def init_ui(self):
        # 使用分割視窗：左邊是控制面板，右邊是圖片展示
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # --- 左側面板 (控制區) ---
        left_panel = QFrame(); left_panel.setObjectName("LeftPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 25, 20, 25); left_layout.setSpacing(15)
        
        # 標題
        lbl_title = QLabel("✨ CTS 影像分割系統"); lbl_title.setFont(QFont("Microsoft JhengHei", 16, QFont.Weight.Bold)); lbl_title.setStyleSheet("color: #2c3e50;")
        left_layout.addWidget(lbl_title)
        
        # 模型選擇下拉選單
        left_layout.addWidget(QLabel("🤖 選擇模型權重 (Select Model):"))
        self.combo_models = QComboBox()
        self.combo_models.currentIndexChanged.connect(self.on_model_changed) # 當選單改變時，觸發 on_model_changed
        left_layout.addWidget(self.combo_models)
        self.lbl_model_status = QLabel("Checking...") # 顯示模型載入狀態
        left_layout.addWidget(self.lbl_model_status)

        # 資料夾選擇按鈕
        left_layout.addWidget(QLabel("📂 資料集路徑 (Dataset Path):"))
        self.btn_root = QPushButton("選擇資料夾 / Select Folder"); self.btn_root.clicked.connect(self.select_root_folder)
        left_layout.addWidget(self.btn_root)
        self.lbl_root_status = QLabel(self.root_dir); self.lbl_root_status.setStyleSheet("color: #95a5a6; font-size: 10px;"); self.lbl_root_status.setWordWrap(True)
        left_layout.addWidget(self.lbl_root_status)

        # 病例選擇下拉選單 (例如: 0, 1, testData)
        left_layout.addWidget(QLabel("👤 選擇病例 (Case ID):"))
        self.combo_cases = QComboBox(); self.combo_cases.currentIndexChanged.connect(self.on_case_changed)
        left_layout.addWidget(self.combo_cases)
        
        # 分數顯示區 (用網格排版顯示 Dice Score)
        score_group = QFrame(); score_layout = QGridLayout(score_group)
        score_layout.setContentsMargins(0,0,0,0); score_layout.setSpacing(10)
        
        # 表頭
        score_layout.addWidget(QLabel(self.get_bilingual_text("組織", "Organ")), 0, 0)
        score_layout.addWidget(QLabel(self.get_bilingual_text("序列平均", "Seq Mean")), 0, 1)
        score_layout.addWidget(QLabel(self.get_bilingual_text("當前切片", "Curr Slice")), 0, 2)
        
        # MN (神經) 分數欄位
        self.lbl_mn_name = QLabel("🟡 正中神經"); self.lbl_mn_name.setStyleSheet("color: #f39c12; font-weight: bold;")
        self.lbl_mean_mn = QLabel("-"); self.lbl_curr_mn = QLabel("0.00")
        score_layout.addWidget(self.lbl_mn_name, 1, 0); score_layout.addWidget(self.lbl_mean_mn, 1, 1); score_layout.addWidget(self.lbl_curr_mn, 1, 2)
        
        # FT (肌腱) 分數欄位
        self.lbl_ft_name = QLabel("🔵 屈肌腱"); self.lbl_ft_name.setStyleSheet("color: #3498db; font-weight: bold;")
        self.lbl_mean_ft = QLabel("-"); self.lbl_curr_ft = QLabel("0.00")
        score_layout.addWidget(self.lbl_ft_name, 2, 0); score_layout.addWidget(self.lbl_mean_ft, 2, 1); score_layout.addWidget(self.lbl_curr_ft, 2, 2)
        
        # CT (隧道) 分數欄位
        self.lbl_ct_name = QLabel("🔴 腕隧道"); self.lbl_ct_name.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.lbl_mean_ct = QLabel("-"); self.lbl_curr_ct = QLabel("0.00")
        score_layout.addWidget(self.lbl_ct_name, 3, 0); score_layout.addWidget(self.lbl_mean_ct, 3, 1); score_layout.addWidget(self.lbl_curr_ct, 3, 2)
        
        left_layout.addWidget(score_group); left_layout.addStretch()

        # --- 右側面板 (圖片區) ---
        right_panel = QFrame(); right_panel.setObjectName("RightPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20); right_layout.setSpacing(15)

        # 圖片網格 (左: 原圖, 中: GT, 右: 預測)
        img_grid = QGridLayout(); img_grid.setSpacing(20)
        self.view_input = QLabel(); self.view_gt = QLabel(); self.view_pred = QLabel()
        labels = [self.view_input, self.view_gt, self.view_pred]
        titles = ["原始 T1 影像 (Original)", "真實標註 (Ground Truth)", "AI 預測結果 (Prediction)"]
        
        for i, (lbl, title) in enumerate(zip(labels, titles)):
            t_lbl = QLabel(title); t_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); t_lbl.setFont(QFont("Microsoft JhengHei", 11, QFont.Weight.Bold))
            img_grid.addWidget(t_lbl, 0, i)
            lbl.setFixedSize(380, 380) # 固定圖片大小
            lbl.setStyleSheet("background-color: black; border-radius: 8px; border: 2px solid #34495e;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_grid.addWidget(lbl, 1, i)

        right_layout.addLayout(img_grid)

        # 底部控制區 (滑動條 Slider)
        control_frame = QFrame(); control_frame.setStyleSheet("background-color: white; border-radius: 10px;")
        control_layout = QHBoxLayout(control_frame)
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self.on_slider_changed) # 當拉動滑桿時，切換圖片
        self.lbl_progress = QLabel("0/0"); self.lbl_progress.setFixedWidth(60); self.lbl_progress.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(QLabel("Slice:")); control_layout.addWidget(self.slider); control_layout.addWidget(self.lbl_progress)
        right_layout.addWidget(control_frame)

        splitter.addWidget(left_panel); splitter.addWidget(right_panel); splitter.setSizes([350, 1100])

    # 檢查模型資料夾，把找到的 .pth 檔加到選單
    def check_models(self):
        self.combo_models.clear(); self.available_folds = []
        if os.path.exists(CHECKPOINT_DIR):
            for f in sorted(os.listdir(CHECKPOINT_DIR)):
                if f.startswith("best_fold_") and f.endswith(".pth"):
                    name = f"Fold {f.split('_')[2].replace('.pth', '')}"
                    self.available_folds.append(name); self.combo_models.addItem(name)
        if not self.available_folds:
            self.lbl_model_status.setText("⚠️ 無模型"); self.lbl_model_status.setStyleSheet("color: red;")

    # ★★★ 當使用者切換模型時 ★★★
    def on_model_changed(self):
        name = self.combo_models.currentText()
        if not name: return
        try:
            # 1. 找出模型檔案路徑
            fold_idx = name.split(" ")[1]
            path = os.path.join(CHECKPOINT_DIR, f"best_fold_{fold_idx}.pth")
            
            # 2. 重新初始化 AI 模型 (清空舊的)
            self.current_model = None
            self.current_model = CTSModel(n_classes=N_CLASSES).to(DEVICE)
            
            # 3. 載入訓練好的權重 (Load Weights)
            self.current_model.load_state_dict(torch.load(path, map_location=DEVICE))
            self.current_model.eval() # 設定為評估模式 (不會更新權重)
            
            self.lbl_model_status.setText(f"✅ Loaded: {name}"); self.lbl_model_status.setStyleSheet("color: green;")
            
            # 4. 如果現在有圖片，立刻重跑一次分割，更新畫面
            if self.image_list: 
                self.calculate_sequence_mean()
                self.run_segmentation(self.slider.value())
        except Exception as e:
            print(f"Error: {e}"); self.lbl_model_status.setText("❌ Load Fail"); self.lbl_model_status.setStyleSheet("color: red;")

    # 讓使用者選擇資料夾
    def select_root_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if d: self.root_dir = d; self.lbl_root_status.setText(d); self.populate_case_combo()

    # 掃描資料夾，列出所有病例 (0, 1, ... 以及 testData)
    def populate_case_combo(self):
        self.combo_cases.blockSignals(True); self.combo_cases.clear()
        if os.path.exists(self.root_dir):
            dirs = []
            for d in os.listdir(self.root_dir):
                if not os.path.isdir(os.path.join(self.root_dir, d)): continue
                if d.isdigit() or d.lower() == "testdata": dirs.append(d)
            # 排序：把 testData 放第一個，其他照數字排
            dirs.sort(key=lambda x: -1 if x.lower() == "testdata" else int(x) if x.isdigit() else 999)
            self.combo_cases.addItems(dirs)
        self.combo_cases.blockSignals(False)
        if self.combo_cases.count(): self.on_case_changed()

    # ★★★ 當使用者切換病例時 ★★★
    def on_case_changed(self):
        case_id = self.combo_cases.currentText()
        if not case_id: return
        base = os.path.join(self.root_dir, case_id)
        self.t1_folder = os.path.join(base, "T1"); self.t2_folder = os.path.join(base, "T2")
        
        # 判斷是「考試資料 (testData)」還是「訓練資料」
        # 考試資料只有一個 GT 資料夾；訓練資料有 MN/FT/CT 三個資料夾
        if os.path.exists(os.path.join(base, "GT")):
            self.gt_folder = os.path.join(base, "GT"); self.use_combined_gt = True
        else:
            self.mn_folder = os.path.join(base, "MN"); self.ft_folder = os.path.join(base, "FT"); self.ct_folder = os.path.join(base, "CT")
            self.use_combined_gt = False

        # 讀取 T1 資料夾裡的所有圖片列表
        if os.path.exists(self.t1_folder):
            self.image_list = sorted([f for f in os.listdir(self.t1_folder) if f.endswith(('.png', '.jpg'))], 
                                     key=lambda x: int(os.path.splitext(x)[0]) if x[0].isdigit() else x)
            # 設定滑桿範圍
            self.slider.setMaximum(len(self.image_list)-1); self.slider.setValue(0); self.slider.setEnabled(True)
            
            # 計算整序列的平均分數，並顯示第一張圖
            self.calculate_sequence_mean()
            self.run_segmentation(0)
        else:
            self.image_list = []; self.slider.setEnabled(False)

    # 當拉動滑桿時執行
    def on_slider_changed(self, val):
        self.lbl_progress.setText(f"{val+1}/{len(self.image_list)}")
        self.run_segmentation(val)

    # ★★★ 核心修正：萬能色彩解析 (Catch-All Strategy) ★★★
    # 用來解決考試資料 GT 顏色不純、被壓縮過的問題
    def parse_colored_gt(self, gt_path):
        img_bgr = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        if img_bgr is None: return np.zeros((512,512), dtype=np.uint8)

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, _ = img_rgb.shape
        mask = np.zeros((h, w), dtype=np.uint8)
        
        R = img_rgb[:,:,0]
        G = img_rgb[:,:,1]
        B = img_rgb[:,:,2]

        # 1. 抓 FT (青色): 綠色與藍色數值高
        mask_ft = (G > 80) & (B > 80)
        mask[mask_ft] = 2

        # 2. 抓 MN (洋紅): 紅色與藍色數值高
        mask_mn = (R > 80) & (B > 80)
        mask[mask_mn] = 1

        # 3. 抓 CT (萬能法)：
        # 只要這個像素有顏色 (RGB任一 > 50)，而且還沒被標記成 MN 或 FT，
        # 我們就認定它是紅色 (CT)！這樣就算紅色有點暗或偏色也能抓到。
        has_color = (R > 50) | (G > 50) | (B > 50)
        mask_ct = has_color & (mask == 0)
        mask[mask_ct] = 3
        
        return mask

    # 讀取 GT (Ground Truth) 遮罩
    def get_gt_mask(self, fname):
        if self.use_combined_gt:
            # 如果是 testData，呼叫上面的萬能解析函式
            p = os.path.join(self.gt_folder, fname)
            if os.path.exists(p): return self.parse_colored_gt(p)
        else:
            # 如果是訓練資料，分別讀取三個資料夾再合成
            final = np.zeros((512, 512), dtype=np.uint8)
            for p, cid in [(os.path.join(self.ct_folder, fname), 3), 
                           (os.path.join(self.ft_folder, fname), 2), 
                           (os.path.join(self.mn_folder, fname), 1)]:
                if os.path.exists(p):
                    m = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if m is not None: final[m > 127] = cid
            return final
        return np.zeros((512,512), dtype=np.uint8)

    # ★★★ AI 推論：讓模型看圖並預測 ★★★
    def predict_mask(self, fname):
        if self.current_model is None: return None
        p1 = os.path.join(self.t1_folder, fname); p2 = os.path.join(self.t2_folder, fname)
        if not os.path.exists(p1) or not os.path.exists(p2): return None
        
        # 讀圖 + CLAHE 增強 (要跟訓練時一致)
        i1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE); i2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        inp = np.stack([clahe.apply(i1), clahe.apply(i2)], axis=0).astype(np.float32) / 255.0
        
        # 轉成 Tensor 丟進 GPU
        t = torch.from_numpy(inp).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad(): # 推論時不需要算梯度
            # 取出預測機率最高的類別 (Argmax)
            pred = torch.argmax(self.current_model(t), dim=1).cpu().numpy()[0]
        return pred

    # 計算 Dice Score (重疊率)
    def calculate_dice(self, pred, target, cid):
        # 根據 PPT 邏輯，CT (隧道) 包含所有內容物 (Union)
        if cid == 3: 
            p = (pred >= 1); t = (target >= 1)
        else: 
            p = (pred == cid); t = (target == cid)
        
        inter = (p & t).sum() # 交集
        union = p.sum() + t.sum() # 聯集
        if union == 0: return 1.0 # 兩邊都沒東西 = 預測正確(全黑) = 100分
        return 2*inter/(union+1e-5) # Dice 公式

    # 計算整個序列 (Sequence) 的平均分數
    def calculate_sequence_mean(self):
        s = {1:[], 2:[], 3:[]}
        for fname in self.image_list:
            p = self.predict_mask(fname); g = self.get_gt_mask(fname)
            if p is None: continue
            for c in [1,2,3]: s[c].append(self.calculate_dice(p, g, c))
        # 更新介面上的平均分數
        self.lbl_mean_mn.setText(f"{np.mean(s[1]):.2f}"); self.lbl_mean_ft.setText(f"{np.mean(s[2]):.2f}"); self.lbl_mean_ct.setText(f"{np.mean(s[3]):.2f}")

    # ★★★ 繪圖函式：畫出漂亮的半透明遮罩 ★★★
    def draw_nice_overlay(self, img_gray, mask):
        # 先轉成彩色圖片
        vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        # 定義顏色：MN=黃, FT=藍, CT=紅
        colors = { 1: (0,255,255), 2: (255,0,0), 3: (0,0,255) } 
        
        for cid, col in colors.items():
            m_u8 = (mask == cid).astype(np.uint8)
            # 1. 畫輪廓線 (實線)
            cnts, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, col, 2)
            
            # 2. 畫半透明填充 (Blend)
            indices = np.where(mask == cid)
            if len(indices[0]) > 0:
                # 運用數學公式：原圖 * 0.6 + 顏色 * 0.4
                roi = vis[indices[0], indices[1]].astype(np.float32)
                blended = roi * 0.6 + np.array(col, dtype=np.float32) * 0.4
                vis[indices[0], indices[1]] = blended.astype(np.uint8)
                
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # ★★★ 主流程：執行一次完整的分割與展示 ★★★
    def run_segmentation(self, idx):
        if not self.image_list: return
        fname = self.image_list[idx]
        
        # 1. AI 預測 + 2. 讀取真實標註
        pred = self.predict_mask(fname); gt = self.get_gt_mask(fname)
        if pred is None: return
        
        # 3. 計算當前切片分數並更新顯示
        d = {c: self.calculate_dice(pred, gt, c) for c in [1,2,3]}
        self.lbl_curr_mn.setText(f"{d[1]:.2f}"); self.lbl_curr_ft.setText(f"{d[2]:.2f}"); self.lbl_curr_ct.setText(f"{d[3]:.2f}")
        
        # 4. 準備圖片用於展示
        p1 = os.path.join(self.t1_folder, fname); i1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        i1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(i1) # 增強對比度讓圖片好看
        
        # 5. 更新三個視窗畫面 (左:原圖, 中:GT, 右:預測)
        self.show_img(cv2.cvtColor(i1, cv2.COLOR_GRAY2RGB), self.view_input)
        self.show_img(self.draw_nice_overlay(i1, gt), self.view_gt)
        self.show_img(self.draw_nice_overlay(i1, pred), self.view_pred)

    # 輔助函式：把 OpenCV 圖片貼到 PyQt 標籤上
    def show_img(self, img, lbl):
        h, w, c = img.shape
        qimg = QImage(img.data, w, h, c*w, QImage.Format.Format_RGB888)
        lbl.setPixmap(QPixmap.fromImage(qimg).scaled(lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

# 程式進入點
if __name__ == "__main__":
    app = QApplication(sys.argv); app.setFont(QFont("Microsoft JhengHei", 10))
    win = MainWindow(); win.show(); sys.exit(app.exec())
