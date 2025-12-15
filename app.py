import sys
import os
import cv2
import torch
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QSlider, QFrame, QGridLayout, QMessageBox, QComboBox, 
                             QSizePolicy, QProgressBar, QSplitter)
from PyQt6.QtGui import QPixmap, QImage, QFont, QColor
from PyQt6.QtCore import Qt
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# 引用資料集設定
try:
    from dataset_cts_v5 import DATA_ROOT as DEFAULT_DATA_ROOT
except ImportError:
    DEFAULT_DATA_ROOT = "./CTS_dataset"

# --- 設定 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ★★★ 關鍵修改：指向你剛剛訓練產生模型的地方 ★★★
CHECKPOINT_DIR = "checkpoints_sota" 
N_CLASSES = 4

# 字型與樣式
STYLE_SHEET = """
    QMainWindow { background-color: #f4f6f9; }
    QFrame#LeftPanel { background-color: white; border-right: 1px solid #d1d5db; }
    QFrame#RightPanel { background-color: #f4f6f9; }
    QLabel { font-family: "Microsoft JhengHei"; color: #333; }
    QComboBox { border: 1px solid #ced4da; border-radius: 5px; padding: 5px; background: white; font-family: "Microsoft JhengHei"; font-size: 13px; }
    QComboBox::drop-down { border: 0px; }
    QPushButton { background-color: #3498db; color: white; border-radius: 5px; padding: 8px; font-family: "Microsoft JhengHei"; font-weight: bold; }
    QPushButton:hover { background-color: #2980b9; }
    QProgressBar { border: none; background-color: #e9ecef; border-radius: 4px; height: 8px; }
    QProgressBar::chunk { background-color: #2ecc71; border-radius: 4px; }
"""

# ===== 模型定義 (必須與訓練時一致) =====
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)

class CTSModel(nn.Module):
    def __init__(self, n_classes=4):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3",
            encoder_weights=None, # 推論時不需要下載權重，因為會載入我們訓練好的 pth
            in_channels=2,
            classes=n_classes,
        )

    def forward(self, x):
        return self.model(x)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DLP Final Project - AI Segmentation Demo")
        self.resize(1450, 850)
        self.setStyleSheet(STYLE_SHEET)

        self.root_dir = DEFAULT_DATA_ROOT
        self.t1_folder = ""; self.t2_folder = ""; self.gt_folder = ""
        self.image_list = []
        
        self.current_model = None
        self.available_folds = []

        self.init_ui()
        self.check_models() # 檢查有哪些 Fold 模型可用
        
        # 預設載入 Fold 1 (如果你剛訓練完的話)
        if "Fold 1" in self.available_folds:
            self.load_model_by_fold("Fold 1")
            self.combo_models.setCurrentText("Fold 1")

        if os.path.exists(self.root_dir):
            self.populate_case_combo()

    def get_bilingual_text(self, cn, en, color="#333", size_cn=11):
        return f"<div style='color: {color}; line-height: 1.3;'><span style='font-size: {size_cn}pt; font-weight: bold;'>{cn}</span><br><span style='font-family: Arial; font-size: 9pt; color: #7f8c8d;'>{en}</span></div>"

    def init_ui(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # Left Panel
        left_panel = QFrame(); left_panel.setObjectName("LeftPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 25, 20, 25); left_layout.setSpacing(15)
        
        lbl_title = QLabel("✨ CTS 影像分割系統"); lbl_title.setFont(QFont("Microsoft JhengHei", 16, QFont.Weight.Bold)); lbl_title.setStyleSheet("color: #2c3e50;")
        left_layout.addWidget(lbl_title)
        
        # 模型選擇下拉選單
        left_layout.addWidget(QLabel("🤖 選擇模型權重 (Select Model):"))
        self.combo_models = QComboBox()
        self.combo_models.currentIndexChanged.connect(self.on_model_changed)
        left_layout.addWidget(self.combo_models)
        
        self.lbl_model_status = QLabel("請選擇模型...")
        left_layout.addWidget(self.lbl_model_status); left_layout.addSpacing(10)

        left_layout.addWidget(QLabel("📂 資料集路徑 (Dataset Path):"))
        self.btn_root = QPushButton("選擇資料夾 / Select Folder"); self.btn_root.clicked.connect(self.select_root_folder)
        left_layout.addWidget(self.btn_root)
        self.lbl_root_status = QLabel(self.root_dir); self.lbl_root_status.setStyleSheet("color: #95a5a6; font-size: 10px;"); self.lbl_root_status.setWordWrap(True)
        left_layout.addWidget(self.lbl_root_status)

        left_layout.addWidget(QLabel("👤 選擇病例 (Case ID):"))
        self.combo_cases = QComboBox(); self.combo_cases.currentIndexChanged.connect(self.on_case_changed)
        left_layout.addWidget(self.combo_cases)
        self.progress_bar = QProgressBar(); left_layout.addWidget(self.progress_bar)
        
        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine); line.setStyleSheet("background-color: #dfe6e9;"); left_layout.addWidget(line); left_layout.addSpacing(10)

        # Score Area
        score_group = QFrame(); score_layout = QGridLayout(score_group)
        score_layout.setContentsMargins(0,0,0,0); score_layout.setSpacing(10)
        score_layout.addWidget(QLabel(self.get_bilingual_text("組織", "Organ")), 0, 0)
        score_layout.addWidget(QLabel(self.get_bilingual_text("序列平均", "Seq Mean")), 0, 1)
        score_layout.addWidget(QLabel(self.get_bilingual_text("當前切片", "Curr Slice")), 0, 2)
        
        self.lbl_mn_name = QLabel("🟡 正中神經"); self.lbl_mn_name.setStyleSheet("color: #f39c12; font-weight: bold;")
        self.lbl_mean_mn = QLabel("-"); self.lbl_curr_mn = QLabel("0.00")
        score_layout.addWidget(self.lbl_mn_name, 1, 0); score_layout.addWidget(self.lbl_mean_mn, 1, 1); score_layout.addWidget(self.lbl_curr_mn, 1, 2)
        
        self.lbl_ft_name = QLabel("🔵 屈肌腱"); self.lbl_ft_name.setStyleSheet("color: #3498db; font-weight: bold;")
        self.lbl_mean_ft = QLabel("-"); self.lbl_curr_ft = QLabel("0.00")
        score_layout.addWidget(self.lbl_ft_name, 2, 0); score_layout.addWidget(self.lbl_mean_ft, 2, 1); score_layout.addWidget(self.lbl_curr_ft, 2, 2)
        
        self.lbl_ct_name = QLabel("🔴 腕隧道"); self.lbl_ct_name.setStyleSheet("color: #e74c3c; font-weight: bold;")
        self.lbl_mean_ct = QLabel("-"); self.lbl_curr_ct = QLabel("0.00")
        score_layout.addWidget(self.lbl_ct_name, 3, 0); score_layout.addWidget(self.lbl_mean_ct, 3, 1); score_layout.addWidget(self.lbl_curr_ct, 3, 2)
        
        left_layout.addWidget(score_group); left_layout.addStretch()
        lbl_footer = QLabel("DLP 2025 Final Project"); lbl_footer.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl_footer.setStyleSheet("color: #bdc3c7; font-size: 10px;")
        left_layout.addWidget(lbl_footer)

        # Right Panel
        right_panel = QFrame(); right_panel.setObjectName("RightPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20); right_layout.setSpacing(15)

        img_grid = QGridLayout(); img_grid.setSpacing(20)
        self.view_input = QLabel(); self.view_gt = QLabel(); self.view_pred = QLabel()
        labels = [self.view_input, self.view_gt, self.view_pred]
        titles = ["原始 T1 影像 (Original)", "真實標註 (Ground Truth)", "AI 預測結果 (Prediction)"]
        
        for i, (lbl, title) in enumerate(zip(labels, titles)):
            t_lbl = QLabel(title); t_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); t_lbl.setFont(QFont("Microsoft JhengHei", 11, QFont.Weight.Bold))
            img_grid.addWidget(t_lbl, 0, i)
            lbl.setFixedSize(380, 380) 
            lbl.setStyleSheet("background-color: black; border-radius: 8px; border: 2px solid #34495e;")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_grid.addWidget(lbl, 1, i)

        img_grid.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addLayout(img_grid)

        control_frame = QFrame(); control_frame.setStyleSheet("background-color: white; border-radius: 10px;")
        control_layout = QHBoxLayout(control_frame)
        lbl_slice = QLabel("切片索引 (Slice Index):"); lbl_slice.setFont(QFont("Microsoft JhengHei", 10, QFont.Weight.Bold))
        self.slider = QSlider(Qt.Orientation.Horizontal); self.slider.setEnabled(False); self.slider.valueChanged.connect(self.on_slider_changed)
        self.slider.setStyleSheet("QSlider::groove:horizontal { height: 6px; background: #bdc3c7; border-radius: 3px; } QSlider::handle:horizontal { background: #3498db; width: 18px; margin: -6px 0; border-radius: 9px; }")
        self.lbl_progress = QLabel("0/0"); self.lbl_progress.setFixedWidth(50); self.lbl_progress.setAlignment(Qt.AlignmentFlag.AlignCenter); self.lbl_progress.setStyleSheet("font-weight: bold; color: #2c3e50;")
        
        control_layout.addWidget(lbl_slice); control_layout.addWidget(self.slider); control_layout.addWidget(self.lbl_progress)
        right_layout.addWidget(control_frame)

        splitter.addWidget(left_panel); splitter.addWidget(right_panel)
        splitter.setSizes([350, 1100]); splitter.setCollapsible(0, False)

    # --- 邏輯修正：動態掃描 checkpoints 資料夾 ---
    def check_models(self):
        self.combo_models.clear()
        self.available_folds = []
        if os.path.exists(CHECKPOINT_DIR):
            files = os.listdir(CHECKPOINT_DIR)
            # 尋找 best_fold_X.pth
            for f in files:
                if f.startswith("best_fold_") and f.endswith(".pth"):
                    fold_num = f.split("_")[2].replace(".pth", "")
                    name = f"Fold {fold_num}"
                    self.available_folds.append(name)
                    self.combo_models.addItem(name)
        
        if not self.available_folds:
            self.lbl_model_status.setText("⚠️ 未找到任何模型檔案")
            self.lbl_model_status.setStyleSheet("color: red;")
            
    def on_model_changed(self):
        name = self.combo_models.currentText()
        if name: self.load_model_by_fold(name)
        # 如果當前有顯示影像，重新預測
        if self.image_list and self.current_model:
            self.calculate_sequence_mean()
            self.run_segmentation(self.slider.value())

    def load_model_by_fold(self, fold_name):
        # fold_name e.g., "Fold 1"
        fold_idx = fold_name.split(" ")[1]
        path = os.path.join(CHECKPOINT_DIR, f"best_fold_{fold_idx}.pth")
        
        if os.path.exists(path):
            try:
                # 重新初始化一個乾淨的模型
                self.current_model = CTSModel(n_classes=N_CLASSES).to(DEVICE)
                self.current_model.load_state_dict(torch.load(path, map_location=DEVICE))
                self.current_model.eval()
                
                self.lbl_model_status.setText(f"✅ 已載入: {fold_name}")
                self.lbl_model_status.setStyleSheet("color: #27ae60; font-weight: bold;")
            except Exception as e:
                print(f"Error loading model: {e}")
                self.lbl_model_status.setText("❌ 模型載入失敗")

    def select_root_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Dataset Root")
        if path: self.root_dir = path; self.lbl_root_status.setText(path); self.populate_case_combo()

    def populate_case_combo(self):
        self.combo_cases.blockSignals(True); self.combo_cases.clear()
        if os.path.exists(self.root_dir):
            subs = sorted([d for d in os.listdir(self.root_dir) if d.isdigit() and os.path.isdir(os.path.join(self.root_dir, d))], key=int)
            self.combo_cases.addItems(subs)
        self.combo_cases.blockSignals(False)
        if self.combo_cases.count() > 0: self.on_case_changed()

    def on_case_changed(self):
        case_id = self.combo_cases.currentText()
        if not case_id: return
        
        self.t1_folder = os.path.join(self.root_dir, case_id, "T1")
        self.t2_folder = os.path.join(self.root_dir, case_id, "T2")
        self.mn_folder = os.path.join(self.root_dir, case_id, "MN")
        self.ft_folder = os.path.join(self.root_dir, case_id, "FT")
        self.ct_folder = os.path.join(self.root_dir, case_id, "CT")

        if os.path.exists(self.t1_folder):
            self.image_list = sorted([f for f in os.listdir(self.t1_folder) if f.endswith(('.png', '.jpg'))], 
                                     key=lambda x: int(os.path.splitext(x)[0]) if x[0].isdigit() else x)
            self.slider.setMaximum(len(self.image_list) - 1)
            self.slider.setValue(0); self.slider.setEnabled(True)
            self.calculate_sequence_mean() 
            self.run_segmentation(0)
        else:
            self.image_list = []; self.slider.setEnabled(False)

    def on_slider_changed(self, val):
        if not self.image_list: return
        self.lbl_progress.setText(f"{val+1}/{len(self.image_list)}")
        self.run_segmentation(val)

    def keep_largest_component(self, mask_class):
        """LCC 去雜訊"""
        mask_class = mask_class.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_class, connectivity=8)
        if num_labels <= 1: return mask_class 
        largest_label_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        new_mask = np.zeros_like(mask_class)
        new_mask[labels == largest_label_idx] = 1
        return new_mask

    def predict_mask(self, fname):
        """執行預測 (TTA + LCC)"""
        if self.current_model is None: return None

        p1 = os.path.join(self.t1_folder, fname); p2 = os.path.join(self.t2_folder, fname)
        if not os.path.exists(p1) or not os.path.exists(p2): return None
        
        img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE); img2 = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        i1_c = clahe.apply(img1); i2_c = clahe.apply(img2)
        inp = np.stack([i1_c, i2_c], axis=0).astype(np.float32) / 255.0
        inp_t = torch.from_numpy(inp).unsqueeze(0).float().to(DEVICE)
        
        # 推論 (TTA: Flip Augmentation)
        with torch.no_grad():
            logits_normal = self.current_model(inp_t)
            inp_flip = torch.flip(inp_t, [3])
            logits_flip = self.current_model(inp_flip)
            logits_flip_back = torch.flip(logits_flip, [3])
            avg_logits = (logits_normal + logits_flip_back) / 2.0

        pred_mask = torch.argmax(avg_logits, dim=1).cpu().numpy()[0]
        
        # 後處理 (LCC)
        mn_mask = (pred_mask == 1).astype(np.uint8)
        mn_clean = self.keep_largest_component(mn_mask)
        ct_mask = (pred_mask == 3).astype(np.uint8)
        ct_clean = self.keep_largest_component(ct_mask)
        
        pred_mask[pred_mask == 1] = 0; pred_mask[pred_mask == 3] = 0
        pred_mask[mn_clean == 1] = 1; pred_mask[ct_clean == 1] = 3
        
        return pred_mask

    def get_gt_mask(self, fname):
        h, w = 512, 512; final = np.zeros((h, w), dtype=np.uint8)
        p_ct = os.path.join(self.ct_folder, fname); p_ft = os.path.join(self.ft_folder, fname); p_mn = os.path.join(self.mn_folder, fname)
        if os.path.exists(p_ct):
            m = cv2.imread(p_ct, cv2.IMREAD_GRAYSCALE); 
            if m is not None: final[m > 127] = 3
        if os.path.exists(p_ft):
            m = cv2.imread(p_ft, cv2.IMREAD_GRAYSCALE); 
            if m is not None: final[m > 127] = 2
        if os.path.exists(p_mn):
            m = cv2.imread(p_mn, cv2.IMREAD_GRAYSCALE); 
            if m is not None: final[m > 127] = 1
        return final

    def calculate_dice(self, pred, target, class_id):
        if class_id == 3: # Union for CT
            p = (pred == 1) | (pred == 2) | (pred == 3); t = (target == 1) | (target == 2) | (target == 3)
        else: p = (pred == class_id); t = (target == class_id)
        inter = (p & t).sum(); union = p.sum() + t.sum()
        if union == 0: return 1.0
        return 2*inter / (union + 1e-5)

    def calculate_sequence_mean(self):
        self.progress_bar.setValue(0); QApplication.processEvents()
        scores = {1: [], 2: [], 3: []}; total = len(self.image_list)
        for i, fname in enumerate(self.image_list):
            pred = self.predict_mask(fname); gt = self.get_gt_mask(fname)
            if pred is None: continue
            scores[1].append(self.calculate_dice(pred, gt, 1))
            scores[2].append(self.calculate_dice(pred, gt, 2))
            scores[3].append(self.calculate_dice(pred, gt, 3))
            if i % 2 == 0: self.progress_bar.setValue(int((i+1)/total*100))
        self.progress_bar.setValue(100)
        self.lbl_mean_mn.setText(f"{np.mean(scores[1]):.2f}"); self.lbl_mean_ft.setText(f"{np.mean(scores[2]):.2f}"); self.lbl_mean_ct.setText(f"{np.mean(scores[3]):.2f}")

    def draw_nice_overlay(self, img_gray, mask):
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB); overlay = np.zeros_like(img_rgb)
        colors = { 1: (0, 255, 255), 2: (255, 0, 0), 3: (0, 0, 255) } # BGR: 黃(MN), 藍(FT), 紅(CT)
        for cid, col in colors.items(): overlay[mask == cid] = col
        output = img_rgb.copy(); cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
        for cid, col in colors.items():
            bin_mask = (mask == cid).astype(np.uint8); contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(output, contours, -1, col, 2)
        return output

    def run_segmentation(self, idx):
        if not self.image_list: return
        fname = self.image_list[idx]; final_pred = self.predict_mask(fname); gt_mask = self.get_gt_mask(fname)
        if final_pred is None: return
        
        d1 = self.calculate_dice(final_pred, gt_mask, 1); d2 = self.calculate_dice(final_pred, gt_mask, 2); d3 = self.calculate_dice(final_pred, gt_mask, 3)
        self.lbl_curr_mn.setText(f"{d1:.2f}"); self.lbl_curr_ft.setText(f"{d2:.2f}"); self.lbl_curr_ct.setText(f"{d3:.2f}")
        
        p1 = os.path.join(self.t1_folder, fname); img1 = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)); img1_c = clahe.apply(img1)
        self.show_on_lbl(cv2.cvtColor(img1_c, cv2.COLOR_GRAY2RGB), self.view_input)
        self.show_on_lbl(self.draw_nice_overlay(img1_c, gt_mask), self.view_gt)
        self.show_on_lbl(self.draw_nice_overlay(img1_c, final_pred), self.view_pred)

    def show_on_lbl(self, img, lbl):
        h, w, c = img.shape
        qimg = QImage(img.data, w, h, c*w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        lbl.setPixmap(pixmap.scaled(lbl.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

if __name__ == "__main__":
    app = QApplication(sys.argv); font = QFont("Microsoft JhengHei", 10); app.setFont(font)
    win = MainWindow(); win.show(); sys.exit(app.exec())