import os
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import json
import datetime
import warnings
import urllib.request
from pathlib import Path

# PyQt6 Imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                             QMessageBox, QStackedWidget, QRadioButton, 
                             QGroupBox, QCheckBox, QFrame, QScrollArea, QDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette

# Matplotlib for PyQt6
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

ROOT = Path.cwd()
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================
# ARQUITETURA DO MODELO PYTORCH
# ============================================

class SkeletonNet(nn.Module):
    def __init__(self, num_features, num_classes, dropout_rate=0.2):
        super(SkeletonNet, self).__init__()
        
        self.entrada = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.bloco1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.reducao1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.bloco2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.saida = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.entrada(x)
        x = x + self.bloco1(x)  
        x = self.reducao1(x)
        x = x + self.bloco2(x)  
        x = self.saida(x)
        return x

# ============================================
# THREAD DA CÂMERA
# ============================================

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    
    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.paused = False

    def run(self):
        # Tentar abrir câmera
        cap = cv2.VideoCapture(0)
        while self._run_flag:
            if not self.paused:
                ret, cv_img = cap.read()
                if ret:
                    self.change_pixmap_signal.emit(cv_img)
            time.sleep(0.03)
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

# ============================================
# CLASSE PRINCIPAL
# ============================================

class PostureEvaluationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Avaliação de Postura - PyQt6 Edition")
        self.setMinimumSize(900, 700)
        
        # Configurações de Path
        self.base_path = ROOT
        self.config_file = os.path.join(self.base_path, 'config.json')
        self.model_filename = ROOT / 'model' / 'modelo_dl_esqueletos.pth'
        self.scaler_filename = ROOT / 'model' / 'scaler_dl.pkl'
        self.encoder_filename = ROOT / 'model' / 'encoder_dl.pkl'
        self.mp_task_filename = 'pose_landmarker.task'
        self.NUM_FEATURES = 99
        
        # Estado
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mediapipe_loaded = False
        self.mlp_loaded = False
        self.detector = None
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.CLASSES = []
        self.current_cv_frame = None
        self.frozen_cv_frame = None
        
        # UI Setup
        self.load_config()
        self.init_ui()
        self.load_models()
        self.apply_theme()
        
    def load_config(self):
        default_config = {
            'theme': 'claro',
            'permissions': {'arquivos': False, 'camera': False}
        }
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
            except:
                self.config = default_config
        else:
            self.config = default_config
        
        self.theme = self.config.get('theme', 'claro')
        self.permissions = self.config.get('permissions', {'arquivos': False, 'camera': False})

    def save_config(self):
        self.config['theme'] = self.theme
        self.config['permissions'] = self.permissions
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=4)

    def init_ui(self):
        self.central_widget = QStackedWidget()
        self.setCentralWidget(self.central_widget)
        
        # Telas
        self.init_home_screen()
        self.init_settings_screen()
        self.init_evaluation_screen()
        self.init_camera_screen()
        
        self.central_widget.setCurrentIndex(0)

    def init_home_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        title = QLabel("AVALIAÇÃO DE POSTURA")
        title.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        title.setStyleSheet("color: #2196F3; margin-bottom: 20px;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.status_lbl = QLabel("Verificando sistema...")
        layout.addWidget(self.status_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        
        btn_eval = QPushButton("📊 Opções de Avaliação")
        btn_eval.setFixedSize(300, 60)
        btn_eval.clicked.connect(lambda: self.central_widget.setCurrentIndex(2))
        layout.addWidget(btn_eval, alignment=Qt.AlignmentFlag.AlignCenter)
        
        btn_settings = QPushButton("⚙️ Configurações")
        btn_settings.setFixedSize(300, 60)
        btn_settings.clicked.connect(lambda: self.central_widget.setCurrentIndex(1))
        layout.addWidget(btn_settings, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.central_widget.addWidget(page)

    def init_settings_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        back_btn = QPushButton("← Voltar")
        back_btn.setFixedWidth(100)
        back_btn.clicked.connect(lambda: self.central_widget.setCurrentIndex(0))
        layout.addWidget(back_btn)
        
        title = QLabel("⚙️ Configurações")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        layout.addWidget(title)
        
        group_theme = QGroupBox("🎨 Tema")
        theme_layout = QHBoxLayout()
        self.radio_claro = QRadioButton("☀️ Claro")
        self.radio_escuro = QRadioButton("🌙 Escuro")
        if self.theme == 'claro': self.radio_claro.setChecked(True)
        else: self.radio_escuro.setChecked(True)
        self.radio_claro.toggled.connect(lambda: self.update_theme('claro'))
        self.radio_escuro.toggled.connect(lambda: self.update_theme('escuro'))
        theme_layout.addWidget(self.radio_claro)
        theme_layout.addWidget(self.radio_escuro)
        group_theme.setLayout(theme_layout)
        layout.addWidget(group_theme)
        
        group_perms = QGroupBox("🔐 Permissões")
        perms_layout = QVBoxLayout()
        self.chk_files = QCheckBox("📁 Acesso a arquivos")
        self.chk_cam = QCheckBox("📷 Acesso a câmera")
        self.chk_files.setChecked(self.permissions['arquivos'])
        self.chk_cam.setChecked(self.permissions['camera'])
        self.chk_files.stateChanged.connect(self.update_permissions)
        self.chk_cam.stateChanged.connect(self.update_permissions)
        perms_layout.addWidget(self.chk_files)
        perms_layout.addWidget(self.chk_cam)
        group_perms.setLayout(perms_layout)
        layout.addWidget(group_perms)
        
        layout.addStretch()
        self.central_widget.addWidget(page)

    def init_evaluation_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        back_btn = QPushButton("← Voltar")
        back_btn.setFixedWidth(100)
        back_btn.clicked.connect(lambda: self.central_widget.setCurrentIndex(0))
        layout.addWidget(back_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        
        title = QLabel("📊 Opções de Avaliação")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        btn_upload = QPushButton("📁 Upload de Arquivo")
        btn_upload.setFixedSize(350, 80)
        btn_upload.clicked.connect(self.upload_file)
        layout.addWidget(btn_upload, alignment=Qt.AlignmentFlag.AlignCenter)
        
        btn_cam = QPushButton("📷 Iniciar Câmera")
        btn_cam.setFixedSize(350, 80)
        btn_cam.clicked.connect(self.start_camera)
        layout.addWidget(btn_cam, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addStretch()
        self.central_widget.addWidget(page)

    def init_camera_screen(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        self.cam_label = QLabel("Iniciando câmera...")
        self.cam_label.setFixedSize(640, 480)
        self.cam_label.setStyleSheet("background-color: black; border: 2px solid #2196F3;")
        layout.addWidget(self.cam_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        controls = QHBoxLayout()
        self.btn_pause = QPushButton("⏸️ Pausar")
        self.btn_classify = QPushButton("📊 Classificar")
        self.btn_classify.setEnabled(False)
        btn_close = QPushButton("❌ Fechar")
        
        self.btn_pause.clicked.connect(self.toggle_camera_pause)
        self.btn_classify.clicked.connect(self.classify_frozen_frame)
        btn_close.clicked.connect(self.stop_camera)
        
        controls.addWidget(self.btn_pause)
        controls.addWidget(self.btn_classify)
        controls.addWidget(btn_close)
        layout.addLayout(controls)
        
        self.cam_result_lbl = QLabel("Aguardando...")
        self.cam_result_lbl.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        layout.addWidget(self.cam_result_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        
        self.central_widget.addWidget(page)

    # ============================================
    # LÓGICA DE NEGÓCIO
    # ============================================

    def load_models(self):
        # MediaPipe
        if not os.path.exists(self.mp_task_filename):
            self.download_mediapipe_model()

        try:
            if os.path.exists(self.mp_task_filename):
                base_options = python.BaseOptions(model_asset_path=str(self.mp_task_filename))
                options = vision.PoseLandmarkerOptions(base_options=base_options, num_poses=1)
                self.detector = vision.PoseLandmarker.create_from_options(options)
                self.mediapipe_loaded = True
        except Exception as e:
            print(f"Erro MediaPipe: {e}")
            
        # PyTorch
        try:
            if os.path.exists(self.model_filename) and os.path.exists(self.scaler_filename) and os.path.exists(self.encoder_filename):
                self.scaler = joblib.load(self.scaler_filename)
                self.label_encoder = joblib.load(self.encoder_filename)
                self.CLASSES = self.label_encoder.classes_
                
                self.model = SkeletonNet(self.NUM_FEATURES, len(self.CLASSES))
                self.model.load_state_dict(torch.load(self.model_filename, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                self.mlp_loaded = True
        except Exception as e:
            print(f"Erro PyTorch: {e}")
            
        self.update_status_text()

    def download_mediapipe_model(self):
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        reply = QMessageBox.question(self, "Modelo Faltando", 
                                   "O arquivo pose_landmarker.task não foi encontrado.\nDeseja baixá-lo agora automaticamente? (aprox. 3MB)",
                                   QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                self.status_lbl.setText("Baixando modelo... aguarde.")
                QApplication.processEvents() # Atualiza a interface
                
                print(f"Baixando modelo de: {url}")
                urllib.request.urlretrieve(url, self.mp_task_filename)
                
                QMessageBox.information(self, "Sucesso", "Modelo baixado com sucesso!")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao baixar modelo automaticamente: {e}\n\nPor favor, baixe manualmente se o erro persistir.")

    def update_status_text(self):
        status = "Sistema: "
        status += "MP ✅ " if self.mediapipe_loaded else "MP ❌ "
        status += "PT ✅" if self.mlp_loaded else "PT ❌"
        self.status_lbl.setText(status)

    def update_theme(self, theme):
        self.theme = theme
        self.save_config()
        self.apply_theme()

    def update_permissions(self):
        self.permissions['arquivos'] = self.chk_files.isChecked()
        self.permissions['camera'] = self.chk_cam.isChecked()
        self.save_config()

    def apply_theme(self):
        if self.theme == 'escuro':
            qss = """
            QMainWindow, QWidget { background-color: #1e1e1e; color: white; }
            QPushButton { background-color: #0d47a1; color: white; border-radius: 5px; padding: 10px; font-weight: bold; }
            QPushButton:hover { background-color: #1565c0; }
            QGroupBox { border: 1px solid #555; margin-top: 10px; padding-top: 10px; font-weight: bold; }
            QLabel { color: #ffffff; }
            """
        else:
            qss = """
            QMainWindow, QWidget { background-color: #f0f0f0; color: black; }
            QPushButton { background-color: #4CAF50; color: white; border-radius: 5px; padding: 10px; font-weight: bold; }
            QPushButton:hover { background-color: #45a049; }
            QGroupBox { border: 1px solid #ccc; margin-top: 10px; padding-top: 10px; font-weight: bold; }
            QLabel { color: #000000; }
            """
        self.setStyleSheet(qss)

    # ============================================
    # CÂMERA E CLASSIFICAÇÃO
    # ============================================

    def start_camera(self):
        if not self.permissions['camera']:
            QMessageBox.warning(self, "Aviso", "Ative a permissão de câmera!")
            return
        
        if not self.mediapipe_loaded or self.detector is None:
            QMessageBox.critical(self, "Erro", "MediaPipe (pose_landmarker.task) não carregado!\nVerifique o arquivo na raiz do projeto.")
            return

        self.central_widget.setCurrentIndex(3)
        self.thread = CameraThread()
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()

    def stop_camera(self):
        if hasattr(self, 'thread'):
            self.thread.stop()
        self.central_widget.setCurrentIndex(2)

    def update_image(self, cv_img):
        self.current_cv_frame = cv_img
        # Processar com skeleton
        processed = self.draw_skeleton(cv_img.copy())
        
        qt_img = self.convert_cv_qt(processed)
        self.cam_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def draw_skeleton(self, frame):
        if not self.mediapipe_loaded or self.detector is None: return frame
        h, w = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            res = self.detector.detect(mp_image)
            if res.pose_landmarks:
                for pose in res.pose_landmarks:
                    for lm in pose:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        color = (0, 255, 0) if lm.visibility > 0.5 else (0, 0, 255)
                        cv2.circle(frame, (cx, cy), 3, color, -1)
        except:
            pass
        return frame

    def toggle_camera_pause(self):
        self.thread.paused = not self.thread.paused
        if self.thread.paused:
            self.btn_pause.setText("▶️ Continuar")
            self.btn_classify.setEnabled(True)
            self.frozen_cv_frame = self.current_cv_frame.copy()
        else:
            self.btn_pause.setText("⏸️ Pausar")
            self.btn_classify.setEnabled(False)
            self.cam_result_lbl.setText("Aguardando...")

    def classify_frozen_frame(self):
        if self.frozen_cv_frame is None or not self.mlp_loaded: return
        
        keypoints = self.extract_keypoints(self.frozen_cv_frame)
        if keypoints is not None:
            res = self.predict(keypoints)
            self.cam_result_lbl.setText(f"🎯 {res['label']} ({res['conf']:.1f}%)")
        else:
            self.cam_result_lbl.setText("❌ Ninguém detectado")

    def extract_keypoints(self, frame):
        if not self.mediapipe_loaded or self.detector is None:
            return None
            
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        try:
            res = self.detector.detect(mp_image)
            if not res.pose_landmarks: return None
            
            feats = []
            for pose in res.pose_landmarks:
                for lm in pose:
                    feats.extend([lm.x, lm.y, lm.visibility])
            
            if len(feats) == 99:
                return np.array(feats).reshape(1, -1)
        except:
            pass
        return None

    def predict(self, keypoints):
        scaled = self.scaler.transform(keypoints)
        tensor = torch.tensor(scaled, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(probs, 1)
        
        label = self.CLASSES[pred.item()]
        return {'label': label, 'conf': conf.item() * 100}

    def upload_file(self):
        if not self.permissions['arquivos']:
            QMessageBox.warning(self, "Aviso", "Ative a permissão de arquivos!")
            return
            
        if not self.mediapipe_loaded or self.detector is None:
            QMessageBox.critical(self, "Erro", "MediaPipe (pose_landmarker.task) não carregado!\nVerifique o arquivo na raiz do projeto.")
            return
        
        file_path, _ = QFileDialog.getOpenFileName(self, "Selecionar Imagem", "", "Imagens (*.png *.jpg *.jpeg)")
        if file_path:
            img = cv2.imread(file_path)
            kp = self.extract_keypoints(img)
            if kp is not None:
                res = self.predict(kp)
                self.show_result_window(file_path, res)
            else:
                QMessageBox.critical(self, "Erro", "Não foi possível detectar pose na imagem.")

    def show_result_window(self, path, res):
        # Janela de resultado visual
        dialog = QDialog(self)
        dialog.setWindowTitle("Resultado da Avaliação")
        dialog.setMinimumSize(800, 650)
        layout = QVBoxLayout(dialog)
        
        # Título
        title = QLabel(f"Postura Detectada: {res['label'].upper()}")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #4CAF50; margin-bottom: 10px;")
        layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Carregar imagem e desenhar esqueleto
        img = cv2.imread(path)
        processed_img = self.draw_skeleton(img.copy())
        
        # Converter para exibição
        qt_img = self.convert_cv_qt(processed_img)
        img_label = QLabel()
        img_label.setPixmap(qt_img.scaled(700, 500, Qt.AspectRatioMode.KeepAspectRatio))
        img_label.setStyleSheet("border: 2px solid #555; background-color: black;")
        layout.addWidget(img_label, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Info
        info_lbl = QLabel(f"Confiança da Inteligência Artificial: {res['conf']:.1f}%")
        info_lbl.setFont(QFont("Arial", 12))
        layout.addWidget(info_lbl, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Botão Fechar
        close_btn = QPushButton("Fechar")
        close_btn.clicked.connect(dialog.accept)
        close_btn.setFixedWidth(200)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        
        dialog.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PostureEvaluationApp()
    window.show()
    sys.exit(app.exec())
