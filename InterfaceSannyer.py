# ============================================
# INTERFACE COMPLETA - PYTORCH EDITION
# ARQUIVOS NA RAIZ: .pth, .pkl, .task
# ============================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
import threading
import time
from PIL import Image, ImageTk
import datetime
import warnings
import json
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path

ROOT = Path.cwd()

# Suprimir warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("="*60)
print("DIAGNÓSTICO DE AMBIENTE")
print("="*60)
print(f"Python: {sys.version}")
print(f"Diretório atual: {os.getcwd()}")

# ============================================
# ARQUITETURA DO MODELO PYTORCH
# ============================================

class SkeletonNet(nn.Module):
    """
    Rede neural para classificação de postura baseada em skeletons.
    Entrada: 99 features (33 keypoints × 3: x, y, visibility)
    Saída: 9 classes (POLAR dataset)
    """
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
        x = x + self.bloco1(x)  # Conexão residual
        x = self.reducao1(x)
        x = x + self.bloco2(x)  # Conexão residual
        x = self.saida(x)
        return x

# ============================================
# CLASSE PRINCIPAL
# ============================================

class PostureEvaluationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Avaliação de Postura - PyTorch Edition")
        self.root.geometry("800x600")
        
        # ============================================
        # CONFIGURAÇÕES - TODOS OS ARQUIVOS NA RAIZ
        # ============================================
        
        self.base_path = ROOT
        self.config_file = os.path.join(self.base_path, 'config.json')
        
        # 🔧 TODOS OS ARQUIVOS NA MESMA PASTA (RAIZ)
        self.model_filename = ROOT /'model'/'modelo_dl_esqueletos.pth'
        self.scaler_filename =ROOT /'model'/ 'scaler_dl.pkl'
        self.encoder_filename = ROOT /'model'/'encoder_dl.pkl'
        self.mp_task_filename = 'pose_landmarker.task'
        
        # 33 keypoints × 3 (x, y, visibility) = 99 features
        self.NUM_FEATURES = 99
        
        print(f"\nDiretório base: {self.base_path}")
        print(f"Arquivos esperados na raiz:")
        print(f"   - {self.model_filename}")
        print(f"   - {self.scaler_filename}")
        print(f"   - {self.encoder_filename}")
        print(f"   - {self.mp_task_filename}")
        
        # ============================================
        # CARREGAR CONFIGURAÇÕES
        # ============================================
        
        self.load_config()
        self.theme = self.config.get('theme', 'claro')
        self.permissions = self.config.get('permissions', {
            "arquivos": False,
            "camera": False
        })
        
        # ============================================
        # HARDWARE
        # ============================================
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"\nDispositivo: {self.device}")
        
        # ============================================
        # VARIÁVEIS
        # ============================================
        
        # Câmera
        self.camera_active = False
        self.camera_paused = False
        self.cap = None
        self.camera_thread = None
        self.current_frame = None
        self.frozen_frame = None
        self.detector = None
        self.camera_running = False
        
        # Modelo
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.CLASSES = []
        self.mlp_loaded = False
        self.mediapipe_loaded = False
        
        # UI
        self.video_label = None
        self.info_label = None
        self.result_label = None
        self.pause_btn = None
        self.classify_btn = None
        
        # ============================================
        # CORES
        # ============================================
        
        self.colors = {
            "claro": {
                "bg": "#f0f0f0", "fg": "#000000",
                "button_bg": "#4CAF50", "button_fg": "#ffffff",
                "frame_bg": "#ffffff", "title_color": "#2196F3"
            },
            "escuro": {
                "bg": "#1e1e1e", "fg": "#ffffff",
                "button_bg": "#0d47a1", "button_fg": "#ffffff",
                "frame_bg": "#2d2d2d", "title_color": "#64B5F6"
            }
        }
        
        # ============================================
        # INICIAR
        # ============================================
        
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.verificar_arquivos()
        self.load_models()
        self.show_initial_screen()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    # ============================================
    # VERIFICAR ARQUIVOS
    # ============================================
    
    def verificar_arquivos(self):
        """Verifica se os arquivos necessários existem na raiz"""
        print("\n" + "="*60)
        print("VERIFICANDO ARQUIVOS NA RAIZ")
        print("="*60)
        
        arquivos = [
            (self.mp_task_filename, 'MediaPipe'),
            (self.model_filename, 'Modelo PyTorch'),
            (self.scaler_filename, 'Scaler'),
            (self.encoder_filename, 'LabelEncoder')
        ]
        
        todos_existem = True
        for nome, desc in arquivos:
            caminho = os.path.join(self.base_path, nome)
            existe = os.path.exists(caminho)
            status = "✅" if existe else "❌"
            print(f"   {status} {desc}: {nome}")
            if not existe:
                todos_existem = False
        
        if not todos_existem:
            print(f"\nAlguns arquivos não foram encontrados!")
            print(f"   Verifique se todos estão em: {self.base_path}")
            print("\n   Estrutura esperada:")
            print(f"   {self.base_path}\\")
            print(f"   ├── {self.mp_task_filename}")
            print(f"   ├── {self.model_filename}")
            print(f"   ├── {self.scaler_filename}")
            print(f"   ├── {self.encoder_filename}")
            print(f"   └── config.json")
        else:
            print("\nTodos os arquivos encontrados na raiz!")
    
    # ============================================
    # SISTEMA DE CONFIGURAÇÕES JSON
    # ============================================
    
    def load_config(self):
        """Carrega as configurações do arquivo JSON"""
        default_config = {
            'theme': 'claro',
            'permissions': {
                'arquivos': False,
                'camera': False
            },
            'last_directory': '',
            'window_size': {'width': 800, 'height': 600}
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.config = json.load(f)
                print(f"Configurações carregadas de: {self.config_file}")
            except Exception as e:
                print(f"Erro ao carregar configurações: {e}")
                self.config = default_config
                self.save_config()
        else:
            self.config = default_config
            self.save_config()
            print(f"Configurações padrão criadas em: {self.config_file}")
    
    def save_config(self):
        """Salva as configurações no arquivo JSON"""
        try:
            self.config['theme'] = self.theme
            self.config['permissions'] = self.permissions
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            print(f"Configurações salvas em: {self.config_file}")
        except Exception as e:
            print(f"Erro ao salvar configurações: {e}")
    
    # ============================================
    # CARREGAR MODELOS
    # ============================================
    
    def load_models(self):
        """Carrega todos os modelos (MediaPipe + PyTorch)"""
        print("\n" + "="*60)
        print("CARREGANDO MODELOS")
        print("="*60)
        print(f"Diretório base: {self.base_path}")
        print(f"ispositivo: {self.device}")
        
        # 1. MediaPipe
        self.load_mediapipe()
        
        # 2. PyTorch
        self.load_pytorch_model()
        
        print("="*60 + "\n")
    
    def load_mediapipe(self):
        """Carrega MediaPipe Pose usando Tasks API"""
        try:
            mediapipe_path = os.path.join(self.base_path, self.mp_task_filename)
            
            if not os.path.exists(mediapipe_path):
                print(f"❌ MediaPipe não encontrado: {mediapipe_path}")
                self.mediapipe_loaded = False
                return
            
            base_options = python.BaseOptions(model_asset_path=mediapipe_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False,
                num_poses=1
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            self.mediapipe_loaded = True
            
            print("✅ MediaPipe Pose carregado com sucesso!")
            print(f"   📁 Arquivo: {os.path.basename(mediapipe_path)}")
            
        except Exception as e:
            print(f"❌ Erro no MediaPipe: {e}")
            self.mediapipe_loaded = False
            self.detector = None
    
    def load_pytorch_model(self):
        """Carrega o modelo PyTorch, Scaler e LabelEncoder"""
        try:
            # 🔧 TODOS OS ARQUIVOS NA RAIZ
            modelo_path = os.path.join(self.base_path, self.model_filename)
            scaler_path = os.path.join(self.base_path, self.scaler_filename)
            encoder_path = os.path.join(self.base_path, self.encoder_filename)
            
            print(f"🔍 Procurando modelo: {modelo_path}")
            print(f"🔍 Procurando scaler: {scaler_path}")
            print(f"🔍 Procurando encoder: {encoder_path}")
            
            # Verificar se os arquivos existem
            if not os.path.exists(modelo_path):
                print(f"❌ Modelo PyTorch não encontrado: {modelo_path}")
                self.mlp_loaded = False
                return
            
            if not os.path.exists(scaler_path):
                print(f"❌ Scaler não encontrado: {scaler_path}")
                self.mlp_loaded = False
                return
            
            if not os.path.exists(encoder_path):
                print(f"❌ LabelEncoder não encontrado: {encoder_path}")
                self.mlp_loaded = False
                return
            
            # Carregar Scaler e LabelEncoder
            print("📦 Carregando scaler...")
            self.scaler = joblib.load(scaler_path)
            print("✅ Scaler carregado")
            
            print("📦 Carregando label encoder...")
            self.label_encoder = joblib.load(encoder_path)
            self.CLASSES = self.label_encoder.classes_
            print(f"✅ LabelEncoder carregado: {self.CLASSES}")
            
            # Carregar modelo PyTorch
            num_classes = len(self.CLASSES)
            print(f"📦 Criando modelo com {num_classes} classes...")
            self.model = SkeletonNet(self.NUM_FEATURES, num_classes)
            
            print(f"📦 Carregando pesos do modelo de: {modelo_path}")
            state_dict = torch.load(modelo_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()  # Modo de avaliação
            
            self.mlp_loaded = True
            
            print(f"✅ Modelo PyTorch carregado com sucesso!")
            print(f"   📁 Arquivo: {os.path.basename(modelo_path)}")
            print(f"   📊 Features: {self.NUM_FEATURES}")
            print(f"   🎯 Classes: {self.CLASSES}")
            print(f"   💻 Dispositivo: {self.device}")
            
        except PermissionError as e:
            print(f"❌ Erro de permissão: {e}")
            print("   ⚠️ O arquivo .pth pode estar bloqueado.")
            print("   Soluções:")
            print("   1. Feche qualquer programa que possa estar usando o arquivo")
            print("   2. Verifique se o arquivo não está como 'Somente leitura'")
            print("   3. Mova o arquivo para a raiz da pasta TCC")
            self.mlp_loaded = False
            
        except Exception as e:
            print(f"❌ Erro no PyTorch: {e}")
            import traceback
            traceback.print_exc()
            self.mlp_loaded = False
            self.model = None
            self.scaler = None
    
    # ============================================
    # EXTRAÇÃO DE KEYPOINTS (99 FEATURES)
    # ============================================
    
    def extract_keypoints_from_frame(self, frame):
        """
        Extrai os 33 keypoints com x, y, visibility = 99 features
        """
        if not self.mediapipe_loaded or self.detector is None:
            return None, None
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            result = self.detector.detect(mp_image)
        except:
            return None, None
        
        if not result.pose_landmarks:
            return None, None
        
        features = []
        keypoints_detectados = []
        
        for pose in result.pose_landmarks:
            for i, lm in enumerate(pose):
                # Padrão do treinamento: x, y, visibility
                features.extend([round(lm.x, 6), round(lm.y, 6), round(lm.visibility, 6)])
                keypoints_detectados.append(i)
        
        if len(features) == 0:
            return None, None
        
        keypoints_array = np.array(features).reshape(1, -1)
        
        # Verificar se tem 99 features
        if keypoints_array.shape[1] != self.NUM_FEATURES:
            print(f"⚠️ Features incorretas: {keypoints_array.shape[1]} (esperado: {self.NUM_FEATURES})")
            return None, None
        
        return keypoints_array, keypoints_detectados
    
    def extract_keypoints_from_image(self, image_path):
        """
        Extrai keypoints de uma imagem
        """
        if not self.mediapipe_loaded or self.detector is None:
            return None, None, None
        
        imagem = cv2.imread(image_path)
        if imagem is None:
            return None, None, None
        
        altura, largura, _ = imagem.shape
        
        imagem_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imagem_rgb)
        
        try:
            resultado = self.detector.detect(mp_image)
        except Exception as e:
            print(f"❌ Erro na detecção: {e}")
            return None, None, None
        
        if not resultado.pose_landmarks:
            return None, None, None
        
        features = []
        keypoints_dict = {}
        keypoints_detectados = []
        
        for pose in resultado.pose_landmarks:
            for i, lm in enumerate(pose):
                x, y = lm.x, lm.y
                vis = lm.visibility
                features.extend([round(x, 6), round(y, 6), round(vis, 6)])
                
                x_px, y_px = int(x * largura), int(y * altura)
                keypoints_dict[f'kp_{i}'] = (x_px, y_px, x, y, vis)
                keypoints_detectados.append(i)
        
        if len(features) == 0:
            return None, None, None
        
        keypoints_array = np.array(features).reshape(1, -1)
        
        if keypoints_array.shape[1] != self.NUM_FEATURES:
            print(f"⚠️ Features incorretas: {keypoints_array.shape[1]} (esperado: {self.NUM_FEATURES})")
            return None, None, None
        
        print(f"✅ Extraídos {len(keypoints_detectados)} keypoints")
        return keypoints_array, keypoints_dict, keypoints_detectados
    
    # ============================================
    # CLASSIFICAÇÃO COM PYTORCH
    # ============================================
    
    def classify_posture(self, keypoints_array):
        """
        Classifica a postura usando o modelo PyTorch
        """
        if not self.mlp_loaded or self.model is None:
            return {
                'classe': 'indisponivel',
                'confianca': 0,
                'mensagem': 'Modelo PyTorch não carregado'
            }
        
        if keypoints_array.shape[1] != self.NUM_FEATURES:
            return {
                'classe': 'erro',
                'confianca': 0,
                'mensagem': f'Features incorretas: {keypoints_array.shape[1]}'
            }
        
        try:
            # 1. Normalizar com o scaler do treinamento
            features_scaled = self.scaler.transform(keypoints_array)
            
            # 2. Converter para tensor PyTorch
            tensor_input = torch.tensor(features_scaled, dtype=torch.float32).to(self.device)
            
            # 3. Fazer predição
            with torch.no_grad():
                outputs = self.model(tensor_input)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(probs, 1)
            
            classe_idx = pred.item()
            conf_val = conf.item()
            classe = self.CLASSES[classe_idx]
            confianca_percent = conf_val * 100
            
            # Mapear para nomes em português (opcional)
            nomes_pt = {
                'bendover': 'Curvado',
                'jump': 'Pulando',
                'lying': 'Deitado',
                'run': 'Correndo',
                'sit': 'Sentado',
                'squat': 'Agachado',
                'stand': 'Em pé',
                'stretch': 'Alongando',
                'walk': 'Andando'
            }
            
            nome_exibicao = nomes_pt.get(classe, classe)
            
            resultado = {
                'classe': classe,
                'classe_exibicao': nome_exibicao,
                'confianca': conf_val,
                'confianca_percent': confianca_percent,
                'probabilidades': probs.cpu().numpy()[0],
                'mensagem': f"{nome_exibicao} ({confianca_percent:.1f}%)"
            }
            
            print(f"\n📊 RESULTADO:")
            print(f"   🎯 Classe: {classe} ({nome_exibicao})")
            print(f"   📊 Confiança: {confianca_percent:.1f}%")
            
            return resultado
            
        except Exception as e:
            print(f"❌ Erro na classificação: {e}")
            import traceback
            traceback.print_exc()
            return {
                'classe': 'erro',
                'confianca': 0,
                'mensagem': f'Erro: {str(e)}'
            }
    
    # ============================================
    # INTERFACE - MÉTODOS VISUAIS
    # ============================================
    
    def get_colors(self):
        return self.colors[self.theme]
    
    def clear_screen(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
    
    def show_initial_screen(self):
        """Tela inicial"""
        self.clear_screen()
        colors = self.get_colors()
        self.main_frame.configure(bg=colors["bg"])
        
        title_frame = tk.Frame(self.main_frame, bg=colors["bg"])
        title_frame.pack(pady=50)
        
        title = tk.Label(title_frame, 
                        text="AVALIAÇÃO DE POSTURA - PYTORCH",
                        font=("Arial", 24, "bold"),
                        fg=colors["title_color"],
                        bg=colors["bg"])
        title.pack()
        
        status_text = "Sistema "
        if self.mediapipe_loaded and self.mlp_loaded:
            status_text += f"✅ completo ({len(self.CLASSES)} classes)"
        elif self.mediapipe_loaded:
            status_text += "⚠️ parcial (apenas keypoints)"
        else:
            status_text += "❌ não configurado"
        
        subtitle = tk.Label(title_frame,
                          text=status_text,
                          font=("Arial", 11),
                          fg=colors["fg"],
                          bg=colors["bg"])
        subtitle.pack(pady=5)
        
        button_frame = tk.Frame(self.main_frame, bg=colors["bg"])
        button_frame.pack(pady=50)
        
        config_btn = tk.Button(button_frame,
                              text="⚙️ Configurações",
                              font=("Arial", 14),
                              bg=colors["button_bg"],
                              fg=colors["button_fg"],
                              padx=30,
                              pady=15,
                              width=25,
                              command=self.show_settings)
        config_btn.pack(pady=10)
        
        eval_btn = tk.Button(button_frame,
                            text="📊 Opções de Avaliação",
                            font=("Arial", 14),
                            bg=colors["button_bg"],
                            fg=colors["button_fg"],
                            padx=30,
                            pady=15,
                            width=25,
                            command=self.show_evaluation_options)
        eval_btn.pack(pady=10)
        
        models_btn = tk.Button(button_frame,
                              text="🔍 Verificar Modelos",
                              font=("Arial", 12),
                              bg="#FF9800",
                              fg="white",
                              padx=30,
                              pady=10,
                              width=25,
                              command=self.show_model_status)
        models_btn.pack(pady=10)
        
        self.update_status_label()
    
    def update_status_label(self):
        colors = self.get_colors()
        
        for widget in self.main_frame.winfo_children():
            if isinstance(widget, tk.Label) and widget.winfo_y() > 500:
                widget.destroy()
        
        status_parts = []
        
        if self.permissions["arquivos"]:
            status_parts.append("📁 ✅")
        else:
            status_parts.append("📁 ❌")
        
        if self.permissions["camera"]:
            status_parts.append("📷 ✅")
        else:
            status_parts.append("📷 ❌")
        
        if self.mediapipe_loaded:
            status_parts.append("📷 MP ✅")
        else:
            status_parts.append("📷 MP ❌")
        
        if self.mlp_loaded:
            status_parts.append(f"🧠 PT ✅ ({len(self.CLASSES)} cls)")
        else:
            status_parts.append("🧠 PT ❌")
        
        status_text = " | ".join(status_parts)
        
        status_label = tk.Label(self.main_frame,
                               text=status_text,
                               font=("Arial", 10),
                               fg=colors["fg"],
                               bg=colors["bg"])
        status_label.pack(side=tk.BOTTOM, pady=10)
    
    # ============================================
    # CONFIGURAÇÕES
    # ============================================
    
    def show_settings(self):
        self.clear_screen()
        colors = self.get_colors()
        self.main_frame.configure(bg=colors["bg"])
        
        top_frame = tk.Frame(self.main_frame, bg=colors["bg"])
        top_frame.pack(fill=tk.X, pady=10, padx=10)
        
        back_btn = tk.Button(top_frame,
                            text="← Voltar",
                            font=("Arial", 11),
                            bg="#f44336",
                            fg="white",
                            padx=15,
                            pady=5,
                            command=self.show_initial_screen)
        back_btn.pack(side=tk.LEFT)
        
        title = tk.Label(self.main_frame,
                        text="⚙️ Configurações",
                        font=("Arial", 20, "bold"),
                        fg=colors["title_color"],
                        bg=colors["bg"])
        title.pack(pady=20)
        
        settings_frame = tk.Frame(self.main_frame, bg=colors["bg"])
        settings_frame.pack(pady=20)
        
        # Tema
        theme_frame = tk.Frame(settings_frame, bg=colors["bg"])
        theme_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(theme_frame,
                text="🎨 Tema:",
                font=("Arial", 12),
                fg=colors["fg"],
                bg=colors["bg"]).pack(side=tk.LEFT, padx=10)
        
        theme_var = tk.StringVar(value=self.theme)
        theme_radio_frame = tk.Frame(theme_frame, bg=colors["bg"])
        theme_radio_frame.pack(side=tk.LEFT, padx=20)
        
        tk.Radiobutton(theme_radio_frame,
                      text="☀️ Claro",
                      variable=theme_var,
                      value="claro",
                      bg=colors["bg"],
                      fg=colors["fg"],
                      selectcolor=colors["bg"],
                      command=lambda: self.change_theme("claro")).pack(side=tk.LEFT, padx=10)
        
        tk.Radiobutton(theme_radio_frame,
                      text="🌙 Escuro",
                      variable=theme_var,
                      value="escuro",
                      bg=colors["bg"],
                      fg=colors["fg"],
                      selectcolor=colors["bg"],
                      command=lambda: self.change_theme("escuro")).pack(side=tk.LEFT, padx=10)
        
        ttk.Separator(settings_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Permissões
        perms_frame = tk.Frame(settings_frame, bg=colors["bg"])
        perms_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(perms_frame,
                text="🔐 Permissões:",
                font=("Arial", 12),
                fg=colors["fg"],
                bg=colors["bg"]).pack(side=tk.LEFT, padx=10)
        
        perms_check_frame = tk.Frame(perms_frame, bg=colors["bg"])
        perms_check_frame.pack(side=tk.LEFT, padx=20)
        
        arquivos_var = tk.BooleanVar(value=self.permissions["arquivos"])
        tk.Checkbutton(perms_check_frame,
                      text="📁 Acesso a arquivos",
                      variable=arquivos_var,
                      bg=colors["bg"],
                      fg=colors["fg"],
                      selectcolor=colors["bg"],
                      font=("Arial", 10),
                      command=lambda: self.toggle_permission("arquivos", arquivos_var.get())).pack(anchor=tk.W, pady=3)
        
        camera_var = tk.BooleanVar(value=self.permissions["camera"])
        tk.Checkbutton(perms_check_frame,
                      text="📷 Acesso a câmera",
                      variable=camera_var,
                      bg=colors["bg"],
                      fg=colors["fg"],
                      selectcolor=colors["bg"],
                      font=("Arial", 10),
                      command=lambda: self.toggle_permission("camera", camera_var.get())).pack(anchor=tk.W, pady=3)
        
        ttk.Separator(settings_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Tutorial
        tutorial_frame = tk.Frame(settings_frame, bg=colors["bg"])
        tutorial_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(tutorial_frame,
                text="📖 Tutorial:",
                font=("Arial", 12),
                fg=colors["fg"],
                bg=colors["bg"]).pack(side=tk.LEFT, padx=10)
        
        tutorial_btn = tk.Button(tutorial_frame,
                                text="Ver Tutorial",
                                bg=colors["button_bg"],
                                fg=colors["button_fg"],
                                font=("Arial", 10),
                                padx=15,
                                command=self.show_tutorial)
        tutorial_btn.pack(side=tk.LEFT, padx=20)
        
        tk.Label(settings_frame,
                text="✅ As configurações são salvas automaticamente",
                font=("Arial", 9),
                fg="#4CAF50",
                bg=colors["bg"]).pack(pady=10)
    
    def change_theme(self, theme):
        self.theme = theme
        self.save_config()
        self.show_settings()
    
    def toggle_permission(self, perm_type, value):
        self.permissions[perm_type] = value
        self.save_config()
        self.update_status_label()
        print(f"Permissão de {perm_type} {'ativada' if value else 'desativada'}")
    
    # ============================================
    # TUTORIAL
    # ============================================
    
    def show_tutorial(self):
        self.clear_screen()
        colors = self.get_colors()
        self.main_frame.configure(bg=colors["bg"])
        
        top_frame = tk.Frame(self.main_frame, bg=colors["bg"])
        top_frame.pack(fill=tk.X, pady=10, padx=10)
        
        back_btn = tk.Button(top_frame,
                            text="← Voltar",
                            font=("Arial", 11),
                            bg="#f44336",
                            fg="white",
                            padx=15,
                            pady=5,
                            command=self.show_settings)
        back_btn.pack(side=tk.LEFT)
        
        title = tk.Label(self.main_frame,
                        text="📖 Tutorial - PyTorch",
                        font=("Arial", 20, "bold"),
                        fg=colors["title_color"],
                        bg=colors["bg"])
        title.pack(pady=20)
        
        tutorial_text = f"""
        Sistema de avaliação de postura com PyTorch.
        
        📁 Diretório dos modelos:
        {self.base_path}
        
        🔑 Features: {self.NUM_FEATURES} (33 keypoints × 3)
        🎯 Classes: {len(self.CLASSES)}
        📊 Dispositivo: {self.device}
        
        📝 Como usar:
        
        1. 📷 Câmera:
           • Pause com 'Pausar'
           • Clique em 'Classificar Frame'
           • Veja o resultado da postura
        
        2. 📁 Upload:
           • Selecione uma imagem (JPG, PNG)
           • O sistema classifica automaticamente
        
        ⚠️ Requisitos:
        • Ative as permissões nas configurações
        • Fique a 1-2 metros da câmera
        """
        
        text_widget = tk.Text(self.main_frame,
                             wrap=tk.WORD,
                             font=("Arial", 11),
                             bg=colors["frame_bg"],
                             fg=colors["fg"],
                             padx=20,
                             pady=20,
                             height=20)
        text_widget.insert("1.0", tutorial_text)
        text_widget.config(state=tk.DISABLED)
        text_widget.pack(fill=tk.BOTH, expand=True, padx=30, pady=10)
    
    # ============================================
    # OPÇÕES DE AVALIAÇÃO
    # ============================================
    
    def show_evaluation_options(self):
        self.clear_screen()
        colors = self.get_colors()
        self.main_frame.configure(bg=colors["bg"])
        
        top_frame = tk.Frame(self.main_frame, bg=colors["bg"])
        top_frame.pack(fill=tk.X, pady=10, padx=10)
        
        back_btn = tk.Button(top_frame,
                            text="← Voltar",
                            font=("Arial", 11),
                            bg="#f44336",
                            fg="white",
                            padx=15,
                            pady=5,
                            command=self.show_initial_screen)
        back_btn.pack(side=tk.LEFT)
        
        title = tk.Label(self.main_frame,
                        text="📊 Opções de Avaliação",
                        font=("Arial", 20, "bold"),
                        fg=colors["title_color"],
                        bg=colors["bg"])
        title.pack(pady=20)
        
        options_frame = tk.Frame(self.main_frame, bg=colors["bg"])
        options_frame.pack(pady=30)
        
        upload_btn = tk.Button(options_frame,
                              text="📁 Upload de Arquivo",
                              font=("Arial", 14),
                              bg=colors["button_bg"],
                              fg=colors["button_fg"],
                              padx=40,
                              pady=20,
                              width=30,
                              command=self.upload_file)
        upload_btn.pack(pady=10)
        
        tk.Label(options_frame,
                text=f"Selecionar imagem para classificar ({len(self.CLASSES)} classes)",
                font=("Arial", 10),
                fg=colors["fg"],
                bg=colors["bg"]).pack(pady=5)
        
        camera_btn = tk.Button(options_frame,
                              text="📷 Iniciar Câmera",
                              font=("Arial", 14),
                              bg=colors["button_bg"],
                              fg=colors["button_fg"],
                              padx=40,
                              pady=20,
                              width=30,
                              command=self.start_camera_interface)
        camera_btn.pack(pady=10)
        
        tk.Label(options_frame,
                text="Análise em tempo real com PyTorch",
                font=("Arial", 10),
                fg=colors["fg"],
                bg=colors["bg"]).pack(pady=5)
    
    # ============================================
    # STATUS DOS MODELOS
    # ============================================
    
    def show_model_status(self):
        """Mostra status dos modelos"""
        status_window = tk.Toplevel(self.root)
        status_window.title("Status dos Modelos - PyTorch")
        status_window.geometry("550x450")
        status_window.configure(bg="#2d2d2d")
        
        title = tk.Label(status_window,
                        text="Status dos Modelos - PyTorch",
                        font=("Arial", 14, "bold"),
                        fg="#64B5F6",
                        bg="#2d2d2d")
        title.pack(pady=15)
        
        status_frame = tk.Frame(status_window, bg="#2d2d2d")
        status_frame.pack(pady=10, padx=20, fill=tk.BOTH)
        
        # MediaPipe
        mp_status = "✅ Carregado" if self.mediapipe_loaded else "❌ Falha"
        mp_color = "#4CAF50" if self.mediapipe_loaded else "#f44336"
        
        tk.Label(status_frame,
                text="📷 MediaPipe Pose:",
                font=("Arial", 11, "bold"),
                fg="white",
                bg="#2d2d2d").pack(anchor=tk.W, pady=5)
        
        tk.Label(status_frame,
                text=mp_status,
                font=("Arial", 11),
                fg=mp_color,
                bg="#2d2d2d").pack(anchor=tk.W, pady=2)
        
        if self.mediapipe_loaded:
            tk.Label(status_frame,
                    text=f"  • Arquivo: {self.mp_task_filename}",
                    font=("Arial", 10),
                    fg="#aaaaaa",
                    bg="#2d2d2d").pack(anchor=tk.W, pady=2)
        
        ttk.Separator(status_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # PyTorch
        mlp_status = "✅ Carregado" if self.mlp_loaded else "❌ Falha"
        mlp_color = "#4CAF50" if self.mlp_loaded else "#f44336"
        
        tk.Label(status_frame,
                text="🧠 Modelo PyTorch:",
                font=("Arial", 11, "bold"),
                fg="white",
                bg="#2d2d2d").pack(anchor=tk.W, pady=5)
        
        tk.Label(status_frame,
                text=mlp_status,
                font=("Arial", 11),
                fg=mlp_color,
                bg="#2d2d2d").pack(anchor=tk.W, pady=2)
        
        if self.mlp_loaded:
            tk.Label(status_frame,
                    text=f"  • Arquivo: {self.model_filename}",
                    font=("Arial", 10),
                    fg="#aaaaaa",
                    bg="#2d2d2d").pack(anchor=tk.W, pady=2)
            tk.Label(status_frame,
                    text=f"  • Features: {self.NUM_FEATURES}",
                    font=("Arial", 10),
                    fg="#aaaaaa",
                    bg="#2d2d2d").pack(anchor=tk.W, pady=2)
            tk.Label(status_frame,
                    text=f"  • Classes: {len(self.CLASSES)}",
                    font=("Arial", 10),
                    fg="#aaaaaa",
                    bg="#2d2d2d").pack(anchor=tk.W, pady=2)
            tk.Label(status_frame,
                    text=f"  • Dispositivo: {self.device}",
                    font=("Arial", 10),
                    fg="#aaaaaa",
                    bg="#2d2d2d").pack(anchor=tk.W, pady=2)
            tk.Label(status_frame,
                    text=f"  • Classes: {', '.join(self.CLASSES)}",
                    font=("Arial", 9),
                    fg="#aaaaaa",
                    bg="#2d2d2d").pack(anchor=tk.W, pady=2)
        else:
            tk.Label(status_frame,
                    text="  • Verifique os arquivos na raiz:",
                    font=("Arial", 10),
                    fg="#FFC107",
                    bg="#2d2d2d").pack(anchor=tk.W, pady=2)
            tk.Label(status_frame,
                    text=f"    {self.base_path}",
                    font=("Arial", 9),
                    fg="#aaaaaa",
                    bg="#2d2d2d").pack(anchor=tk.W, pady=2)
            tk.Label(status_frame,
                    text=f"    - {self.model_filename}",
                    font=("Arial", 9),
                    fg="#aaaaaa",
                    bg="#2d2d2d").pack(anchor=tk.W, pady=2)
            tk.Label(status_frame,
                    text=f"    - {self.scaler_filename}",
                    font=("Arial", 9),
                    fg="#aaaaaa",
                    bg="#2d2d2d").pack(anchor=tk.W, pady=2)
            tk.Label(status_frame,
                    text=f"    - {self.encoder_filename}",
                    font=("Arial", 9),
                    fg="#aaaaaa",
                    bg="#2d2d2d").pack(anchor=tk.W, pady=2)
        
        close_btn = tk.Button(status_window,
                             text="Fechar",
                             bg="#4CAF50",
                             fg="white",
                             font=("Arial", 10),
                             padx=20,
                             command=status_window.destroy)
        close_btn.pack(pady=15)
    
    # ============================================
    # RESULTADO
    # ============================================
    
    def show_result_window(self, image_path, keypoints_dict, keypoints_detectados, resultado):
        """Janela de resultado com skeleton e legenda"""
        
        result_window = tk.Toplevel(self.root)
        result_window.title("Resultado da Avaliação - PyTorch")
        result_window.geometry("1100x800")
        result_window.configure(bg="#2d2d2d")
        result_window.minsize(900, 650)
        
        top_frame = tk.Frame(result_window, bg="#2d2d2d")
        top_frame.pack(fill=tk.X, pady=10, padx=10)
        
        back_btn = tk.Button(top_frame,
                            text="← Voltar",
                            font=("Arial", 11),
                            bg="#f44336",
                            fg="white",
                            padx=20,
                            pady=8,
                            command=lambda: self.close_result_window(result_window))
        back_btn.pack(side=tk.LEFT)
        
        main_container = tk.Frame(result_window, bg="#2d2d2d")
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        main_container.grid_columnconfigure(0, weight=65)
        main_container.grid_columnconfigure(1, weight=35)
        main_container.grid_rowconfigure(0, weight=1)
        
        image_frame = tk.Frame(main_container, bg="#1e1e1e", relief=tk.RAISED, bd=2)
        image_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        fig, ax = plt.subplots(figsize=(10, 9))
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        imagem = cv2.imread(image_path)
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        ax.imshow(imagem)
        
        # Desenhar keypoints detectados
        if keypoints_dict and keypoints_detectados:
            for i, kp_idx in enumerate(keypoints_detectados):
                if kp_idx in keypoints_dict:
                    x, y, x_norm, y_norm, vis = keypoints_dict[kp_idx]
                    cor = 'cyan' if vis > 0.5 else 'gray'
                    ax.plot(x, y, 'o', markersize=8, color=cor, 
                           markeredgecolor='white', markeredgewidth=1)
                    if i % 5 == 0:
                        ax.annotate(str(kp_idx), (x+5, y-5), fontsize=7, color='white')
        
        ax.axis('off')
        plt.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=image_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Legenda
        legend_frame = tk.Frame(main_container, bg="#2d2d2d")
        legend_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)
        
        result_title = tk.Label(legend_frame,
                               text="RESULTADO - PYTORCH",
                               font=("Arial", 18, "bold"),
                               fg="#64B5F6",
                               bg="#2d2d2d")
        result_title.pack(pady=(0, 10))
        
        ttk.Separator(legend_frame, orient='horizontal').pack(fill=tk.X, pady=5)
        
        if 'classe' in resultado and resultado['classe'] in self.CLASSES:
            cor = "#4CAF50"
            classe_text = f"POSTURA: {resultado.get('classe_exibicao', resultado['classe']).upper()}"
        else:
            cor = "#FF9800"
            classe_text = f"ERRO: {resultado.get('mensagem', 'Erro desconhecido')}"
        
        tk.Label(legend_frame,
                text=classe_text,
                font=("Arial", 16, "bold"),
                fg=cor,
                bg="#2d2d2d").pack(pady=5)
        
        tk.Label(legend_frame,
                text=f"Confiança: {resultado['confianca_percent']:.1f}%",
                font=("Arial", 14),
                fg="white",
                bg="#2d2d2d").pack(pady=5)
        
        ttk.Separator(legend_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        # Top 3 probabilidades
        if 'probabilidades' in resultado:
            tk.Label(legend_frame,
                    text="Top 3 classes:",
                    font=("Arial", 12, "bold"),
                    fg="white",
                    bg="#2d2d2d").pack(pady=5)
            
            probs = resultado['probabilidades']
            top_indices = np.argsort(probs)[::-1][:3]
            
            nomes_pt = {
                'bendover': 'Curvado',
                'jump': 'Pulando',
                'lying': 'Deitado',
                'run': 'Correndo',
                'sit': 'Sentado',
                'squat': 'Agachado',
                'stand': 'Em pé',
                'stretch': 'Alongando',
                'walk': 'Andando'
            }
            
            for idx in top_indices:
                classe = self.CLASSES[idx]
                prob = probs[idx] * 100
                nome_exibicao = nomes_pt.get(classe, classe)
                cor = "#4CAF50" if idx == top_indices[0] else "#aaaaaa"
                tk.Label(legend_frame,
                        text=f"{nome_exibicao}: {prob:.1f}%",
                        font=("Arial", 11),
                        fg=cor,
                        bg="#2d2d2d",
                        wraplength=300,
                        justify=tk.LEFT).pack(anchor=tk.W, pady=2)
        
        ttk.Separator(legend_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        tk.Label(legend_frame,
                text=f"Keypoints detectados: {len(keypoints_detectados)}/33",
                font=("Arial", 12, "bold"),
                fg="white",
                bg="#2d2d2d").pack(pady=5)
        
        tk.Label(legend_frame,
                text=f"Features: {self.NUM_FEATURES} (33 × 3)",
                font=("Arial", 10),
                fg="#aaaaaa",
                bg="#2d2d2d").pack(pady=2)
        
        ttk.Separator(legend_frame, orient='horizontal').pack(fill=tk.X, pady=10)
        
        save_btn = tk.Button(legend_frame,
                            text="💾 SALVAR RESULTADO",
                            font=("Arial", 14, "bold"),
                            bg="#2196F3",
                            fg="white",
                            padx=20,
                            pady=12,
                            relief=tk.RAISED,
                            bd=3,
                            command=lambda: self.save_analysis_result(image_path, resultado))
        save_btn.pack(fill=tk.X, pady=10)
        
        self.result_window = result_window
    
    def close_result_window(self, window):
        try:
            plt.close('all')
            window.destroy()
        except:
            pass
        self.show_evaluation_options()
    
    def save_analysis_result(self, image_path, resultado):
        try:
            save_dir = "analises"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            classe = resultado.get('classe', 'indefinido')
            confianca = int(resultado.get('confianca_percent', 0))
            
            filename = os.path.join(save_dir, f"analise_{classe}_{confianca}pc_{timestamp}.png")
            plt.savefig(filename, bbox_inches='tight', dpi=150)
            messagebox.showinfo("✅ Sucesso", f"Análise salva em:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("❌ Erro", f"Erro ao salvar: {str(e)}")
    
    # ============================================
    # CÂMERA
    # ============================================
    
    def start_camera_interface(self):
        if not self.permissions["camera"]:
            messagebox.showwarning("Permissão Negada", 
                                 "Ative a permissão de acesso à câmera nas configurações!")
            return
        
        if not self.mediapipe_loaded or self.detector is None:
            messagebox.showerror("Erro", "MediaPipe não carregado!")
            return
        
        self.camera_window = tk.Toplevel(self.root)
        self.camera_window.title("Câmera - Avaliação de Postura")
        self.camera_window.geometry("700x700")
        self.camera_window.protocol("WM_DELETE_WINDOW", self.close_camera)
        
        colors = self.get_colors()
        self.camera_window.configure(bg=colors["bg"])
        
        video_frame = tk.Frame(self.camera_window, bg="black")
        video_frame.pack(pady=10)
        
        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack()
        
        control_frame = tk.Frame(self.camera_window, bg=colors["bg"])
        control_frame.pack(pady=10)
        
        self.pause_btn = tk.Button(control_frame,
                                  text="⏸️ Pausar",
                                  font=("Arial", 11),
                                  bg=colors["button_bg"],
                                  fg=colors["button_fg"],
                                  padx=15,
                                  pady=8,
                                  command=self.toggle_pause)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.classify_btn = tk.Button(control_frame,
                                     text="📊 Classificar Frame",
                                     font=("Arial", 11),
                                     bg="#2196F3",
                                     fg="white",
                                     padx=15,
                                     pady=8,
                                     command=self.classify_frozen_frame,
                                     state=tk.DISABLED)
        self.classify_btn.pack(side=tk.LEFT, padx=5)
        
        save_btn = tk.Button(control_frame,
                            text="💾 Salvar Frame",
                            font=("Arial", 11),
                            bg=colors["button_bg"],
                            fg=colors["button_fg"],
                            padx=15,
                            pady=8,
                            command=self.save_frame)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = tk.Button(control_frame,
                             text="❌ Fechar",
                             font=("Arial", 11),
                             bg="#f44336",
                             fg="white",
                             padx=15,
                             pady=8,
                             command=self.close_camera)
        close_btn.pack(side=tk.LEFT, padx=5)
        
        self.info_label = tk.Label(self.camera_window,
                                  text="🔄 Iniciando câmera...",
                                  font=("Arial", 10),
                                  fg=colors["fg"],
                                  bg=colors["bg"])
        self.info_label.pack(pady=5)
        
        self.result_label = tk.Label(self.camera_window,
                                    text="Aguardando classificação...",
                                    font=("Arial", 12, "bold"),
                                    fg=colors["title_color"],
                                    bg=colors["bg"])
        self.result_label.pack(pady=5)
        
        self.camera_running = True
        self.camera_paused = False
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
    
    def camera_loop(self):
        """Loop principal da câmera"""
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Erro", "Não foi possível acessar a câmera!"))
            self.camera_running = False
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Variáveis para análise
        fps = 0
        frame_count = 0
        start_time = cv2.getTickCount()
        
        while self.camera_running:
            if not self.camera_paused:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Calcular FPS
                frame_count += 1
                tempo_atual = cv2.getTickCount()
                tempo_decorrido = (tempo_atual - start_time) / cv2.getTickFrequency()
                if tempo_decorrido >= 1.0:
                    fps = frame_count / tempo_decorrido
                    frame_count = 0
                    start_time = tempo_atual
                
                # Processar frame
                processed_frame = self.process_frame_with_skeleton(frame, fps)
                self.current_frame = processed_frame
                self.frozen_frame = processed_frame.copy()
                
                self.display_frame(processed_frame)
                self.root.after(0, self.update_info)
            else:
                if self.frozen_frame is not None:
                    self.display_frame(self.frozen_frame)
                time.sleep(0.05)
            
            time.sleep(0.033)
        
        if self.cap:
            self.cap.release()
            self.cap = None
    
    def process_frame_with_skeleton(self, frame, fps):
        """Processa o frame mostrando o skeleton (33 pontos)"""
        if not self.mediapipe_loaded or self.detector is None:
            cv2.putText(frame, "MediaPipe não carregado!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame
        
        h, w = frame.shape[:2]
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        try:
            result = self.detector.detect(mp_image)
        except:
            return frame
        
        img_landmarks = frame.copy()
        pontos_detectados = []
        
        if result.pose_landmarks:
            for pose in result.pose_landmarks:
                for i, lm in enumerate(pose):
                    x, y = int(lm.x * w), int(lm.y * h)
                    pontos_detectados.append(i)
                    
                    # Cor baseada na visibilidade
                    if lm.visibility > 0.5:
                        cor = (0, 255, 0)  # Verde - visível
                    else:
                        cor = (0, 0, 255)  # Vermelho - não visível
                    
                    cv2.circle(img_landmarks, (x, y), 4, cor, -1)
                    cv2.circle(img_landmarks, (x, y), 6, (255, 255, 255), 1)
                    if i % 5 == 0:  # Mostrar alguns números
                        cv2.putText(img_landmarks, str(i), (x+8, y-8), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Informações
        overlay = img_landmarks.copy()
        cv2.rectangle(overlay, (5, 5), (300, 80), (0, 0, 0), -1)
        img_landmarks = cv2.addWeighted(overlay, 0.6, img_landmarks, 0.4, 0)
        
        status = "PAUSADO" if self.camera_paused else "ATIVO"
        cv2.putText(img_landmarks, f"Status: {status}", (15, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img_landmarks, f"FPS: {fps:.1f}", (15, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(img_landmarks, f"Keypoints: {len(pontos_detectados)}/33", (15, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        if not result.pose_landmarks:
            cv2.putText(img_landmarks, "Nenhuma pessoa detectada", (15, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return img_landmarks
    
    def display_frame(self, frame):
        try:
            if not hasattr(self, 'video_label') or not self.video_label.winfo_exists():
                return
            
            frame_resized = cv2.resize(frame, (640, 480))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.video_label.config(image=imgtk)
            self.video_label.image = imgtk
        except Exception as e:
            pass
    
    def update_info(self):
        try:
            if hasattr(self, 'info_label') and self.info_label.winfo_exists():
                status = "PAUSADO" if self.camera_paused else "ATIVO"
                self.info_label.config(text=f"{status} | Pause para classificar")
        except:
            pass
    
    def toggle_pause(self):
        self.camera_paused = not self.camera_paused
        
        if self.camera_paused:
            self.pause_btn.config(text="▶️ Continuar")
            if self.current_frame is not None:
                self.frozen_frame = self.current_frame.copy()
            self.info_label.config(text="⏸️ Frame pausado - Clique em 'Classificar Frame'")
            self.classify_btn.config(state=tk.NORMAL)
        else:
            self.pause_btn.config(text="⏸️ Pausar")
            self.frozen_frame = None
            self.info_label.config(text="▶️ Câmera ativa")
            self.classify_btn.config(state=tk.DISABLED)
            self.result_label.config(text="")
    
    def classify_frozen_frame(self):
        if not self.camera_paused:
            messagebox.showinfo("Aviso", "Pause a câmera primeiro!")
            return
        
        if self.frozen_frame is None:
            messagebox.showwarning("Aviso", "Nenhum frame disponível!")
            return
        
        if not self.mlp_loaded:
            messagebox.showerror("Erro", "Modelo PyTorch não carregado!")
            return
        
        # Extrair keypoints (99 features)
        keypoints_array, keypoints_detectados = self.extract_keypoints_from_frame(self.frozen_frame)
        
        if keypoints_array is None:
            self.result_label.config(text="❌ Nenhuma pessoa detectada!", fg="red")
            return
        
        # Classificar
        resultado = self.classify_posture(keypoints_array)
        
        if resultado and 'classe' in resultado and resultado['classe'] in self.CLASSES:
            cor = "green"
            nome_exibicao = resultado.get('classe_exibicao', resultado['classe'])
            self.result_label.config(
                text=f"🎯 {nome_exibicao} ({resultado['confianca_percent']:.1f}%)",
                fg=cor
            )
            
            # Salvar o frame classificado
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = "capturas_classificadas"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            filename = os.path.join(save_dir, f"classificacao_{resultado['classe']}_{timestamp}.png")
            cv2.imwrite(filename, self.frozen_frame)
            print(f"Frame classificado salvo: {filename}")
        else:
            msg = resultado.get('mensagem', 'Erro na classificação') if resultado else 'Erro desconhecido'
            self.result_label.config(text=f"❌ {msg}", fg="red")
    
    def save_frame(self):
        if self.current_frame is None:
            messagebox.showwarning("Aviso", "Nenhum frame disponível!")
            return
        
        save_dir = "capturas"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_dir, f"postura_{timestamp}.png")
        
        cv2.imwrite(filename, self.current_frame)
        messagebox.showinfo("Sucesso", f"✅ Frame salvo em:\n{filename}")
    
    def close_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if hasattr(self, 'camera_window') and self.camera_window.winfo_exists():
            self.camera_window.destroy()
    
    def on_closing(self):
        self.close_camera()
        self.save_config()
        self.root.destroy()
    
    # ============================================
    # UPLOAD DE ARQUIVO
    # ============================================
    
    def upload_file(self):
        if not self.permissions["arquivos"]:
            messagebox.showwarning("Permissão Negada", 
                                 "Ative a permissão de acesso a arquivos!")
            return
        
        filetypes = (('Imagens', '*.jpg *.jpeg *.png'), ('Todos', '*.*'))
        filename = filedialog.askopenfilename(title='Selecionar imagem', filetypes=filetypes)
        
        if filename:
            try:
                file_size = os.path.getsize(filename) / (1024 * 1024)
                if file_size > 300:
                    messagebox.showerror("Erro", f"Arquivo muito grande: {file_size:.1f}MB")
                    return
                
                self.process_uploaded_image(filename)
                
            except Exception as e:
                messagebox.showerror("Erro", str(e))
    
    def process_uploaded_image(self, image_path):
        if not self.mlp_loaded:
            messagebox.showerror("Erro", "Modelo PyTorch não carregado!")
            return
        
        # Extrair keypoints (99 features)
        keypoints_array, keypoints_dict, keypoints_detectados = self.extract_keypoints_from_image(image_path)
        
        if keypoints_array is None:
            messagebox.showerror("Erro", 
                               "Não foi possível detectar os keypoints!\n"
                               "Certifique-se que a imagem mostra uma pessoa claramente.")
            return
        
        if keypoints_array.shape[1] != self.NUM_FEATURES:
            messagebox.showerror("Erro", 
                               f"Features incorretas!\n"
                               f"Esperado: {self.NUM_FEATURES}\n"
                               f"Obtido: {keypoints_array.shape[1]}")
            return
        
        # Classificar
        resultado = self.classify_posture(keypoints_array)
        self.show_result_window(image_path, keypoints_dict, keypoints_detectados, resultado)

# ============================================
# FUNÇÃO PRINCIPAL
# ============================================

def main():
    root = tk.Tk()
    app = PostureEvaluationApp(root)
    root.mainloop()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎯 SISTEMA DE AVALIAÇÃO DE POSTURA - PYTORCH")
    print("="*60)
    print(f"📊 Features: 99 (33 keypoints × 3)")
    print(f"🎯 Classes: 9 (POLAR dataset)")
    print("="*60)
    main()