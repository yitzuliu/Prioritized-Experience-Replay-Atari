"""
Configuration file for DQN training on Atari Ice Hockey.

This file contains all hyperparameters and settings used throughout the project,
organized by category for better readability and management.

DQN 在 Atari 冰球遊戲上訓練的配置文件。

本文件包含項目中使用的所有超參數和設置，按類別組織以提高可讀性和可管理性。
"""
import os
import sys
###############################
# GAME ENVIRONMENT SETTINGS
# 遊戲環境設置
###############################

# Basic environment settings
# 基本環境設置
ENV_NAME = 'ALE/IceHockey-v5'  # 遊戲環境名稱
ACTION_SPACE_SIZE = 18  # IceHockey has 18 possible actions (冰球遊戲有18個可能的動作)

# Game difficulty settings
# 遊戲難度設置
DIFFICULTY = 0  # 0=Easy, 1=Normal, 2=Hard, 3=Expert (0=簡單, 1=普通, 2=困難, 3=專家)
MODE = 0  # 0=Default mode, values vary by game (0=默認模式，具體影響因遊戲而異)

# Frame processing settings
# 幀處理設置
FRAME_WIDTH = 84  # Downscaled from native 160 (從原始160縮小到84)
FRAME_HEIGHT = 84  # Downscaled from native 210 (從原始210縮小到84)
FRAME_STACK = 4  # Number of frames to stack together (堆疊幀數量)
FRAME_SKIP = 4  # Matches the environment's built-in frameskip for v5 (與環境內建的v5版本跳幀設置匹配)
NOOP_MAX = 30  # Maximum number of no-op actions at the start of an episode (開始時最大無操作動作數)

# Visualization settings
# 視覺化設置
RENDER_MODE = None  # No rendering during training for maximum speed (訓練期間不渲染以提高速度)
TRAINING_MODE = True  # Ensure render_mode is None during training (確保訓練模式下不渲染)

###############################
# DEEP Q-LEARNING PARAMETERS
# 深度Q學習參數
###############################

# Core DQN parameters
# 核心DQN參數
LEARNING_RATE = 0.00005  # Standard learning rate for Adam optimizer (Adam優化器的標準學習率)
GAMMA = 0.99  # Standard discount factor (標準折扣因子)
BATCH_SIZE = 512  # Standard batch size (標準批次大小)
MEMORY_CAPACITY = 200000  # for better experience diversity 
TARGET_UPDATE_FREQUENCY = 10000  # Update target network 
TRAINING_EPISODES = 10000  # Total number of training episodes (訓練總回合數)

# Exploration parameters
# 探索參數
EPSILON_START = 1.0  # Initial exploration rate (初始探索率：完全隨機)
EPSILON_END = 0.01  # Lower final exploration rate for better policy (最終較低的探索率，有利於更好的策略)
EPSILON_DECAY = 400000  # Slower decay over more steps for better exploration (較慢的衰減速率，分佈更多步數以改善探索)
DEFAULT_EVALUATE_MODE = False  # Default evaluation mode (默認評估模式)

# Training control parameters
# 訓練控制參數
LEARNING_STARTS = 28000  # Wait for more experiences before starting learning (開始學習前等待的經驗數量)
UPDATE_FREQUENCY = 1  # Standard update frequency (標準更新頻率)
SAVE_FREQUENCY = 400  # Save more frequently to prevent data loss (頻繁保存以防止數據丟失)

###############################
# PRIORITIZED EXPERIENCE REPLAY PARAMETERS
# 優先經驗回放參數
###############################

# Whether to use Prioritized Experience Replay
# 是否使用優先經驗回放
USE_PER = True  # Set to False to use standard uniform sampling (設為False則使用標準均勻採樣)

# PER hyperparameters
# PER超參數
ALPHA = 0.6  # Priority exponent - controls how much prioritization is used (優先級指數 - 控制優先級的使用程度)
BETA_START = 0.4  # Initial importance sampling weight (初始重要性採樣權重)
BETA_FRAMES = 1000000  # Number of frames over which beta will be annealed to 1.0 (β值增長到1.0所需的幀數)
EPSILON_PER = 1e-6  # Small constant to add to TD-errors to ensure non-zero priority (添加到TD誤差的小常數，確保優先級非零)

# SumTree settings
# 總和樹設置
TREE_CAPACITY = MEMORY_CAPACITY  # Size of the sum tree (總和樹的大小)
DEFAULT_NEW_PRIORITY = 1.0  # Default priority for new experiences (新經驗的默認優先級)

###############################
# NEURAL NETWORK SETTINGS
# 神經網絡設置
###############################

USE_ONE_CONV_LAYER = True  # Use 1 convolutional layer (使用1個卷積層)
USE_TWO_CONV_LAYERS = False  # Use 2 convolutional layers (使用2個卷積層)
USE_THREE_CONV_LAYERS = False  # Using full 3-layer architecture (hardware can handle it) (使用完整的3層架構，由硬件支持)

# Network architecture parameters (previously hardcoded in q_network.py)
# 網絡架構參數
CONV1_CHANNELS = 32  # First convolutional layer output channels (第一卷積層輸出通道數)
CONV1_KERNEL_SIZE = 8  # First convolutional layer kernel size (第一卷積層內核大小)
CONV1_STRIDE = 4  # First convolutional layer stride (第一卷積層步幅)

CONV2_CHANNELS = 64  # Second convolutional layer output channels (第二卷積層輸出通道數)
CONV2_KERNEL_SIZE = 4  # Second convolutional layer kernel size (第二卷積層內核大小)
CONV2_STRIDE = 2  # Second convolutional layer stride (第二卷積層步幅)

CONV3_CHANNELS = 64  # Third convolutional layer output channels (第三卷積層輸出通道數)
CONV3_KERNEL_SIZE = 3  # Third convolutional layer kernel size (第三卷積層內核大小)
CONV3_STRIDE = 1  # Third convolutional layer stride (第三卷積層步幅)

FC_SIZE = 512  # Size of fully connected layer (全連接層大小)
GRAD_CLIP_NORM = 10.0  # Maximum gradient norm for gradient clipping (梯度裁剪的最大梯度范數)

# Evaluation settings (used in evaluate.py)
# 評估設置（在 evaluate.py 中使用）
EVAL_EPISODES = 30  # Number of episodes for evaluation (評估的回合數)
EVAL_FREQUENCY = 500  # How often to evaluate during training (training.py doesn't fully implement this) (訓練期間評估的頻率)

###############################
# SYSTEM AND OPTIMIZATION
# 系統與優化
###############################

# Memory optimization
# 記憶體優化
MEMORY_IMPLEMENTATION = "optimized"  # Using memory-efficient implementation (使用記憶體高效的實現)

# Video recording settings
# 視頻記錄設置
ENABLE_VIDEO_RECORDING = False  # Enable/disable video recording functionality (啟用/禁用視頻記錄功能)
VIDEO_FPS = 30  # Frames per second for recorded videos (記錄視頻的每秒幀數)
VIDEO_LENGTH_LIMIT = 3600  # Maximum video length in seconds (最大視頻長度，單位為秒)

# Directory configurations
# 目錄配置 - All directories now under 'result' folder
RESULT_DIR = "result"  # Main result directory
LOG_DIR = os.path.join(RESULT_DIR, "logs")  # Logs directory
MODEL_DIR = os.path.join(RESULT_DIR, "models")  # Models directory
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")  # Checkpoints directory
PLOT_DIR = os.path.join(RESULT_DIR, "plots")  # Plots directory
DATA_DIR = os.path.join(RESULT_DIR, "data")  # Data directory
VIDEO_DIR = os.path.join(RESULT_DIR, "videos")  # Video recording directory

