"""
Configuration file for DQN training on Atari Ice Hockey.

This file contains all hyperparameters and settings used throughout the project,
organized by category for better readability and management.

DQN 在 Atari 冰球遊戲上訓練的配置文件。

本文件包含項目中使用的所有超參數和設置，按類別組織以提高可讀性和可管理性。
"""
import os

###############################
# GAME ENVIRONMENT SETTINGS
# 遊戲環境設置
###############################

# Basic environment settings
# 基本環境設置
ENV_NAME = 'ALE/IceHockey-v5'  # 遊戲環境名稱
ACTION_SPACE_SIZE = 18  # IceHockey has 18 possible actions (冰球遊戲有18個可能的動作)
DIFFICULTY = 0  # Game difficulty level from 0-4, where 0 is easiest (遊戲難度等級，0-4，0為最簡單)

# Frame processing settings
# 幀處理設置
FRAME_WIDTH = 84  # Downscaled from native 160 (從原始160縮小到84)
FRAME_HEIGHT = 84  # Downscaled from native 210 (從原始210縮小到84)
FRAME_STACK = 3  # Number of frames to stack together (堆疊幀數量) - 從4減少到3以減少記憶體使用
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
BATCH_SIZE = 64  # Reduced batch size to save memory (減小批次大小以節省記憶體)
MEMORY_CAPACITY = 50000  # Reduced memory capacity (減小記憶容量)
TARGET_UPDATE_FREQUENCY = 20000  # Update target network less frequently to reduce computation (較低頻率更新目標網路以減少計算)
TRAINING_EPISODES = 10000  # Total number of training episodes (訓練總回合數)

# Exploration parameters
# 探索參數
EPSILON_START = 1.0  # Initial exploration rate (初始探索率：完全隨機)
EPSILON_END = 0.01  # Lower final exploration rate for better policy (最終較低的探索率，有利於更好的策略)
EPSILON_DECAY = 1000000   # Slower decay over more steps for better exploration (較慢的衰減速率，分佈更多步數以改善探索)
DEFAULT_EVALUATE_MODE = False  # Default evaluation mode (默認評估模式)

# Training control parameters
# 訓練控制參數
LEARNING_STARTS = 20000  # Wait for more experiences before starting learning (開始學習前等待的經驗數量)
UPDATE_FREQUENCY = 4  # Update less frequently to reduce computation (降低更新頻率以減少計算)
SAVE_FREQUENCY = 100  # Save more frequently to prevent data loss (頻繁保存以防止數據丟失)

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
BETA_FRAMES = 1000000  # Number of frames over which beta will be annealed to a value of 1.0 (β值增長到1.0所需的幀數)
EPSILON_PER = 1e-6  # Small constant to add to TD-errors to ensure non-zero priority (添加到TD誤差的小常數，確保優先級非零)

# SumTree settings
# 總和樹設置
TREE_CAPACITY = MEMORY_CAPACITY  # Size of the sum tree (總和樹的大小)
DEFAULT_NEW_PRIORITY = 1.0  # Default priority for new experiences (新經驗的默認優先級)

###############################
# NEURAL NETWORK SETTINGS
# 神經網絡設置
###############################

USE_ONE_CONV_LAYER = False  # Use 1 convolutional layer (使用1個卷積層)
USE_TWO_CONV_LAYERS = False  # Use 2 convolutional layers (使用2個卷積層)
USE_THREE_CONV_LAYERS = True  # Using full 3-layer architecture (hardware can handle it) (使用完整的3層架構，由硬件支持)

# First convolutional layer parameters
# 第一卷積層參數
CONV1_CHANNELS = 32  # First convolutional layer output channels (第一卷積層輸出通道數)
CONV1_KERNEL_SIZE = 8  # First convolutional layer kernel size (第一卷積層內核大小)
CONV1_STRIDE = 4  # First convolutional layer stride (第一卷積層步幅)

# Second convolutional layer parameters
# 第二卷積層參數
CONV2_CHANNELS = 64  # Second convolutional layer output channels (第二卷積層輸出通道數)
CONV2_KERNEL_SIZE = 4  # Second convolutional layer kernel size (第二卷積層內核大小)
CONV2_STRIDE = 2  # Second convolutional layer stride (第二卷積層步幅)

# Third convolutional layer parameters
# 第三卷積層參數
CONV3_CHANNELS = 64  # Third convolutional layer output channels (第三卷積層輸出通道數)
CONV3_KERNEL_SIZE = 3  # Third convolutional layer kernel size (第三卷積層內核大小)
CONV3_STRIDE = 1  # Third convolutional layer stride (第三卷積層步幅)

# Fully connected layer and gradient settings
# 全連接層和梯度設置
FC_SIZE = 256  # Reduced size of fully connected layer (縮小全連接層大小以減少參數數量)
GRAD_CLIP_NORM = 10.0  # Maximum gradient norm for gradient clipping (梯度裁剪的最大梯度范數)

# Evaluation settings
# 評估設置
EVAL_EPISODES = 30  # Number of episodes for evaluation (評估的回合數)
EVAL_FREQUENCY = 500  # How often to evaluate during training (訓練期間評估的頻率)

###############################
# SYSTEM AND OPTIMIZATION
# 系統與優化
###############################

# System resource management
# 系統資源管理
MEMORY_CHECK_INTERVAL = 300  # Memory check interval in seconds (內存檢查間隔，單位秒)
MEMORY_THRESHOLD_PERCENT = 75  # Memory usage threshold percentage (內存使用閾值百分比)

# Directory configurations
# 目錄配置 - All directories now under 'result' folder
RESULT_DIR = "result"  # Main result directory
LOG_DIR = os.path.join(RESULT_DIR, "logs")  # Logs directory
MODEL_DIR = os.path.join(RESULT_DIR, "models")  # Models directory
CHECKPOINT_DIR = os.path.join(RESULT_DIR, "checkpoints")  # Checkpoints directory
PLOT_DIR = os.path.join(RESULT_DIR, "plots")  # Plots directory
DATA_DIR = os.path.join(RESULT_DIR, "data")  # Data directory
VIDEO_DIR = os.path.join(RESULT_DIR, "videos")  # Video recording directory

# Logger settings
# 日誌記錄器設置
LOGGER_SAVE_INTERVAL = 10  # How often to save logger data (every N episodes) (每N回合保存一次日誌數據)
LOGGER_MEMORY_WINDOW = 1000  # Maximum number of records to keep in memory (內存中保留的最大記錄數量)
LOGGER_BATCH_SIZE = 50  # Number of records to accumulate before writing to disk (累積多少記錄後寫入磁盤)
LOGGER_DETAILED_INTERVAL = 10  # How often to print detailed progress (每隔多少回合打印詳細進度)
LOGGER_MAJOR_METRICS = ["reward", "loss", "epsilon", "beta"]  # Suggested major metrics for visualization (建議用於可視化的主要指標)
