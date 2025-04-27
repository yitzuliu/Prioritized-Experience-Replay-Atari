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
ENV_NAME = 'ALE/IceHockey-v5'  # 遊戲環境名稱 (更改此值可切換到其他 Atari 遊戲)
ACTION_SPACE_SIZE = 18  # IceHockey has 18 possible actions (冰球遊戲有18個可能的動作，與遊戲環境綁定)
DIFFICULTY = 0  # Game difficulty level from 0-4, where 0 is easiest (遊戲難度等級，0-4，0為最簡單，提高可增加挑戰性)

# Frame processing settings
# 幀處理設置
FRAME_WIDTH = 84  # Downscaled from native 160 (從原始160縮小到84，增加可提供更詳細的輸入但增加計算負擔)
FRAME_HEIGHT = 84  # Downscaled from native 210 (從原始210縮小到84，增加可提供更詳細的輸入但增加計算負擔)
FRAME_STACK = 4  # Number of frames to stack (幀堆疊數量，增加可提高時間序列信息捕捉能力，但增加計算負擔)
FRAME_SKIP = 4  # Matches the environment's built-in frameskip for v5 (與環境內建的v5版本跳幀設置匹配，減少可增加訓練精度但降低速度)
NOOP_MAX = 30  # Maximum number of no-op actions at the start of an episode (開始時最大無操作動作數，增加可提高初始狀態多樣性)

# Visualization settings
# 視覺化設置
RENDER_MODE = None  # No rendering during training for maximum speed (訓練期間不渲染以提高速度，設為'human'可觀察但會降低速度)
TRAINING_MODE = True  # Ensure render_mode is None during training (確保訓練模式下不渲染，設為False會影響性能)

###############################
# DEEP Q-LEARNING PARAMETERS
# 深度Q學習參數
###############################

# Core DQN parameters
# 核心DQN參數
LEARNING_RATE = 0.00025  # Learning rate for Adam optimizer (Adam優化器的學習率，降低以提高穩定性)
GAMMA = 0.99  # Discount factor (折扣因子，保持為0.99以平衡短期與長期獎勵)
BATCH_SIZE = 32  # Batch size for training (批次大小，降低可減少記憶體使用並提高更新頻率)
MEMORY_CAPACITY = 100000  # Experience replay memory size (降低記憶體容量以減少RAM使用)
TARGET_UPDATE_FREQUENCY = 8000  # Target network update frequency (更頻繁地更新目標網絡以加速學習 來自Nature DQN論文的合適值)
TRAINING_EPISODES = 5000  # Total number of training episodes (降低以更快完成訓練)

# Exploration parameters
# 探索參數
EPSILON_START = 1.0  # Initial exploration rate (初始探索率，完全隨機)
EPSILON_END = 0.1  # Final exploration rate (提高最終探索率以持續探索環境)
EPSILON_DECAY = 500000  # Steps over which epsilon decays (加速探索率衰減以更快利用學到的策略)
DEFAULT_EVALUATE_MODE = False  # Default evaluation mode (默認評估模式，True時關閉探索僅用於評估，不影響訓練)

# Training control parameters
# 訓練控制參數
LEARNING_STARTS = 10000  # Steps before starting learning (論文中常用的值)
UPDATE_FREQUENCY = 4  # Network update frequency (與Nature DQN論文一致)

###############################
# PRIORITIZED EXPERIENCE REPLAY PARAMETERS
# 優先經驗回放參數
###############################

# Whether to use Prioritized Experience Replay
# 是否使用優先經驗回放
USE_PER = True  # Enable/disable PER (設為False則使用標準均勻採樣，PER通常提高樣本效率但增加計算複雜度)

# PER hyperparameters
# PER超參數
ALPHA = 0.6  # Priority exponent (優先級指數，增加會強化高誤差樣本的優先級，降低會趨近於均勻採樣)
BETA_START = 0.4  # Initial importance sampling weight (初始重要性採樣權重，提高可減少偏差但可能降低學習效率)
BETA_FRAMES = 200000  # Frames over which beta increases to 1.0 (β值增長到1.0所需的幀數，增加會使偏差校正更平滑)
EPSILON_PER = 1e-6  # Small constant for TD-errors (TD誤差的小常數，確保優先級非零，調整通常影響不大)

# SumTree settings
# 總和樹設置
TREE_CAPACITY = MEMORY_CAPACITY  # Size of the sum tree (與記憶體容量保持一致)
DEFAULT_NEW_PRIORITY = 1.0  # Default priority for new experiences (新經驗的默認優先級)

# PER Logging Configuration
PER_LOG_FREQUENCY = 100  # How often to log PER metrics (in steps) (記錄PER指標的頻率，以步數為單位)
PER_BATCH_SIZE = 50  # Batch size for PER data writes (number of records) (PER數據寫入的批次大小)

###############################
# NEURAL NETWORK SETTINGS
# 神經網絡設置
###############################

USE_ONE_CONV_LAYER = False  # Use 1 convolutional layer (使用1個卷積層，簡化模型降低計算量，但可能降低性能)
USE_TWO_CONV_LAYERS = False  # Use 2 convolutional layers (使用2個卷積層，平衡複雜度和性能)
USE_THREE_CONV_LAYERS = True  # Use full 3-layer architecture (使用3層架構，增加模型複雜度和表達能力，提高性能但需要更多計算資源)

# First convolutional layer parameters
# 第一卷積層參數
CONV1_CHANNELS = 32  # First conv layer output channels (第一卷積層輸出通道數，增加可提取更多特徵但增加計算負擔)
CONV1_KERNEL_SIZE = 8  # First conv layer kernel size (第一卷積層內核大小，增加可捕捉更大範圍的特徵)
CONV1_STRIDE = 4  # First conv layer stride (第一卷積層步幅，減少可保留更多信息但增加特徵圖大小和計算量)

# Second convolutional layer parameters
# 第二卷積層參數
CONV2_CHANNELS = 64  # Second conv layer output channels (第二卷積層輸出通道數，增加可提取更複雜特徵但增加參數量)
CONV2_KERNEL_SIZE = 4  # Second conv layer kernel size (第二卷積層內核大小，調整可改變特徵提取範圍和細節捕捉能力)
CONV2_STRIDE = 2  # Second conv layer stride (第二卷積層步幅，調整可平衡特徵精細度和計算效率)

# Third convolutional layer parameters
# 第三卷積層參數
CONV3_CHANNELS = 64  # Third conv layer output channels (第三卷積層輸出通道數，增加可提取更高級特徵但增加參數量)
CONV3_KERNEL_SIZE = 3  # Third conv layer kernel size (第三卷積層內核大小，調整可改變特徵提取範圍和細節程度)
CONV3_STRIDE = 1  # Third conv layer stride (第三卷積層步幅，保持為1通常可保留更多空間信息)

# Fully connected layer and gradient settings
# 全連接層和梯度設置
FC_SIZE = 256  # Fully connected layer size (降低全連接層大小以減少參數量)
GRAD_CLIP_NORM = 5.0  # Maximum gradient norm (降低梯度范數上限以提高穩定性)

# Evaluation settings
# 評估設置
EVAL_EPISODES = 20  # Number of episodes for evaluation (評估的回合數，增加可提高評估準確性但耗時更長)
EVAL_FREQUENCY = 100  # Evaluation frequency during training (訓練期間評估的頻率，降低可更頻繁評估但延長訓練時間)

###############################
# LOGGER SETTINGS
# 日誌記錄器設置
###############################

# System resource management
# 系統資源管理
MEMORY_THRESHOLD_PERCENT = 75  # Memory usage threshold % (內存使用閾值百分比，降低可更保守地使用記憶體但可能限制性能)

# Directory configurations
# 目錄配置
RESULTS_DIR = "results"  # Main results directory (主要結果目錄，可更改路徑但需確保權限)
LOG_DIR = os.path.join(RESULTS_DIR, "logs")  # Logs directory (日誌目錄，存儲訓練過程記錄)
MODEL_DIR = os.path.join(RESULTS_DIR, "models")  # Models directory (模型目錄，存儲訓練的模型)
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")  # Plots directory (圖表目錄，存儲視覺化結果)
DATA_DIR = os.path.join(RESULTS_DIR, "data")  # Data directory (數據目錄，存儲原始數據和處理結果)

# Logger settings
# 日誌記錄器設置
ENABLE_FILE_LOGGING = False  # Enable file logging (啟用文件日誌記錄，設為True可將日誌寫入文件)
LOGGER_SAVE_INTERVAL = 50  # Logger save interval (日誌保存間隔，降低可更頻繁保存但增加I/O操作)
LOGGER_MEMORY_WINDOW = 1000  # Max records in memory (內存中保留的最大記錄數量，增加可記錄更多歷史但增加記憶體使用)
LOGGER_BATCH_SIZE = 30  # Records before disk write (累積多少記錄後寫入磁盤，增加可減少I/O頻率但延遲數據持久化)
LOGGER_DETAILED_INTERVAL = 10  # Detailed progress print frequency (詳細進度打印頻率，降低可獲得更頻繁的進度更新)
LOGGER_MAJOR_METRICS = ["reward", "loss", "epsilon", "beta"]  # Major metrics for visualization (主要視覺化指標，可根據需求添加或刪除)
VISUALIZATION_SAVE_INTERVAL = 100  # How often to save intermediate visualizations (每隔多少回合保存中間可視化結果，降低可更頻繁保存但增加I/O操作)
