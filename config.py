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
ENV_NAME = 'ALE/IceHockey-v5'  # Environment name (游戲環境名稱)
ACTION_SPACE_SIZE = 18  # Number of possible actions in Ice Hockey (冰球游戲可執行的動作數量)
DIFFICULTY = 0  # Game difficulty level from 0-4, 0 is easiest (游戲難度等級，0-4，0為最簡單) - 增加難度可測試算法在更複雜環境中的表現，但會增加學習難度

# Frame processing settings
# 幀處理設置
FRAME_WIDTH = 84  # Width of processed frame, downscaled from 160 (處理後的幀寬度，從160縮小) - 增加會提供更詳細的視覺信息，但增加網絡計算負擔
FRAME_HEIGHT = 84  # Height of processed frame, downscaled from 210 (處理後的幀高度，從210縮小) - 同上，更高解析度需要更大的網絡結構
FRAME_STACK = 4  # Number of frames to stack for temporal information (堆疊的幀數量，用於時間序列信息) - 增加可捕捉更長時間序列但增加記憶體使用，減少可能導致時間信息不足
FRAME_SKIP = 4  # Frames to skip between agent actions (智能體動作之間跳過的幀數) - 減少可增加決策頻率但訓練較慢，增加可加速訓練但可能錯過重要狀態
NOOP_MAX = 30  # Maximum number of no-op actions at episode start (回合開始時最大無操作動作數) - 增加有助於探索更多起始狀態，但可能浪費訓練步數

# Visualization settings
# 視覺化設置
RENDER_MODE = None  # Render mode for visualization, None for training speed (渲染模式，訓練時設為None以提高速度) - 設為'human'可觀察智能體但會大幅降低訓練速度
TRAINING_MODE = True  # Training mode flag to disable rendering (訓練模式標誌，禁用渲染) - 設為False將啟用渲染，僅用於測試

###############################
# DEEP Q-LEARNING PARAMETERS
# 深度Q學習參數
###############################

# Core DQN parameters
# 核心DQN參數
LEARNING_RATE = 0.00025  # Learning rate for Adam optimizer (Adam優化器的學習率) - 增加可加快學習但可能不穩定，減少可提高穩定性但學習較慢
GAMMA = 0.99  # Discount factor for future rewards (未來獎勵的折扣因子) - 增加使智能體更重視長期獎勵，減少則更關注即時獎勵
BATCH_SIZE = 64  # Batch size for training updates (訓練更新的批次大小) - 增加可提供更穩定的梯度估計但增加記憶體消耗，減少可能導致訓練不穩定
MEMORY_CAPACITY = 200000  # Experience replay memory capacity (經驗回放記憶體容量) - 增加可儲存更多多樣化經驗但增加記憶體用量，減少可能導致過度擬合近期經驗
TARGET_UPDATE_FREQUENCY = 8000  # Steps between target network updates (目標網絡更新間隔步數) - 增加提高訓練穩定性但降低學習速度，減少提高學習速度但可能導致不穩定振盪
TRAINING_EPISODES = 10000  # Total training episodes (總訓練回合數) - 增加可提高最終性能但延長訓練時間，減少可加快實驗但可能無法充分學習

# Exploration parameters
# 探索參數
EPSILON_START = 1.0  # Initial exploration rate (初始探索率) - 保持較高可確保初期充分探索環境，降低則更早利用已知策略
EPSILON_END = 0.1  # Final exploration rate (最終探索率) - 增加可確保持續探索新策略，減少則更專注於利用學到的策略
EPSILON_DECAY = 1000000  # Steps over which epsilon decays (epsilon衰減的步數) - 增加使探索率下降更慢，確保更長時間的探索，減少則更快地專注於利用學到的策略
DEFAULT_EVALUATE_MODE = False  # Default evaluation mode (默認評估模式) - 設為True時禁用探索，僅用於評估不影響訓練

# Training control parameters
# 訓練控制參數
LEARNING_STARTS = 10000  # Steps before starting learning (開始學習前的步數) - 增加可收集更多隨機經驗確保多樣性，減少可加速開始學習
UPDATE_FREQUENCY = 4  # Steps between network updates (網絡更新間隔步數) - 減少可更頻繁更新網絡加速學習，增加可減少計算負擔但可能減慢學習

###############################
# PRIORITIZED EXPERIENCE REPLAY PARAMETERS
# 優先經驗回放參數
###############################

# Whether to use Prioritized Experience Replay
# 是否使用優先經驗回放
USE_PER = True  # Enable Prioritized Experience Replay (啟用優先經驗回放) - 設為False時使用標準均勻採樣，PER通常提升樣本效率但增加計算複雜度

# PER hyperparameters
# PER超參數
ALPHA = 0.6  # Priority exponent for sampling probability (優先級指數，用於採樣概率) - 增加強化高誤差樣本採樣頻率，減少使採樣更接近均勻
BETA_START = 0.4  # Initial importance sampling weight value (初始重要性採樣權重值) - 增加可減少初期偏差但可能減慢收斂，保持低值使初期學習更聚焦於高誤差樣本
BETA_FRAMES = 400000  # Frames over which beta increases to 1.0 (beta增加到1.0的幀數) - 增加使偏差校正更平緩但延長非均衡學習階段，減少可加速達到無偏學習
EPSILON_PER = 1e-6  # Small constant for priority calculation (優先級計算的小常數) - 確保所有經驗都有非零優先級，防止某些經驗永不被採樣

# SumTree settings
# 總和樹設置
TREE_CAPACITY = MEMORY_CAPACITY  # Size of the sum tree (與記憶體容量一致) - 應與記憶體容量保持一致，否則可能導致記憶體使用不一致
DEFAULT_NEW_PRIORITY = 1.0  # Default priority for new experiences (新經驗的默認優先級) - 設置新經驗的初始優先級，影響新加入經驗被採樣的機會

# PER Logging Configuration
PER_LOG_FREQUENCY = 100  # Steps between PER metrics logging (PER指標記錄的步數間隔) - 減少可提供更詳細的PER指標記錄但增加日誌大小
PER_BATCH_SIZE = 50  # Batch size for PER data writes (PER數據寫入的批次大小) - 增加可減少I/O操作但增加記憶體使用，減少可減輕記憶體負擔但增加I/O頻率

###############################
# NEURAL NETWORK SETTINGS
# 神經網絡設置
###############################

USE_ONE_CONV_LAYER = False  # Whether to use a single convolutional layer (是否使用單個卷積層) - 簡化模型結構，降低計算需求但可能減弱特徵提取能力
USE_TWO_CONV_LAYERS = False  # Whether to use two convolutional layers (是否使用兩個卷積層) - 中等複雜度，平衡計算效率和特徵提取能力
USE_THREE_CONV_LAYERS = True  # Whether to use three convolutional layers (是否使用三個卷積層) - 增加模型複雜度和表達能力，提高特徵提取但增加計算需求

# First convolutional layer parameters
# 第一卷積層參數
CONV1_CHANNELS = 32  # Number of filters in first conv layer (第一卷積層過濾器數量) - 增加可提取更多基本特徵但增加計算量
CONV1_KERNEL_SIZE = 8  # Kernel size for first conv layer (第一卷積層核大小) - 增加可捕捉更大視野範圍的特徵，減少則更專注於細節
CONV1_STRIDE = 4  # Stride for first conv layer (第一卷積層步幅) - 增加可減少輸出特徵圖大小節省計算，減少可保留更多空間信息但增加計算量

# Second convolutional layer parameters
# 第二卷積層參數
CONV2_CHANNELS = 64  # Number of filters in second conv layer (第二卷積層過濾器數量) - 增加可提取更複雜的特徵組合但增加計算需求
CONV2_KERNEL_SIZE = 4  # Kernel size for second conv layer (第二卷積層核大小) - 影響中階特徵的感受野大小
CONV2_STRIDE = 2  # Stride for second conv layer (第二卷積層步幅) - 控制特徵圖尺寸減少的速率

# Third convolutional layer parameters
# 第三卷積層參數
CONV3_CHANNELS = 64  # Number of filters in third conv layer (第三卷積層過濾器數量) - 提取高級特徵的能力，增加可改善複雜模式識別
CONV3_KERNEL_SIZE = 3  # Kernel size for third conv layer (第三卷積層核大小) - 較小的核專注於精細特徵整合
CONV3_STRIDE = 1  # Stride for third conv layer (第三卷積層步幅) - 保持為1通常可在最後卷積層保留更完整的空間信息

# Fully connected layer and gradient settings
# 全連接層和梯度設置
FC_SIZE = 512  # Size of fully connected layer (全連接層大小) - 增加提高模型表示能力但增加參數量，減少可降低過擬合風險但可能降低表達能力
GRAD_CLIP_NORM = 5.0  # Gradient clipping norm (梯度裁剪范數) - 防止梯度爆炸，增加允許更大的更新步長但可能導致不穩定，減少提高穩定性但可能減緩學習

# Evaluation settings
# 評估設置
EVAL_EPISODES = 20  # Number of episodes for each evaluation (每次評估的回合數) - 增加可提供更可靠的評估結果但延長評估時間
EVAL_FREQUENCY = 100  # Episodes between evaluations (評估間隔的回合數) - 減少可更頻繁評估訓練進度但延長總訓練時間，增加可加速訓練但降低進度監控頻率

###############################
# LOGGER SETTINGS
# 日誌記錄器設置
###############################

# System resource management
# 系統資源管理
MEMORY_THRESHOLD_PERCENT = 75  # Memory usage threshold percentage (內存使用閾值百分比) - 降低可更保守地控制記憶體使用但可能限制性能，增加允許使用更多記憶體但風險更高

# Directory configurations
# 目錄配置
RESULTS_DIR = "results"  # Main results directory (主要結果目錄) - 保存所有實驗結果的主目錄
LOG_DIR = os.path.join(RESULTS_DIR, "logs")  # Directory for log files (日誌文件目錄)
MODEL_DIR = os.path.join(RESULTS_DIR, "models")  # Directory for model checkpoints (模型檢查點目錄)
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")  # Directory for visualization plots (可視化圖表目錄)
DATA_DIR = os.path.join(RESULTS_DIR, "data")  # Directory for data storage (數據存儲目錄)

# Logger settings
# 日誌記錄器設置
ENABLE_FILE_LOGGING = False  # Whether to write logs to files (是否將日誌寫入文件) - 啟用可保存完整訓練記錄但增加I/O操作，禁用可減輕系統負擔但丟失歷史記錄
LOGGER_SAVE_INTERVAL = 50  # Episodes between logger saves (日誌保存間隔的回合數) - 減少可更頻繁保存進度但增加I/O操作，增加可減輕I/O負擔但風險更高
LOGGER_MEMORY_WINDOW = 1000  # Maximum records kept in memory (内存中保留的最大記錄數) - 增加可存儲更長的歷史記錄但增加記憶體使用，減少可節省記憶體但限制可存取的歷史數據
LOGGER_BATCH_SIZE = 30  # Records to accumulate before writing (累積多少記錄後寫入磁盤) - 增加可減少I/O頻率但延遲數據持久化，減少可更快保存記錄但增加I/O頻率
LOGGER_DETAILED_INTERVAL = 10  # Episodes between detailed reports (詳細報告間隔的回合數) - 減少提供更頻繁的詳細進度報告但增加輸出量，增加可減少輸出量但降低監控粒度
LOGGER_MAJOR_METRICS = ["reward", "loss", "epsilon", "beta"]  # Main metrics to plot (主要繪圖指標) - 自定義要在概覽圖中顯示的主要指標
VISUALIZATION_SAVE_INTERVAL = 100  # Episodes between visualization saves (可視化保存間隔的回合數) - 減少可更頻繁生成可視化但增加I/O和計算負擔，增加可減輕負擔但減少視覺反饋
