"""
Configuration file for DQN training on Atari games.

This file contains all hyperparameters and settings used throughout the project,
organized by category for better readability and management.

DQN 在 Atari 遊戲上訓練的配置文件。

本文件包含項目中使用的所有超參數和設置，按類別組織以提高可讀性和可管理性。
"""
import os
import warnings

###############################
# GAME ENVIRONMENT SETTINGS
# 遊戲環境設置
###############################

# Basic environment settings
# 基本環境設置
ENV_NAME = 'ALE/MsPacman-v5'  # Environment name (游戲環境名稱)
ACTION_SPACE_SIZE = 9  # Number of possible actions in the game (遊戲中可能的動作數量)
DIFFICULTY = 0  # Game difficulty level from 0-4, 0 is easiest (游戲難度等級，0-4，0為最簡單) - Testing with higher difficulty can evaluate algorithm performance in more complex environments, but increases learning difficulty

# Frame processing settings
# 幀處理設置
FRAME_WIDTH = 84  # Width of processed frame, downscaled from 160 (處理後的幀寬度，從160縮小) - Increasing provides more detailed visual information but increases network computational burden
FRAME_HEIGHT = 84  # Height of processed frame, downscaled from 210 (處理後的幀高度，從210縮小) - Same as above, higher resolution requires larger network structure
FRAME_STACK = 4  # Number of frames to stack for temporal information (堆疊的幀數量，用於時間序列信息) - Increasing can capture longer temporal sequences but increases memory usage, decreasing may lead to insufficient temporal information
FRAME_SKIP = 4  # Frames to skip between agent actions (智能體動作之間跳過的幀數) - Decreasing can increase decision frequency but slower training, increasing can accelerate training but may miss important states
NOOP_MAX = 30  # Maximum number of no-op actions at episode start (回合開始時最大無操作動作數) - Increasing helps explore more initial states but may waste training steps

# Visualization settings
# 視覺化設置
RENDER_MODE = None  # Render mode for visualization, None for training speed (渲染模式，訓練時設為None以提高速度) - Set to 'human' to observe agent but will significantly reduce training speed
TRAINING_MODE = True  # Training mode flag to disable rendering (訓練模式標誌，禁用渲染) - Set to False will enable rendering, for testing only

###############################
# DEEP Q-LEARNING PARAMETERS
# 深度Q學習參數
###############################

# Core DQN parameters
# 核心DQN參數
LEARNING_RATE = 0.0001  # Learning rate for Adam optimizer (Adam優化器的學習率) - Increasing can accelerate learning but may be unstable, decreasing can improve stability but slower learning
GAMMA = 0.99  # Discount factor for future rewards (未來獎勵的折扣因子) - Increasing makes agent value long-term rewards more, decreasing focuses more on immediate rewards
BATCH_SIZE = 32  # Batch size for training updates (訓練更新的批次大小) - Increasing provides more stable gradient estimation but increases memory consumption, decreasing may lead to unstable training
MEMORY_CAPACITY = 500000  # Experience replay memory capacity (經驗回放記憶體容量) - Increasing can store more diverse experiences but increases memory usage, decreasing may lead to overfitting to recent experiences
TARGET_UPDATE_FREQUENCY = 10000  # Steps between target network updates (目標網絡更新間隔步數) - Increasing improves training stability but reduces learning speed, decreasing improves learning speed but may cause unstable oscillations
TRAINING_EPISODES = 100000  # Total training episodes (總訓練回合數) - Increasing can improve final performance but extends training time, decreasing can speed up experiments but may not learn sufficiently

# Exploration parameters
# 探索參數
EPSILON_START = 1.0  # Initial exploration rate (初始探索率) - Keeping high ensures thorough early exploration of environment, lowering leads to earlier exploitation of known strategies
EPSILON_END = 0.05  # Final exploration rate (最終探索率) - Increasing ensures continued exploration of new strategies, decreasing focuses more on exploiting learned strategies
EPSILON_DECAY = 1700000  # Steps over which epsilon decays (epsilon衰減的步數) - Increasing makes exploration rate decay slower, ensuring longer exploration period, decreasing focuses faster on exploiting learned strategies
DEFAULT_EVALUATE_MODE = False  # Default evaluation mode (默認評估模式) - When set to True disables exploration, for evaluation only without affecting training

# Training control parameters
# 訓練控制參數
LEARNING_STARTS = 20000  # Steps before starting learning (開始學習前的步數) - Increasing can collect more random experiences ensuring diversity, decreasing can accelerate start of learning
UPDATE_FREQUENCY = 2  # Steps between network updates (網絡更新間隔步數) - Decreasing can update network more frequently to accelerate learning, increasing can reduce computational burden but may slow learning

###############################
# PRIORITIZED EXPERIENCE REPLAY PARAMETERS
# 優先經驗回放參數
###############################

# Whether to use Prioritized Experience Replay
# 是否使用優先經驗回放
USE_PER = True  # Enable Prioritized Experience Replay (啟用優先經驗回放) - When set to False uses standard uniform sampling, PER typically improves sample efficiency but increases computational complexity

# PER hyperparameters
# PER超參數
ALPHA = 0.6  # Priority exponent for sampling probability (優先級指數，用於採樣概率) - Increasing reinforces high-error sample sampling frequency, decreasing makes sampling closer to uniform
BETA_START = 0.4  # Initial importance sampling weight value (初始重要性採樣權重值) - Increasing can reduce initial bias but may slow convergence, keeping low value makes early learning more focused on high-error samples
BETA_FRAMES = 1700000  # Frames over which beta increases to 1.0 (beta增加到1.0的幀數) - Increasing makes bias correction more gradual but extends unbalanced learning phase, decreasing can accelerate reaching unbiased learning
BETA_EXPONENT = 1.02  # Exponent for beta increase (beta增加的指數) - Increasing can make beta grow faster, decreasing makes growth more gradual
EPSILON_PER = 1e-6  # Small constant for priority calculation (優先級計算的小常數) - Ensures all experiences have non-zero priority, preventing some experiences from never being sampled

# SumTree settings
# 總和樹設置
TREE_CAPACITY = MEMORY_CAPACITY  # Size of the sum tree (與記憶體容量一致) - Should remain consistent with memory capacity, otherwise may lead to inconsistent memory usage
DEFAULT_NEW_PRIORITY = 1.0  # Default priority for new experiences (新經驗的默認優先級) - Sets initial priority for new experiences, affects the chance of newly added experiences being sampled

# PER Logging Configuration
PER_LOG_FREQUENCY = 100  # Steps between PER metrics logging (PER指標記錄的步數間隔) - Decreasing can provide more detailed PER metrics logging but increases log size
PER_BATCH_SIZE = 50  # Batch size for PER data writes (PER數據寫入的批次大小) - Increasing can reduce I/O operations but increases memory usage, decreasing can reduce memory burden but increases I/O frequency

###############################
# NEURAL NETWORK SETTINGS
# 神經網絡設置
###############################

USE_ONE_CONV_LAYER = False  # Whether to use a single convolutional layer (是否使用單個卷積層) - Simplifies model structure, reduces computational requirements but may weaken feature extraction capability
USE_TWO_CONV_LAYERS = True  # Whether to use two convolutional layers (是否使用兩個卷積層) - Medium complexity, balances computational efficiency and feature extraction capability
USE_THREE_CONV_LAYERS = False  # Whether to use three convolutional layers (是否使用三個卷積層) - Increases model complexity and expressiveness, improves feature extraction but increases computational requirements

# First convolutional layer parameters
# 第一卷積層參數
CONV1_CHANNELS = 32  # Number of filters in first conv layer (第一卷積層過濾器數量) - Increasing can extract more basic features but increases computation
CONV1_KERNEL_SIZE = 8  # Kernel size for first conv layer (第一卷積層核大小) - Increasing can capture features with larger field of view, decreasing focuses more on details
CONV1_STRIDE = 4  # Stride for first conv layer (第一卷積層步幅) - Increasing can reduce output feature map size saving computation, decreasing can preserve more spatial information but increases computation

# Second convolutional layer parameters
# 第二卷積層參數
CONV2_CHANNELS = 64  # Number of filters in second conv layer (第二卷積層過濾器數量) - Increasing can extract more complex feature combinations but increases computational requirements
CONV2_KERNEL_SIZE = 4  # Kernel size for second conv layer (第二卷積層核大小) - Affects receptive field size of mid-level features
CONV2_STRIDE = 2  # Stride for second conv layer (第二卷積層步幅) - Controls the rate of feature map size reduction

# Third convolutional layer parameters
# 第三卷積層參數
CONV3_CHANNELS = 64  # Number of filters in third conv layer (第三卷積層過濾器數量) - High-level feature extraction capability, increasing can improve complex pattern recognition
CONV3_KERNEL_SIZE = 3  # Kernel size for third conv layer (第三卷積層核大小) - Smaller kernels focus on fine feature integration
CONV3_STRIDE = 1  # Stride for third conv layer (第三卷積層步幅) - Keeping at 1 typically preserves more complete spatial information in the final convolutional layer

# Fully connected layer and gradient settings
# 全連接層和梯度設置
FC_SIZE = 512  # Size of fully connected layer (全連接層大小) - Increasing improves model representation capability but increases parameter count, decreasing can reduce overfitting risk but may lower expressiveness
GRAD_CLIP_NORM = 5.0  # Gradient clipping norm (梯度裁剪范數) - Prevents gradient explosion, increasing allows larger update steps but may cause instability, decreasing improves stability but may slow learning

# Evaluation settings
# 評估設置
EVAL_EPISODES = 100  # Number of episodes for each evaluation (每次評估的回合數) - Increasing can provide more reliable evaluation results but extends evaluation time
EVAL_FREQUENCY = 100  # Episodes between evaluations (評估間隔的回合數) - Decreasing can evaluate training progress more frequently but extends total training time, increasing can accelerate training but reduces progress monitoring frequency

###############################
# LOGGER SETTINGS
# 日誌記錄器設置
###############################

# System resource management
# 系統資源管理
MEMORY_THRESHOLD_PERCENT = 90  # Memory usage threshold percentage (內存使用閾值百分比) - Decreasing can control memory usage more conservatively but may limit performance, increasing allows more memory usage but higher risk

# Directory configurations
# 目錄配置
RESULTS_DIR = "results"  # Main results directory (主要結果目錄) - 保存所有實驗結果的主目錄
LOG_DIR = os.path.join(RESULTS_DIR, "logs")  # Directory for log files (日誌文件目錄)
MODEL_DIR = os.path.join(RESULTS_DIR, "models")  # Directory for model checkpoints (模型檢查點目錄)
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")  # Directory for visualization plots (可視化圖表目錄)
DATA_DIR = os.path.join(RESULTS_DIR, "data")  # Directory for data storage (數據存儲目錄)

# Logger settings
# 日誌記錄器設置
ENABLE_FILE_LOGGING = False  # Whether to write logs to files (是否將日誌寫入文件) - Enabling can save complete training records but increases I/O operations, disabling can reduce system burden but loses historical records
LOGGER_SAVE_INTERVAL = 100  # Episodes between logger saves (日誌保存間隔的回合數) - Decreasing can save progress more frequently but increases I/O operations, increasing can reduce I/O burden but higher risk
LOGGER_MEMORY_WINDOW = 1000  # Maximum records kept in memory (内存中保留的最大記錄數) - Increasing can store longer historical records but increases memory usage, decreasing can save memory but limits accessible historical data
LOGGER_BATCH_SIZE = 50  # Records to accumulate before writing (累積多少記錄後寫入磁盤) - Increasing can reduce I/O frequency but delays data persistence, decreasing can save records faster but increases I/O frequency
LOGGER_DETAILED_INTERVAL = 50  # Episodes between detailed reports (詳細報告間隔的回合數) - Decreasing provides more frequent detailed progress reports but increases output volume, increasing can reduce output volume but lowers monitoring granularity
LOGGER_MAJOR_METRICS = ["reward", "loss", "epsilon", "beta"]  # Main metrics to plot (主要繪圖指標) - Customize main metrics to display in overview plots
VISUALIZATION_SAVE_INTERVAL = 1000  # Episodes between visualization saves (可視化保存間隔的回合數) - Decreasing can generate visualizations more frequently but increases I/O and computational burden, increasing can reduce burden but decreases visual feedback

VISUALIZATION_SPECIFIC_EXPERIMENT = '20250430_014335' # Specific training run for visualization (可視化的特定訓練運行) - Used to specify specific training run for visualization, typically for comparing results from different runs

###############################
# CONFIGURATION VALIDATION
# 配置驗證
###############################

def validate_config():
    """
    Validate configuration parameters for consistency and sanity.
    
    驗證配置參數的一致性和合理性。
    
    Returns:
        list: List of validation errors, empty if all valid
    """
    errors = []
    warnings_list = []
    
    # Environment validation
    if not isinstance(ENV_NAME, str) or not ENV_NAME.strip():
        errors.append("ENV_NAME must be a non-empty string")
    
    if ACTION_SPACE_SIZE <= 0:
        errors.append("ACTION_SPACE_SIZE must be positive")
    
    if DIFFICULTY < 0 or DIFFICULTY > 4:
        errors.append("DIFFICULTY must be between 0 and 4")
    
    # Frame processing validation
    if FRAME_WIDTH <= 0 or FRAME_HEIGHT <= 0:
        errors.append("FRAME_WIDTH and FRAME_HEIGHT must be positive")
    
    if FRAME_STACK <= 0:
        errors.append("FRAME_STACK must be positive")
    
    if FRAME_SKIP <= 0:
        errors.append("FRAME_SKIP must be positive")
    
    # DQN parameters validation
    if LEARNING_RATE <= 0 or LEARNING_RATE > 1:
        errors.append("LEARNING_RATE must be between 0 and 1")
    
    if GAMMA < 0 or GAMMA > 1:
        errors.append("GAMMA must be between 0 and 1")
    
    if BATCH_SIZE <= 0:
        errors.append("BATCH_SIZE must be positive")
    
    if MEMORY_CAPACITY <= BATCH_SIZE:
        errors.append("MEMORY_CAPACITY must be larger than BATCH_SIZE")
    
    if TARGET_UPDATE_FREQUENCY <= 0:
        errors.append("TARGET_UPDATE_FREQUENCY must be positive")
    
    if TRAINING_EPISODES <= 0:
        errors.append("TRAINING_EPISODES must be positive")
    
    # Exploration parameters validation
    if EPSILON_START < 0 or EPSILON_START > 1:
        errors.append("EPSILON_START must be between 0 and 1")
    
    if EPSILON_END < 0 or EPSILON_END > 1:
        errors.append("EPSILON_END must be between 0 and 1")
    
    if EPSILON_END >= EPSILON_START:
        warnings_list.append("EPSILON_END should be less than EPSILON_START for decay")
    
    if EPSILON_DECAY <= 0:
        errors.append("EPSILON_DECAY must be positive")
    
    if LEARNING_STARTS < 0:
        errors.append("LEARNING_STARTS must be non-negative")
    
    if UPDATE_FREQUENCY <= 0:
        errors.append("UPDATE_FREQUENCY must be positive")
    
    # PER parameters validation
    if USE_PER:
        if ALPHA < 0:
            errors.append("ALPHA must be non-negative")
        
        if BETA_START < 0 or BETA_START > 1:
            errors.append("BETA_START must be between 0 and 1")
        
        if BETA_FRAMES <= 0:
            errors.append("BETA_FRAMES must be positive")
        
        if EPSILON_PER <= 0:
            errors.append("EPSILON_PER must be positive")
        
        if TREE_CAPACITY != MEMORY_CAPACITY:
            warnings_list.append("TREE_CAPACITY should equal MEMORY_CAPACITY")
    
    # Neural network validation
    conv_layers = sum([USE_ONE_CONV_LAYER, USE_TWO_CONV_LAYERS, USE_THREE_CONV_LAYERS])
    if conv_layers != 1:
        errors.append("Exactly one of USE_ONE_CONV_LAYER, USE_TWO_CONV_LAYERS, USE_THREE_CONV_LAYERS must be True")
    
    if FC_SIZE <= 0:
        errors.append("FC_SIZE must be positive")
    
    if GRAD_CLIP_NORM <= 0:
        errors.append("GRAD_CLIP_NORM must be positive")
    
    # Evaluation settings validation
    if EVAL_EPISODES <= 0:
        errors.append("EVAL_EPISODES must be positive")
    
    if EVAL_FREQUENCY <= 0:
        errors.append("EVAL_FREQUENCY must be positive")
    
    # Logger settings validation
    if MEMORY_THRESHOLD_PERCENT < 50 or MEMORY_THRESHOLD_PERCENT > 100:
        warnings_list.append("MEMORY_THRESHOLD_PERCENT should be between 50-100")
    
    # Print warnings
    if warnings_list:
        print("Configuration warnings:")
        for warning in warnings_list:
            print(f"  WARNING: {warning}")
    
    return errors

def get_config_summary():
    """
    Get a summary of key configuration parameters.
    
    獲取關鍵配置參數的摘要。
    
    Returns:
        dict: Summary of configuration
    """
    return {
        'environment': {
            'name': ENV_NAME,
            'action_space': ACTION_SPACE_SIZE,
            'frame_size': f"{FRAME_WIDTH}x{FRAME_HEIGHT}",
            'frame_stack': FRAME_STACK,
            'difficulty': DIFFICULTY
        },
        'training': {
            'episodes': TRAINING_EPISODES,
            'learning_rate': LEARNING_RATE,
            'batch_size': BATCH_SIZE,
            'memory_capacity': MEMORY_CAPACITY,
            'gamma': GAMMA
        },
        'exploration': {
            'epsilon_start': EPSILON_START,
            'epsilon_end': EPSILON_END,
            'epsilon_decay': EPSILON_DECAY
        },
        'per': {
            'enabled': USE_PER,
            'alpha': ALPHA if USE_PER else None,
            'beta_start': BETA_START if USE_PER else None,
            'beta_frames': BETA_FRAMES if USE_PER else None
        }
    }

# Validate configuration on import
_validation_errors = validate_config()
if _validation_errors:
    error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in _validation_errors)
    raise ValueError(error_msg)