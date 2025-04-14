# DQN with Prioritized Experience Replay for Atari Ice Hockey

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-red.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.28+-green.svg)](https://gymnasium.farama.org/)

This project implements Deep Q-Networks with Prioritized Experience Replay (PER) to play Atari Ice Hockey. By prioritizing experiences, PER can improve DQN's learning efficiency by focusing on more valuable transitions.

*這個項目實現了帶優先經驗回放 (Prioritized Experience Replay, PER) 的深度 Q 網絡 (DQN) 來玩 Atari 冰球遊戲。通過使用優先級來選擇經驗，優先經驗回放可以提高 DQN 的學習效率，關注更有價值的轉換。*

![Ice Hockey Game](https://gymnasium.farama.org/_images/ice_hockey.gif)

## Features

- **Prioritized Experience Replay** - Using SumTree data structure for O(log n) priority-based sampling
- **Visualization Tools** - Training progress, reward curves, priority distributions
- **Cross-platform Compatibility** - Support for CPU, NVIDIA GPUs, and Apple Silicon (MPS)
- **Comprehensive Educational Comments** - Beginner-friendly reinforcement learning code
- **Detailed Pseudocode** - Helps understand algorithm principles

*特點：優先經驗回放、可視化工具、跨平台兼容、完整的教學註釋、詳細的偽代碼*

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.8 or higher
- Gymnasium with Atari support

### Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/dqn-per-ice-hockey.git
   cd dqn-per-ice-hockey
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Atari ROM support:
   ```bash
   pip install gymnasium[atari,accept-rom-license]
   ```

*安裝：克隆此倉庫，安裝依賴，安裝 Atari ROM 支持*

## Usage

### Training the agent

To train a DQN agent with PER using default parameters:
```bash
python train.py
```

Additional training options:
```bash
# Without Prioritized Experience Replay
python train.py --use-per False

# Display training visualization
python train.py --render

# Specify number of training episodes
python train.py --episodes 1000

# Force CPU usage
python train.py --cpu
```

*訓練：使用默認參數訓練、不使用優先經驗回放、顯示訓練可視化、指定訓練回合數、強制使用CPU*

### Evaluating the agent

To evaluate a trained model:
```bash
python evaluate.py --model checkpoints/dqn_per_TIMESTAMP_best.pth --render
```

Additional evaluation options:
```bash
# Slow playback for better observation
python evaluate.py --model checkpoints/dqn_per_TIMESTAMP_best.pth --render --slow

# Record videos
python evaluate.py --model checkpoints/dqn_per_TIMESTAMP_best.pth --record

# Specify number of evaluation episodes
python evaluate.py --model checkpoints/dqn_per_TIMESTAMP_best.pth --episodes 20
```

*評估：評估訓練好的模型、慢速播放、錄製影片、指定評估回合數*

## Project Structure

```
.
├── config.py                 # Configuration parameters
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
├── src/                      # Source code
│   ├── agent/                # Agent implementation
│   │   ├── __init__.py
│   │   ├── dqn_agent.py      # DQN agent
│   │   └── q_network.py      # Q network architecture
│   ├── environment/          # Environment wrappers
│   │   ├── __init__.py
│   │   └── env_wrappers.py   # Atari environment wrappers
│   ├── memory/               # Memory module
│   │   ├── __init__.py
│   │   ├── per_memory.py     # Prioritized experience replay
│   │   └── sumtree.py        # SumTree data structure
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── device_utils.py   # Device detection
│       ├── logger.py         # Logging
│       └── visualization.py  # Visualization
├── logs/                     # Logs directory
├── checkpoints/              # Model checkpoints
├── results/                  # Results and plots
└── README.md                 # This document
```

*項目結構：配置參數、訓練腳本、評估腳本、源代碼、日誌目錄、模型檢查點、結果和圖表*

## Prioritized Experience Replay Algorithm

Prioritized Experience Replay is an improved experience replay technique that assigns higher sampling probabilities to more valuable experiences. The method is based on the following key ideas:

1. **Priority Calculation**:
   - Using absolute TD-error as priority: `p_i = |δ_i|^α + ε`
   - Where `α` controls priority strength and `ε` ensures non-zero probability

2. **Efficient Sampling**:
   - Using SumTree data structure for O(log n) priority-based sampling
   - Dividing total priority into B segments, sampling from each segment

3. **Importance Sampling**:
   - Using importance sampling weights to correct priority-induced bias: `w_i = (N · P(i))^(-β)`
   - `β` gradually increases from initial value to 1, balancing early exploration and later convergence

*優先經驗回放算法：優先級計算、高效抽樣、重要性採樣*

## Experimental Results

Prioritized Experience Replay significantly improves DQN's learning efficiency on Atari Ice Hockey, demonstrating:

- **Faster Convergence** - PER typically reaches higher rewards faster than standard DQN
- **More Stable Learning Curves** - Reduced fluctuations during training
- **Better Final Performance** - Achieves better results with the same number of training steps

*實驗結果：更快的收斂速度、更穩定的學習曲線、更好的最終表現*

## Acknowledgements

- DeepMind's DQN paper: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- PER paper: [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952) (Schaul et al., 2015)
- Gymnasium team for the Atari environments

*致謝：DeepMind 的 DQN 論文、PER 論文、Gymnasium 團隊*

## License

MIT License

## Contact

For any questions, please create an issue or contact me via email.

*聯繫方式：如有任何問題，請創建 issue 或通過電子郵件聯繫我。*