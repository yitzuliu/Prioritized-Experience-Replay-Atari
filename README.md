# Prioritized Experience Replay (PER) for Atari Games

Training a Deep Q-Network (DQN) to play Atari games using Prioritized Experience Replay (PER) technique.

使用優先經驗回放 (PER) 技術訓練深度 Q 網絡 (DQN) 學習玩 Atari 遊戲。

![Atari Game Example](https://gymnasium.farama.org/_images/ms_pacman.gif)

## 📝 Project Overview (專案概述)

This project implements a Deep Q-Network (DQN) with Prioritized Experience Replay (PER) to play Atari games. PER is an enhanced experience replay mechanism that prioritizes sampling of high-value experiences based on their importance (measured by TD-error). This approach significantly improves DQN's learning efficiency and performance.

本專案實現了一個帶有優先經驗回放 (PER) 的深度 Q 網絡 (DQN) 來玩 Atari 遊戲。優先經驗回放是一種改進型經驗回放機制，它根據樣本的重要性（由 TD 誤差測量）來優先採樣高價值的經驗。這種方法可以顯著提高 DQN 的學習效率和性能。

### 💡 Key Features (主要特點)

- **Complete PER Implementation**: Using SumTree data structure for efficient priority-based experience storage and sampling
- **Multi-Platform Compatibility**: Automatic detection and utilization of CPU, CUDA (NVIDIA GPU), or MPS (Apple Silicon)
- **Detailed Visualization**: Provides visualizations of training progress, rewards, losses, priority distributions, and TD errors
- **Educational Implementation**: Includes detailed bilingual (English/Chinese) comments and algorithm explanations suitable for learning and research
- **Efficient Training**: Optimized environment preprocessing, experience sampling, and model architecture
- **Graceful Interruption Handling**: Training can be safely interrupted (Ctrl+C) with automatic checkpoint saving and resuming capability

*主要特點：完整的 PER 實現、多平台兼容、詳細視覺化、教育性實現、高效訓練與優雅的中斷處理*

## 🛠️ Installation & Setup (安裝與設置)

### Prerequisites (前提條件)

- Python 3.8+
- PyTorch 2.0+
- Gymnasium (newer version of OpenAI Gym)
- NumPy, Matplotlib, OpenCV
- Other dependencies listed in requirements.txt

### Installation Steps (安裝步驟)

1. Clone the repository
```bash
git clone https://github.com/yourusername/atari-per-dqn.git
cd atari-per-dqn
```

2. Install required dependencies
```bash
pip install -r requirements.txt
```

## 📊 Project Structure (專案結構)

```
.
├── config.py                    # Configuration file with all hyperparameters
├── train.py                     # Training script to start the training process
├── src/
│   ├── dqn_agent.py             # DQN agent implementation
│   ├── per_memory.py            # Prioritized Experience Replay memory
│   ├── sumtree.py               # SumTree data structure implementation
│   ├── q_network.py             # Q-Network neural network architecture
│   ├── env_wrappers.py          # Atari environment wrappers
│   ├── device_utils.py          # Device detection and optimization utilities
│   ├── logger.py                # Training logging tools
│   └── visualization.py         # Training metrics visualization tools
├── results/                      # Directory for storing training results
│   ├── data/                    # Training data
│   ├── logs/                    # Training logs
│   ├── models/                  # Saved models
│   └── plots/                   # Generated plots
└── CHECKLIST.md                 # Implementation plan and pseudocode
```

## 🚀 Usage (使用方法)

### Training a Model (訓練模型)

To train a new model from scratch:

```bash
python train.py
```

### Visualizing Agent Performance (可視化智能體表現)

To observe the performance of a trained agent, use the `visualize_agent.py` script. This script allows you to load a specific experiment and checkpoint to visualize gameplay.

#### Example
```bash
python visualize_agent.py --exp_id exp_20250430_014335 --checkpoint checkpoint_1000.pt --speed 0.5 --episodes 5 --difficulty 2
```

- `--exp_id`: Specify the experiment ID (e.g., `exp_20250430_014335`).
- `--checkpoint`: Specify the checkpoint file to load (e.g., `checkpoint_1000.pt`).
- `--speed`: Adjust the game speed (e.g., `0.5` for slow motion).
- `--episodes`: Number of episodes to visualize (default: 3).
- `--difficulty`: Set the game difficulty level (0-4).

This script is useful for debugging and analyzing the agent's behavior in the environment.

### Algorithm Description (演算法說明)

This project implements the DQN algorithm with Prioritized Experience Replay:

1. **Prioritized Experience Replay (PER)**:
   - Efficiently stores and samples experiences using a SumTree data structure
   - Transitions are assigned priorities based on TD-error: p = (|δ|+ε)^α
   - Importance sampling weights w = (N·P(i))^(-β) correct the bias introduced
   - β gradually increases from 0.4 to 1.0 over time

2. **DQN Architecture**:
   - 3-layer convolutional neural network followed by fully connected layers
   - Dual network architecture (policy and target networks) with PER integration
   - ε-greedy exploration strategy with decaying ε value
   - Target network updates every 20,000 steps

## 📈 Experimental Results (實驗結果)

The training process generates various visualizations stored in the `results/plots` directory:

- Reward curves: Shows average rewards obtained per episode
- Loss curves: Shows how the network training loss changes
- Priority distribution: Shows the distribution of priorities in the experience replay
- Exploration rate curve: Shows how ε value changes over time

## 🔍 Algorithm Details (算法細節)

### SumTree Data Structure

The implemented SumTree has the following main operations:
- `add`: Add new experience with its priority
- `update_priority`: Update the priority of an experience
- `get_experience_by_priority`: Retrieve experience based on priority value

### Prioritized Experience Replay Mechanism

The PER memory implements:
- Priority calculation based on TD-errors
- Computation and proper application of importance sampling weights
- Linear annealing strategy for β value
- Efficient batch sampling

## 🤝 Contributing (貢獻)

Contributions and improvements are welcome! If you're interested in improving this project, please follow these steps:

1. Fork the repository
2. Create a new branch for your feature (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's coding style and includes appropriate tests.

貢獻和改進是受歡迎的！如果您對改進此專案感興趣，請按照以下步驟操作：

1. Fork 此存儲庫
2. 為您的功能創建一個新分支 (`git checkout -b feature/amazing-feature`)
3. 提交您的更改 (`git commit -m '添加一些令人驚嘆的功能'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 打開一個拉取請求

請確保您的代碼遵循項目的編碼風格並包含適當的測試。

## 📄 License (許可證)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

本專案採用 MIT 許可證 - 詳情請參閱 [LICENSE](LICENSE) 文件。

## 👨‍💻 Author (作者)

Yitzu Liu - [@yitzuliu](https://github.com/yitzuliu)

## 🙏 Acknowledgements (致謝)

- This project is based on the [DQN paper](https://www.nature.com/articles/nature14236) by DeepMind and the [PER paper](https://arxiv.org/abs/1511.05952) by Google DeepMind
- Uses Atari game environments provided by [OpenAI Gymnasium](https://gymnasium.farama.org/)
- Special thanks to all researchers and developers in the reinforcement learning community

## 📊 Project Status (專案狀態)

This project is currently in active development. Core features are implemented and working, but optimizations and additional features are planned.

目前該專案正在積極開發中。核心功能已經實現並正常運作，但計劃進行優化和添加其他功能。