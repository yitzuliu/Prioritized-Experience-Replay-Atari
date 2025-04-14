# Prioritized Experience Replay DQN for Atari Ice Hockey - Implementation Plan

This document outlines the detailed plan for implementing a DQN with Prioritized Experience Replay (PER) for playing Atari Ice Hockey.

*此文件記錄實現帶有優先經驗回放 (PER) 的 DQN 來玩 Atari 冰球遊戲的詳細計劃。*

## Implementation Goals

1. **Beginner-friendly Algorithm Implementation**
   - Focus on clarity and readability of algorithm implementation
   - Provide educational-level English comments with Chinese annotations
   - Include pseudocode for each core algorithm
   - Ensure perfect execution beyond 5,000 steps for observable learning effects

2. **Data Visualization**
   - Provide reward curves during the training process
   - Display experience replay priority distribution
   - Present charts for loss function and other relevant metrics

3. **Multi-platform Compatibility**
   - Ensure code runs on CPU, GPU, and Apple M-series chips
   - Automatically detect available devices and optimize performance

4. **GitHub Preparation**
   - Complete documentation and examples
   - Well-organized code structure

*實作目標：初學者友好的演算法實現、資料可視化、多平台兼容性、GitHub準備*

## Pseudocode

### Prioritized Experience Replay (PER) Pseudocode
```
Algorithm: Prioritized Experience Replay
1. Initialize SumTree with capacity N
2. For each transition:
   a. Calculate priority p = |δ|^α + ε  (where δ is TD-error)
   b. Store transition with priority p in SumTree
3. When sampling batch of size B:
   a. Divide total priority sum into B segments
   b. From each segment i, sample value v_i uniformly
   c. Retrieve transition with priority proportional to v_i
   d. Calculate importance sampling weight w_i = (N · P(i))^(-β)
   e. Normalize weights to max weight
4. When updating network:
   a. Use importance weights to scale loss: L = Σ w_i · (TD-error_i)^2
   b. Update priorities in SumTree with new |TD-error|^α + ε
```

### DQN with PER Pseudocode
```
Algorithm: Deep Q-Network with Prioritized Experience Replay
1. Initialize:
   - Primary Q-network Q with random weights θ
   - Target Q-network Q' with weights θ' = θ
   - Prioritized Experience Replay memory D
   
2. For each episode:
   a. Initialize state s from environment
   b. For each step t:
      i. Select action a_t using ε-greedy policy
         - With probability ε: choose random action
         - Otherwise: choose a_t = argmax_a Q(s_t, a; θ)
      ii. Execute a_t, observe r_t and s_{t+1}
      iii. Store (s_t, a_t, r_t, s_{t+1}, done) in D with max priority
      iv. If enough samples in D, perform learning:
          - Sample batch using PER algorithm
          - For each transition j:
            * Compute y_j = r_j if done
                         = r_j + γ·max_a Q'(s_{j+1}, a; θ') otherwise
            * Compute TD-error δ_j = y_j - Q(s_j, a_j; θ)
          - Update Q using loss L = 1/B · Σ w_j · (δ_j)^2
          - Update priorities in D with new |TD-error|
      v. Every C steps, update target network: θ' ← θ
      vi. If done, break
```

### SumTree Data Structure Pseudocode
```
Data Structure: SumTree
1. Initialize:
   - Complete binary tree with capacity N leaves
   - Internal nodes store sum of child priorities
   - Leaf nodes store priorities of transitions
   
2. Operations:
   a. update(index, priority):
      - Set leaf value at index to priority
      - Propagate change upwards, updating all parent sums
      
   b. total():
      - Return value at root (total priority sum)
      
   c. sample(value):
      - Starting from root, traverse tree:
        * If value <= left_child.sum, go left
        * Else, subtract left_child.sum from value, go right
      - Return index, priority and data at leaf
```

*偽代碼：優先經驗回放算法、DQN與PER結合算法、SumTree數據結構*

## Detailed Plan and Progress Tracking

### 1. Basic Architecture Setup
- [x] Analyze existing file structure
- [x] Enhance config.py to add PER configuration parameters
- [x] Design interactions between agent, environment, memory modules

### 2. Implement Prioritized Experience Replay Memory
- [x] Create memory/sumtree.py implementing SumTree data structure
- [x] Create memory/per_memory.py implementing PER
- [x] Implement priority calculation and update mechanisms
- [x] Implement priority-based sampling
- [x] Implement importance sampling weight calculation

### 3. Implement DQN Agent
- [x] Create agent/q_network.py implementing DQN neural network architecture
- [x] Create agent/dqn_agent.py implementing DQN agent
- [x] Integrate PER with DQN

### 4. Environment Processing
- [x] Check and optimize environment wrappers (environment/env_wrappers.py)
- [x] Ensure game environment is properly configured

### 5. Tools and Utilities
- [x] Establish utils/ directory for auxiliary functions
- [x] Implement device detection supporting CPU/GPU/M-series chips
- [x] Build data collection and visualization tools

### 6. Training and Evaluation Scripts
- [x] Create train.py main training script
- [x] Create evaluate.py evaluation script
- [x] Implement phased model saving and training restoration functions

### 7. Visualization and Analysis
- [x] Implement training process data recording
- [x] Implement charts for rewards, losses, priority distributions
- [x] Build training result analysis tools

### 8. Documentation and Examples
- [x] Create detailed README.md file
- [x] Provide usage examples and running instructions
- [x] Add performance analysis and result discussion

### 9. Testing and Optimization
- [x] Conduct complete testing to ensure operation beyond 5,000 steps
- [x] Optimize memory and computational efficiency
- [x] Ensure cross-platform compatibility

### 10. GitHub Preparation
- [x] Create .gitignore file
- [x] Organize code and documentation
- [x] Final checks and adjustments

*進度追蹤：基本架構設置、優先經驗回放記憶體實現、DQN智能體實現、環境處理、工具和輔助功能、訓練和評估腳本、可視化和分析、文檔和範例、測試和優化、GitHub準備*