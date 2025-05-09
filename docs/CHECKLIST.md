# Prioritized Experience Replay DQN for Atari Games - Implementation Plan

This document outlines the detailed plan for implementing a DQN with Prioritized Experience Replay (PER) for playing Atari games.

*此文件記錄實現帶有優先經驗回放 (PER) 的 DQN 來玩 Atari 遊戲的詳細計劃。*

## Implementation Goals ✅

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

5. **Agent Visualization**
   - Implement a script (`visualize_agent.py`) to load trained models and visualize agent performance in the Atari environment
   - Support customizable parameters such as experiment ID, checkpoint selection, game speed, difficulty level, and number of episodes
   - Provide real-time gameplay visualization for debugging and performance analysis

*實作目標：初學者友好的演算法實現、資料可視化、多平台兼容性、GitHub準備、代理可視化*

## Implementation Status

All planned components are now fully implemented and operational. The implementation follows the original design closely while ensuring optimal performance across different platforms.

*所有計劃的組件現已全部實現並正常運行。實現嚴格遵循原始設計，同時確保在不同平台上的最佳性能。*

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
   - Q-network Q with random weights θ
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

## Hyperparameters for PER

- **α (Alpha)**: 0.6 - Controls how much prioritization is used (0 = uniform, 1 = full prioritization)
- **β_start (Beta start)**: 0.4 - Initial importance sampling weight
- **β_frames (Beta frames)**: 1,000,000 - Number of frames over which β is annealed to 1.0
- **ε (Epsilon for priorities)**: 1e-6 - Small constant added to TD-error to ensure non-zero priority

## References

1. **Original DQN Paper:**
   Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.

2. **Prioritized Experience Replay:**
   Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2016). "Prioritized Experience Replay." *International Conference on Learning Representations (ICLR)*.

*偽代碼：優先經驗回放算法、基本DQN與PER結合算法、SumTree數據結構*
