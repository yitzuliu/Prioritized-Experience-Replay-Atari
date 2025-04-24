"""
Prioritized Experience Replay (PER) Memory implementation.

This module implements a PER memory buffer using a SumTree for efficient
priority-based sampling. It allows transitions to be stored with priorities
and sampled according to their importance.

優先經驗回放 (PER) 記憶體實現。

此模組使用 SumTree 實現 PER 記憶體緩衝區，用於高效的基於優先級的採樣。
它允許經驗轉換以優先級存儲，並根據其重要性進行採樣。
"""

import numpy as np
import os
import sys
from collections import namedtuple

# Add parent directory to path to import config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from src.sumtree import SumTree

# Define a transition class using namedtuple for efficiency and readability
Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'next_state', 'done'))


class PERMemory:
    """
    Prioritized Experience Replay Memory.
    
    Stores transitions with priorities and samples them based on their
    importance using a SumTree data structure.
    
    優先經驗回放記憶體。
    
    使用優先級存儲轉換，並使用 SumTree 數據結構根據其重要性進行採樣。
    """
    
    def __init__(self, memory_capacity=config.MEMORY_CAPACITY, 
                 alpha=config.ALPHA, 
                 beta_start=config.BETA_START, 
                 beta_frames=config.BETA_FRAMES,
                 epsilon=config.EPSILON_PER):
        """
        Initialize the PER memory.
        
        Args:
            memory_capacity: Maximum capacity of the memory
            alpha: Priority exponent (controls how much prioritization is used)
            beta_start: Initial importance-sampling weight coefficient
            beta_frames: Number of frames over which beta will anneal from beta_start to 1.0
            epsilon: Small constant added to priorities to ensure non-zero probability
            
        初始化 PER 記憶體。
        
        參數：
            memory_capacity: 記憶體的最大容量
            alpha: 優先級指數（控制使用多少優先級）
            beta_start: 初始重要性採樣權重係數
            beta_frames: beta 將從 beta_start 增加到 1.0 的幀數
            epsilon: 添加到優先級的小常數，確保非零概率
        """
        # Initialize the SumTree
        self.sumtree = SumTree(memory_capacity)
        
        # Store parameters
        self.memory_capacity = memory_capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        
        # Start from beta_start and anneal towards 1.0
        self.beta = beta_start
        
        # Track total frames to calculate current beta
        self.frame_count = 0
        
        # Maximum priority to use for new experiences
        self.max_priority = 1.0
    
    def update_beta(self, frame_idx):
        """
        Update the beta parameter based on current training progress.
        
        Args:
            frame_idx: Current frame index
            
        根據當前訓練進度更新 beta 參數。
        
        參數：
            frame_idx: 當前幀索引
        """
        self.frame_count = frame_idx
        # Linear annealing from beta_start to 1.0
        self.beta = min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def _calculate_priority(self, td_error):
        """
        Calculate priority from TD-error.
        
        Args:
            td_error: TD-error
            
        Returns:
            float: Priority value
            
        根據 TD 誤差計算優先級。
        
        參數：
            td_error: TD 誤差
            
        返回：
            float: 優先級值
        """
        # Convert TD-error to priority using the formula from the PER paper
        # priority = (|δ| + ε)^α
        return (np.abs(td_error) + self.epsilon) ** self.alpha
    
    def add(self, state, action, reward, next_state, done, td_error=None):
        """
        Add a new transition to the memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
            td_error: TD-error. If None, max priority is used
            
        將新的轉換添加到記憶體中。
        
        參數：
            state: 當前狀態
            action: 採取的動作
            reward: 獲得的獎勵
            next_state: 下一個狀態
            done: 回合是否結束
            td_error: TD 誤差。如果為 None，則使用最大優先級
        """
        # Create transition
        transition = Transition(state, action, reward, next_state, done)
        
        # Calculate priority
        if td_error is None:
            priority = self.max_priority
        else:
            priority = self._calculate_priority(td_error)
            self.max_priority = max(self.max_priority, priority)
        
        # Add to SumTree
        self.sumtree.add(priority, transition)
    
    def sample(self, batch_size):
        """
        Sample a batch of transitions based on their priorities.
        
        Args:
            batch_size: Size of the batch to sample
            
        Returns:
            tuple: (batch_indices, batch_weights, batch_transitions)
                batch_indices: Indices of sampled transitions in the tree
                batch_weights: Importance sampling weights for the batch
                batch_transitions: Sampled transitions
                
        根據優先級採樣一批轉換。
        
        參數：
            batch_size: 要採樣的批次大小
            
        返回：
            tuple: (batch_indices, batch_weights, batch_transitions)
                batch_indices: 樹中採樣的轉換索引
                batch_weights: 批次的重要性採樣權重
                batch_transitions: 採樣的轉換
        """
        # Lists to store the sampled data
        batch_indices = np.zeros(batch_size, dtype=np.int32)
        batch_weights = np.zeros(batch_size, dtype=np.float32)
        batch_transitions = []
        
        # Calculate segment size
        segment_size = self.sumtree.total_priority() / batch_size
        
        # Current beta for importance sampling weights
        current_beta = self.beta
        
        # Calculate the maximum weight for normalization
        min_prob = np.min(self.sumtree.get_all_priorities()) / self.sumtree.total_priority()
        max_weight = (min_prob * self.sumtree.experience_count) ** (-current_beta)
        
        # Sample from each segment
        for i in range(batch_size):
            # Get the segment bounds
            segment_start = segment_size * i
            segment_end = segment_size * (i + 1)
            
            # Sample a value from the segment
            value = np.random.uniform(segment_start, segment_end)
            
            # Get the transition, its index and priority
            idx, priority, transition = self.sumtree.get_experience_by_priority(value)
            
            # Store the index
            batch_indices[i] = idx
            
            # Calculate sampling weight
            sample_prob = priority / self.sumtree.total_priority()
            weight = (sample_prob * self.sumtree.experience_count) ** (-current_beta)
            
            # Normalize weight to be <= 1
            batch_weights[i] = weight / max_weight
            
            # Store the transition
            batch_transitions.append(transition)
        
        return batch_indices, batch_weights, batch_transitions
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities of transitions based on new TD-errors.
        
        Args:
            indices: Tree indices of transitions to update
            td_errors: New TD-errors
            
        根據新的 TD 誤差更新轉換的優先級。
        
        參數：
            indices: 要更新的轉換的樹索引
            td_errors: 新的 TD 誤差
        """
        for idx, error in zip(indices, td_errors):
            # Calculate new priority
            priority = self._calculate_priority(error)
            
            # Update max priority
            self.max_priority = max(self.max_priority, priority)
            
            # Update priority in the tree
            self.sumtree.update_priority(idx, priority)
    
    def __len__(self):
        """
        Get the current size of the memory.
        
        Returns:
            int: Number of stored transitions
            
        獲取記憶體的當前大小。
        
        返回：
            int: 存儲的轉換數量
        """
        return self.sumtree.experience_count


# For testing purposes
if __name__ == "__main__":
    import time
    
    # Create a small memory buffer for demonstration
    memory = PERMemory(memory_capacity=100)
    
    # Add some random transitions
    for i in range(50):
        state = np.random.rand(4, 84, 84)  # Random state (4 stacked frames)
        action = np.random.randint(0, 18)  # Random action (18 possible actions)
        reward = np.random.rand() * 2 - 1  # Random reward between -1 and 1
        next_state = np.random.rand(4, 84, 84)  # Random next state
        done = np.random.random() > 0.8  # 20% chance of episode ending
        
        # Random TD error (just for demonstration)
        td_error = np.random.rand() * 2  # Random error between 0 and 2
        
        # Add transition with its error
        memory.add(state, action, reward, next_state, done, td_error)
    
    print(f"Memory size: {len(memory)}")
    print(f"Initial beta: {memory.beta}")
    
    # Update beta based on progress
    memory.update_beta(config.BETA_FRAMES // 2)
    print(f"Updated beta (halfway): {memory.beta}")
    
    # Sample a batch
    start_time = time.time()
    batch_indices, batch_weights, batch = memory.sample(32)
    end_time = time.time()
    
    print(f"\nSampled batch in {(end_time - start_time)*1000:.2f} ms")
    print(f"Batch size: {len(batch)}")
    print(f"First few batch weights: {batch_weights[:5]}")
    
    # Update priorities with new errors
    new_errors = np.random.rand(32) * 2  # Random new errors
    memory.update_priorities(batch_indices, new_errors)
    
    print("\nPriorities updated")
    
    # Sample again to show the effect of updated priorities
    _, new_weights, _ = memory.sample(32)
    print(f"New first few batch weights: {new_weights[:5]}")
    
    # Check max priority
    print(f"\nMax priority: {memory.max_priority}")