"""
Prioritized Experience Replay (PER) Memory Implementation.

This file implements the PER memory buffer as described in the paper:
"Prioritized Experience Replay" by Schaul et al. (2016)
https://arxiv.org/abs/1511.05952

優先經驗回放 (PER) 記憶體實現。

實現了 Schaul 等人 (2016) 提出的 PER 記憶體緩衝區，
論文地址：https://arxiv.org/abs/1511.05952

Pseudocode for PER:
1. Initialize SumTree with capacity N
2. For each transition, calculate priority = |TD error|^α + ε
3. Store transition with its priority in the SumTree
4. When sampling:
   - Sample a value S in range [0, total_priority]
   - Retrieve transition with priority proportional to S
   - Calculate importance sampling weight w = (N * P(i))^(-β)
5. Return batch of transitions and their importance weights
"""

import numpy as np
import torch
import config
from config import (
    MEMORY_CAPACITY, BATCH_SIZE, GAMMA, ALPHA, BETA_START, 
    BETA_FRAMES, EPSILON_PER, TREE_CAPACITY, 
    DEFAULT_NEW_PRIORITY
)
from src.memory.sumtree import SumTree
from src.utils.device_utils import get_device


class PrioritizedReplayMemory:
    """
    Prioritized Experience Replay (PER) Memory.
    
    Stores transitions and allows sampling based on their TD-error (priority),
    which leads to more efficient learning by focusing on important experiences.
    
    優先經驗回放 (PER) 記憶體。
    
    存儲轉換並允許基於其 TD 誤差（優先級）進行採樣，
    通過專注於重要經驗，從而實現更高效的學習。
    """
    
    def __init__(self, capacity=MEMORY_CAPACITY, alpha=ALPHA, 
                 beta_start=BETA_START, beta_frames=BETA_FRAMES, 
                 epsilon=EPSILON_PER):
        """
        Initialize PER memory.
        
        Args:
            capacity (int): Maximum number of transitions to store
            alpha (float): How much prioritization to use (0=none, 1=full)
            beta_start (float): Initial importance sampling weight (0=none, 1=full)
            beta_frames (int): Number of frames over which beta will be annealed to 1
            epsilon (float): Small constant added to TD-errors to ensure non-zero priority
            
        初始化 PER 記憶體。
        
        參數：
            capacity (int)：要存儲的最大轉換數量
            alpha (float)：使用優先級的程度 (0=無，1=完全)
            beta_start (float)：初始重要性採樣權重 (0=無，1=完全)
            beta_frames (int)：β值增長到1.0所需的幀數
            epsilon (float)：添加到TD誤差的小常數，確保優先級非零
        """
        # Initialize the SumTree
        self.tree = SumTree(capacity)
        
        # PER hyperparameters
        self.alpha = alpha  # Priority exponent
        self.beta = beta_start  # Importance sampling exponent
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon  # Small constant for non-zero priority
        
        # To track how many frames have been processed for beta annealing
        self.frame_count = 0
        
        # Max priority for new experiences
        self.max_priority = DEFAULT_NEW_PRIORITY
    
    def _get_priority(self, td_error):
        """
        Calculate priority from TD-error.
        
        Args:
            td_error (float): The TD-error of a transition
            
        Returns:
            float: Priority value
            
        從 TD 誤差計算優先級。
        
        參數：
            td_error (float)：轉換的 TD 誤差
            
        返回：
            float：優先級值
        """
        # Convert TD-error to priority according to the PER formula
        # P(i) = (|δ_i| + ε)^α
        return (np.abs(td_error) + self.epsilon) ** self.alpha
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        
        Args:
            state: The current state
            action: The action taken
            reward: The reward received
            next_state: The next state
            done: Whether the episode ended
            
        將新經驗添加到記憶體中。
        
        參數：
            state：當前狀態
            action：採取的動作
            reward：獲得的獎勵
            next_state：下一個狀態
            done：回合是否結束
        """
        # Store experience as a tuple
        experience = (state, action, reward, next_state, done)
        
        # Use max priority for new experiences to ensure they're sampled at least once
        self.tree.add(self.max_priority, experience)
    
    def sample(self, batch_size=BATCH_SIZE):
        """
        Sample a batch of experiences based on priority.
        
        Args:
            batch_size (int): Number of experiences to sample
            
        Returns:
            tuple: (batch, indices, importance_weights)
                - batch: Tuple of states, actions, rewards, next_states, dones
                - indices: Indices of sampled experiences
                - importance_weights: Importance sampling weights
                
        根據優先級採樣一批經驗。
        
        參數：
            batch_size (int)：要採樣的經驗數量
            
        返回：
            tuple：(batch, indices, importance_weights)
                - batch：狀態、動作、獎勵、下一狀態和完成標誌的元組
                - indices：採樣的經驗的索引
                - importance_weights：重要性採樣權重
        """
        # Lists to store batch data
        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices = []
        priorities = []
        
        # Calculate segment size
        segment_size = self.tree.total() / batch_size
        
        # Update beta value (annealing from beta_start to 1 over beta_frames)
        self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * 
                       (self.frame_count / self.beta_frames))
        self.frame_count += 1
        
        # Sample from each segment
        for i in range(batch_size):
            # Get a value from the current segment
            a, b = segment_size * i, segment_size * (i + 1)
            value = np.random.uniform(a, b)
            
            # Get index, priority, and experience from the tree
            idx, priority, experience = self.tree.sample(value)
            
            # Store the index and priority
            indices.append(idx)
            priorities.append(priority)
            
            # Unpack the experience
            state, action, reward, next_state, done = experience
            
            # Store in batch lists
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        
        # Convert to numpy arrays for batch processing
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        dones = np.array(dones, dtype=np.float32)
        
        # Convert to PyTorch tensors
        device = get_device()  # Get device directly from device_utils
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Calculate importance sampling weights
        # The importance sampling weight is w_i = (N * P(i))^(-β)
        # where N is the memory size and P(i) is the probability of sampling transition i
        sampling_probabilities = np.array(priorities) / self.tree.total()
        importance_weights = (self.tree.size * sampling_probabilities) ** -self.beta
        
        # Normalize weights so they only scale the update downwards
        importance_weights = importance_weights / importance_weights.max()
        
        # Convert weights to PyTorch tensor
        importance_weights = torch.FloatTensor(importance_weights).to(device)
        
        # Return batch, indices, and weights
        batch = (states, actions, rewards, next_states, dones)
        return batch, indices, importance_weights
    
    def update_priorities(self, indices, td_errors):
        """
        Update priorities of sampled transitions.
        
        Args:
            indices (list): Indices of sampled transitions
            td_errors (list): TD-errors of sampled transitions
            
        更新採樣轉換的優先級。
        
        參數：
            indices (list)：採樣轉換的索引
            td_errors (list)：採樣轉換的 TD 誤差
        """
        for idx, td_error in zip(indices, td_errors):
            # Calculate new priority
            priority = self._get_priority(td_error)
            
            # Update priority in the tree
            self.tree.update(idx, priority)
            
            # Update max priority for new experiences
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        """
        Return the current size of the memory.
        
        Returns:
            int: Current memory size
            
        返回記憶體的當前大小。
        
        返回：
            int：當前記憶體大小
        """
        return self.tree.size