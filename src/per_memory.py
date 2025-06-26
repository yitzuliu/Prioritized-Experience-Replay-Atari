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
import gc
import psutil
import warnings
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
    Enhanced Prioritized Experience Replay Memory with optimization features.
    
    Stores transitions with priorities and samples them based on their
    importance using a SumTree data structure. Includes memory monitoring,
    batch optimization, and efficient priority management.
    
    增強型優先經驗回放記憶體，具有優化功能。
    
    使用 SumTree 數據結構根據重要性存儲和採樣轉換。包括記憶體監控、
    批處理優化和高效的優先級管理。
    """
    
    def __init__(self, memory_capacity=config.MEMORY_CAPACITY, 
                 alpha=config.ALPHA, 
                 beta_start=config.BETA_START, 
                 beta_frames=config.BETA_FRAMES,
                 epsilon=config.EPSILON_PER):
        """
        Initialize the enhanced PER memory.
        
        Args:
            memory_capacity: Maximum capacity of the memory
            alpha: Priority exponent (controls how much prioritization is used)
            beta_start: Initial importance-sampling weight coefficient
            beta_frames: Number of frames over which beta will anneal from beta_start to 1.0
            epsilon: Small constant added to priorities to ensure non-zero probability
            
        初始化增強型 PER 記憶體。
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
        
        # Enhanced features
        self._priority_cache = {}  # Cache frequently used priorities
        self._batch_cache_size = 32  # Size of priority cache
        self._memory_warning_threshold = 0.9  # Memory warning at 90%
        self._last_gc_frame = 0  # Track garbage collection timing
        self._gc_frequency = 10000  # Garbage collect every N frames
        self._performance_stats = {
            'samples_drawn': 0,
            'priorities_updated': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def get_memory_usage(self):
        """
        Get current memory usage statistics.
        
        獲取當前記憶體使用統計。
        
        Returns:
            dict: Memory usage statistics
        """
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        return {
            'rss_bytes': memory_info.rss,
            'vms_bytes': memory_info.vms,
            'percent': memory_percent,
            'tree_size': self.sumtree.experience_count,
            'cache_size': len(self._priority_cache)
        }
    
    def _check_memory_and_cleanup(self):
        """
        Check memory usage and perform cleanup if necessary.
        
        檢查記憶體使用並在必要時執行清理。
        """
        memory_stats = self.get_memory_usage()
        
        if memory_stats['percent'] > self._memory_warning_threshold * 100:
            warnings.warn(f"High memory usage: {memory_stats['percent']:.1f}%")
            
            # Clear priority cache
            self._priority_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            # Update last GC frame
            self._last_gc_frame = self.frame_count
    
    def _periodic_cleanup(self):
        """
        Perform periodic cleanup operations.
        
        執行定期清理操作。
        """
        if self.frame_count - self._last_gc_frame >= self._gc_frequency:
            # Clear cache periodically
            if len(self._priority_cache) > self._batch_cache_size * 2:
                # Keep only the most recent entries
                items = list(self._priority_cache.items())
                self._priority_cache = dict(items[-self._batch_cache_size:])
            
            # Run garbage collection
            gc.collect()
            self._last_gc_frame = self.frame_count
    
    def update_beta(self, frame_idx):
        """
        Update the beta parameter based on current training progress.

        Args:
            frame_idx: Current frame index

        根據當前訓練進度更新 beta 參數。
        """
        self.frame_count = frame_idx
        
        # Calculate the progress of training
        progress = min(1.0, frame_idx / self.beta_frames)
        # Adjust beta using a non-linear function
        adjusted_progress = progress ** config.BETA_EXPONENT  # Non-linear scaling
        self.beta = min(1.0, self.beta_start + adjusted_progress * (1.0 - self.beta_start))
        
        # Perform periodic cleanup
        self._periodic_cleanup()
    
    def _calculate_priority(self, td_error):
        """
        Calculate priority from TD-error with caching.
        
        根據 TD 誤差計算優先級（帶緩存）。
        
        Args:
            td_error: TD-error
            
        Returns:
            float: Priority value
        """
        # Round TD error for caching (reduces cache size)
        rounded_error = round(float(td_error), 6)
        
        # Check cache first
        if rounded_error in self._priority_cache:
            self._performance_stats['cache_hits'] += 1
            return self._priority_cache[rounded_error]
        
        # Calculate priority using the formula from the PER paper
        # priority = (|δ| + ε)^α
        priority = (np.abs(rounded_error) + self.epsilon) ** self.alpha
        
        # Cache the result if cache isn't too large
        if len(self._priority_cache) < self._batch_cache_size * 2:
            self._priority_cache[rounded_error] = priority
        
        self._performance_stats['cache_misses'] += 1
        return priority
    
    def add(self, state, action, reward, next_state, done, td_error=None):
        """
        Add a new transition to the memory with enhanced error handling.
        
        將新的轉換添加到記憶體中（帶增強錯誤處理）。
        """
        try:
            # Validate inputs
            if state is None or next_state is None:
                raise ValueError("State and next_state cannot be None")
            
            if not isinstance(action, (int, np.integer)):
                raise ValueError("Action must be an integer")
            
            if not isinstance(reward, (int, float, np.number)):
                raise ValueError("Reward must be a number")
            
            if not isinstance(done, (bool, np.bool_)):
                raise ValueError("Done must be a boolean")
            
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
            
            # Check memory usage periodically
            if self.sumtree.experience_count % 1000 == 0:
                self._check_memory_and_cleanup()
                
        except Exception as e:
            warnings.warn(f"Error adding transition to memory: {str(e)}")
            raise
    
    def sample(self, batch_size):
        """
        Enhanced sample method with better error handling and optimization.
        
        增強的採樣方法，具有更好的錯誤處理和優化。
        """
        if self.sumtree.experience_count < batch_size:
            raise ValueError(f"Not enough samples in memory: {self.sumtree.experience_count} < {batch_size}")
        
        try:
            # Lists to store the sampled data
            batch_indices = np.zeros(batch_size, dtype=np.int32)
            batch_weights = np.zeros(batch_size, dtype=np.float32)
            batch_transitions = []
            
            # Calculate segment size
            total_priority = self.sumtree.total_priority()
            if total_priority <= 0:
                raise ValueError("Total priority must be positive")
            
            segment_size = total_priority / batch_size
            
            # Current beta for importance sampling weights
            current_beta = self.beta
            
            # Calculate the maximum weight for normalization
            all_priorities = self.sumtree.get_all_priorities()
            if len(all_priorities) == 0:
                raise ValueError("No priorities available")
            
            min_prob = np.min(all_priorities[all_priorities > 0]) / total_priority
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
                
                if transition is None:
                    raise ValueError(f"Retrieved None transition at index {idx}")
                
                # Store the index
                batch_indices[i] = idx
                
                # Calculate sampling weight
                sample_prob = priority / total_priority
                weight = (sample_prob * self.sumtree.experience_count) ** (-current_beta)
                
                # Normalize weight to be <= 1
                batch_weights[i] = weight / max_weight
                
                # Store the transition
                batch_transitions.append(transition)
            
            self._performance_stats['samples_drawn'] += batch_size
            return batch_indices, batch_weights, batch_transitions
            
        except Exception as e:
            warnings.warn(f"Error sampling from memory: {str(e)}")
            raise
    
    def update_priorities(self, indices, td_errors):
        """
        Enhanced priority update with batch processing and error handling.
        
        增強的優先級更新，具有批處理和錯誤處理。
        """
        try:
            if len(indices) != len(td_errors):
                raise ValueError("Indices and td_errors must have the same length")
            
            # Batch process priorities for efficiency
            new_priorities = []
            for error in td_errors:
                if not isinstance(error, (int, float, np.number)):
                    warnings.warn(f"Invalid TD error type: {type(error)}, converting to float")
                    error = float(error)
                
                priority = self._calculate_priority(error)
                new_priorities.append(priority)
                
                # Update max priority
                self.max_priority = max(self.max_priority, priority)
            
            # Update priorities in the tree
            for idx, priority in zip(indices, new_priorities):
                self.sumtree.update_priority(idx, priority)
            
            self._performance_stats['priorities_updated'] += len(indices)
            
        except Exception as e:
            warnings.warn(f"Error updating priorities: {str(e)}")
            raise
    
    def get_performance_stats(self):
        """
        Get performance statistics for monitoring and debugging.
        
        獲取性能統計信息用於監控和調試。
        
        Returns:
            dict: Performance statistics
        """
        cache_hit_rate = 0.0
        if self._performance_stats['cache_hits'] + self._performance_stats['cache_misses'] > 0:
            cache_hit_rate = self._performance_stats['cache_hits'] / (
                self._performance_stats['cache_hits'] + self._performance_stats['cache_misses']
            )
        
        return {
            'samples_drawn': self._performance_stats['samples_drawn'],
            'priorities_updated': self._performance_stats['priorities_updated'],
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self._priority_cache),
            'memory_usage': self.get_memory_usage(),
            'max_priority': self.max_priority,
            'beta': self.beta
        }
    
    def __len__(self):
        """
        Get the current size of the memory.
        
        Returns:
            int: Number of stored transitions
            
        獲取記憶體的當前大小。
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