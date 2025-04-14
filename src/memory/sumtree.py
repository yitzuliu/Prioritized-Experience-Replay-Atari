"""
SumTree Data Structure for Prioritized Experience Replay.

This file implements a complete binary tree with sum aggregation, which is used
to efficiently sample experiences according to their priorities.

完全二叉樹數據結構，用於高效地根據優先級對經驗進行採樣。

Pseudocode:
1. SumTree structure:
   - A complete binary tree where leaves contain priorities and data
   - Each internal node stores the sum of its children's values
   - Allows O(log n) updates and priority-proportional sampling

2. Main operations:
   - update(idx, priority): Update priority at leaf node with index idx
   - total(): Get total priority sum
   - sample(value): Sample a leaf based on a value in range [0, total]
"""

import numpy as np


class SumTree:
    """
    SumTree: A binary tree data structure where the parent's value is the sum of its children.
    
    This data structure is used for efficient sampling based on priorities in 
    Prioritized Experience Replay (PER).
    
    SumTree：一種二叉樹數據結構，其中父節點的值是其子節點值的總和。
    
    此數據結構用於在優先經驗回放 (PER) 中基於優先級進行高效採樣。
    """
    
    def __init__(self, capacity):
        """
        Initialize a SumTree with a given capacity.
        
        Args:
            capacity (int): Number of leaf nodes (experiences) that can be stored
            
        初始化具有給定容量的 SumTree。
        
        參數：
            capacity (int)：可以存儲的葉節點（經驗）數量
        """
        # Number of leaf nodes (experiences)
        self.capacity = capacity
        
        # Total nodes in the tree (internal nodes + leaf nodes)
        # For a complete binary tree with n leaf nodes, the total number of nodes is 2n-1
        # 樹中的節點總數（內部節點+葉節點）
        # 對於具有n個葉節點的完全二叉樹，節點總數為2n-1
        self.tree = np.zeros(2 * capacity - 1)
        
        # Array to store experience data
        # 存儲經驗數據的數組
        self.data = np.zeros(capacity, dtype=object)
        
        # Next position in the tree to be updated (for circular buffer behavior)
        # 樹中下一個要更新的位置（用於循環緩衝行為）
        self.next_idx = 0
        
        # Current size (number of filled positions)
        # 當前大小（已填充的位置數）
        self.size = 0
    
    def _propagate_update(self, idx, change):
        """
        Propagate the change up through the tree to update parent sums.
        
        Args:
            idx (int): The index of the node that was updated
            change (float): The amount by which the node value changed
            
        將變化向上傳播通過樹以更新父節點的總和。
        
        參數：
            idx (int)：更新的節點索引
            change (float)：節點值變化的量
        """
        # Get the parent index
        parent = (idx - 1) // 2
        
        # Update the parent value with the change
        self.tree[parent] += change
        
        # Continue propagating upward if not at the root
        if parent != 0:
            self._propagate_update(parent, change)
    
    def update(self, idx, priority):
        """
        Update the priority of an experience at the specified tree index.
        
        Args:
            idx (int): The index of the experience in the tree
            priority (float): The new priority value
            
        更新樹中指定索引處經驗的優先級。
        
        參數：
            idx (int)：樹中經驗的索引
            priority (float)：新的優先級值
        """
        # Calculate the index in the tree array
        tree_idx = idx + self.capacity - 1
        
        # Calculate the change in priority
        change = priority - self.tree[tree_idx]
        
        # Update the leaf node
        self.tree[tree_idx] = priority
        
        # Propagate the change up through the tree
        self._propagate_update(tree_idx, change)
    
    def add(self, priority, data):
        """
        Add a new experience with its priority to the tree.
        
        Args:
            priority (float): The priority of the experience
            data (object): The experience data to store
            
        將具有優先級的新經驗添加到樹中。
        
        參數：
            priority (float)：經驗的優先級
            data (object)：要存儲的經驗數據
        """
        # Get the index in the data array
        idx = self.next_idx
        
        # Store the data
        self.data[idx] = data
        
        # Update the priority in the tree
        self.update(idx, priority)
        
        # Move to the next position
        self.next_idx = (self.next_idx + 1) % self.capacity
        
        # Update the current size
        if self.size < self.capacity:
            self.size += 1
    
    def _get_leaf(self, value):
        """
        Find the leaf node based on a value (priority).
        
        Args:
            value (float): A value in range [0, total_priority]
            
        Returns:
            int: The tree index of the chosen leaf
            float: The priority value of the chosen leaf
            object: The data stored at the chosen leaf
            
        基於優先級值尋找葉節點。
        
        參數：
            value (float)：範圍在[0，總優先級]之間的值
            
        返回：
            int：所選葉節點的樹索引
            float：所選葉節點的優先級值
            object：存儲在所選葉節點的數據
        """
        # Start from the root
        parent_idx = 0
        
        while True:
            # Get indices of left and right children
            left_idx = 2 * parent_idx + 1
            right_idx = left_idx + 1
            
            # If we reach a leaf node, break
            if left_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
                
            # Move to left or right child based on value
            if value <= self.tree[left_idx]:
                parent_idx = left_idx
            else:
                value -= self.tree[left_idx]  # Subtract left child's value
                parent_idx = right_idx
        
        # Calculate the data index
        data_idx = leaf_idx - self.capacity + 1
        
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]
    
    def sample(self, value):
        """
        Sample a leaf node based on a value in range [0, total].
        
        Args:
            value (float): A value in range [0, total_priority]
            
        Returns:
            int: The data index of the chosen leaf
            float: The priority value of the chosen leaf
            object: The data stored at the chosen leaf
            
        根據範圍在[0，總優先級]之間的值採樣葉節點。
        
        參數：
            value (float)：範圍在[0，總優先級]之間的值
            
        返回：
            int：所選葉節點的數據索引
            float：所選葉節點的優先級值
            object：存儲在所選葉節點的數據
        """
        # Get the leaf node
        leaf_idx, priority, data = self._get_leaf(value)
        
        # Calculate data index
        data_idx = leaf_idx - self.capacity + 1
        
        return data_idx, priority, data
    
    def total(self):
        """
        Return the total priority of the tree (sum of all priorities).
        
        Returns:
            float: The total priority
            
        返回樹的總優先級（所有優先級的總和）。
        
        返回：
            float：總優先級
        """
        # The root node contains the sum of all priorities
        return self.tree[0]
        
    def get_leaf_values(self):
        """
        Get all leaf node values (priorities) for visualization.
        
        Returns:
            ndarray: Priority values of all leaf nodes
            
        獲取所有葉節點值（優先級）以進行可視化。
        
        返回：
            ndarray：所有葉節點的優先級值
        """
        # Return all leaf node values
        return self.tree[self.capacity-1:2*self.capacity-1]