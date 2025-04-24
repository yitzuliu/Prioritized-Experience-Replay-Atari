"""
SumTree implementation for Prioritized Experience Replay.

This module implements a binary sum tree data structure that efficiently
stores priorities and allows sampling transitions in proportion to their priorities.

用於優先經驗回放的總和樹實現。

此模組實現了一個二叉總和樹數據結構，能夠高效地存儲優先級，
並允許按照優先級比例採樣轉換。
"""

import numpy as np
import os
import sys

# Add parent directory to path to import config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class SumTree:
    """
    A binary sum tree data structure for efficient priority-based sampling.
    
    The leaf nodes contain priorities of transitions, and each internal node
    contains the sum of priorities of its children.
    
    一種二叉總和樹數據結構，用於高效的基於優先級的採樣。
    
    葉節點包含轉換的優先級，每個內部節點包含其子節點優先級的總和。
    """
    
    def __init__(self, memory_capacity=config.TREE_CAPACITY):
        """
        Initialize the SumTree with the specified capacity.
        
        Args:
            memory_capacity (int): Number of leaf nodes (capacity for storing experiences)
            
        使用指定的容量初始化總和樹。
        
        參數：
            memory_capacity (int): 葉節點數量（存儲經驗的容量）
        """
        # Number of leaf nodes that will contain experiences
        self.memory_capacity = memory_capacity
        
        # Total size of the tree (nodes + leaves)
        # A binary tree with n leaves has 2n-1 nodes in total
        self.tree_size = 2 * memory_capacity - 1
        
        # Initialize the tree with zeros
        # The first capacity-1 elements are nodes, the rest are leaves
        self.priority_tree = np.zeros(self.tree_size)
        
        # Data storage for experiences
        self.experience_data = np.zeros(memory_capacity, dtype=object)
        
        # Index to write next experience
        self.next_write_index = 0
        
        # Number of stored experiences (used for importance sampling)
        self.experience_count = 0
    
    def _propagate_priority_change(self, tree_index, priority_change):
        """
        Propagate the priority change up the tree.
        
        Args:
            tree_index (int): Index of the node that was updated
            priority_change (float): The change in priority value
            
        將優先級變化向上傳播到樹中。
        
        參數：
            tree_index (int): 已更新節點的索引
            priority_change (float): 優先級值的變化
        """
        # Get parent index
        parent_index = (tree_index - 1) // 2
        
        # Update parent value
        self.priority_tree[parent_index] += priority_change
        
        # If not root, propagate change to parent
        if parent_index != 0:
            self._propagate_priority_change(parent_index, priority_change)
    
    def _find_priority_leaf_index(self, tree_index, cumulative_value):
        """
        Find the leaf index that contains the priority corresponding to a value.
        
        Args:
            tree_index (int): Current node index
            cumulative_value (float): Value to find in the tree
            
        Returns:
            int: Index of the leaf node
            
        查找包含與值對應的優先級的葉索引。
        
        參數：
            tree_index (int): 當前節點索引
            cumulative_value (float): 要在樹中查找的值
            
        返回：
            int: 葉節點的索引
        """
        # Get indices of left and right children
        left_child_index = 2 * tree_index + 1
        right_child_index = left_child_index + 1
        
        # If leaf node, return index
        if left_child_index >= self.tree_size:
            return tree_index
        
        # Check which child to traverse
        if cumulative_value <= self.priority_tree[left_child_index]:
            return self._find_priority_leaf_index(left_child_index, cumulative_value)
        else:
            return self._find_priority_leaf_index(right_child_index, 
                                               cumulative_value - self.priority_tree[left_child_index])
    
    def total_priority(self):
        """
        Get the total priority stored in the tree.
        
        Returns:
            float: Total priority
            
        獲取存儲在樹中的總優先級。
        
        返回：
            float: 總優先級
        """
        return self.priority_tree[0]
    
    def add(self, priority, experience_data):
        """
        Add a new experience with its priority.
        
        Args:
            priority (float): Priority of the experience
            experience_data (object): Experience data to store
            
        添加一個新的經驗及其優先級。
        
        參數：
            priority (float): 經驗的優先級
            experience_data (object): 要存儲的經驗數據
        """
        # Get the leaf index
        leaf_index = self.next_write_index + self.memory_capacity - 1
        
        # Store the data
        self.experience_data[self.next_write_index] = experience_data
        
        # Update the tree with the new priority
        self.update_priority(leaf_index, priority)
        
        # Move to next write position
        self.next_write_index = (self.next_write_index + 1) % self.memory_capacity
        
        # Update experience_count
        if self.experience_count < self.memory_capacity:
            self.experience_count += 1
    
    def update_priority(self, tree_index, new_priority):
        """
        Update the priority of a leaf node.
        
        Args:
            tree_index (int): Index of the leaf in the tree
            new_priority (float): New priority value
            
        更新葉節點的優先級。
        
        參數：
            tree_index (int): 樹中葉的索引
            new_priority (float): 新的優先級值
        """
        # Calculate the change in priority
        priority_change = new_priority - self.priority_tree[tree_index]
        
        # Update the leaf
        self.priority_tree[tree_index] = new_priority
        
        # Propagate the change through the tree
        self._propagate_priority_change(tree_index, priority_change)
    
    def get_experience_by_priority(self, priority_value):
        """
        Get an experience based on a priority value.
        
        Args:
            priority_value (float): Value to sample (between 0 and total priority)
            
        Returns:
            tuple: (tree_index, priority, experience)
                tree_index: Index of the leaf in the tree
                priority: Priority of the experience
                experience: The experience data
                
        基於優先級值獲取經驗。
        
        參數：
            priority_value (float): 要採樣的值（介於 0 和總優先級之間）
            
        返回：
            tuple: (tree_index, priority, experience)
                tree_index: 樹中葉的索引
                priority: 經驗的優先級
                experience: 經驗數據
        """
        # Get the leaf index
        leaf_index = self._find_priority_leaf_index(0, priority_value)
        
        # Get the data index
        data_index = leaf_index - self.memory_capacity + 1
        
        return (leaf_index, self.priority_tree[leaf_index], self.experience_data[data_index])
    
    def get_all_priorities(self):
        """
        Get all priorities from leaf nodes.
        
        Returns:
            ndarray: Array of all priorities
            
        從葉節點獲取所有優先級。
        
        返回：
            ndarray: 所有優先級的數組
        """
        return self.priority_tree[self.memory_capacity - 1:self.memory_capacity - 1 + self.experience_count]


# For testing purposes
if __name__ == "__main__":
    # Create a small SumTree for demonstration
    sumtree = SumTree(memory_capacity=8)
    
    # Add some experiences with priorities
    for i in range(8):
        priority = (i + 1) * 10  # Priorities: 10, 20, 30, ..., 80
        experience = f"Experience {i}"
        sumtree.add(priority, experience)
    
    print("Total priority:", sumtree.total_priority())
    print("All priorities:", sumtree.get_all_priorities())
    
    # Sample experiences based on priority
    print("\nSampling based on priority:")
    for _ in range(5):
        # Sample a random value between 0 and total priority
        random_priority = np.random.uniform(0, sumtree.total_priority())
        leaf_index, priority, experience = sumtree.get_experience_by_priority(random_priority)
        print(f"Sampled value: {random_priority:.2f}, Got priority: {priority}, Data: {experience}")
    
    # Update some priorities
    print("\nUpdating priorities:")
    sumtree.update_priority(4 + sumtree.memory_capacity - 1, 100)  # Update Experience 4's priority to 100
    print("New total priority:", sumtree.total_priority())
    print("New all priorities:", sumtree.get_all_priorities())
    
    # Sample again after updates
    print("\nSampling after updates:")
    for _ in range(5):
        random_priority = np.random.uniform(0, sumtree.total_priority())
        leaf_index, priority, experience = sumtree.get_experience_by_priority(random_priority)
        print(f"Sampled value: {random_priority:.2f}, Got priority: {priority}, Data: {experience}")