"""
Memory module initialization.

This module contains different types of experience replay memories used in DQN.

記憶體模塊初始化。

該模塊包含在DQN中使用的不同類型的經驗回放記憶體。
"""

from src.memory.sumtree import SumTree
from src.memory.per_memory import PrioritizedReplayMemory