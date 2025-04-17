"""
Deep Q-Network Model for Atari Ice Hockey.

This file implements the neural network architecture used in the DQN algorithm.
It can be configured to use 1, 2, or 3 convolutional layers followed by fully connected layers.

深度 Q 網絡模型，用於 Atari 冰球遊戲。

該文件實現了DQN算法中使用的神經網絡架構。
可以配置為使用1、2或3個卷積層，後接全連接層。

Pseudocode for DQN architecture:
1. Input: Stack of 4 frames (84x84x4)
2. Conv layers with ReLU activation:
   - Conv1: 32 filters, 8x8 kernel, stride 4
   - Conv2: 64 filters, 4x4 kernel, stride 2
   - Conv3: 64 filters, 3x3 kernel, stride 1
3. Flatten layer
4. Fully connected layer: 512 neurons with ReLU
5. Output layer: action_space_size neurons (Q-values for each action)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import (
    FRAME_HEIGHT, FRAME_WIDTH, FRAME_STACK, ACTION_SPACE_SIZE,
    CONV1_CHANNELS, CONV1_KERNEL_SIZE, CONV1_STRIDE,
    CONV2_CHANNELS, CONV2_KERNEL_SIZE, CONV2_STRIDE,
    CONV3_CHANNELS, CONV3_KERNEL_SIZE, CONV3_STRIDE,
    FC_SIZE, USE_ONE_CONV_LAYER, USE_TWO_CONV_LAYERS, USE_THREE_CONV_LAYERS
)


class DQN(nn.Module):
    """
    Deep Q-Network (DQN) architecture for learning to play Atari Ice Hockey.
    
    This network maps stacked game frames to Q-values for each possible action.
    The architecture can be configured to use 1, 2, or 3 convolutional layers.
    
    深度 Q 網絡 (DQN) 架構，用於學習玩 Atari 冰球遊戲。
    
    該網絡將堆疊的遊戲幀映射到每個可能動作的 Q 值。
    可以配置為使用 1、2 或 3 個卷積層。
    """
    
    def __init__(self):
        """
        Initialize the DQN model with the configured architecture.
        
        使用配置的架構初始化 DQN 模型。
        """
        super(DQN, self).__init__()
        
        # /* DQN ARCHITECTURE - Steps 1-2 */
        # Define convolutional layers based on configuration
        # Input: Stack of 4 frames (84x84x4)
        # Conv layers with ReLU activation
        if USE_ONE_CONV_LAYER:
            # Simple network with just one convolutional layer (for weaker hardware)
            # Conv1: 32 filters, 8x8 kernel, stride 4
            self.conv1 = nn.Conv2d(
                in_channels=FRAME_STACK, 
                out_channels=CONV1_CHANNELS, 
                kernel_size=CONV1_KERNEL_SIZE, 
                stride=CONV1_STRIDE
            )
            
            # Calculate the size of the flattened features after convolution
            conv_output_size = self._calculate_conv_output_size(
                [FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH],
                [(CONV1_CHANNELS, CONV1_KERNEL_SIZE, CONV1_STRIDE)]
            )
            
        elif USE_TWO_CONV_LAYERS:
            # Medium network with two convolutional layers
            # Conv1: 32 filters, 8x8 kernel, stride 4
            self.conv1 = nn.Conv2d(
                in_channels=FRAME_STACK, 
                out_channels=CONV1_CHANNELS, 
                kernel_size=CONV1_KERNEL_SIZE, 
                stride=CONV1_STRIDE
            )
            # Conv2: 64 filters, 4x4 kernel, stride 2
            self.conv2 = nn.Conv2d(
                in_channels=CONV1_CHANNELS, 
                out_channels=CONV2_CHANNELS, 
                kernel_size=CONV2_KERNEL_SIZE, 
                stride=CONV2_STRIDE
            )
            
            # Calculate the size of the flattened features after convolution
            conv_output_size = self._calculate_conv_output_size(
                [FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH],
                [(CONV1_CHANNELS, CONV1_KERNEL_SIZE, CONV1_STRIDE),
                 (CONV2_CHANNELS, CONV2_KERNEL_SIZE, CONV2_STRIDE)]
            )
            
        else:  # Default to three convolutional layers (USE_THREE_CONV_LAYERS)
            # Full network with three convolutional layers (DeepMind's original architecture)
            # Conv1: 32 filters, 8x8 kernel, stride 4
            self.conv1 = nn.Conv2d(
                in_channels=FRAME_STACK, 
                out_channels=CONV1_CHANNELS, 
                kernel_size=CONV1_KERNEL_SIZE, 
                stride=CONV1_STRIDE
            )
            # Conv2: 64 filters, 4x4 kernel, stride 2
            self.conv2 = nn.Conv2d(
                in_channels=CONV1_CHANNELS, 
                out_channels=CONV2_CHANNELS, 
                kernel_size=CONV2_KERNEL_SIZE, 
                stride=CONV2_STRIDE
            )
            # Conv3: 64 filters, 3x3 kernel, stride 1
            self.conv3 = nn.Conv2d(
                in_channels=CONV2_CHANNELS, 
                out_channels=CONV3_CHANNELS, 
                kernel_size=CONV3_KERNEL_SIZE, 
                stride=CONV3_STRIDE
            )
            
            # Calculate the size of the flattened features after convolution
            conv_output_size = self._calculate_conv_output_size(
                [FRAME_STACK, FRAME_HEIGHT, FRAME_WIDTH],
                [(CONV1_CHANNELS, CONV1_KERNEL_SIZE, CONV1_STRIDE),
                 (CONV2_CHANNELS, CONV2_KERNEL_SIZE, CONV2_STRIDE),
                 (CONV3_CHANNELS, CONV3_KERNEL_SIZE, CONV3_STRIDE)]
            )
        
        # /* DQN ARCHITECTURE - Steps 3-5 */
        # Step 3: Flatten layer (handled in forward method)
        # Step 4: Fully connected layer with 512 neurons and ReLU activation
        self.fc1 = nn.Linear(conv_output_size, FC_SIZE)
        # Step 5: Output layer with action_space_size neurons (Q-values for each action)
        self.fc2 = nn.Linear(FC_SIZE, ACTION_SPACE_SIZE)
    
    def _calculate_conv_output_size(self, input_dims, conv_params):
        """
        Calculate the size of the flattened features after convolution.
        
        Args:
            input_dims (list): Input dimensions [channels, height, width]
            conv_params (list): List of tuples (out_channels, kernel_size, stride)
            
        Returns:
            int: Size of flattened features
            
        計算卷積後展平特徵的大小。
        
        參數：
            input_dims (list)：輸入維度 [通道數, 高度, 寬度]
            conv_params (list)：元組列表 (輸出通道數, 內核大小, 步幅)
            
        返回：
            int：展平特徵的大小
        """
        c, h, w = input_dims
        
        # Apply each convolution
        for _, k, s in conv_params:
            # Output size formula: ((W-K+2P)/S)+1
            h = (h - k) // s + 1
            w = (w - k) // s + 1
        
        # Get the last output channels
        c = conv_params[-1][0]
        
        # Return flattened size
        return c * h * w
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, stack_size, height, width]
            
        Returns:
            Tensor: Q-values for each action
            
        通過網絡的前向傳播。
        
        參數：
            x (Tensor)：形狀為 [batch_size, stack_size, height, width] 的輸入張量
            
        返回：
            Tensor：每個動作的 Q 值
        """
        # Apply convolutional layers based on configuration
        if USE_ONE_CONV_LAYER:
            # One conv layer
            x = F.relu(self.conv1(x))
            
        elif USE_TWO_CONV_LAYERS:
            # Two conv layers
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            
        else:  # Default to three convolutional layers
            # Three conv layers (DeepMind's original architecture)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # No activation on the output layer (raw Q-values)
        
        return x