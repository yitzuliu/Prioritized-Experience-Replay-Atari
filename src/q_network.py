"""
Deep Q-Network (DQN) architecture implementation.

This module defines the neural network model used by the DQN agent
to approximate the Q-function, mapping states to action values.

深度 Q 網絡 (DQN) 架構實現。

本模組定義了 DQN 智能體用來近似 Q 函數的神經網絡模型，
該模型將狀態映射到動作價值。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys

# Add parent directory to path to import config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from src.device_utils import get_device


class QNetwork(nn.Module):
    """
    Deep Q-Network for approximating the Q-function.
    
    This network takes an input state (stacked frames) and outputs
    a vector of action values.
    
    用於近似 Q 函數的深度 Q 網絡。
    
    此網絡接受輸入狀態（堆疊的幀）並輸出動作價值向量。
    """
    
    def __init__(self, input_shape, action_space_size, 
                 use_one_conv=config.USE_ONE_CONV_LAYER,
                 use_two_conv=config.USE_TWO_CONV_LAYERS,
                 use_three_conv=config.USE_THREE_CONV_LAYERS):
        """
        Initialize the Q-Network.
        
        Args:
            input_shape: Shape of the input state (channels, height, width)
            action_space_size: Number of possible actions
            use_one_conv: Whether to use a single convolutional layer
            use_two_conv: Whether to use two convolutional layers
            use_three_conv: Whether to use three convolutional layers
            
        初始化 Q 網絡。
        
        參數：
            input_shape: 輸入狀態的形狀（通道數, 高度, 寬度）
            action_space_size: 可能的動作數量
            use_one_conv: 是否使用單個卷積層
            use_two_conv: 是否使用兩個卷積層
            use_three_conv: 是否使用三個卷積層
        """
        super(QNetwork, self).__init__()
        
        # Convolutional layers will differ based on configuration
        self.use_one_conv = use_one_conv
        self.use_two_conv = use_two_conv
        self.use_three_conv = use_three_conv
        
        # Calculate input channels (number of stacked frames)
        in_channels = input_shape[0]
        
        # First convolutional layer (always used)
        self.conv1 = nn.Conv2d(in_channels, config.CONV1_CHANNELS, 
                               kernel_size=config.CONV1_KERNEL_SIZE, 
                               stride=config.CONV1_STRIDE)
        
        # Second and third convolutional layers (optional)
        if use_two_conv or use_three_conv:
            self.conv2 = nn.Conv2d(config.CONV1_CHANNELS, config.CONV2_CHANNELS, 
                                   kernel_size=config.CONV2_KERNEL_SIZE, 
                                   stride=config.CONV2_STRIDE)
        
        if use_three_conv:
            self.conv3 = nn.Conv2d(config.CONV2_CHANNELS, config.CONV3_CHANNELS, 
                                   kernel_size=config.CONV3_KERNEL_SIZE, 
                                   stride=config.CONV3_STRIDE)
        
        # Calculate the size of flattened features after convolution
        # This depends on the number of convolutional layers used
        if use_three_conv:
            # After 3 convolutional layers
            self.feature_size = self._get_conv_output_size(input_shape)
        elif use_two_conv:
            # After 2 convolutional layers
            self.feature_size = self._get_conv_output_size(input_shape, use_three_conv=False)
        else:
            # After 1 convolutional layer
            self.feature_size = self._get_conv_output_size(input_shape, use_two_conv=False, use_three_conv=False)
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, config.FC_SIZE)
        self.fc2 = nn.Linear(config.FC_SIZE, action_space_size)
        
        # Initialize weights
        self._initialize_weights()
        
        # Move model to the appropriate device
        self.device = get_device()
        self.to(self.device)
    
    def _get_conv_output_size(self, input_shape, use_two_conv=True, use_three_conv=True):
        """
        Calculate the size of the flattened features after convolution.
        
        Args:
            input_shape: Input shape (channels, height, width)
            use_two_conv: Whether to apply the second convolutional layer
            use_three_conv: Whether to apply the third convolutional layer
            
        Returns:
            int: Size of flattened features
            
        計算卷積後平坦特徵的大小。
        
        參數：
            input_shape: 輸入形狀（通道數, 高度, 寬度）
            use_two_conv: 是否應用第二個卷積層
            use_three_conv: 是否應用第三個卷積層
            
        返回：
            int: 平坦特徵的大小
        """
        # Create a dummy input tensor
        x = torch.zeros(1, *input_shape)
        
        # Apply convolutional layers
        x = F.relu(self.conv1(x))
        
        if use_two_conv:
            x = F.relu(self.conv2(x))
        
        if use_three_conv:
            x = F.relu(self.conv3(x))
        
        # Get flattened size
        return int(np.prod(x.size()[1:]))
    
    def _initialize_weights(self):
        """
        Initialize network weights using He initialization.
        
        使用 He 初始化來初始化網絡權重。
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # He initialization for ReLU activations
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input state tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Action values of shape (batch_size, action_space_size)
            
        通過網絡進行前向傳遞。
        
        參數：
            x: 形狀為（批次大小, 通道數, 高度, 寬度）的輸入狀態張量
            
        返回：
            torch.Tensor: 形狀為（批次大小, 動作空間大小）的動作價值
        """
        # Ensure input is a float tensor on the right device
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        
        x = x.to(self.device)
        
        # Apply convolutional layers with ReLU activations
        x = F.relu(self.conv1(x))
        
        if self.use_two_conv or self.use_three_conv:
            x = F.relu(self.conv2(x))
        
        if self.use_three_conv:
            x = F.relu(self.conv3(x))
        
        # Flatten the features
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x


# For testing purposes
if __name__ == "__main__":
    # Test the Q-Network with different configurations
    # Assuming input shape is (4, 84, 84) - 4 stacked frames of 84x84 pixels
    input_shape = (4, 84, 84)
    action_space_size = config.ACTION_SPACE_SIZE
    
    # Create a batch of sample inputs
    batch_size = 32
    sample_input = torch.randn(batch_size, *input_shape)
    
    # Test with different convolutional configurations
    print("Testing Q-Network with different configurations:")
    
    # 1-conv layer network
    net1 = QNetwork(input_shape, action_space_size, use_one_conv=True, use_two_conv=False, use_three_conv=False)
    output1 = net1(sample_input)
    print(f"1-Conv Layer Network - Output shape: {output1.shape}")
    print(f"Feature size after convolutions: {net1.feature_size}")
    
    # 2-conv layer network
    net2 = QNetwork(input_shape, action_space_size, use_one_conv=False, use_two_conv=True, use_three_conv=False)
    output2 = net2(sample_input)
    print(f"2-Conv Layer Network - Output shape: {output2.shape}")
    print(f"Feature size after convolutions: {net2.feature_size}")
    
    # 3-conv layer network (default)
    net3 = QNetwork(input_shape, action_space_size, use_one_conv=False, use_two_conv=False, use_three_conv=True)
    output3 = net3(sample_input)
    print(f"3-Conv Layer Network - Output shape: {output3.shape}")
    print(f"Feature size after convolutions: {net3.feature_size}")
    
    # Print the device being used
    print(f"Network is on device: {net3.device}")
    
    # Print model summary (number of parameters)
    total_params = sum(p.numel() for p in net3.parameters())
    trainable_params = sum(p.numel() for p in net3.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")