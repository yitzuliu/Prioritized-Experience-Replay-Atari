"""
Atari environment wrappers for preprocessing and optimizing the training process.

This module contains various wrapper classes that modify the environment
to make it more suitable for reinforcement learning algorithms.

Atari 環境包裝器，用於預處理和優化訓練過程。

此模組包含各種修改環境的包裝器類，使其更適合強化學習算法。
"""

import gymnasium as gym 
from gymnasium import spaces
import ale_py
import numpy as np
import cv2
import os
import sys

# Add parent directory to path to import config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    
    通過在重置時執行隨機數量的無操作來對初始狀態進行採樣。
    """
    def __init__(self, env, max_no_op_steps=config.NOOP_MAX):
        """
        Initialize the NoopResetEnv wrapper.
        
        Args:
            env: The environment to wrap
            max_no_op_steps: Maximum number of no-ops to execute on reset
            
        初始化 NoopResetEnv 包裝器。
        
        參數：
            env: 要包裝的環境
            max_no_op_steps: 重置時執行的最大無操作數量
        """
        super(NoopResetEnv, self).__init__(env)
        self.max_no_op_steps = max_no_op_steps
        self.no_op_action = 0  # The action that does nothing
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """
        Reset the environment with a random number of no-ops.
        
        Returns:
            tuple: (observation, info)
            
        使用隨機數量的無操作重置環境。
        
        返回：
            tuple: (觀察值, 資訊)
        """
        observation, info = self.env.reset(**kwargs)
        
        # Sample number of no-ops to perform
        num_no_ops = np.random.randint(1, self.max_no_op_steps + 1)
        
        # Perform the no-ops
        for _ in range(num_no_ops):
            observation, reward, terminated, truncated, info = self.env.step(self.no_op_action)
            if terminated or truncated:
                observation, info = self.env.reset(**kwargs)
                
        return observation, info

    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: The action to take
            
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
            
        在環境中執行一個步驟。
        
        參數：
            action: 要執行的動作
            
        返回：
            tuple: (觀察值, 獎勵, 終止, 截斷, 資訊)
        """
        return self.env.step(action)


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame and max over the last 2 frames.
    
    僅返回每 `skip` 幀並對最近 2 幀進行最大值操作。
    """
    def __init__(self, env, skip_frames=config.FRAME_SKIP):
        """
        Initialize the MaxAndSkipEnv wrapper.
        
        Args:
            env: The environment to wrap
            skip_frames: Number of frames to skip
            
        初始化 MaxAndSkipEnv 包裝器。
        
        參數：
            env: 要包裝的環境
            skip_frames: 要跳過的幀數
        """
        super(MaxAndSkipEnv, self).__init__(env)
        self.skip_frames = skip_frames
        self.frame_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        
    def step(self, action):
        """
        Repeat action, sum reward, and max over last observations.
        
        Args:
            action: The action to take
            
        Returns:
            tuple: (max_frame, total_reward, terminated, truncated, info)
            
        重複動作，累加獎勵，並對最近的觀察進行最大值操作。
        
        參數：
            action: 要執行的動作
            
        返回：
            tuple: (最大幀, 總獎勵, 終止, 截斷, 資訊)
        """
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        
        # Repeat action for skip frames and sum rewards
        for frame_idx in range(self.skip_frames):
            observation, reward, terminated, truncated, info = self.env.step(action)
            is_done = terminated or truncated
            
            if frame_idx == self.skip_frames - 2:
                self.frame_buffer[0] = observation
            if frame_idx == self.skip_frames - 1:
                self.frame_buffer[1] = observation
                
            total_reward += reward
            if is_done:
                break
                
        # Max over the last 2 frames to reduce flickering
        max_frame = self.frame_buffer.max(axis=0)
        
        return max_frame, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """
        Reset the environment.
        
        Returns:
            tuple: (observation, info)
            
        重置環境。
        
        返回：
            tuple: (觀察值, 資訊)
        """
        return self.env.reset(**kwargs)


class ResizeFrame(gym.ObservationWrapper):
    """
    Resize observation frames to specified dimensions and convert to CHW format.
    
    將觀察幀調整為指定尺寸，並轉換為CHW格式。
    """
    def __init__(self, env, frame_width=config.FRAME_WIDTH, frame_height=config.FRAME_HEIGHT):
        """
        Initialize the ResizeFrame wrapper.
        
        Args:
            env: The environment to wrap
            frame_width: Target width
            frame_height: Target height
            
        初始化 ResizeFrame 包裝器。
        
        參數：
            env: 要包裝的環境
            frame_width: 目標寬度
            frame_height: 目標高度
        """
        super(ResizeFrame, self).__init__(env)
        self.frame_width = frame_width
        self.frame_height = frame_height
        # 注意這裡改變了shape從(H,W,C)到(C,H,W)格式
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(1, self.frame_height, self.frame_width), 
            dtype=np.uint8
        )

    def observation(self, observation):
        """
        Process the observation through grayscale conversion and resizing.
        
        Args:
            observation: The raw observation from the environment
            
        Returns:
            ndarray: Processed observation in CHW format
            
        通過灰度轉換和調整大小處理觀察。
        
        參數：
            observation: 來自環境的原始觀察
            
        返回：
            ndarray: CHW格式的處理後觀察
        """
        # Convert to grayscale
        grayscale_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        
        # Resize to desired dimensions
        resized_frame = cv2.resize(
            grayscale_frame, (self.frame_width, self.frame_height), 
            interpolation=cv2.INTER_AREA
        )
        
        # Return the frame in CHW format (Channel, Height, Width)
        # 這樣就不需要在DQN中再做轉換了
        return resized_frame[None, :, :]  # Add channel dimension first


class NormalizedFrame(gym.ObservationWrapper):
    """
    Normalize pixel values to range [0, 1].
    
    將像素值標準化到 [0, 1] 範圍。
    """
    def __init__(self, env):
        """
        Initialize the NormalizedFrame wrapper.
        
        Args:
            env: The environment to wrap
            
        初始化 NormalizedFrame 包裝器。
        
        參數：
            env: 要包裝的環境
        """
        super(NormalizedFrame, self).__init__(env)
        self.observation_space = spaces.Box(
            low=0, high=1, 
            shape=env.observation_space.shape, 
            dtype=np.float32
        )

    def observation(self, observation):
        """
        Normalize the observation.
        
        Args:
            observation: The raw observation from the environment
            
        Returns:
            ndarray: Normalized observation
            
        標準化觀察。
        
        參數：
            observation: 來自環境的原始觀察
            
        返回：
            ndarray: 標準化的觀察
        """
        # Normalize pixel values from [0, 255] to [0, 1]
        normalized_frame = np.array(observation).astype(np.float32) / 255.0
        return normalized_frame
    

class FrameStack(gym.Wrapper):
    """
    Stack n_frames frames together along the channel dimension.
    
    將 n_frames 幀沿通道維度堆疊在一起。
    """
    def __init__(self, env, num_frames=config.FRAME_STACK):
        """
        Initialize the FrameStack wrapper.
        
        Args:
            env: The environment to wrap
            num_frames: Number of frames to stack
            
        初始化 FrameStack 包裝器。
        
        參數：
            env: 要包裝的環境
            num_frames: 要堆疊的幀數
        """
        super(FrameStack, self).__init__(env)
        self.num_frames = num_frames
        self.frames = None
        
        # Update observation space for CHW format
        # 原始觀察值的形狀應該是(C, H, W)
        obs_shape = self.observation_space.shape
        # 堆疊後的形狀將是(C*num_frames, H, W)
        self.stack_axis = 0  # 沿著通道維度堆疊
        stacked_obs_shape = (obs_shape[0] * self.num_frames, obs_shape[1], obs_shape[2])
        
        self.observation_space = spaces.Box(
            low=np.min(self.observation_space.low),
            high=np.max(self.observation_space.high),
            shape=stacked_obs_shape,
            dtype=self.observation_space.dtype
        )
        
    def reset(self, **kwargs):
        """
        Reset the environment and initialize frame stack.
        
        Returns:
            tuple: (stacked_frames, info)
            
        重置環境並初始化幀堆疊。
        
        返回：
            tuple: (堆疊幀, 資訊)
        """
        observation, info = self.env.reset(**kwargs)
        
        # 初始化幀堆疊，複製相同的幀num_frames次
        self.frames = np.concatenate([observation] * self.num_frames, axis=self.stack_axis)
        return self.frames, info
    
    def step(self, action):
        """
        Take a step and update the frame stack.
        
        Args:
            action: The action to take
            
        Returns:
            tuple: (stacked_frames, reward, terminated, truncated, info)
            
        執行一個步驟並更新幀堆疊。
        
        參數：
            action: 要執行的動作
            
        返回：
            tuple: (堆疊幀, 獎勵, 終止, 截斷, 資訊)
        """
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # 更新幀堆疊：移除最舊的幀並添加最新的幀
        # 對於CHW格式，我們在第一個維度（通道）上操作
        old_frames = self.frames[observation.shape[0]:] if observation.shape[0] > 0 else self.frames[1:]
        self.frames = np.concatenate([old_frames, observation], axis=self.stack_axis)
        
        return self.frames, reward, terminated, truncated, info


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip rewards to {+1, 0, -1} by their sign.
    
    通過符號將獎勵裁剪為 {+1, 0, -1}。
    """
    def __init__(self, env):
        """
        Initialize the ClipRewardEnv wrapper.
        
        Args:
            env: The environment to wrap
            
        初始化 ClipRewardEnv 包裝器。
        
        參數：
            env: 要包裝的環境
        """
        super(ClipRewardEnv, self).__init__(env)
        
    def reward(self, reward):
        """
        Clip the reward.
        
        Args:
            reward: The original reward
            
        Returns:
            float: The clipped reward
            
        裁剪獎勵。
        
        參數：
            reward: 原始獎勵
            
        返回：
            float: 裁剪後的獎勵
        """
        return np.sign(reward)


def make_atari_env(env_name=config.ENV_NAME, render_mode=config.RENDER_MODE, training=config.TRAINING_MODE, difficulty=config.DIFFICULTY):
    """
    Create and configure Atari environment with all necessary wrappers.
    
    Args:
        env_name: The name of the Atari environment
        render_mode: The render mode for the environment
        training: Whether the environment is for training or evaluation
        difficulty: The difficulty level of the game (0-4, where 0 is easiest)
        
    Returns:
        env: The wrapped Atari environment
        
    創建並配置具有所有必要包裝器的 Atari 環境。
    
    參數：
        env_name: Atari 環境的名稱
        render_mode: 環境的渲染模式
        training: 環境是用於訓練還是評估
        difficulty: 遊戲的難度級別（0-4，其中 0 最簡單）
        
    返回：
        env: 包裝後的 Atari 環境
    """
    # Set render mode to None during training
    if training:
        render_mode = None
    
    # Create the Atari environment with specified difficulty
    env = gym.make(env_name, render_mode=render_mode, difficulty=difficulty)
    
    # Apply wrappers in the correct order
    env = NoopResetEnv(env)
    env = MaxAndSkipEnv(env)
    env = ResizeFrame(env)
    
    # Only normalize and clip reward during training
    if training:
        env = NormalizedFrame(env)
        env = ClipRewardEnv(env)
    
    env = FrameStack(env)
    
    return env


if __name__ == "__main__":
    """Test the environment wrappers by rendering a few frames."""
    # Create the Atari environment with rendering enabled for testing
    env = make_atari_env(render_mode="human", training=False, difficulty=0)
    observation, info = env.reset()
    
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    # Take some random actions
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()