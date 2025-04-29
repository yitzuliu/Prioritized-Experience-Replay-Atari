"""
Game environment wrappers for preprocessing and optimizing the training process.

This module contains various wrapper classes that modify the environment
to make it more suitable for reinforcement learning algorithms.

遊戲環境包裝器，用於預處理和優化訓練過程。

此模組包含各種修改環境的包裝器類，使其更適合強化學習算法。
"""

import gymnasium as gym 
from gymnasium import spaces
from gymnasium.wrappers import GrayscaleObservation, ResizeObservation,FrameStackObservation, AtariPreprocessing

import ale_py
import numpy as np
import cv2
import os
import sys

# Add parent directory to path to import config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def make_atari_env(env_name=config.ENV_NAME, render_mode=config.RENDER_MODE, training=config.TRAINING_MODE, difficulty=config.DIFFICULTY):
    """
    Create and configure game environment with all necessary wrappers.
    
    Args:
        env_name: The name of the game environment
        render_mode: The render mode for the environment
        training: Whether the environment is for training or evaluation
        difficulty: The difficulty level of the game (0-4, where 0 is easiest)
        
    Returns:
        env: The wrapped game environment
        
    創建並配置具有所有必要包裝器的遊戲環境。
    
    參數：
        env_name: 遊戲環境的名稱
        render_mode: 環境的渲染模式
        training: 環境是用於訓練還是評估
        difficulty: 遊戲的難度級別（0-4，其中 0 最簡單）
        
    返回：
        env: 包裝後的遊戲環境
    """
    # Set render mode to None during training
    if training:
        render_mode = None

    env = gym.make(env_name, 
                   render_mode=render_mode,
                   difficulty=difficulty,
                   obs_type="rgb",
                   frameskip=1, 
                   full_action_space=False)
    
    # Apply AtariPreprocessing
    env = AtariPreprocessing(env, 
                            noop_max=config.NOOP_MAX,
                            frame_skip=config.FRAME_SKIP, 
                            screen_size=config.FRAME_WIDTH,
                            terminal_on_life_loss=True,
                            grayscale_obs=True,
                            grayscale_newaxis=True,
                            scale_obs=False)
    
    # FrameStack wrapper
    env = FrameStackObservation(env, stack_size=config.FRAME_STACK)
    
    print(f"Environment: {env_name}")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")

    return env


if __name__ == "__main__":
    """Test the environment wrappers to make sure the game runs normally."""

    env = make_atari_env(render_mode="human", training=False, difficulty=0)
    observation, info = env.reset()
    
    print("Observation space:", env.observation_space.shape)
    print("Action space:", env.action_space)
    print("Action meanings:", env.unwrapped.get_action_meanings())
    
    total_reward = 0
    step = 0
    
    # Simple testing loop
    while True:
        try:
            # Take meaningful actions (excluding NOOP most of the time for testing purposes)
            action = np.random.randint(1, env.action_space.n) if np.random.random() < 0.9 else 0
            
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Print status occasionally
            if step % 20 == 0:
                print(f"Step {step}, Action: {action}, Reward: {reward}, Total: {total_reward}")
            
            if terminated or truncated:
                print(f"Episode finished after {step} steps with total reward: {total_reward}")
                observation, info = env.reset()
                total_reward = 0
                step = 0
                
        except KeyboardInterrupt:
            print("Test interrupted by user")
            break
    
    env.close()

