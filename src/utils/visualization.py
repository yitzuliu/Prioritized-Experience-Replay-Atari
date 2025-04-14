"""
Visualization utilities for DQN with Prioritized Experience Replay.

This module provides functions for visualizing training metrics, priority
distributions, and other data to help understand the learning process.

DQN 優先經驗回放的可視化工具。

此模組提供用於可視化訓練指標、優先級分佈和其他數據的函數，以幫助理解學習過程。
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
from datetime import datetime
import config


def setup_plotting_style():
    """
    Set up consistent plotting style for all visualizations.
    
    設置所有可視化的一致繪圖風格。
    """
    # Use seaborn style for better aesthetics
    sns.set(style="darkgrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12


def plot_training_metrics(episode_rewards, episode_lengths, losses=None, smoothing=0.9, save_path=None):
    """
    Plot training metrics including rewards, episode lengths, and losses.
    
    Args:
        episode_rewards (list): List of episode rewards
        episode_lengths (list): List of episode lengths (steps per episode)
        losses (list, optional): List of loss values during training
        smoothing (float): Smoothing factor for the plots (0-1)
        save_path (str, optional): Path to save the plot image
    
    繪製包括獎勵、回合長度和損失在內的訓練指標。
    
    參數：
        episode_rewards (list)：回合獎勵列表
        episode_lengths (list)：回合長度列表（每回合步數）
        losses (list, optional)：訓練期間的損失值列表
        smoothing (float)：繪圖的平滑因子（0-1）
        save_path (str, optional)：保存繪圖圖像的路徑
    """
    setup_plotting_style()
    
    # Create figure with appropriate subplots
    n_plots = 3 if losses is not None else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots), sharex=True)
    
    # Calculate smoothed values
    def smooth(data, alpha=smoothing):
        """Apply exponential smoothing to data series"""
        smoothed = []
        if not data:
            return smoothed
        value = data[0]
        for point in data:
            value = alpha * value + (1 - alpha) * point
            smoothed.append(value)
        return smoothed
    
    # Plot episode rewards
    ax = axes[0]
    episodes = list(range(1, len(episode_rewards) + 1))
    ax.plot(episodes, episode_rewards, alpha=0.3, label='Raw')
    if len(episode_rewards) >= 2:  # Need at least 2 points for smoothing
        ax.plot(episodes, smooth(episode_rewards), label=f'Smoothed (α={smoothing})',
                linewidth=2)
    ax.set_ylabel('Reward')
    ax.set_title('Training Rewards per Episode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot episode lengths
    ax = axes[1]
    ax.plot(episodes, episode_lengths, alpha=0.3, label='Raw')
    if len(episode_lengths) >= 2:
        ax.plot(episodes, smooth(episode_lengths), label=f'Smoothed (α={smoothing})',
                linewidth=2)
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length (Steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot losses if provided
    if losses is not None:
        ax = axes[2]
        loss_steps = list(range(1, len(losses) + 1))
        ax.plot(loss_steps, losses, alpha=0.3, label='Raw')
        if len(losses) >= 2:
            ax.plot(loss_steps, smooth(losses), label=f'Smoothed (α={smoothing})',
                    linewidth=2)
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Final x-axis label
    axes[-1].set_xlabel('Episodes')
    
    # Adjust layout and display
    plt.tight_layout()
    
    # Save if path is specified
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_priority_distribution(priorities, save_path=None):
    """
    Plot the distribution of priorities in the replay buffer.
    
    Args:
        priorities (numpy.ndarray): Array of priorities from the replay buffer
        save_path (str, optional): Path to save the plot image
        
    繪製回放緩衝區中優先級的分佈。
    
    參數：
        priorities (numpy.ndarray)：來自回放緩衝區的優先級數組
        save_path (str, optional)：保存繪圖圖像的路徑
    """
    setup_plotting_style()
    
    plt.figure(figsize=(10, 6))
    
    # Plot priority distribution as a histogram
    sns.histplot(priorities, bins=50, kde=True)
    plt.title('Priority Distribution in Replay Buffer')
    plt.xlabel('Priority Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    if len(priorities) > 0:
        plt.axvline(np.mean(priorities), color='r', linestyle='--', label=f'Mean: {np.mean(priorities):.4f}')
        plt.axvline(np.median(priorities), color='g', linestyle='--', label=f'Median: {np.median(priorities):.4f}')
        plt.legend()
    
    plt.tight_layout()
    
    # Save if path is specified
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_comparative_rewards(uniform_rewards, per_rewards, smoothing=0.9, save_path=None):
    """
    Plot comparative results between uniform sampling and prioritized experience replay.
    
    Args:
        uniform_rewards (list): List of episode rewards using uniform sampling
        per_rewards (list): List of episode rewards using prioritized experience replay
        smoothing (float): Smoothing factor for the plots (0-1)
        save_path (str, optional): Path to save the plot image
        
    繪製均勻採樣和優先經驗回放之間的比較結果。
    
    參數：
        uniform_rewards (list)：使用均勻採樣的回合獎勵列表
        per_rewards (list)：使用優先經驗回放的回合獎勵列表
        smoothing (float)：繪圖的平滑因子（0-1）
        save_path (str, optional)：保存繪圖圖像的路徑
    """
    setup_plotting_style()
    
    # Apply smoothing
    def smooth(data, alpha=smoothing):
        """Apply exponential smoothing to data series"""
        smoothed = []
        if not data:
            return smoothed
        value = data[0]
        for point in data:
            value = alpha * value + (1 - alpha) * point
            smoothed.append(value)
        return smoothed
    
    plt.figure(figsize=(12, 6))
    
    # Plot both reward curves
    max_len = max(len(uniform_rewards), len(per_rewards))
    
    if uniform_rewards:
        uniform_episodes = list(range(1, len(uniform_rewards) + 1))
        plt.plot(uniform_episodes, smooth(uniform_rewards), 
                 label='Uniform Sampling', color='blue', linewidth=2)
    
    if per_rewards:
        per_episodes = list(range(1, len(per_rewards) + 1))
        plt.plot(per_episodes, smooth(per_rewards), 
                 label='Prioritized Experience Replay', color='red', linewidth=2)
    
    plt.title('Training Rewards: Uniform Sampling vs. Prioritized Experience Replay')
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Reward')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set consistent x-axis limits
    plt.xlim(1, max_len)
    
    plt.tight_layout()
    
    # Save if path is specified
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_training_plots(stats, run_name=None, output_dir=None):
    """
    Generate and save all training plots at once.
    
    Args:
        stats (dict): Dictionary containing training statistics
        run_name (str, optional): Name for this training run
        output_dir (str, optional): Directory to save plots
        
    一次性生成並保存所有訓練繪圖。
    
    參數：
        stats (dict)：包含訓練統計數據的字典
        run_name (str, optional)：此訓練運行的名稱
        output_dir (str, optional)：保存繪圖的目錄
    """
    # Set up output directory
    if output_dir is None:
        output_dir = config.RESULT_DIR  # Fixed: Use RESULT_DIR instead of RESULTS_DIR
    
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    # Create output directory if it doesn't exist
    results_dir = os.path.join(output_dir, run_name)
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot and save training metrics
    if all(key in stats for key in ['episode_rewards', 'episode_lengths']):
        losses = stats.get('losses', None)
        plot_training_metrics(
            stats['episode_rewards'], 
            stats['episode_lengths'], 
            losses=losses,
            save_path=os.path.join(results_dir, 'training_metrics.png')
        )
    
    # Plot and save priority distribution if available
    if 'priorities' in stats:
        plot_priority_distribution(
            stats['priorities'],
            save_path=os.path.join(results_dir, 'priority_distribution.png')
        )
    
    # Plot and save comparative results if available
    if all(key in stats for key in ['uniform_rewards', 'per_rewards']):
        plot_comparative_rewards(
            stats['uniform_rewards'],
            stats['per_rewards'],
            save_path=os.path.join(results_dir, 'comparative_rewards.png')
        )
    
    print(f"All plots saved to {results_dir}")
    return results_dir