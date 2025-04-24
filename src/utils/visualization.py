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
import time
import traceback
import logging
from datetime import datetime
import config
from functools import wraps


def retry_on_io_error(max_retries=5, delay=2.0, backoff_factor=1.5):
    """
    Decorator to retry file operations on I/O errors with exponential backoff.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        delay (float): Initial delay between retries in seconds
        backoff_factor (float): Factor by which to increase delay after each failure
        
    裝飾器，用於在 I/O 錯誤時重試文件操作，採用指數退避策略。
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (IOError, OSError) as e:
                    last_exception = e
                    error_code = getattr(e, 'errno', None)
                    error_msg = str(e)
                    
                    # Log error details for debugging
                    logging.warning(f"I/O error during {func.__name__}: [Errno {error_code}] {error_msg}. "
                                    f"Attempt {attempt+1}/{max_retries}")
                    
                    if attempt < max_retries - 1:
                        logging.info(f"Waiting {current_delay:.1f}s before retry...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor  # Exponential backoff
            
            # If all attempts fail, log error but don't interrupt training
            logging.error(f"Failed to execute {func.__name__} after {max_retries} attempts: {str(last_exception)}")
            return None
        return wrapper
    return decorator


def setup_plotting_style():
    """
    Set up consistent plotting style for all visualizations.
    
    設置所有可視化的一致繪圖風格。
    """
    # Use a simple clean style without background color
    plt.style.use('classic')
    plt.rcParams['figure.figsize'] = (12, 10)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10


def create_combined_plot(stats, smoothing=0.9, save_path=None):
    """
    Create a 4-panel combined plot showing all important metrics in one figure.
    
    Args:
        stats (dict): Dictionary containing training statistics
        smoothing (float): Smoothing factor for the plots (0-1)
        save_path (str, optional): Path to save the plot image
        
    創建一個4面板組合圖，在一個圖中顯示所有重要指標。
    
    參數：
        stats (dict)：包含訓練統計數據的字典
        smoothing (float)：繪圖的平滑因子（0-1）
        save_path (str, optional)：保存繪圖圖像的路徑
    """
    setup_plotting_style()
    
    # Create a figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Smoothing function for data series
    def smooth(data, alpha=smoothing):
        """Apply exponential smoothing to data series"""
        smoothed = []
        if not data or len(data) == 0:
            return smoothed
        value = data[0]
        for point in data:
            value = alpha * value + (1 - alpha) * point
            smoothed.append(value)
        return smoothed
    
    # 1. Episode Rewards Plot (Top Left)
    ax = axes[0, 0]
    has_labels = False
    if 'episode_rewards' in stats and len(stats['episode_rewards']) > 0:
        episode_rewards = stats['episode_rewards']
        episodes = list(range(1, len(episode_rewards) + 1))
        ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', label='Raw')
        has_labels = True
        if len(episode_rewards) >= 2:
            ax.plot(episodes, smooth(episode_rewards), color='darkblue',
                    linewidth=2, label=f'Smoothed (α={smoothing})')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Reward')
    ax.set_title('Training Rewards')
    if has_labels:  # Only add legend if we have labeled plots
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Episode Length Plot (Top Right)
    ax = axes[0, 1]
    has_labels = False
    if 'episode_lengths' in stats and len(stats['episode_lengths']) > 0:
        episode_lengths = stats['episode_lengths']
        episodes = list(range(1, len(episode_lengths) + 1))
        ax.plot(episodes, episode_lengths, alpha=0.3, color='green', label='Raw')
        has_labels = True
        if len(episode_lengths) >= 2:
            ax.plot(episodes, smooth(episode_lengths), color='darkgreen',
                    linewidth=2, label=f'Smoothed (α={smoothing})')
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Steps')
    ax.set_title('Episode Length')
    if has_labels:  # Only add legend if we have labeled plots
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Loss Plot (Bottom Left)
    ax = axes[1, 0]
    has_labels = False
    if 'losses' in stats and len(stats['losses']) > 0:
        losses = stats['losses']
        steps = list(range(1, len(losses) + 1))
        ax.plot(steps, losses, alpha=0.3, color='red', label='Raw')
        has_labels = True
        if len(losses) >= 2:
            ax.plot(steps, smooth(losses), color='darkred',
                    linewidth=2, label=f'Smoothed (α={smoothing})')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    if has_labels:  # Only add legend if we have labeled plots
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Priority Distribution / Comparative Plot (Bottom Right)
    ax = axes[1, 1]
    has_labels = False
    
    if 'priorities' in stats and len(stats['priorities']) > 0:
        # Plot priority distribution as a histogram
        priorities = np.array(stats['priorities'])
        if len(priorities) > 0:
            # Use simple histogram without KDE
            ax.hist(priorities, bins=30, alpha=0.7, color='purple', density=True)
            
            # Add mean and median lines
            mean_priority = np.mean(priorities)
            median_priority = np.median(priorities)
            ax.axvline(mean_priority, color='red', linestyle='--', label=f'Mean: {mean_priority:.4f}')
            ax.axvline(median_priority, color='green', linestyle='--', label=f'Median: {median_priority:.4f}')
            has_labels = True
            
        ax.set_xlabel('Priority Value')
        ax.set_ylabel('Density')
        ax.set_title('Priority Distribution')
    elif all(key in stats for key in ['uniform_rewards', 'per_rewards']):
        # Alternative: Plot comparative rewards if available
        uniform_rewards = stats['uniform_rewards']
        per_rewards = stats['per_rewards']
        
        if uniform_rewards:
            uniform_episodes = list(range(1, len(uniform_rewards) + 1))
            ax.plot(uniform_episodes, smooth(uniform_rewards), 
                     label='Uniform Sampling', color='blue', linewidth=2)
            has_labels = True
        
        if per_rewards:
            per_episodes = list(range(1, len(per_rewards) + 1))
            ax.plot(per_episodes, smooth(per_rewards), 
                     label='PER', color='red', linewidth=2)
            has_labels = True
            
        ax.set_xlabel('Episodes')
        ax.set_ylabel('Smoothed Reward')
        ax.set_title('Uniform vs. PER Rewards')
    else:
        # If neither priorities nor comparative rewards available
        ax.text(0.5, 0.5, "No priority or comparative data available", 
                horizontalalignment='center', verticalalignment='center')
        ax.set_title('No Data Available')
    
    if has_labels:  # Only add legend if we have labeled plots
        ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add overall title
    fig.suptitle('DQN with PER Training Metrics', fontsize=16)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the overall title
    
    # Save if path is specified
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Try to save the figure directly without using temporary files
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        except (IOError, OSError) as e:
            logging.error(f"Error saving plot to {save_path}: {str(e)}")
            
            # Try alternative save location in same filesystem
            try:
                alt_dir = os.path.dirname(os.path.dirname(save_path))
                alt_filename = f"backup_plot_{int(time.time())}.png"
                alt_path = os.path.join(alt_dir, alt_filename)
                plt.savefig(alt_path, dpi=300, bbox_inches='tight')
                logging.info(f"Plot saved to alternative location: {alt_path}")
            except Exception as backup_e:
                logging.error(f"Failed to save plot to alternative location: {str(backup_e)}")
    
    plt.close(fig)
    
    return fig


@retry_on_io_error(max_retries=5, delay=3.0, backoff_factor=2.0)
def save_training_plots(stats, run_name=None, output_dir=None):
    """
    Generate and save combined training plot.
    
    Args:
        stats (dict): Dictionary containing training statistics
        run_name (str, optional): Name for this training run
        output_dir (str, optional): Directory to save plots
        
    生成並保存組合訓練圖。
    
    參數：
        stats (dict)：包含訓練統計數據的字典
        run_name (str, optional)：此訓練運行的名稱
        output_dir (str, optional)：保存繪圖的目錄
    """
    # Set up output directory
    if output_dir is None:
        output_dir = config.PLOT_DIR  # Use dedicated plot directory instead of general result dir
    
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{timestamp}"
    
    # Create output directory if it doesn't exist
    results_dir = os.path.join(output_dir, run_name)
    try:
        os.makedirs(results_dir, exist_ok=True)
    except (IOError, OSError) as e:
        logging.error(f"Failed to create plot directory at {results_dir}: {str(e)}")
        # Try to use parent directory as fallback
        results_dir = output_dir
    
    # Create and save combined plot
    save_path = os.path.join(results_dir, 'training_metrics_combined.png')
    create_combined_plot(stats, save_path=save_path)
    
    print(f"Combined plot saved to {save_path}")
    return results_dir