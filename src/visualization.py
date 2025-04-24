"""
Visualization utilities for DQN with Prioritized Experience Replay.

This module provides visualization functions for training metrics recorded
by the logger, helping to analyze and understand the training process.

DQN 優先經驗回放的可視化工具。

此模組提供用於可視化由日誌記錄的訓練指標的功能，幫助分析和理解訓練過程。
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import sys
from typing import Dict, List, Tuple, Any, Optional, Union
import datetime

# Add parent directory to path to import config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from src.logger import Logger


class Visualizer:
    """
    Generates visualizations from training data.
    
    This class provides various plotting functions to visualize training metrics
    such as rewards, losses, exploration rates, and PER-specific metrics.
    
    生成訓練數據的可視化。
    
    此類提供各種繪圖功能，用於可視化訓練指標，例如獎勵、損失、探索率和PER特有指標。
    """
    
    def __init__(self, plot_dir: str = config.PLOT_DIR, 
                 logger_instance: Optional[Logger] = None,
                 experiment_name: Optional[str] = None,
                 data_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            plot_dir: Directory to save plot images
            logger_instance: Optional Logger instance to get data directly
            experiment_name: Name of the experiment to visualize (required if logger_instance not provided)
            data_dir: Directory containing logger data (required if logger_instance not provided)
        """
        self.plot_dir = plot_dir
        self.logger = logger_instance
        self.experiment_name = experiment_name
        
        # If logger instance not provided, we'll need to load data from files
        if self.logger is None:
            if experiment_name is None:
                raise ValueError("Must provide either a logger instance or an experiment name")
                
            if data_dir is None:
                self.data_dir = os.path.join(config.DATA_DIR, experiment_name)
            else:
                self.data_dir = os.path.join(data_dir, experiment_name)
                
            self.experiment_name = experiment_name
            self.episode_data_path = os.path.join(self.data_dir, "episode_data.jsonl")
            self.per_data_path = os.path.join(self.data_dir, "per_data.jsonl")
        else:
            self.experiment_name = logger_instance.experiment_name
            self.data_dir = logger_instance.data_dir
        
        # Create plots directory if it doesn't exist
        self.plots_dir = os.path.join(plot_dir, self.experiment_name)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Data containers
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_losses = []
        self.epsilon_values = []  # (step, epsilon)
        self.beta_values = []  # (step, beta)
        self.priority_means = []  # (step, mean_priority)
        self.priority_maxes = []  # (step, max_priority)
        self.td_error_means = []  # (step, mean_td_error)
        self.is_weight_means = []  # (step, mean_is_weight)
        
        # Load data if logger not provided
        if self.logger is None:
            self._load_data_from_files()
    
    def _load_data_from_files(self):
        """Load training data from JSON files."""
        # Load episode data
        if os.path.exists(self.episode_data_path):
            try:
                rewards = []
                lengths = []
                losses = []
                
                with open(self.episode_data_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            rewards.append(data.get('reward', 0))
                            lengths.append(data.get('steps', 0))
                            if 'loss' in data:
                                losses.append(data.get('loss', 0))
                
                self.episode_rewards = rewards
                self.episode_lengths = lengths
                self.episode_losses = losses
                print(f"Loaded data for {len(rewards)} episodes from {self.episode_data_path}")
            except Exception as e:
                print(f"Error loading episode data: {e}")
        else:
            print(f"Episode data file not found: {self.episode_data_path}")
        
        # Load PER data
        if os.path.exists(self.per_data_path):
            try:
                epsilon_values = []
                beta_values = []
                priority_means = []
                priority_maxes = []
                td_error_means = []
                is_weight_means = []
                
                with open(self.per_data_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            step = data.get('step', 0)
                            
                            if 'epsilon' in data:
                                epsilon_values.append((step, data['epsilon']))
                            
                            if 'beta' in data:
                                beta_values.append((step, data['beta']))
                            
                            if 'mean_priority' in data:
                                priority_means.append((step, data['mean_priority']))
                            
                            if 'max_priority' in data:
                                priority_maxes.append((step, data['max_priority']))
                            
                            if 'mean_td_error' in data:
                                td_error_means.append((step, data['mean_td_error']))
                            
                            if 'mean_is_weight' in data:
                                is_weight_means.append((step, data['mean_is_weight']))
                
                self.epsilon_values = sorted(epsilon_values, key=lambda x: x[0])
                self.beta_values = sorted(beta_values, key=lambda x: x[0])
                self.priority_means = sorted(priority_means, key=lambda x: x[0])
                self.priority_maxes = sorted(priority_maxes, key=lambda x: x[0])
                self.td_error_means = sorted(td_error_means, key=lambda x: x[0])
                self.is_weight_means = sorted(is_weight_means, key=lambda x: x[0])
                print(f"Loaded PER data from {self.per_data_path}")
            except Exception as e:
                print(f"Error loading PER data: {e}")
        else:
            print(f"PER data file not found: {self.per_data_path}")
    
    def _get_data(self):
        """Get training data either from logger or loaded files."""
        if self.logger is not None:
            # Get data directly from logger
            data = self.logger.get_training_data()
            self.episode_rewards = data.get('rewards', [])
            self.episode_lengths = data.get('lengths', [])
            self.episode_losses = data.get('losses', [])
            self.epsilon_values = data.get('epsilon_values', [])
            self.beta_values = data.get('beta_values', [])
            self.priority_means = data.get('priority_means', [])
            self.priority_maxes = data.get('priority_maxes', [])
            self.td_error_means = data.get('td_error_means', [])
            self.is_weight_means = data.get('is_weight_means', [])
    
    def plot_rewards(self, window_size: int = 100, save: bool = True, show: bool = False) -> str:
        """
        Plot episode rewards over time.
        
        Args:
            window_size: Size of the moving average window
            save: Whether to save the plot to a file
            show: Whether to display the plot
            
        Returns:
            str: Path to the saved plot file, or empty string if not saved
        """
        # Update data before plotting
        self._get_data()
        
        if not self.episode_rewards:
            print("No reward data available")
            return ""
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot raw rewards
        episodes = range(1, len(self.episode_rewards) + 1)
        ax.plot(episodes, self.episode_rewards, alpha=0.3, color='lightblue', label='Episode Reward')
        
        # Calculate and plot moving average
        if len(self.episode_rewards) >= window_size:
            moving_avg = np.convolve(self.episode_rewards, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            ax.plot(range(window_size, len(self.episode_rewards) + 1), 
                   moving_avg, 
                   color='blue', 
                   linewidth=2, 
                   label=f'{window_size}-Episode Moving Average')
        
        # Add labels and title
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title(f'Training Rewards - {self.experiment_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Add some statistics as text
        if self.episode_rewards:
            stats_text = (
                f"Episodes: {len(self.episode_rewards)}\n"
                f"Max Reward: {max(self.episode_rewards):.2f}\n"
                f"Recent Avg: {np.mean(self.episode_rewards[-100:]):.2f}"
            )
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rewards_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=120)
            print(f"Saved rewards plot to {filepath}")
            
                    # Show plot
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return filepath if save else ""
    
    def plot_losses(self, window_size: int = 100, save: bool = True, show: bool = False) -> str:
        """
        Plot training losses over time.
        
        Args:
            window_size: Size of the moving average window
            save: Whether to save the plot to a file
            show: Whether to display the plot
            
        Returns:
            str: Path to the saved plot file, or empty string if not saved
        """
        # Update data before plotting
        self._get_data()
        
        if not self.episode_losses:
            print("No loss data available")
            return ""
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot raw losses
        episodes = range(1, len(self.episode_losses) + 1)
        ax.plot(episodes, self.episode_losses, alpha=0.3, color='lightcoral', label='Episode Loss')
        
        # Calculate and plot moving average
        if len(self.episode_losses) >= window_size:
            moving_avg = np.convolve(self.episode_losses, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            ax.plot(range(window_size, len(self.episode_losses) + 1), 
                   moving_avg, 
                   color='red', 
                   linewidth=2, 
                   label=f'{window_size}-Episode Moving Average')
        
        # Add labels and title
        ax.set_xlabel('Episode')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training Losses - {self.experiment_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Use log scale for loss plots
        ax.set_yscale('log')
        
        # Add some statistics as text
        if self.episode_losses:
            stats_text = (
                f"Episodes: {len(self.episode_losses)}\n"
                f"Min Loss: {min(self.episode_losses):.6f}\n"
                f"Recent Avg: {np.mean(self.episode_losses[-100:]):.6f}"
            )
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Save plot
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"losses_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=120)
            print(f"Saved losses plot to {filepath}")
            
                    # Show plot
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return filepath if save else ""
    
    def plot_epsilon(self, save: bool = True, show: bool = False) -> str:
        """
        Plot epsilon decay over training steps.
        
        Args:
            save: Whether to save the plot to a file
            show: Whether to display the plot
            
        Returns:
            str: Path to the saved plot file, or empty string if not saved
        """
        # Update data before plotting
        self._get_data()
        
        if not self.epsilon_values:
            print("No epsilon data available")
            return ""
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data
        steps, epsilons = zip(*self.epsilon_values)
        
        # Plot epsilon decay
        ax.plot(steps, epsilons, color='green', linewidth=2, label='Epsilon')
        
        # Add labels and title
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Epsilon Value')
        ax.set_title(f'Exploration Rate (Epsilon) - {self.experiment_name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Format x-axis to show steps in thousands
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
        
        # Save plot
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"epsilon_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout()
            plt.savefig(filepath, dpi=120)
            print(f"Saved epsilon plot to {filepath}")
            
                    # Show plot
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return filepath if save else ""
    
    def plot_per_metrics(self, save: bool = True, show: bool = False) -> str:
        """
        Plot PER-specific metrics (beta, priorities, TD errors).
        
        Args:
            save: Whether to save the plot to a file
            show: Whether to display the plot
            
        Returns:
            str: Path to the saved plot file, or empty string if not saved
        """
        # Update data before plotting
        self._get_data()
        
        if not self.beta_values and not self.priority_means:
            print("No PER metrics data available")
            return ""
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # Plot beta annealing
        if self.beta_values:
            steps, betas = zip(*self.beta_values)
            axs[0].plot(steps, betas, color='purple', linewidth=2, label='Beta')
            axs[0].set_ylabel('Beta Value')
            axs[0].set_title('Importance Sampling Weight (Beta)')
            axs[0].grid(True, alpha=0.3)
            axs[0].legend()
        
        # Plot priority statistics
        if self.priority_means:
            steps_p, priorities = zip(*self.priority_means)
            axs[1].plot(steps_p, priorities, color='blue', linewidth=2, label='Mean Priority')
            
            if self.priority_maxes:
                steps_m, max_priorities = zip(*self.priority_maxes)
                axs[1].plot(steps_m, max_priorities, color='navy', linewidth=1, 
                         alpha=0.7, label='Max Priority')
            
            axs[1].set_ylabel('Priority Value')
            axs[1].set_title('Priority Distribution')
            axs[1].grid(True, alpha=0.3)
            axs[1].legend()
            
            # Use log scale for priorities
            axs[1].set_yscale('log')
        
        # Plot TD error
        if self.td_error_means:
            steps_td, td_errors = zip(*self.td_error_means)
            axs[2].plot(steps_td, td_errors, color='orange', linewidth=2, label='Mean |TD Error|')
            axs[2].set_xlabel('Training Steps')
            axs[2].set_ylabel('TD Error')
            axs[2].set_title('Temporal Difference Error')
            axs[2].grid(True, alpha=0.3)
            axs[2].legend()
            
            # Use log scale for TD errors
            axs[2].set_yscale('log')
        
        # Format x-axis to show steps in thousands
        for ax in axs:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
        
        # Overall title
        fig.suptitle(f'Prioritized Experience Replay Metrics - {self.experiment_name}', 
                    fontsize=16, y=0.98)
        
        # Save plot
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"per_metrics_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the overall title
            plt.savefig(filepath, dpi=120)
            print(f"Saved PER metrics plot to {filepath}")
            
                    # Show plot
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return filepath if save else ""
    
    def plot_training_overview(self, window_size: int = 100, save: bool = True, show: bool = False) -> str:
        """
        Plot the major metrics defined in config.LOGGER_MAJOR_METRICS in a single figure with 2x2 layout.
        
        This creates a combined plot of the main metrics (typically reward, loss, epsilon, beta)
        for easy comparison and overview, arranged in a 2x2 grid (upper-two, lower-two).
        
        Args:
            window_size: Size of the moving average window
            save: Whether to save the plot to a file
            show: Whether to display the plot
            
        Returns:
            str: Path to the saved plot file, or empty string if not saved
        """
        # Update data before plotting
        self._get_data()
        
        # Check if we have any data
        if not (self.episode_rewards or self.episode_losses or self.epsilon_values or self.beta_values):
            print("No major metrics data available")
            return ""
        
        # Create figure with 2x2 subplot layout
        fig, axs = plt.subplots(2, 2, figsize=(14, 12), sharex=False)
        axs = axs.flatten()  # Flatten to make indexing easier
        
        for i, metric_name in enumerate(config.LOGGER_MAJOR_METRICS):
            if i >= 4:  # Only support up to 4 metrics in 2x2 layout
                print(f"Warning: Only showing first 4 metrics in 2x2 layout (ignoring {metric_name})")
                break
                
            if metric_name == "reward" and self.episode_rewards:
                # Plot rewards
                episodes = range(1, len(self.episode_rewards) + 1)
                axs[i].plot(episodes, self.episode_rewards, alpha=0.3, color='lightblue', label='Episode Reward')
                
                if len(self.episode_rewards) >= window_size:
                    moving_avg = np.convolve(self.episode_rewards, 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
                    axs[i].plot(range(window_size, len(self.episode_rewards) + 1), 
                              moving_avg, 
                              color='blue', 
                              linewidth=2, 
                              label=f'{window_size}-Ep Avg')
                
                axs[i].set_ylabel('Reward')
                axs[i].set_title('Training Rewards')
                axs[i].set_xlabel('Episode')
                
                # Add some statistics
                if self.episode_rewards:
                    stats_text = (
                        f"Max: {max(self.episode_rewards):.2f}, "
                        f"Recent Avg: {np.mean(self.episode_rewards[-100:]):.2f}"
                    )
                    axs[i].text(0.02, 0.95, stats_text, transform=axs[i].transAxes, 
                              verticalalignment='top', 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
            elif metric_name == "loss" and self.episode_losses:
                # Plot losses
                episodes = range(1, len(self.episode_losses) + 1)
                axs[i].plot(episodes, self.episode_losses, alpha=0.3, color='lightcoral', label='Episode Loss')
                
                if len(self.episode_losses) >= window_size:
                    moving_avg = np.convolve(self.episode_losses, 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
                    axs[i].plot(range(window_size, len(self.episode_losses) + 1), 
                              moving_avg, 
                              color='red', 
                              linewidth=2, 
                              label=f'{window_size}-Ep Avg')
                
                axs[i].set_ylabel('Loss')
                axs[i].set_title('Training Losses')
                axs[i].set_xlabel('Episode')
                axs[i].set_yscale('log')
                
                # Add some statistics
                if self.episode_losses:
                    stats_text = (
                        f"Min: {min(self.episode_losses):.6f}, "
                        f"Recent Avg: {np.mean(self.episode_losses[-100:]):.6f}"
                    )
                    axs[i].text(0.02, 0.95, stats_text, transform=axs[i].transAxes, 
                              verticalalignment='top', 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
            elif metric_name == "epsilon" and self.epsilon_values:
                # Plot epsilon
                steps, epsilons = zip(*self.epsilon_values)
                axs[i].plot(steps, epsilons, color='green', linewidth=2, label='Epsilon')
                axs[i].set_ylabel('Epsilon Value')
                axs[i].set_title('Exploration Rate (Epsilon)')
                axs[i].set_xlabel('Training Steps')
                
                # Format x-axis to show steps in thousands
                axs[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
                
                # Add some statistics
                if self.epsilon_values:
                    stats_text = (
                        f"Current: {self.epsilon_values[-1][1]:.4f}, "
                        f"Initial: {self.epsilon_values[0][1]:.4f}"
                    )
                    axs[i].text(0.02, 0.95, stats_text, transform=axs[i].transAxes, 
                              verticalalignment='top', 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
            elif metric_name == "beta" and self.beta_values:
                # Plot beta
                steps, betas = zip(*self.beta_values)
                axs[i].plot(steps, betas, color='purple', linewidth=2, label='Beta')
                axs[i].set_ylabel('Beta Value')
                axs[i].set_title('Importance Sampling Weight (Beta)')
                axs[i].set_xlabel('Training Steps')
                
                # Format x-axis to show steps in thousands
                axs[i].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
                
                # Add some statistics
                if self.beta_values:
                    stats_text = (
                        f"Current: {self.beta_values[-1][1]:.4f}, "
                        f"Initial: {self.beta_values[0][1]:.4f}"
                    )
                    axs[i].text(0.02, 0.95, stats_text, transform=axs[i].transAxes, 
                              verticalalignment='top', 
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            axs[i].grid(True, alpha=0.3)
            
            # Only add legend if there are labeled artists
            handles, labels = axs[i].get_legend_handles_labels()
            if handles and labels:
                axs[i].legend()
        
        # Overall title
        fig.suptitle(f'Major Training Metrics - {self.experiment_name}', fontsize=16, y=0.98)
        
        # Save plot
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"major_metrics_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the overall title
            plt.savefig(filepath, dpi=120)
            print(f"Saved major metrics plot to {filepath}")
                    
        # Show plot
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return filepath if save else ""
    
    def generate_all_plots(self, show: bool = False) -> List[str]:
        """
        Generate all available plots.
        
        Args:
            show: Whether to display the plots
            
        Returns:
            List[str]: Paths to all generated plot files
        """
        plot_files = []
        
        # Generate each type of plot
        reward_plot = self.plot_rewards(save=True, show=show)
        if reward_plot:
            plot_files.append(reward_plot)
            
        loss_plot = self.plot_losses(save=True, show=show)
        if loss_plot:
            plot_files.append(loss_plot)
            
        epsilon_plot = self.plot_epsilon(save=True, show=show)
        if epsilon_plot:
            plot_files.append(epsilon_plot)
            
        per_plot = self.plot_per_metrics(save=True, show=show)
        if per_plot:
            plot_files.append(per_plot)
        
        major_metrics_plot = self.plot_training_overview(save=True, show=show)
        if major_metrics_plot:
            plot_files.append(major_metrics_plot)
            
        return plot_files


# For testing purposes
if __name__ == "__main__":
    # Try to load an experiment if one exists
    result_dir = config.RESULT_DIR
    data_dir = config.DATA_DIR
    
    # Check if any experiments exist
    if os.path.exists(data_dir):
        experiments = [d for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d))]
        
        if experiments:
            # Use the most recent experiment
            experiments.sort()
            latest_experiment = experiments[-1]
            print(f"Using latest experiment: {latest_experiment}")
            
            # Create visualizer
            vis = Visualizer(experiment_name=latest_experiment)
            
            # Generate all plots
            plot_files = vis.generate_all_plots(show=True)
            print(f"Generated {len(plot_files)} plots")
        else:
            print("No experiments found. Run training first to generate data.")
    else:
        print(f"Data directory {data_dir} not found.")