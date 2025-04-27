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
import seaborn as sns

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
        self.epsilon_values = []
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
                epsilon = []
                
                with open(self.episode_data_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            rewards.append(data.get('reward', 0))
                            lengths.append(data.get('steps', 0))
                            if 'loss' in data:
                                losses.append(data.get('loss', 0))
                            if 'epsilon' in data:
                                epsilon.append(data.get('epsilon', 0))
                
                self.episode_rewards = rewards
                self.episode_lengths = lengths
                self.episode_losses = losses
                self.epsilon_values = epsilon
            except Exception as e:
                print(f"Error loading episode data: {e}")
        
        # Load PER data
        if os.path.exists(self.per_data_path):
            try:
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
                            
                            if 'beta' in data:
                                beta_values.append((step, data['beta']))
                            
                            if 'mean_priority' in data:
                                priority_means.append((step, data['mean_priority']))
                            elif 'priority_mean' in data:
                                priority_means.append((step, data['priority_mean']))
                            
                            if 'max_priority' in data:
                                priority_maxes.append((step, data['max_priority']))
                            elif 'priority_max' in data:
                                priority_maxes.append((step, data['priority_max']))
                            
                            if 'mean_td_error' in data:
                                td_error_means.append((step, data['mean_td_error']))
                            elif 'td_error_mean' in data:
                                td_error_means.append((step, data['td_error_mean']))
                            
                            if 'mean_is_weight' in data:
                                is_weight_means.append((step, data['mean_is_weight']))
                            elif 'is_weight_mean' in data:
                                is_weight_means.append((step, data['is_weight_mean']))
                
                self.beta_values = sorted(beta_values, key=lambda x: x[0])
                self.priority_means = sorted(priority_means, key=lambda x: x[0])
                self.priority_maxes = sorted(priority_maxes, key=lambda x: x[0])
                self.td_error_means = sorted(td_error_means, key=lambda x: x[0])
                self.is_weight_means = sorted(is_weight_means, key=lambda x: x[0])
            except Exception as e:
                print(f"Error loading PER data: {e}")
    
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
    
    def setup_plot_style(self):
        """Set up clean, professional plotting style as specified in the style guide."""
        # Reset style to defaults first
        plt.rcdefaults()
        
        # Set color palette according to the style guide with darker raw data colors
        self.colors = {
            'reward_raw': '#79b6e3',      # Darker lightblue
            'reward_avg': '#0066cc',      # Stronger blue
            'loss_raw': '#f48fb1',        # Darker pink
            'loss_avg': '#cc0000',        # Stronger red
            'epsilon': '#2ecc71',         # Green
            'beta': '#8e44ad',            # Purple
            'td_error': '#e67e22',        # Orange
            'priority_mean': '#00acc1',   # Darker cyan
            'priority_max': '#0d47a1',    # Darker blue
            'background': '#f8f9fa'       # Light gray background
        }
        
        # Optimized font sizes for better readability
        self.font_sizes = {
            'title': 18,                  # Larger main title
            'subtitle': 15,               # Larger subplot titles
            'axis_label': 13,             # Slightly larger axis labels
            'tick_label': 11,             # Slightly larger tick labels
            'legend': 11,                 # Slightly larger legend text
            'stats': 10                   # Slightly larger stats text
        }
        
        # Optimized figure sizes
        self.fig_sizes = {
            'single': (14, 8),            # Wider single plots
            'overview': (16, 14),         # Keep 2x2 overview size
            'multi': (14, 16)             # Slightly wider and taller multi plots
        }
        
        # Configure matplotlib settings for optimal presentation
        plt.style.use('ggplot')
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        plt.rcParams['figure.figsize'] = self.fig_sizes['single']
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = self.colors['background']
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['grid.linewidth'] = 0.8    # Slightly thicker grid lines
        plt.rcParams['axes.linewidth'] = 1.2    # Slightly thicker axis borders
        plt.rcParams['lines.linewidth'] = 2.0   # Default line thickness
        plt.rcParams['axes.titlepad'] = 12      # More padding for titles
        plt.rcParams['axes.labelpad'] = 8       # More padding for axis labels

    def configure_axis(self, ax, title, xlabel, ylabel, log_scale=False):
        """Configure axis with consistent styling according to the style guide."""
        # Set labels and title with improved styling
        ax.set_title(title, fontsize=self.font_sizes['subtitle'], fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=self.font_sizes['axis_label'])
        ax.set_ylabel(ylabel, fontsize=self.font_sizes['axis_label'])
        ax.tick_params(labelsize=self.font_sizes['tick_label'], length=5, width=1.2)
        
        # Set grid with improved styling
        ax.grid(True, alpha=0.3, linewidth=0.8)
        
        # Apply log scale if specified
        if log_scale:
            ax.set_yscale('log')
        
        # Remove top and right spines for cleaner look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)

    def plot_rewards(self, window_size: int = 100, save: bool = True, show: bool = False) -> str:
        """
        Plot episode rewards over time with moving average.
        
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
            return ""
        
        # Set up plot style
        self.setup_plot_style()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_sizes['single'])
        
        # Plot raw rewards with improved styling
        episodes = range(1, len(self.episode_rewards) + 1)
        ax.plot(episodes, self.episode_rewards, alpha=0.4, color=self.colors['reward_raw'], 
                linewidth=1.2, label='Episode Reward')
        
        # Calculate and plot moving average with improved styling
        if len(self.episode_rewards) >= window_size:
            moving_avg = np.convolve(self.episode_rewards, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            ax.plot(range(window_size, len(self.episode_rewards) + 1), 
                   moving_avg, 
                   color=self.colors['reward_avg'], 
                   linewidth=2.5, 
                   label=f'{window_size}-Episode Avg')
        
        # Configure axis styling
        self.configure_axis(ax, 'Training Rewards', 'Episode', 'Reward')
        
        # Add some statistics as text with improved styling
        if self.episode_rewards:
            max_reward = max(self.episode_rewards)
            recent_avg = np.mean(self.episode_rewards[-min(100, len(self.episode_rewards)):])
            stats_text = f"Max: {max_reward:.2f}, Recent Avg: {recent_avg:.2f}"
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=self.font_sizes['stats'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
        
        # Add legend with improved styling
        ax.legend(fontsize=self.font_sizes['legend'], framealpha=0.9, loc='lower right')
        
        # Set overall title
        fig.suptitle(f'Training Rewards - {self.experiment_name}', 
                    fontsize=self.font_sizes['title'], fontweight='bold')
        
        # Save plot with improved styling
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rewards_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
        # Show plot
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return filepath if save else ""

    def plot_losses(self, window_size: int = 100, save: bool = True, show: bool = False) -> str:
        """
        Plot training losses over time with moving average.
        
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
            return ""
        
        # Set up plot style
        self.setup_plot_style()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_sizes['single'])
        
        # Plot raw losses with improved styling
        episodes = range(1, len(self.episode_losses) + 1)
        ax.plot(episodes, self.episode_losses, alpha=0.4, color=self.colors['loss_raw'], 
                linewidth=1.2, label='Episode Loss')
        
        # Calculate and plot moving average with improved styling
        if len(self.episode_losses) >= window_size:
            moving_avg = np.convolve(self.episode_losses, 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            ax.plot(range(window_size, len(self.episode_losses) + 1), 
                   moving_avg, 
                   color=self.colors['loss_avg'], 
                   linewidth=2.5, 
                   label=f'{window_size}-Episode Avg')
        
        # Configure axis styling with log scale for loss
        self.configure_axis(ax, 'Training Losses', 'Episode', 'Loss', log_scale=True)
        
        # Add some statistics as text with improved styling
        if self.episode_losses:
            min_loss = min(self.episode_losses)
            recent_avg = np.mean(self.episode_losses[-min(100, len(self.episode_losses)):])
            stats_text = f"Min: {min_loss:.6f}, Recent Avg: {recent_avg:.6f}"
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=self.font_sizes['stats'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
        
        # Add legend with improved styling
        ax.legend(fontsize=self.font_sizes['legend'], framealpha=0.9, loc='upper right')
        
        # Set overall title
        fig.suptitle(f'Training Losses - {self.experiment_name}', 
                    fontsize=self.font_sizes['title'], fontweight='bold')
        
        # Save plot with improved styling
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"losses_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
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
            return ""
        
        # Set up plot style
        self.setup_plot_style()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.fig_sizes['single'])
        
        # Plot epsilon values
        episodes = range(1, len(self.epsilon_values) + 1)
        ax.plot(episodes, self.epsilon_values, color=self.colors['epsilon'], 
                linewidth=2, label='Epsilon')
        
        # Configure axis styling
        self.configure_axis(ax, 'Exploration Rate (Epsilon)', 'Training Steps', 'Epsilon Value')
        
        # Add some statistics as text
        if self.epsilon_values:
            current_epsilon = self.epsilon_values[-1] if self.epsilon_values else 0
            initial_epsilon = self.epsilon_values[0] if self.epsilon_values else 0
            stats_text = f"Current: {current_epsilon:.4f}, Initial: {initial_epsilon:.4f}"
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=self.font_sizes['stats'],
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
        
        # Add legend
        ax.legend(fontsize=self.font_sizes['legend'], framealpha=0.9)
        
        # Format x-axis to show steps in thousands
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
        
        # Set overall title
        fig.suptitle(f'Exploration Rate (Epsilon) - {self.experiment_name}', 
                    fontsize=self.font_sizes['title'], fontweight='bold')
        
        # Save plot
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"epsilon_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the title
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
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
            return ""
        
        # Set up plot style
        self.setup_plot_style()
        
        # Create figure with multiple subplots
        fig, axs = plt.subplots(3, 1, figsize=self.fig_sizes['multi'])
        
        # Plot beta annealing
        if self.beta_values:
            steps, betas = zip(*self.beta_values)
            axs[0].plot(steps, betas, color=self.colors['beta'], linewidth=2, label='Beta')
            self.configure_axis(axs[0], 'Importance Sampling Weight (Beta)', '', 'Beta Value')
            
            # Add some statistics
            current_beta = betas[-1] if betas else 0
            initial_beta = betas[0] if betas else 0
            stats_text = f"Current: {current_beta:.4f}, Initial: {initial_beta:.4f}"
            axs[0].text(0.02, 0.95, stats_text, transform=axs[0].transAxes, 
                       verticalalignment='top', fontsize=self.font_sizes['stats'],
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
            axs[0].legend(fontsize=self.font_sizes['legend'], framealpha=0.9)
        else:
            self.configure_axis(axs[0], 'Importance Sampling Weight (Beta) - No Data', '', '')
            axs[0].text(0.5, 0.5, "No Beta data available", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axs[0].transAxes, fontsize=12)
        
        # Plot priority statistics
        if self.priority_means:
            steps_p, priorities = zip(*self.priority_means)
            axs[1].plot(steps_p, priorities, color=self.colors['priority_mean'], 
                       linewidth=2, label='Mean Priority')
            
            if self.priority_maxes:
                steps_m, max_priorities = zip(*self.priority_maxes)
                axs[1].plot(steps_m, max_priorities, color=self.colors['priority_max'], 
                           linewidth=1.5, alpha=0.7, label='Max Priority')
            
            self.configure_axis(axs[1], 'Priority Distribution', '', 'Priority Value', log_scale=True)
            axs[1].legend(fontsize=self.font_sizes['legend'], framealpha=0.9)
        else:
            self.configure_axis(axs[1], 'Priority Distribution - No Data', '', '')
            axs[1].text(0.5, 0.5, "No Priority data available", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axs[1].transAxes, fontsize=12)
        
        # Plot TD error
        if self.td_error_means:
            steps_td, td_errors = zip(*self.td_error_means)
            axs[2].plot(steps_td, td_errors, color=self.colors['td_error'], linewidth=2, label='Mean |TD Error|')
            self.configure_axis(axs[2], 'Temporal Difference Error', 'Training Steps', 'TD Error', log_scale=True)
            axs[2].legend(fontsize=self.font_sizes['legend'], framealpha=0.9)
        else:
            self.configure_axis(axs[2], 'Temporal Difference Error - No Data', 'Training Steps', '')
            axs[2].text(0.5, 0.5, "No TD error data available", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axs[2].transAxes, fontsize=12)
        
        # Format x-axis to show steps in thousands for all subplots
        for ax in axs:
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
        
        # Overall title
        fig.suptitle(f'Prioritized Experience Replay Metrics - {self.experiment_name}', 
                    fontsize=self.font_sizes['title'], fontweight='bold')
        
        # Save plot
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"per_metrics_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the overall title
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
        
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
        for easy comparison and overview, arranged in a 2x2 grid.
        
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
            return ""
        
        # Set up plot style
        self.setup_plot_style()
        
        # Create figure with 2x2 subplot layout
        fig, axs = plt.subplots(2, 2, figsize=self.fig_sizes['overview'], sharex=False)
        fig.subplots_adjust(hspace=0.25, wspace=0.2)  # Adjust spacing between subplots
        axs = axs.flatten()  # Flatten to make indexing easier
        
        metrics_plotted = 0  # Counter for actually plotted metrics
        
        for i, metric_name in enumerate(config.LOGGER_MAJOR_METRICS):
            if i >= 4:  # Only support up to 4 metrics in 2x2 layout
                break
                
            if metric_name == "reward" and self.episode_rewards:
                # Plot rewards in the top-left position
                episodes = range(1, len(self.episode_rewards) + 1)
                axs[metrics_plotted].plot(episodes, self.episode_rewards, alpha=0.4, color=self.colors['reward_raw'], 
                          linewidth=1.2, label='Episode Reward')
                
                # Calculate and plot moving average if we have enough data
                if len(self.episode_rewards) >= window_size:
                    moving_avg = np.convolve(self.episode_rewards, 
                                          np.ones(window_size)/window_size, 
                                          mode='valid')
                    axs[metrics_plotted].plot(range(window_size, len(self.episode_rewards) + 1), 
                              moving_avg, 
                              color=self.colors['reward_avg'], 
                              linewidth=2.5, 
                              label=f'{window_size}-Ep Avg')
                
                self.configure_axis(axs[metrics_plotted], 'Training Rewards', 'Episode', 'Reward')
                
                # Add some statistics
                if self.episode_rewards:
                    max_reward = max(self.episode_rewards)
                    recent_avg = np.mean(self.episode_rewards[-min(100, len(self.episode_rewards)):])
                    stats_text = f"Max: {max_reward:.2f}, Recent Avg: {recent_avg:.2f}"
                    axs[metrics_plotted].text(0.02, 0.95, stats_text, transform=axs[metrics_plotted].transAxes, 
                              verticalalignment='top', fontsize=self.font_sizes['stats'],
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
                
                # Add legend
                handles, labels = axs[metrics_plotted].get_legend_handles_labels()
                if handles and labels:
                    axs[metrics_plotted].legend(fontsize=self.font_sizes['legend'], framealpha=0.9, loc='best')
                    
                metrics_plotted += 1
                
            elif metric_name == "loss" and self.episode_losses:
                # Plot losses
                episodes = range(1, len(self.episode_losses) + 1)
                axs[metrics_plotted].plot(episodes, self.episode_losses, alpha=0.4, color=self.colors['loss_raw'], 
                          linewidth=1.2, label='Episode Loss')
                
                # Calculate and plot moving average if we have enough data
                if len(self.episode_losses) >= window_size:
                    moving_avg = np.convolve(self.episode_losses, np.ones(window_size)/window_size, mode='valid')
                    axs[metrics_plotted].plot(range(window_size, len(self.episode_losses) + 1), 
                              moving_avg, color=self.colors['loss_avg'], 
                              linewidth=2.5, label=f'{window_size}-Ep Avg')
                
                self.configure_axis(axs[metrics_plotted], 'Training Losses', 'Episode', 'Loss', log_scale=True)
                
                # Add some statistics
                if self.episode_losses:
                    min_loss = min(self.episode_losses)
                    recent_avg = np.mean(self.episode_losses[-min(100, len(self.episode_losses)):])
                    stats_text = f"Min: {min_loss:.6f}, Recent Avg: {recent_avg:.6f}"
                    axs[metrics_plotted].text(0.02, 0.95, stats_text, transform=axs[metrics_plotted].transAxes, 
                              verticalalignment='top', fontsize=self.font_sizes['stats'],
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
                
                # Add legend
                handles, labels = axs[metrics_plotted].get_legend_handles_labels()
                if handles and labels:
                    axs[metrics_plotted].legend(fontsize=self.font_sizes['legend'], framealpha=0.9, loc='best')
                    
                metrics_plotted += 1
                
            elif metric_name == "epsilon" and self.epsilon_values:
                # Plot epsilon
                steps = range(1, len(self.epsilon_values) + 1)
                axs[metrics_plotted].plot(steps, self.epsilon_values, color=self.colors['epsilon'], 
                          linewidth=2, label='Epsilon')
                
                self.configure_axis(axs[metrics_plotted], 'Exploration Rate (Epsilon)', 'Training Steps', 'Epsilon Value')
                
                # Format x-axis to show steps in thousands
                axs[metrics_plotted].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
                
                # Add some statistics
                if self.epsilon_values:
                    current_epsilon = self.epsilon_values[-1] if self.epsilon_values else 0
                    initial_epsilon = self.epsilon_values[0] if self.epsilon_values else 0
                    stats_text = f"Initial: {initial_epsilon:.4f}, Current: {current_epsilon:.4f}"
                    axs[metrics_plotted].text(0.02, 0.95, stats_text, transform=axs[metrics_plotted].transAxes, 
                              verticalalignment='top', fontsize=self.font_sizes['stats'],
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
                
                # Add legend
                handles, labels = axs[metrics_plotted].get_legend_handles_labels()
                if handles and labels:
                    axs[metrics_plotted].legend(fontsize=self.font_sizes['legend'], framealpha=0.9, loc='best')
                    
                metrics_plotted += 1
                
            elif metric_name == "beta" and self.beta_values:
                # Plot beta
                steps, betas = zip(*self.beta_values)
                axs[metrics_plotted].plot(steps, betas, color=self.colors['beta'], linewidth=2, label='Beta')
                
                self.configure_axis(axs[metrics_plotted], 'Importance Sampling Weight (Beta)', 'Training Steps', 'Beta Value')
                
                # Format x-axis to show steps in thousands
                axs[metrics_plotted].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1000:.0f}k'))
                
                # Add some statistics
                if self.beta_values:
                    initial_beta = self.beta_values[0][1] if self.beta_values else 0
                    current_beta = self.beta_values[-1][1] if self.beta_values else 0
                    stats_text = f"Initial: {initial_beta:.4f}, Current: {current_beta:.4f}"
                    axs[metrics_plotted].text(0.02, 0.95, stats_text, transform=axs[metrics_plotted].transAxes, 
                              verticalalignment='top', fontsize=self.font_sizes['stats'],
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.4, edgecolor='#dddddd'))
                
                # Add legend
                handles, labels = axs[metrics_plotted].get_legend_handles_labels()
                if handles and labels:
                    axs[metrics_plotted].legend(fontsize=self.font_sizes['legend'], framealpha=0.9, loc='best')
                    
                metrics_plotted += 1
        
        # If we have remaining empty plots, fill them with placeholders
        for i in range(metrics_plotted, 4):
            axs[i].text(0.5, 0.5, "No data available", 
                      horizontalalignment='center', verticalalignment='center',
                      transform=axs[i].transAxes, fontsize=12)
            self.configure_axis(axs[i], "Empty Plot", "", "")
        
        # Overall title with improved styling
        fig.suptitle(f'Major Training Metrics - {self.experiment_name}', 
                     fontsize=self.font_sizes['title'], fontweight='bold', y=0.98)
        
        # Save plot with improved styling
        if save:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"major_metrics_{timestamp}.png"
            filepath = os.path.join(self.plots_dir, filename)
            plt.tight_layout(rect=[0, 0, 1, 0.96])  # Make room for the title as specified in test.md
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    
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
    results_dir = config.RESULTS_DIR
    data_dir = config.DATA_DIR
    
    # Check if any experiments exist
    if os.path.exists(data_dir):
        experiments = [d for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d))]
        
        if experiments:
            # Use the most recent experiment
            experiments.sort()
            latest_experiment = experiments[-1]
            
            # Create visualizer
            vis = Visualizer(experiment_name=latest_experiment)
            
            # Generate all plots
            plot_files = vis.generate_all_plots(show=True)
        else:
            print("No experiments found. Run training first to generate data.")
    else:
        print(f"Data directory {data_dir} not found.")