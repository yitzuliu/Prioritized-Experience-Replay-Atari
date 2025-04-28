"""
Training logger for DQN with Prioritized Experience Replay.

This module provides logging, visualization, and data storage functionality 
for tracking the training process of the DQN agent with PER.

DQN 優先經驗回放訓練日誌記錄器。

此模組提供日誌記錄、數據存儲和訓練跟踪功能，用於記錄使用 PER 的 DQN 智能體的訓練過程。
"""

import os
import json
import time
import datetime
import numpy as np
from collections import deque
import sys
from typing import Dict, List, Tuple, Any, Optional, Union

# Add parent directory to path to import config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


class Logger:
    """
    Logger for DQN training with PER.
    
    Provides functionality for:
    - Recording training metrics
    - Progress reporting
    - Data storage with memory optimization
    - Data interfaces for visualization
    
    DQN 使用 PER 的訓練日誌記錄器。
    
    提供以下功能：
    - 記錄訓練指標
    - 進度報告
    - 記憶體優化的數據存儲
    - 向可視化模組提供數據接口
    """
    
    def __init__(self, log_dir: str = config.LOG_DIR, 
                 data_dir: str = config.DATA_DIR,
                 experiment_name: str = None,
                 enable_file_logging: bool = config.ENABLE_FILE_LOGGING,
                 save_interval: int = config.LOGGER_SAVE_INTERVAL, 
                 memory_window: int = config.LOGGER_MEMORY_WINDOW,
                 batch_size: int = config.LOGGER_BATCH_SIZE,
                 per_log_frequency: int = config.PER_LOG_FREQUENCY):
        """
        Initialize the logger.
        
        Args:
            log_dir: Directory for storing logs
            data_dir: Directory for storing data
            experiment_name: Name of the experiment (defaults to timestamp)
            enable_file_logging: Whether to enable file logging
            save_interval: Interval for auto-saving data (episodes)
            memory_window: Maximum number of records to keep in memory
            batch_size: Number of records to accumulate before writing to disk
            per_log_frequency: How frequently to log PER updates (steps)
        """
        # Create a unique experiment name if not provided
        if experiment_name is None:
            self.experiment_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            self.experiment_name = experiment_name
            
        # Set up directories
        self.log_dir = os.path.join(log_dir, self.experiment_name)
        self.data_dir = os.path.join(data_dir, self.experiment_name)
        
        # Create directories if they don't exist
        for directory in [self.log_dir, self.data_dir]:
            os.makedirs(directory, exist_ok=True)
            
        # Set parameters
        self.save_interval = save_interval
        self.memory_window = memory_window
        self.batch_size = batch_size
        self.per_log_frequency = per_log_frequency
        self.enable_file_logging = enable_file_logging
        
        # Initialize training metrics (only keep recent data in memory)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_losses: List[float] = []
        self.epsilon_values: List[float] = []  # 改為簡單列表而不是元組列表
        self.beta_values: List[Tuple[int, float]] = []  # (step, beta)
        
        # PER specific metrics - store summaries instead of raw data
        self.priority_means: List[Tuple[int, float]] = []  # (step, mean_priority)
        self.priority_maxes: List[Tuple[int, float]] = []  # (step, max_priority)
        self.td_error_means: List[Tuple[int, float]] = []  # (step, mean_td_error)
        self.is_weight_means: List[Tuple[int, float]] = []  # (step, mean_is_weight)
        
        # Store tracking variables for last logged values to reduce redundancy
        self.last_logged_beta = None
        self.last_logged_epsilon = None
        
        # Training progress
        self.total_steps: int = 0
        self.current_episode: int = 0
        self.start_time = time.time()
        self.episode_start_time = time.time()
        
        # Best evaluation reward tracking
        self.best_eval_reward: float = float('-inf')
        
        # For tracking moving averages
        self.reward_window = deque(maxlen=100)
        self.loss_window = deque(maxlen=100)
        
        # Data buffer for batch writing
        self.data_buffer = []
        self.buffer_count = 0
        
        # Buffer for PER metrics batch writing
        self.per_data_buffer = []
        self.per_buffer_count = 0
        
        # File paths
        self.log_file_path = os.path.join(self.log_dir, "training_log.txt")
        self.episode_data_path = os.path.join(self.data_dir, "episode_data.jsonl")
        self.per_data_path = os.path.join(self.data_dir, "per_data.jsonl")
        
        # Create the log file and write header
        if enable_file_logging:
            with open(self.log_file_path, "w") as f:
                f.write(f"Training Log - Experiment: {self.experiment_name}\n")
                f.write(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
        
        # Log initialization message
        self.log_text(f"Logger initialized. Experiment: {self.experiment_name}")
        
    def log_episode_start(self, episode_num: int):
        """
        Log the start of a new episode.
        
        Args:
            episode_num: The episode number
        """
        self.current_episode = episode_num
        self.episode_start_time = time.time()
    
    def log_episode_end(self, episode_num: int, 
                        total_reward: float, 
                        steps: int, 
                        avg_loss: Optional[float] = None,
                        epsilon: Optional[float] = None,
                        beta: Optional[float] = None):  # Added beta parameter
        """
        Log the end of an episode with metrics.
        
        Args:
            episode_num: Episode number
            total_reward: Total reward obtained in the episode
            steps: Number of steps taken in the episode
            avg_loss: Average loss during the episode (if available)
            epsilon: Current epsilon value (if available)
            beta: Current beta value for PER (if available)
        """
        # Calculate episode duration
        duration = time.time() - self.episode_start_time
        
        # Update metrics
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(steps)
        self.reward_window.append(total_reward)
        
        if avg_loss is not None:
            self.episode_losses.append(avg_loss)
            self.loss_window.append(avg_loss)
            
        # Update total steps
        self.total_steps += steps
        
        # Calculate moving averages
        avg_reward_100 = np.mean(self.reward_window) if self.reward_window else 0.0
        avg_loss_100 = np.mean(self.loss_window) if self.loss_window and avg_loss is not None else 0.0
        
        # Create episode data record
        episode_data = {
            "episode": episode_num,
            "reward": total_reward,
            "steps": steps,
            "duration": duration,
            "avg_reward_100": avg_reward_100,
            "timestamp": time.time()
        }
        
        if avg_loss is not None:
            episode_data["loss"] = avg_loss
            episode_data["avg_loss_100"] = avg_loss_100
            
        if epsilon is not None:
            episode_data["epsilon"] = epsilon
            
        # Add beta to episode data if available
        if beta is not None:
            episode_data["beta"] = beta
            
        # Add to buffer for batch writing
        self.data_buffer.append(episode_data)
        self.buffer_count += 1
        
        # Format the log line
        loss_str = f"{avg_loss:.6f}" if avg_loss is not None else "N/A"
        epsilon_str = f"{epsilon:.4f}" if epsilon is not None else "N/A"
        beta_str = f"{beta:.4f}" if beta is not None else "N/A"
        
        # Create log line with the requested format
        log_line = f"episode {episode_num} | steps {steps} | reward {total_reward:.2f} | loss {loss_str}"
        
        # Only add epsilon and beta if they're provided, with updated format
        if epsilon is not None:
            log_line += f" | Epsilon {epsilon_str}"
        if beta is not None:
            log_line += f" | Beta {beta_str}"
            
        log_line += f" | during {duration:.2f}s"
        
        # Print to console
        print(log_line)
        
        # Write to log file with the exact same format (no timestamp prefix)
        if self.enable_file_logging:
            try:
                with open(self.log_file_path, "a") as f:
                    f.write(f"{log_line}\n")
            except OSError as e:
                print(f"WARNING: Failed to write to log file: {str(e)}. Continuing without logging to file.")
            
        # Print detailed progress report at intervals
        if episode_num % config.LOGGER_DETAILED_INTERVAL == 0:
            self.log_text(self._format_progress_report(detailed=True))
            
        # Perform batch write if buffer is full
        if self.buffer_count >= self.batch_size:
            self._batch_write()
            
        # Manage memory usage
        self.limit_memory_usage()
    
    def log_step(self, step_num: int, reward: float, loss: Optional[float] = None):
        """
        Log a single step within an episode (optional, for detailed logging).
        
        Args:
            step_num: Step number within the episode
            reward: Reward received at this step
            loss: Loss at this step (if available)
        """
        # This is a lightweight log that doesn't write to file to avoid performance impact
        pass
    
    def log_per_update(self, step_num: int, 
                      beta: float, 
                      priorities: np.ndarray, 
                      td_errors: np.ndarray,
                      is_weights: np.ndarray):
        """
        Log PER-specific metrics after memory update.
        
        Args:
            step_num: Global step number
            beta: Current beta value
            priorities: Array of priorities
            td_errors: Array of TD errors
            is_weights: Array of importance sampling weights
        """
        # Check if beta value has changed significantly before logging
        # This reduces redundant logging when beta changes very slowly
        significant_beta_change = (
            self.last_logged_beta is None or 
            abs(beta - self.last_logged_beta) > 0.01 or
            step_num % (self.per_log_frequency * 10) == 0
        )
        
        if significant_beta_change:
            self.beta_values.append((step_num, beta))
            self.last_logged_beta = beta
        
        # Only log at specified frequency to reduce data volume
        # Calculate and record summary statistics instead of raw data
        mean_priority = float(np.mean(priorities))
        max_priority = float(np.max(priorities)) 
        mean_td_error = float(np.mean(np.abs(td_errors)))
        mean_is_weight = float(np.mean(is_weights))
        
        # Record the summary statistics
        self.priority_means.append((step_num, mean_priority))
        self.priority_maxes.append((step_num, max_priority))
        self.td_error_means.append((step_num, mean_td_error))
        self.is_weight_means.append((step_num, mean_is_weight))
        
        # Create PER data record
        per_data = {
            "step": step_num,
            "beta": beta,
            "mean_priority": mean_priority,
            "max_priority": max_priority,
            "mean_td_error": mean_td_error,
            "mean_is_weight": mean_is_weight,
            "timestamp": time.time()
        }
        
        # Add to buffer instead of writing directly
        self.per_data_buffer.append(per_data)
        self.per_buffer_count += 1
        
        # Perform batch write if buffer has enough entries
        if self.per_buffer_count >= config.PER_BATCH_SIZE:
            self._batch_write_per()
        
        # Only log to console for significant changes (less frequently)
        if step_num % (self.per_log_frequency * 20) == 0:
            self.log_text(
                f"PER Update - Step: {step_num}, Beta: {beta:.4f}, "
                f"Mean Priority: {mean_priority:.6f}, Max Priority: {max_priority:.6f}"
            )
    
        # Perform batch write if buffer has enough entries, avoiding loss of data
        if self.per_buffer_count >= config.PER_BATCH_SIZE or (self.per_buffer_count >= config.PER_BATCH_SIZE / 2 and step_num % (self.per_log_frequency * 5) == 0):
            self._batch_write_per()


    def log_epsilon(self, step_num: int, epsilon: float):
        """
        Log the current epsilon value.
        
        Args:
            step_num: Global step number
            epsilon: Current epsilon value
        """
        # Check if epsilon value has changed significantly before logging
        # This reduces redundant logging when epsilon changes very slowly
        significant_epsilon_change = (
            self.last_logged_epsilon is None or 
            abs(epsilon - self.last_logged_epsilon) > 0.01 or
            step_num % 5000 == 0  # Force log every 5000 steps
        )
        
        if significant_epsilon_change:
            self.epsilon_values.append(epsilon)  # 只保存epsilon值，不保存step
            self.last_logged_epsilon = epsilon
    
    def limit_memory_usage(self):
        """
        Limit memory usage by keeping only recent records in memory.
        
        Older records are kept on disk but removed from memory.
        """
        # If we have more episode records than the memory window, trim the oldest
        if len(self.episode_rewards) > self.memory_window:
            # Calculate how many records to trim
            trim_count = len(self.episode_rewards) - self.memory_window
            
            # Trim basic episode metrics
            self.episode_rewards = self.episode_rewards[trim_count:]
            self.episode_lengths = self.episode_lengths[trim_count:]
            self.episode_losses = self.episode_losses[trim_count:]
            self.epsilon_values = self.epsilon_values[trim_count:]  # 也修剪epsilon值
            
        # Similarly trim PER metrics if they grow too large
        max_per_records = self.memory_window * 5  # Reduced from 10 to 5 for memory efficiency
        
        # More aggressive trimming for time-series metrics
        for metric_list in [self.beta_values, self.priority_means, 
                           self.priority_maxes, self.td_error_means, self.is_weight_means]:
            if len(metric_list) > max_per_records:
                # Keep first few, last few, and evenly sample the middle for visualization
                if len(metric_list) > max_per_records * 2:
                    # If we have a lot of data, keep the first 10%, last 30%, and sample 60% from the middle
                    first_n = max(10, int(max_per_records * 0.1))
                    last_n = max(30, int(max_per_records * 0.3))
                    middle_n = max_per_records - first_n - last_n
                    
                    # Get indices for the middle section to sample
                    middle_start = first_n
                    middle_end = len(metric_list) - last_n
                    middle_indices = np.linspace(middle_start, middle_end-1, middle_n, dtype=int)
                    
                    # Combine first, sampled middle, and last sections
                    metric_list[:] = (
                        metric_list[:first_n] + 
                        [metric_list[i] for i in middle_indices] + 
                        metric_list[-last_n:]
                    )
                else:
                    # For smaller amounts, just keep the most recent
                    metric_list[:] = metric_list[-max_per_records:]
    
    def get_training_data(self, metric_name=None, start=None, end=None):
        """
        Get training data for visualization or analysis.
        
        Args:
            metric_name: Name of the metric to retrieve (None for all metrics)
            start: Start index/episode (None for beginning)
            end: End index/episode (None for end)
            
        Returns:
            dict: The requested training data
        """
        # First, ensure all data is written to disk for consistency
        self._batch_write()
        self._batch_write_per()
        
        # Define the data dictionary with all metrics
        data = {
            "rewards": self.episode_rewards,
            "lengths": self.episode_lengths,
            "losses": self.episode_losses,
            "epsilon_values": self.epsilon_values,
            "beta_values": self.beta_values,
            "priority_means": self.priority_means,
            "priority_maxes": self.priority_maxes,
            "td_error_means": self.td_error_means,
            "is_weight_means": self.is_weight_means
        }
        
        # If specific metric requested, return just that
        if metric_name and metric_name in data:
            metric_data = data[metric_name]
            
            # Apply slicing if start/end specified
            if start is not None or end is not None:
                start = start or 0
                end = end or len(metric_data)
                return metric_data[start:end]
            return metric_data
            
        # Otherwise return all data (with optional slicing for episode-based metrics)
        if start is not None or end is not None:
            start = start or 0
            end = end or len(self.episode_rewards)
            
            # Apply slicing only to episode-based metrics
            sliced_data = data.copy()
            sliced_data["rewards"] = data["rewards"][start:end]
            sliced_data["lengths"] = data["lengths"][start:end]
            sliced_data["losses"] = data["losses"][start:end]
            
            return sliced_data
            
        return data
    
    def get_training_summary(self):
        """
        Get a summary of the training progress.
        
        Returns:
            dict: Summary statistics of the training
        """
        # Calculate duration
        duration = time.time() - self.start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Calculate statistics
        current_reward_avg = np.mean(self.reward_window) if self.reward_window else 0
        best_reward = max(self.episode_rewards) if self.episode_rewards else 0
        best_episode = self.episode_rewards.index(best_reward) + 1 if self.episode_rewards else 0
        
        summary = {
            "experiment_name": self.experiment_name,
            "episodes_completed": self.current_episode,
            "total_steps": self.total_steps,
            "duration": f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}",
            "current_reward_avg": current_reward_avg,
            "best_reward": best_reward,
            "best_episode": best_episode,
            "best_eval_reward": self.best_eval_reward,
            "last_rewards": self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
        }
        
        return summary
        
    def log_text(self, message: str):
        """
        Log a message to both console and log file.
        
        Args:
            message: The message to log
        """
        # Print to console directly without timestamp
        print(message)
        
        if self.enable_file_logging:
            try:
                # Write to log file without timestamp
                with open(self.log_file_path, "a") as f:
                    f.write(message + "\n")
            except OSError as e:
                print(f"WARNING: Failed to write to log file: {str(e)}. Continuing without logging to file.")
    
    def update_best_eval_reward(self, new_best_reward):
        """
        Update the best evaluation reward value.
        Called by training scripts when a better model performance is discovered.
        
        Args:
            new_best_reward: New best evaluation reward value
        """
        if new_best_reward > self.best_eval_reward:
            self.best_eval_reward = new_best_reward
            self.log_text(f"Updated best evaluation reward to: {new_best_reward:.2f}")
    
    # === Internal helper methods ===
    
    def _batch_write(self):
        """Perform batch writing of accumulated episode data to disk."""
        if not self.data_buffer:
            return
            
        # Append to episode data file
        try:
            with open(self.episode_data_path, 'a') as f:
                for record in self.data_buffer:
                    f.write(json.dumps(record) + '\n')
        except OSError as e:
            print(f"WARNING: Failed to write to episode data file: {str(e)}. Continuing without logging to file.")
                
        # Clear the buffer
        self.data_buffer = []
        self.buffer_count = 0
    
    def _batch_write_per(self):
        """Perform batch writing of accumulated PER data to disk."""
        if not self.per_data_buffer:
            return
            
        # Append to PER data file
        try:
            with open(self.per_data_path, 'a') as f:
                for record in self.per_data_buffer:
                    f.write(json.dumps(record) + '\n')
        except OSError as e:
            print(f"WARNING: Failed to write to PER data file: {str(e)}. Continuing without logging to file.")
                
        # Clear the buffer
        self.per_data_buffer = []
        self.per_buffer_count = 0
    
    def _format_progress_report(self, detailed=False):
        """
        Format a progress report string.
        
        Args:
            detailed: Whether to include detailed statistics
            
        Returns:
            str: Formatted progress report
        """
        # Calculate training duration
        duration = time.time() - self.start_time
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        # Calculate percentage complete
        percent_complete = (self.current_episode / config.TRAINING_EPISODES) * 100
        
        # Add separator line
        report = ["=================================================================="]
        
        # Basic progress information with percentage
        report.append(f"Training Progress - Episode {self.current_episode}/{config.TRAINING_EPISODES} ({percent_complete:.1f}%)")
        report.append(f"Total Steps: {self.total_steps}")
        
        # Add reward statistics
        if self.episode_rewards:
            avg_reward = np.mean(self.reward_window) if self.reward_window else 0
            report.append(f"Avg Reward (last 100): {avg_reward:.2f}")
            
        # Add exploration statistics (moved up)
        if self.epsilon_values:
            latest_epsilon = self.epsilon_values[-1]
            report.append(f"Current Epsilon: {latest_epsilon:.4f}")
        
        # Add loss statistics if available
        if self.episode_losses:
            avg_loss = np.mean(self.loss_window) if self.loss_window else 0
            report.append(f"Avg Loss (last 100): {avg_loss:.6f}")
        
        # Add elapsed time
        report.append(f"Elapsed time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
        
        # Add best evaluation reward
        if self.best_eval_reward > float('-inf'):
            report.append(f"Best Evaluation Reward: {self.best_eval_reward:.2f}")
        
        # Add last 5 rewards
        if self.episode_rewards:
            last_rewards = [f"{r:.1f}" for r in self.episode_rewards[-5:]]
            report.append(f"Last 5 Rewards: [{', '.join(last_rewards)}]")
        
        # Add PER statistics (only if detailed and available)
        if detailed:
            if self.beta_values:
                latest_beta = self.beta_values[-1][1]
                beta_progress = (latest_beta - config.BETA_START) / (1.0 - config.BETA_START) * 100
                report.append(f"Current Beta: {latest_beta:.4f} ({beta_progress:.1f}% to 1.0)")
                
            if self.priority_means:
                latest_priority = self.priority_means[-1][1]
                report.append(f"Current Avg Priority: {latest_priority:.6f}")
                
            if self.td_error_means:
                latest_td_error = self.td_error_means[-1][1]
                report.append(f"Current Avg TD Error: {latest_td_error:.6f}")
                
            if self.is_weight_means:
                latest_is_weight = self.is_weight_means[-1][1]
                report.append(f"Current Avg IS Weight: {latest_is_weight:.6f}")
        
        # Add bottom separator
        report.append("==================================================================")
        
        return "\n".join(report)


# For testing purposes
if __name__ == "__main__":
    # Create a logger
    logger = Logger(experiment_name="test_experiment")
    
    # Simulate some training episodes
    for episode in range(1, 11):
        logger.log_episode_start(episode)
        
        # Simulate steps within the episode
        episode_reward = 0
        step_losses = []
        
        for step in range(100):
            # Simulate a step
            reward = np.random.normal(0, 1)
            loss = 1.0 / (episode + step/100)  # Decreasing loss
            
            episode_reward += reward
            step_losses.append(loss)
            
            # Log epsilon
            epsilon = max(0.1, 1.0 - episode * 0.1)
            logger.log_epsilon(episode * 100 + step, epsilon)
            
            # Simulate PER updates every 10 steps
            if step % 10 == 0:
                # Simulate PER data
                beta = min(1.0, 0.4 + episode * 0.06)
                priorities = np.random.exponential(1.0 / (episode + 1), size=32)
                td_errors = np.random.normal(0, 1.0 / (episode + 1), size=32)
                is_weights = np.random.uniform(0.5, 1.0, size=32)
                
                logger.log_per_update(
                    episode * 100 + step, 
                    beta, 
                    priorities, 
                    td_errors, 
                    is_weights
                )
                
        # Log episode completion
        avg_loss = np.mean(step_losses)
        logger.log_episode_end(episode, episode_reward, 100, avg_loss, epsilon, beta)
        
    # Get and print training summary
    summary = logger.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
        
    # Test data retrieval for visualization
    reward_data = logger.get_training_data("rewards")
    print(f"\nRetrieved {len(reward_data)} reward records for visualization")