"""
Logger for DQN with Prioritized Experience Replay.

This module provides logging functionality for training metrics and data
visualization, helping track and analyze the learning process.

DQN 優先經驗回放的日誌記錄器。

此模組提供用於訓練指標和數據可視化的日誌記錄功能，幫助跟踪和分析學習過程。
"""

import os
import time
import numpy as np
import json
import csv
from datetime import datetime
import config


class Logger:
    """
    Logger class for tracking and storing training statistics.
    
    This class collects, stores, and exports training statistics for
    visualization and analysis.
    
    用於跟踪和存儲訓練統計數據的日誌記錄器類。
    
    此類收集、存儲和導出用於可視化和分析的訓練統計數據。
    """
    
    def __init__(self, log_dir=None, experiment_name=None):
        """
        Initialize the logger.
        
        Args:
            log_dir (str, optional): Directory for storing logs
            experiment_name (str, optional): Name for this experiment
            
        初始化日誌記錄器。
        
        參數：
            log_dir (str, optional)：存儲日誌的目錄
            experiment_name (str, optional)：此實驗的名稱
        """
        # Set up log directory
        if log_dir is None:
            log_dir = config.LOG_DIR
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"dqn_per_{timestamp}"
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize statistics dictionaries
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.priorities = []
        self.eval_rewards = []
        self.td_errors = []
        
        # Initialize timing statistics
        self.start_time = time.time()
        self.episode_start_time = self.start_time
        
        # CSV file paths for continuous logging
        self.episode_csv_path = os.path.join(self.log_dir, 'episode_stats.csv')
        self.eval_csv_path = os.path.join(self.log_dir, 'eval_stats.csv')
        self.training_csv_path = os.path.join(self.log_dir, 'training_stats.csv')
        
        # Initialize CSV files with headers
        with open(self.episode_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'length', 'duration_seconds', 'epsilon'])
        
        with open(self.eval_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'avg_reward', 'std_reward', 'min_reward', 'max_reward', 'avg_length'])
        
        with open(self.training_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss', 'avg_td_error', 'max_priority'])
        
        print(f"Logger initialized at {self.log_dir}")
    
    def log_episode(self, episode, reward, length, epsilon):
        """
        Log statistics for a training episode.
        
        Args:
            episode (int): Episode number
            reward (float): Episode total reward
            length (int): Episode length in steps
            epsilon (float): Current epsilon value
            
        記錄訓練回合的統計數據。
        
        參數：
            episode (int)：回合編號
            reward (float)：回合總獎勵
            length (int)：回合步數長度
            epsilon (float)：當前 epsilon 值
        """
        # Calculate episode duration
        now = time.time()
        duration = now - self.episode_start_time
        self.episode_start_time = now
        
        # Store in memory
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # Log to CSV file
        with open(self.episode_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, length, duration, epsilon])
        
        # Print progress
        if episode % 10 == 0:
            elapsed = now - self.start_time
            avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            print(f"Episode {episode} | Current Reward: {reward:.1f} | Current Steps: {length} | "
                  f"Epsilon: {epsilon:.4f} | Last 10 Eps Avg Reward: {avg_reward:.1f} | Total Training Time: {elapsed:.0f}s")
    
    def log_evaluation(self, episode, rewards, lengths):
        """
        Log statistics from an evaluation run.
        
        Args:
            episode (int): Current episode number
            rewards (list): List of rewards from evaluation episodes
            lengths (list): List of episode lengths from evaluation episodes
            
        記錄評估運行的統計數據。
        
        參數：
            episode (int)：當前回合編號
            rewards (list)：評估回合的獎勵列表
            lengths (list)：評估回合的長度列表
        """
        # Calculate statistics
        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        avg_length = np.mean(lengths)
        
        # Store in memory
        self.eval_rewards.append(avg_reward)
        
        # Log to CSV file
        with open(self.eval_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, avg_reward, std_reward, min_reward, max_reward, avg_length])
        
        # Print evaluation results
        print(f"\n--- Evaluation after Episode {episode} ---")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Min/Max Reward: {min_reward:.1f} / {max_reward:.1f}")
        print(f"Average Length: {avg_length:.1f}")
        print("-" * 40 + "\n")
    
    def log_training_step(self, step, loss, td_errors=None, max_priority=None):
        """
        Log statistics for a training step.
        
        Args:
            step (int): Current step number
            loss (float): Loss value for this step
            td_errors (list, optional): TD errors for this batch
            max_priority (float, optional): Maximum priority value
            
        記錄訓練步驟的統計數據。
        
        參數：
            step (int)：當前步驟編號
            loss (float)：此步驟的損失值
            td_errors (list, optional)：此批次的 TD 誤差
            max_priority (float, optional)：最大優先級值
        """
        # Store in memory
        self.losses.append(loss)
        
        # Store TD errors if provided
        avg_td_error = None
        if td_errors is not None and len(td_errors) > 0:
            avg_td_error = np.mean(np.abs(td_errors))
            self.td_errors.extend(np.abs(td_errors).flatten())
        
        # Log to CSV file (only on regular intervals to avoid excessive I/O)
        if step % 100 == 0:
            with open(self.training_csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([step, loss, avg_td_error, max_priority])
    
    def log_priorities(self, priorities):
        """
        Log the current distribution of priorities from the replay buffer.
        
        Args:
            priorities (numpy.ndarray): Array of current priorities
            
        記錄回放緩衝區的當前優先級分佈。
        
        參數：
            priorities (numpy.ndarray)：當前優先級數組
        """
        # Store priorities (we don't need to store all of them, a sample is enough)
        if len(priorities) > 1000:
            # Sample a subset to avoid excessive memory usage
            indices = np.random.choice(len(priorities), size=1000, replace=False)
            sampled_priorities = priorities[indices]
        else:
            sampled_priorities = priorities
        
        self.priorities = sampled_priorities.copy()
    
    def get_stats(self):
        """
        Get all logged statistics as a dictionary.
        
        Returns:
            dict: All collected statistics
            
        以字典形式獲取所有記錄的統計數據。
        
        返回：
            dict：所有收集的統計數據
        """
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'priorities': self.priorities,
            'eval_rewards': self.eval_rewards,
            'td_errors': self.td_errors
        }
        return stats
    
    def save_stats(self):
        """
        Save all statistics to JSON file.
        
        Returns:
            str: Path to the saved statistics file
            
        將所有統計數據保存到 JSON 文件。
        
        返回：
            str：保存的統計數據文件的路徑
        """
        stats = self.get_stats()
        
        # Convert numpy arrays to lists for JSON serialization
        for key, value in stats.items():
            if isinstance(value, np.ndarray):
                stats[key] = value.tolist()
        
        # Save to file
        stats_path = os.path.join(self.log_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
        
        print(f"Statistics saved to {stats_path}")
        return stats_path