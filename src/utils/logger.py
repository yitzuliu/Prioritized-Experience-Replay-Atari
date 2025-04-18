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
import logging
import traceback
import tempfile
import shutil
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
                    
                    # 記錄錯誤詳細信息以便調試
                    logging.warning(f"I/O error during {func.__name__}: [Errno {error_code}] {error_msg}. "
                                    f"Attempt {attempt+1}/{max_retries}")
                    
                    if attempt < max_retries - 1:
                        logging.info(f"Waiting {current_delay:.1f}s before retry...")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor  # 指數退避
            
            # 如果所有嘗試都失敗，記錄錯誤但不中斷訓練
            logging.error(f"Failed to execute {func.__name__} after {max_retries} attempts: {str(last_exception)}")
            return None
        return wrapper
    return decorator


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
        # Configure basic logging
        logging.basicConfig(level=logging.INFO, 
                           format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Set up log directory
        if log_dir is None:
            log_dir = config.LOG_DIR
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"dqn_per_{timestamp}"
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 建立一個臨時目錄來處理寫入操作
        self.temp_dir = tempfile.mkdtemp(prefix=f"dqn_per_temp_{experiment_name}_")
        logging.info(f"Created temporary log directory at {self.temp_dir}")
        
        # 只保留用於圖表和恢復訓練的數據
        self.episode_rewards = []   # 用於繪製獎勵圖表
        self.episode_lengths = []   # 用於繪製回合長度圖表
        self.eval_rewards = []      # 用於繪製評估獎勵圖表
        
        # 這些數據用於恢復訓練
        self.latest_episode = 0     # 最新的回合數
        self.total_steps = 0        # 總步數
        self.best_eval_reward = float('-inf')  # 最佳評估獎勵
        
        # 初始化計時器
        self.start_time = time.time()
        self.episode_start_time = self.start_time
        
        # 緊急備份路徑
        self.emergency_log_path = os.path.expanduser("~/dqn_per_emergency_log.json")
        
        # 最小化日誌檔案 - 只保留用於圖表和恢復的資料
        print(f"Logger initialized at {self.log_dir} - Minimized logging mode (only chart data and recovery info)")
        print(f"All CSV logging disabled to prevent I/O errors")
    
    def __del__(self):
        """清理臨時文件夾，確保數據被保存"""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logging.info(f"Cleaned up temporary directory {self.temp_dir}")
        except Exception as e:
            logging.error(f"Error cleaning up temp directory: {e}")
    
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
        
        # Print progress
        if episode % 10 == 0:
            elapsed = now - self.start_time
            avg_reward = np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0
            print(f"Episode {episode} | Current Reward: {reward:.1f} | Current Steps: {length} | "
                  f"Epsilon: {epsilon:.4f} | Last 10 Eps Avg Reward: {avg_reward:.1f} | Total Training Time: {elapsed:.0f}s", 
                  flush=True)
    
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
        
        # Print evaluation results
        print(f"\n--- Evaluation after Episode {episode} ---", flush=True)
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}", flush=True)
        print(f"Min/Max Reward: {min_reward:.1f} / {max_reward:.1f}", flush=True)
        print(f"Average Length: {avg_length:.1f}", flush=True)
        print("-" * 40 + "\n", flush=True)
    
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
            'eval_rewards': self.eval_rewards,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return stats
    
    @retry_on_io_error(max_retries=5, delay=3.0, backoff_factor=2.0)
    def save_stats(self):
        """
        Save all statistics to JSON file.
        
        Returns:
            str: Path to the saved statistics file
            
        將所有統計數據保存到 JSON 文件。
        
        返回：
            str：保存的統計數據文件的路徑
        """
        # 縮減統計數據，只保留重要資訊
        compact_stats = {
            'episode_rewards': self.episode_rewards[-50:] if len(self.episode_rewards) > 50 else self.episode_rewards,
            'episode_lengths': self.episode_lengths[-50:] if len(self.episode_lengths) > 50 else self.episode_lengths,
            'eval_rewards': self.eval_rewards,
            'max_reward': max(self.episode_rewards) if self.episode_rewards else 0,
            'recent_avg_reward': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_episodes': len(self.episode_rewards)
        }
        
        # 將 numpy 數組轉換為列表以便 JSON 序列化
        for key, value in compact_stats.items():
            if isinstance(value, np.ndarray):
                compact_stats[key] = value.tolist()
            elif isinstance(value, np.float64) or isinstance(value, np.float32):
                compact_stats[key] = float(value)
        
        # 直接保存到永久位置
        stats_path = os.path.join(self.log_dir, 'stats.json')
        
        # 使用安全的寫入方式 - 先寫入臨時文件，然後移動替換
        temp_stats_path = os.path.join(self.temp_dir, 'stats.json.tmp')
        
        try:
            # 寫入臨時文件
            with open(temp_stats_path, 'w') as f:
                json.dump(compact_stats, f)
            
            # 移動替換目標文件
            shutil.move(temp_stats_path, stats_path)
            
            print(f"Simplified statistics saved to {stats_path}")
            
            # 同時保存一個應急備份
            with open(self.emergency_log_path, 'w') as f:
                backup_stats = {
                    'timestamp': compact_stats.get('timestamp'),
                    'episode_count': compact_stats.get('total_episodes', 0),
                    'last_rewards': compact_stats.get('episode_rewards', [])[-10:],
                    'experiment_name': os.path.basename(self.log_dir),
                    'max_reward': compact_stats.get('max_reward', 0),
                    'recent_avg_reward': compact_stats.get('recent_avg_reward', 0)
                }
                json.dump(backup_stats, f)
            
            return stats_path
            
        except (IOError, OSError) as e:
            logging.error(f"Error saving stats to {stats_path}: {str(e)}")
            logging.error(traceback.format_exc())
            
            # 基本備份 - 嘗試直接保存到用戶主目錄
            try:
                backup_path = os.path.join(os.path.expanduser("~"), f"dqn_per_backup_{int(time.time())}.json")
                with open(backup_path, 'w') as f:
                    json.dump(backup_stats, f)
                print(f"Emergency backup saved to {backup_path}")
                return backup_path
            except Exception as backup_e:
                logging.error(f"Failed to save backup: {backup_e}")
                print("Failed to save any statistics")
                return None
    
    @retry_on_io_error(max_retries=5, delay=3.0, backoff_factor=2.0)
    def save_recovery_info(self):
        """
        Save recovery information for resuming training later.
        
        Returns:
            str: Path to the saved recovery file
            
        保存恢復信息以便稍後恢復訓練。
        
        返回：
            str：保存的恢復文件的路徑
        """
        recovery_info = {
            'latest_episode': self.latest_episode,
            'total_steps': self.total_steps,
            'best_eval_reward': float(self.best_eval_reward) if self.best_eval_reward != float('-inf') else None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'elapsed_time': time.time() - self.start_time
        }
        
        # 保存到恢復文件
        recovery_path = os.path.join(self.log_dir, 'recovery_info.json')
        temp_recovery_path = os.path.join(self.temp_dir, 'recovery_info.json.tmp')
        
        try:
            # 寫入臨時文件
            with open(temp_recovery_path, 'w') as f:
                json.dump(recovery_info, f)
            
            # 移動替換目標文件
            shutil.move(temp_recovery_path, recovery_path)
            logging.info(f"Recovery information saved to {recovery_path}")
            
            return recovery_path
        except (IOError, OSError) as e:
            logging.error(f"Error saving recovery info: {str(e)}")
            return None
    
    def update_training_progress(self, episode, total_steps, best_reward=None):
        """
        Update training progress information (for recovery purposes).
        
        Args:
            episode (int): Current episode number
            total_steps (int): Total training steps so far
            best_reward (float, optional): Best evaluation reward so far
            
        更新訓練進度信息（用於恢復訓練）。
        
        參數：
            episode (int)：當前回合編號
            total_steps (int)：到目前為止的總訓練步數
            best_reward (float, optional)：到目前為止的最佳評估獎勵
        """
        self.latest_episode = episode
        self.total_steps = total_steps
        
        if best_reward is not None and best_reward > self.best_eval_reward:
            self.best_eval_reward = best_reward
    
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
        # 只保留在記憶體中，不寫入文件
        # 在簡化版本中，我們不再儲存每一步的損失值，而是只記錄用於繪圖的資料
    
    def log_priorities(self, priorities):
        """
        Log the current distribution of priorities from the replay buffer.
        
        Args:
            priorities (numpy.ndarray): Array of current priorities
            
        記錄回放緩衝區的當前優先級分佈。
        
        參數：
            priorities (numpy.ndarray)：當前優先級數組
        """
        # 在簡化版本中，我們不再存儲優先級分佈
        # 如果視覺化需要，可以在未來添加這部分功能
        pass