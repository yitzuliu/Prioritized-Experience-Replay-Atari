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
        
        # 建立一個臨時目錄來處理寫入操作，降低主存儲區的I/O負擔
        self.temp_dir = tempfile.mkdtemp(prefix=f"dqn_per_temp_{experiment_name}_")
        logging.info(f"Created temporary log directory at {self.temp_dir}")
        
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
        self.last_flush_time = self.start_time
                
        # 增加緩衝區刷新控制
        self.flush_interval = 2400  # 10分鐘刷新一次臨時文件到主存儲
        self.emergency_log_path = os.path.expanduser("~/dqn_per_emergency_log.json")
                
        # CSV file paths for continuous logging - use temp dir for frequent writes
        self.temp_episode_csv_path = os.path.join(self.temp_dir, 'episode_stats.csv')
        self.temp_eval_csv_path = os.path.join(self.temp_dir, 'eval_stats.csv')
        self.temp_training_csv_path = os.path.join(self.temp_dir, 'training_stats.csv')
        
        # Final paths in the log directory
        self.episode_csv_path = os.path.join(self.log_dir, 'episode_stats.csv')
        self.eval_csv_path = os.path.join(self.log_dir, 'eval_stats.csv')
        self.training_csv_path = os.path.join(self.log_dir, 'training_stats.csv')
        
        # In-memory queue for data waiting to be written
        self.episode_queue = []
        self.eval_queue = []
        self.training_queue = []
        
        # 調整緩衝區閾值 - 減少寫入頻率
        self.episode_buffer_threshold = 20   # 累積20條記錄再寫入
        self.training_buffer_threshold = 500  # 累積500步再寫入
        
        # Initialize CSV files with headers
        self._init_csv_files()
        
        print(f"Logger initialized at {self.log_dir} (temporary files at {self.temp_dir})")
            
    def __del__(self):
        """清理臨時文件夾，確保數據被保存"""
        self._flush_to_permanent_storage()
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logging.info(f"Cleaned up temporary directory {self.temp_dir}")
        except Exception as e:
            logging.error(f"Error cleaning up temp directory: {e}")
    
    @retry_on_io_error()
    def _init_csv_files(self):
        """Initialize CSV files with headers if they don't exist."""
        # 檢查是否是恢復訓練
        is_resuming = os.path.exists(self.episode_csv_path)
        
        # 如果是恢復訓練，先將現有文件複製到臨時目錄
        if is_resuming:
            try:
                if os.path.exists(self.episode_csv_path):
                    shutil.copy2(self.episode_csv_path, self.temp_episode_csv_path)
                if os.path.exists(self.eval_csv_path):
                    shutil.copy2(self.eval_csv_path, self.temp_eval_csv_path)
                if os.path.exists(self.training_csv_path):
                    shutil.copy2(self.training_csv_path, self.temp_training_csv_path)
                logging.info("Copied existing log files to temp directory")
            except Exception as e:
                logging.error(f"Error copying existing logs: {e}")
        
        # 設置寫入模式
        file_mode = 'a' if is_resuming else 'w'
        
        try:
            # 初始化回合統計CSV
            if not os.path.exists(self.temp_episode_csv_path) or os.path.getsize(self.temp_episode_csv_path) == 0:
                with open(self.temp_episode_csv_path, file_mode, newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['episode', 'reward', 'length', 'duration_seconds', 'epsilon'])
            
            # 初始化評估統計CSV
            if not os.path.exists(self.temp_eval_csv_path) or os.path.getsize(self.temp_eval_csv_path) == 0:
                with open(self.temp_eval_csv_path, file_mode, newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['episode', 'avg_reward', 'std_reward', 'min_reward', 'max_reward', 'avg_length'])
            
            # 初始化訓練步驟統計CSV
            if not os.path.exists(self.temp_training_csv_path) or os.path.getsize(self.temp_training_csv_path) == 0:
                with open(self.temp_training_csv_path, file_mode, newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step', 'loss', 'avg_td_error', 'max_priority'])
        except (IOError, OSError) as e:
            logging.error(f"Error initializing CSV files: {str(e)}")
            logging.error(traceback.format_exc())
    
    @retry_on_io_error()
    def _write_to_csv(self, file_path, rows):
        """
        Write rows to a CSV file with error handling.
        
        Args:
            file_path (str): Path to the CSV file
            rows (list): List of rows to write
        """
        if not rows:
            return
        
        # 如果文件不存在，先創建並寫入表頭
        if not os.path.exists(file_path):
            is_episode = 'episode_stats' in file_path
            is_eval = 'eval_stats' in file_path
            is_training = 'training_stats' in file_path
            
            headers = []
            if is_episode:
                headers = ['episode', 'reward', 'length', 'duration_seconds', 'epsilon']
            elif is_eval:
                headers = ['episode', 'avg_reward', 'std_reward', 'min_reward', 'max_reward', 'avg_length']
            elif is_training:
                headers = ['step', 'loss', 'avg_td_error', 'max_priority']
            
            if headers:
                try:
                    with open(file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                except Exception as e:
                    logging.error(f"Failed to create file {file_path} with headers: {e}")
                    return False
            
        # 使用"寫後保留"策略: 先寫到一個臨時文件，成功後再替換原文件
        temp_file = f"{file_path}.tmp"
        try:
            # 如果原始文件存在，先複製內容
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                shutil.copy2(file_path, temp_file)
            else:
                # 否則創建一個新文件
                with open(temp_file, 'w', newline='') as f:
                    pass
            
            # 追加新行
            with open(temp_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for row in rows:
                    writer.writerow(row)
            
            # 替換原文件 (原子操作)
            shutil.move(temp_file, file_path)
            return True
            
        except (IOError, OSError) as e:
            logging.error(f"Error writing to {file_path}: {str(e)}")
            
            # 如果在替換文件時出錯，嘗試直接附加到原文件
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'a', newline='') as f:
                        writer = csv.writer(f)
                        for row in rows:
                            writer.writerow(row)
                    return True
            except Exception as inner_e:
                logging.error(f"Fallback write failed: {inner_e}")
            
            # 如果所有方法都失敗了
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
            return False
    
    def _flush_to_permanent_storage(self):
        """將臨時文件寫入到永久存儲"""
        # 先將內存中的數據寫到臨時文件
        if self.episode_queue:
            self._write_to_csv(self.temp_episode_csv_path, self.episode_queue)
            self.episode_queue = []
        
        if self.eval_queue:
            self._write_to_csv(self.temp_eval_csv_path, self.eval_queue)
            self.eval_queue = []
        
        if self.training_queue:
            self._write_to_csv(self.temp_training_csv_path, self.training_queue)
            self.training_queue = []
        
        # 然後將臨時文件複製到永久存儲
        try:
            if os.path.exists(self.temp_episode_csv_path):
                shutil.copy2(self.temp_episode_csv_path, self.episode_csv_path)
            
            if os.path.exists(self.temp_eval_csv_path):
                shutil.copy2(self.temp_eval_csv_path, self.eval_csv_path)
            
            if os.path.exists(self.temp_training_csv_path):
                shutil.copy2(self.temp_training_csv_path, self.training_csv_path)
            
            logging.info("Successfully flushed logs to permanent storage")
            self.last_flush_time = time.time()
        except (IOError, OSError) as e:
            logging.error(f"Error flushing to permanent storage: {e}")
    
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
        
        # Add to queue for writing
        self.episode_queue.append([episode, reward, length, duration, epsilon])
        
        # 檢查是否該刷新到臨時文件 (每20回合)
        if len(self.episode_queue) >= self.episode_buffer_threshold:
            self._write_to_csv(self.temp_episode_csv_path, self.episode_queue)
            self.episode_queue = []
        
        # 檢查是否應該將臨時文件刷新到永久存儲
        if now - self.last_flush_time > self.flush_interval:
            self._flush_to_permanent_storage()
        
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
        
        # Add to queue and write immediately (evaluation happens less frequently)
        self.eval_queue.append([episode, avg_reward, std_reward, min_reward, max_reward, avg_length])
        self._write_to_csv(self.temp_eval_csv_path, self.eval_queue)
        self.eval_queue = []
        
        # Print evaluation results
        print(f"\n--- Evaluation after Episode {episode} ---")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Min/Max Reward: {min_reward:.1f} / {max_reward:.1f}")
        print(f"Average Length: {avg_length:.1f}")
        print("-" * 40 + "\n")
        
        # 評估時始終刷新到永久存儲，因為評估結果很重要
        self._flush_to_permanent_storage()
    
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
        
        # Always add the data point to our queue
        self.training_queue.append([step, loss, avg_td_error, max_priority])
        
        # 減少寫入頻率 - 每500步或每1000步
        if len(self.training_queue) >= self.training_buffer_threshold or step % 1000 == 0:
            success = self._write_to_csv(self.temp_training_csv_path, self.training_queue)
            if success:
                self.training_queue = []  # Only clear if write was successful
    
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
            'td_errors': self.td_errors,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        return stats
    
    @retry_on_io_error(max_retries=10, delay=3.0, backoff_factor=2.0)
    def save_stats(self):
        """
        Save all statistics to JSON file.
        
        Returns:
            str: Path to the saved statistics file
            
        將所有統計數據保存到 JSON 文件。
        
        返回：
            str：保存的統計數據文件的路徑
        """
        # 先刷新內存隊列到臨時文件
        if self.episode_queue:
            self._write_to_csv(self.temp_episode_csv_path, self.episode_queue)
            self.episode_queue = []
        
        if self.eval_queue:
            self._write_to_csv(self.temp_eval_csv_path, self.eval_queue)
            self.eval_queue = []
        
        if self.training_queue:
            self._write_to_csv(self.temp_training_csv_path, self.training_queue)
            self.training_queue = []
        
        # 然後將臨時文件刷新到永久存儲
        self._flush_to_permanent_storage()
        
        # 獲取統計數據並準備保存
        stats = self.get_stats()
        
        # Convert numpy arrays to lists for JSON serialization
        for key, value in stats.items():
            if isinstance(value, np.ndarray):
                stats[key] = value.tolist()
        
        # Save using write-then-move pattern (atomic operation)
        stats_path = os.path.join(self.log_dir, 'stats.json')
        temp_stats_path = os.path.join(self.temp_dir, 'stats.json.tmp')
        
        try:
            # 先寫入臨時文件
            with open(temp_stats_path, 'w') as f:
                json.dump(stats, f)
            
            # 然後移動替換目標文件
            shutil.move(temp_stats_path, stats_path)
            print(f"Statistics saved to {stats_path}")
            
            # 成功保存後，同時寫入一個應急備份
            with open(self.emergency_log_path, 'w') as f:
                backup_stats = {
                    'timestamp': stats.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    'episode_count': len(stats.get('episode_rewards', [])),
                    'last_rewards': stats.get('episode_rewards', [])[-10:] if stats.get('episode_rewards') else [],
                    'experiment_name': os.path.basename(self.log_dir)
                }
                json.dump(backup_stats, f)
            
            return stats_path
        except (IOError, OSError) as e:
            logging.error(f"Error saving stats to {stats_path}: {str(e)}")
            logging.error(traceback.format_exc())
            
            # 嘗試直接保存到家目錄的備份位置
            backup_path = os.path.join(os.path.expanduser("~"), f"dqn_per_stats_backup_{int(time.time())}.json")
            try:
                with open(backup_path, 'w') as f:
                    json.dump(stats, f)
                print(f"Statistics saved to backup location: {backup_path}")
                return backup_path
            except Exception as backup_e:
                logging.error(f"Failed to save backup: {backup_e}")
                print("Failed to save statistics to backup location")
                return None