"""
Logger for DQN with Prioritized Experience Replay.

This module provides efficient logging functionality for training metrics.
Designed for minimal memory usage and reliable I/O operations during long training sessions.
Focus on essential data storage and recovery capabilities.

DQN 優先經驗回放的輕量化日誌記錄器。

此模組提供高效的訓練指標日誌記錄功能。
專為長時間訓練過程中的最小內存使用和可靠I/O操作而設計。
專注於必要數據存儲和恢復功能。
"""

import os
import time
import numpy as np
import json
from datetime import datetime
import logging
import traceback
from functools import wraps
import gc  # Garbage collection
import config
import threading
import queue
from collections import deque


def retry_on_io_error(max_retries=5, delay=2.0, backoff_factor=1.5):
    """
    Decorator to retry file operations on I/O errors with exponential backoff.
    
    Args:
        max_retries (int): Maximum number of retry attempts
        delay (float): Initial delay between retries in seconds
        backoff_factor (float): Factor by which to increase delay after each failure
        
    I/O錯誤時使用指數退避策略重試文件操作的裝飾器。
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
                    
                    logging.warning(f"I/O error during {func.__name__}: [Errno {error_code}] {error_msg}. "
                                   f"Attempt {attempt+1}/{max_retries}")
                    
                    if attempt < max_retries - 1:
                        time.sleep(current_delay)
                        current_delay *= backoff_factor  # Exponential backoff
            
            # If all attempts fail, log error
            logging.error(f"Failed to execute {func.__name__} after {max_retries} attempts: {str(last_exception)}")
            return None
        return wrapper
    return decorator


class AsyncWriter:
    """
    Asynchronous file writer that handles I/O operations in a separate thread.
    This prevents I/O operations from blocking the main training process.
    
    異步文件寫入器，在單獨線程中處理I/O操作。
    防止I/O操作阻塞主訓練過程。
    """
    def __init__(self, max_queue_size=100):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.is_running = True
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()
        logging.info("Async writer thread started")
    
    def _worker_loop(self):
        """Worker thread that processes write operations."""
        while self.is_running:
            try:
                task = self.queue.get(timeout=1.0)
                if task is None:  # Sentinel to stop the thread
                    break
                    
                func, args, kwargs = task
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"Error in async writer: {str(e)}")
                finally:
                    self.queue.task_done()
            except queue.Empty:
                continue
    
    def write(self, func, *args, **kwargs):
        """
        Queue a write operation to be executed asynchronously.
        
        Args:
            func: The function to execute
            *args, **kwargs: Arguments to pass to the function
        
        Returns:
            bool: Whether the task was queued successfully
        """
        try:
            self.queue.put((func, args, kwargs), block=False)
            return True
        except queue.Full:
            logging.warning("Async writer queue is full, skipping write operation")
            return False
    
    def stop(self):
        """Stop the worker thread and wait for pending operations to complete."""
        logging.info("Stopping async writer thread...")
        self.is_running = False
        self.queue.put(None)  # Send sentinel to stop the thread
        self.worker.join(timeout=5.0)
        if self.worker.is_alive():
            logging.warning("Async writer thread did not terminate gracefully")
        else:
            logging.info("Async writer thread stopped")


class SegmentedFileManager:
    """
    Manages data storage across multiple segment files to prevent large 
    monolithic files and reduce memory pressure during I/O operations.
    
    分段文件管理器，跨多個段文件存儲數據，防止大型單體文件，
    並減少I/O操作期間的內存壓力。
    """
    def __init__(self, base_path, prefix="segment_", max_segment_size=10000):
        self.base_path = base_path
        self.prefix = prefix
        self.max_segment_size = max_segment_size
        self.current_segment = 0
        self.segment_counts = {}  # Track item count in each segment
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.base_path), exist_ok=True)
        
        # Scan for existing segments
        self._scan_existing_segments()
    
    def _scan_existing_segments(self):
        """Scan for existing segment files and update internal state."""
        dir_path = os.path.dirname(self.base_path)
        base_name = os.path.basename(self.base_path)
        
        # Look for segment files matching the pattern
        for filename in os.listdir(dir_path):
            if filename.startswith(f"{base_name}_{self.prefix}") and filename.endswith(".json"):
                try:
                    segment_num = int(filename.split('_')[-1].split('.')[0])
                    self.current_segment = max(self.current_segment, segment_num + 1)
                    
                    # Load segment to count items
                    segment_path = os.path.join(dir_path, filename)
                    try:
                        with open(segment_path, 'r') as f:
                            data = json.load(f)
                            # Store count of first array found (assuming it's episode data)
                            for key, value in data.items():
                                if isinstance(value, list):
                                    self.segment_counts[segment_num] = len(value)
                                    break
                    except (IOError, json.JSONDecodeError):
                        # If we can't read the file, assume it's empty or corrupted
                        self.segment_counts[segment_num] = 0
                except ValueError:
                    continue
    
    def get_segment_path(self, segment_num):
        """Get the file path for a specific segment number."""
        dir_path = os.path.dirname(self.base_path)
        base_name = os.path.basename(self.base_path)
        return os.path.join(dir_path, f"{base_name}_{self.prefix}{segment_num}.json")
    
    def write_segment(self, segment_num, data):
        """Write data to a specific segment file."""
        segment_path = self.get_segment_path(segment_num)
        
        # Write using temporary file and atomic replacement
        temp_path = f"{segment_path}.tmp"
        with open(temp_path, 'w') as f:
            json.dump(data, f)
        
        # Atomic replacement
        if os.path.exists(segment_path):
            os.replace(temp_path, segment_path)
        else:
            os.rename(temp_path, segment_path)
            
        # Update segment count if we have episode data
        for key, value in data.items():
            if isinstance(value, list):
                self.segment_counts[segment_num] = len(value)
                break
        
        return True
    
    def read_segment(self, segment_num):
        """Read data from a specific segment file."""
        segment_path = self.get_segment_path(segment_num)
        if not os.path.exists(segment_path):
            return None
            
        try:
            with open(segment_path, 'r') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            logging.error(f"Error reading segment {segment_num}: {str(e)}")
            return None
    
    def get_all_segments(self):
        """Get a list of all valid segment numbers."""
        return sorted(self.segment_counts.keys())
    
    def append_data(self, key, values):
        """
        Append values to the specified key in the current segment.
        If the segment is full, create a new segment.
        """
        # Get or create current segment data
        current_data = self.read_segment(self.current_segment) or {}
        
        # Initialize the key if it doesn't exist
        if key not in current_data:
            current_data[key] = []
        
        # Append values
        current_data[key].extend(values)
        
        # Check if we need to start a new segment
        for k, v in current_data.items():
            if isinstance(v, list) and len(v) >= self.max_segment_size:
                # Save current segment
                success = self.write_segment(self.current_segment, current_data)
                if success:
                    # Start a new segment
                    self.current_segment += 1
                    return self.append_data(key, [])  # Create the new segment with an empty list
                break
        
        # Save the updated segment
        return self.write_segment(self.current_segment, current_data)
    
    def read_all_data(self):
        """
        Read and combine data from all segments.
        This operation can be memory-intensive for large datasets.
        """
        combined_data = {}
        
        for segment_num in self.get_all_segments():
            segment_data = self.read_segment(segment_num)
            if not segment_data:
                continue
                
            # Combine data from this segment
            for key, values in segment_data.items():
                if key not in combined_data:
                    combined_data[key] = []
                
                if isinstance(values, list):
                    combined_data[key].extend(values)
        
        return combined_data


class Logger:
    """
    Lightweight logger for tracking training statistics.
    
    Focuses on memory efficiency and I/O reliability for long training sessions.
    
    輕量化訓練統計記錄器。
    
    專注於長時間訓練過程中的內存效率和I/O可靠性。
    """
    
    def __init__(self, log_dir=None, experiment_name=None):
        """
        Initialize the lightweight logger.
        
        Args:
            log_dir (str, optional): Directory for storing logs
            experiment_name (str, optional): Name for this experiment
        """
        # Configure basic logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Set up log directory
        if log_dir is None:
            log_dir = config.LOG_DIR
        
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"dqn_per_{timestamp}"
        
        self.log_dir = os.path.join(log_dir, experiment_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Use circular buffer for in-memory storage to limit memory usage
        self.episode_rewards = deque(maxlen=config.DEFAULT_BUFFER_SIZE)
        self.episode_lengths = deque(maxlen=config.DEFAULT_BUFFER_SIZE)
        self.eval_rewards = []  # List of tuples (episode, reward) - typically small
        
        # Training state
        self.episode_counter = 0
        self.latest_episode = 0
        self.total_steps = 0
        self.best_eval_reward = float('-inf')
        
        # Incremental update tracking
        self.last_saved_episode = 0
        self.data_modified = False
        
        # File paths
        self.stats_base_path = os.path.join(self.log_dir, 'stats')
        self.recovery_path = os.path.join(self.log_dir, 'recovery_info.json')
        self.backup_dir = os.path.dirname(log_dir)
        
        # Create segmented file manager for episode data
        self.segment_mgr = SegmentedFileManager(
            self.stats_base_path, 
            prefix="episodes_", 
            max_segment_size=1000  # Store 1000 episodes per segment
        )
        
        # Create async writer
        self.async_writer = AsyncWriter()
        
        # Memory management settings
        self.memory_check_interval = config.MEMORY_CHECK_INTERVAL  # 使用統一配置的內存檢查間隔
        self.last_memory_check = time.time()
        self.memory_threshold = config.MEMORY_THRESHOLD_PERCENT  # 使用統一配置的內存閾值
        
        # Timers
        self.start_time = time.time()
        self.episode_start_time = self.start_time
        self.last_write_time = self.start_time
        self.last_recovery_save_time = self.start_time  # 添加恢復信息保存計時器
        
        # Dynamic writing frequency (adjusted based on memory pressure)
        self.min_write_interval = config.MIN_WRITE_INTERVAL
        self.max_write_interval = config.MAX_WRITE_INTERVAL
        self.current_write_interval = config.DEFAULT_WRITE_INTERVAL
        self.recovery_save_interval = config.RECOVERY_SAVE_INTERVAL  # 添加恢復信息保存間隔
        
        # Buffer size management
        self.buffer_size = config.DEFAULT_BUFFER_SIZE  # 初始緩衝區大小
        self.min_buffer_size = config.MIN_BUFFER_SIZE
        self.max_buffer_size = config.MAX_BUFFER_SIZE
        
        logging.info(f"Enhanced logger initialized at {self.log_dir}")
        logging.info(f"Using segmented files and async writing for memory efficiency")
    
    def _adjust_write_interval(self, memory_usage):
        """
        Dynamically adjust writing interval based on memory pressure.
        
        Args:
            memory_usage (float): Current memory usage percentage
        """
        if memory_usage > self.memory_threshold:
            # High memory usage - write more frequently
            self.current_write_interval = max(
                self.min_write_interval,
                self.current_write_interval * 0.5  # Cut interval in half
            )
            self.buffer_size = max(self.min_buffer_size, self.buffer_size // 2)  # Reduce buffer size
            
            # Update deque max sizes
            new_episode_rewards = deque(self.episode_rewards, maxlen=self.buffer_size)
            new_episode_lengths = deque(self.episode_lengths, maxlen=self.buffer_size)
            
            self.episode_rewards = new_episode_rewards
            self.episode_lengths = new_episode_lengths
            
            logging.info(f"High memory usage ({memory_usage:.1f}%) - "
                        f"decreased write interval to {self.current_write_interval:.1f}s, "
                        f"buffer size to {self.buffer_size}")
        elif memory_usage < self.memory_threshold * 0.7:
            # Low memory usage - can write less frequently
            self.current_write_interval = min(
                self.max_write_interval,
                self.current_write_interval * 1.2  # Increase interval by 20%
            )
            self.buffer_size = min(self.max_buffer_size, self.buffer_size * 2)  # Increase buffer size
            
            # Update deque max sizes
            new_episode_rewards = deque(self.episode_rewards, maxlen=self.buffer_size)
            new_episode_lengths = deque(self.episode_lengths, maxlen=self.buffer_size)
            
            self.episode_rewards = new_episode_rewards
            self.episode_lengths = new_episode_lengths
            
            logging.info(f"Low memory usage ({memory_usage:.1f}%) - "
                        f"increased write interval to {self.current_write_interval:.1f}s, "
                        f"buffer size to {self.buffer_size}")
    
    def _check_memory_usage(self):
        """
        Check current memory usage and adjust writing parameters if necessary.
        Returns the current memory usage percentage.
        """
        now = time.time()
        if now - self.last_memory_check < self.memory_check_interval:
            return None
            
        self.last_memory_check = now
        
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            system_memory = psutil.virtual_memory()
            
            # Calculate memory usage
            process_mb = memory_info.rss / (1024 * 1024)
            memory_percent = system_memory.percent
            
            # Log memory usage
            logging.info(f"Memory usage: {process_mb:.1f} MB, System: {memory_percent:.1f}%")
            
            # Adjust parameters based on memory usage
            self._adjust_write_interval(memory_percent)
            
            return memory_percent
        except ImportError:
            logging.warning("psutil not available - cannot monitor memory usage")
            return None
    
    def _should_write_now(self, force=False):
        """
        Determine if we should write data to disk now based on:
        1. Time since last write
        2. Memory pressure
        3. Force flag (for critical data)
        """
        now = time.time()
        
        # Always write if forced (for critical data like evaluation results)
        if force:
            return True
            
        # Check memory and potentially adjust writing interval
        memory_percent = self._check_memory_usage()
        
        # Write if we've exceeded the current time interval
        time_based = now - self.last_write_time >= self.current_write_interval
        
        # Write if memory usage is high (over threshold)
        memory_based = memory_percent is not None and memory_percent > self.memory_threshold
        
        return time_based or memory_based
    
    def log_episode(self, episode, reward, length, epsilon):
        """
        Log essential statistics for a training episode.
        
        Args:
            episode (int): Episode number
            reward (float): Episode total reward
            length (int): Episode length in steps
            epsilon (float): Current exploration rate
        """
        # Update state
        now = time.time()
        duration = now - self.episode_start_time
        self.episode_start_time = now
        self.episode_counter = episode
        self.data_modified = True
        
        # Store in circular buffer
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        
        # Stream-write data periodically or based on memory pressure
        if self._should_write_now():
            self._write_episode_data()
            
        # Print progress periodically
        if episode % 10 == 0:
            elapsed = now - self.start_time
            avg_reward = np.mean(list(self.episode_rewards)[-10:]) if len(self.episode_rewards) >= 10 else np.mean(list(self.episode_rewards))
            print(f"Episode {episode} | Reward: {reward:.1f} | Steps: {length} | "
                  f"Eps: {epsilon:.4f} | Avg10: {avg_reward:.1f} | Time: {elapsed:.0f}s", 
                  flush=True)
    
    @retry_on_io_error(max_retries=5, delay=2.0, backoff_factor=1.5)
    def _write_episode_data(self):
        """
        Write accumulated episode data to disk using segmented files.
        Uses incremental updates to only write new data.
        
        Returns:
            bool: Whether the operation was successful
        """
        if not self.data_modified:
            return True
            
        # Prepare episode data in the expected format for segment manager
        formatted_episodes = [
            {'reward': r, 'length': l, 'episode': self.last_saved_episode + i + 1} 
            for i, (r, l) in enumerate(zip(
                list(self.episode_rewards), 
                list(self.episode_lengths)
            ))
        ]
        
        # Directly write to segments (we're already in the retry-decorated method)
        success = self.segment_mgr.append_data('episodes', formatted_episodes)
        
        if success:
            # Update tracking state
            self.last_write_time = time.time()
            self.last_saved_episode = self.episode_counter
            self.data_modified = False
            
            # Force garbage collection after writing
            gc.collect()
            
            return True
        else:
            logging.error("Failed to write episode data to segments")
            return False
    
    @retry_on_io_error(max_retries=5, delay=2.0, backoff_factor=1.5)
    def _write_evaluation_data(self, force=False):
        """Write evaluation data to a separate file."""
        if not self.eval_rewards and not force:
            return
            
        eval_path = os.path.join(self.log_dir, 'eval_data.json')
        
        # Prepare data
        eval_data = {
            'eval_episodes': [ep for ep, _ in self.eval_rewards],
            'eval_rewards': [reward for _, reward in self.eval_rewards],
            'best_eval_reward': float(self.best_eval_reward) if self.best_eval_reward != float('-inf') else None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Write the file using atomic file writing pattern
        temp_path = f"{eval_path}.tmp"
        with open(temp_path, 'w') as f:
            json.dump(eval_data, f)
            
        # Atomic replacement
        if os.path.exists(eval_path):
            os.replace(temp_path, eval_path)
        else:
            os.rename(temp_path, eval_path)
            
        logging.info(f"Evaluation data saved to {eval_path}")
        return True
    
    def get_stats(self):
        """
        Get essential statistics for visualization.
        Reads from segmented files and merges with in-memory data.
        
        Returns:
            dict: Formatted statistics
        """
        # Read all data from segments (this could be memory-intensive)
        segment_data = self.segment_mgr.read_all_data()
        
        # Extract episode data
        episode_data = []
        if 'episodes' in segment_data:
            episode_data = sorted(segment_data['episodes'], key=lambda x: x.get('episode', 0))
        
        # Read evaluation data
        eval_path = os.path.join(self.log_dir, 'eval_data.json')
        eval_data = {}
        if os.path.exists(eval_path):
            try:
                with open(eval_path, 'r') as f:
                    eval_data = json.load(f)
            except (IOError, json.JSONDecodeError):
                pass
        
        # Create current in-memory stats
        in_memory_rewards = list(self.episode_rewards)
        in_memory_lengths = list(self.episode_lengths)
        
        # Combine data for return
        combined_rewards = []
        combined_lengths = []
        
        # Add data from segments
        if episode_data:
            combined_rewards.extend([ep.get('reward', 0) for ep in episode_data])
            combined_lengths.extend([ep.get('length', 0) for ep in episode_data])
        
        # Add in-memory data, avoiding duplicates
        last_episode_in_segments = episode_data[-1].get('episode', 0) if episode_data else 0
        if self.episode_counter > last_episode_in_segments:
            # Calculate how many episodes from memory need to be added
            episodes_to_add = self.episode_counter - last_episode_in_segments
            episodes_to_add = min(episodes_to_add, len(in_memory_rewards))
            
            # Add the most recent in-memory episodes
            combined_rewards.extend(in_memory_rewards[-episodes_to_add:])
            combined_lengths.extend(in_memory_lengths[-episodes_to_add:])
        
        # Get evaluation data
        eval_episodes = eval_data.get('eval_episodes', [])
        if not eval_episodes:
            eval_episodes = [ep for ep, _ in self.eval_rewards]
            eval_rewards = [reward for _, reward in self.eval_rewards]
        else:
            eval_rewards = eval_data.get('eval_rewards', [])
            
            # Add any in-memory evaluation data that's newer
            if self.eval_rewards:
                max_saved_ep = max(eval_episodes) if eval_episodes else 0
                new_evals = [(ep, reward) for ep, reward in self.eval_rewards if ep > max_saved_ep]
                
                if new_evals:
                    new_episodes, new_rewards = zip(*new_evals)
                    eval_episodes.extend(new_episodes)
                    eval_rewards.extend(new_rewards)
        
        # Return formatted stats
        return {
            'episode_rewards': combined_rewards,
            'episode_lengths': combined_lengths,
            'eval_episodes': eval_episodes,
            'eval_rewards': eval_rewards,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'total_episodes': self.episode_counter
        }
    
    @retry_on_io_error(max_retries=5, delay=3.0, backoff_factor=2.0)
    def _save_recovery_info(self):
        """
        Save minimal recovery information for resuming training.
        
        Returns:
            bool: Whether save was successful
        """
        recovery_info = {
            'latest_episode': self.latest_episode,
            'total_steps': self.total_steps,
            'best_eval_reward': float(self.best_eval_reward) if self.best_eval_reward != float('-inf') else None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'elapsed_time': time.time() - self.start_time
        }
        
        try:
            # Use a temporary file to prevent corruption
            temp_path = f"{self.recovery_path}.tmp"
            with open(temp_path, 'w') as f:
                json.dump(recovery_info, f)
                
            # Atomic replacement
            if os.path.exists(self.recovery_path):
                os.replace(temp_path, self.recovery_path)
            else:
                os.rename(temp_path, self.recovery_path)
                
            logging.info(f"Recovery info saved to {self.recovery_path}")
            return True
            
        except (IOError, OSError) as e:
            logging.error(f"Error saving recovery info: {str(e)}")
            # Try emergency backup
            try:
                backup_path = os.path.join(self.backup_dir, f"dqn_recovery_{int(time.time())}.json")
                with open(backup_path, 'w') as f:
                    json.dump(recovery_info, f)
                logging.info(f"Emergency recovery backup saved to {backup_path}")
                return True
            except Exception as backup_e:
                logging.error(f"Failed to create emergency backup: {str(backup_e)}")
                return False
    
    def update_training_progress(self, episode, total_steps, best_reward):
        """
        Update training progress information for recovery.
        
        Args:
            episode (int): Current episode number
            total_steps (int): Total steps taken in training so far
            best_reward (float): Best evaluation reward achieved
            
        Returns:
            bool: Whether the update was successful
        """
        # Update internal state
        self.latest_episode = episode
        self.total_steps = total_steps
        if best_reward > self.best_eval_reward:
            self.best_eval_reward = best_reward
        
        # 檢查距離上次保存恢復信息的時間間隔
        now = time.time()
        time_to_save = now - self.last_recovery_save_time >= self.recovery_save_interval
        
        # 只有在滿足時間間隔或有重要更新（如最佳獎勵變化）時才保存
        force_save = best_reward > self.best_eval_reward
        
        if time_to_save or force_save:
            self.last_recovery_save_time = now
            return self.save_recovery_info()
        
        return True  # 不需要保存也視為成功
    
    def save_recovery_info(self):
        """
        Public method to save recovery information.
        Persists current training state for resuming later.
        
        Returns:
            bool: Whether the save was successful
        """
        return self._save_recovery_info()
    
    def log_training_step(self, step, loss):
        """
        Log statistics for an individual training step.
        
        Args:
            step (int): Current training step
            loss (float): Loss value from the current optimization step
        """
        # Convert loss to float if it's a tensor
        loss_value = float(loss) if hasattr(loss, 'item') else loss
        
        # Only log every N steps to avoid I/O overhead
        log_frequency = config.TRAINING_STEP_LOG_FREQUENCY if hasattr(config, 'TRAINING_STEP_LOG_FREQUENCY') else 100
        
        # If we should log this step
        if step % log_frequency == 0:
            # Prepare data entry for this step
            step_data = {
                'step': step,
                'loss': loss_value,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Write asynchronously
            self.async_writer.write(self._append_training_step_data, step_data)
            
            # Print periodic update (similar to log_episode style)
            if step % (log_frequency * 10) == 0:
                elapsed = time.time() - self.start_time
                print(f"Training step {step} | Loss: {loss_value:.6f} | Time: {elapsed:.0f}s", flush=True)
    
    @retry_on_io_error(max_retries=5, delay=2.0, backoff_factor=1.5)
    def _append_training_step_data(self, step_data):
        """
        Append training step data to a file with error handling.
        
        Args:
            step_data (dict): Data for the current training step
            
        Returns:
            bool: Whether the operation was successful
        """
        steps_path = os.path.join(self.log_dir, 'training_steps.json')
        
        # Create file if it doesn't exist
        if not os.path.exists(steps_path):
            # Initialize with an empty structure
            with open(steps_path, 'w') as f:
                json.dump({'steps': []}, f)
        
        # Read the current file
        with open(steps_path, 'r') as f:
            data = json.load(f)
            
        # Ensure the structure is correct
        if 'steps' not in data:
            data['steps'] = []
        
        # Append the new step data
        data['steps'].append(step_data)
        
        # Keep only the last N entries to avoid file size blowup
        max_entries = config.MAX_TRAINING_STEP_ENTRIES if hasattr(config, 'MAX_TRAINING_STEP_ENTRIES') else 1000
        if len(data['steps']) > max_entries:
            data['steps'] = data['steps'][-max_entries:]
        
        # Write using the atomic pattern
        temp_path = f"{steps_path}.tmp"
        with open(temp_path, 'w') as f:
            json.dump(data, f)
            
        # Atomic replacement
        if os.path.exists(steps_path):
            os.replace(temp_path, steps_path)
        else:
            os.rename(temp_path, steps_path)
            
        return True
    
    def save_data(self):
        """
        Force save all data to disk.
        Use this to explicitly save current state regardless of auto-save settings.
        
        Returns:
            bool: Whether all operations were successful
        """
        success = True
        
        # Write episode data if modified
        if self.data_modified:
            self._write_episode_data()
        
        # Write evaluation data
        if self.eval_rewards:
            self._write_evaluation_data(force=True)
        
        # Force save recovery info
        recovery_success = self.save_recovery_info()
        success = success and recovery_success
        
        logging.info("Data explicitly saved to disk" if success else "Some data could not be saved")
        return success
        
    def log_priorities(self, priorities):
        """
        Log priority values from the PER memory.
        
        Args:
            priorities (list): List of priority values from PER memory
        """
        # Store priorities for plotting
        # We'll just use the latest batch since storing all would be too much
        # Sample a subset if the array is too large
        if len(priorities) > 10000:
            indices = np.random.choice(len(priorities), 10000, replace=False)
            priorities = priorities[indices]
            
        # Store in a separate file
        priorities_path = os.path.join(self.log_dir, 'priorities.json')
        
        # Create a simple dict with the priorities and timestamp
        data = {
            'priorities': priorities.tolist() if isinstance(priorities, np.ndarray) else priorities,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'episode': self.episode_counter
        }
        
        # Write asynchronously
        self.async_writer.write(self._write_priorities_data, data)
    
    @retry_on_io_error(max_retries=5, delay=2.0, backoff_factor=1.5)
    def _write_priorities_data(self, data):
        """
        Write priorities data to a file with error handling.
        
        Args:
            data (dict): Priority data to write
            
        Returns:
            bool: Whether the operation was successful
        """
        priorities_path = os.path.join(self.log_dir, 'priorities.json')
        
        # Use atomic file write pattern
        temp_path = f"{priorities_path}.tmp"
        with open(temp_path, 'w') as f:
            json.dump(data, f)
            
        # Atomic replacement
        if os.path.exists(priorities_path):
            os.replace(temp_path, priorities_path)
        else:
            os.rename(temp_path, priorities_path)
            
        logging.info(f"Priorities data saved to {priorities_path}")
        return True
    
    def __del__(self):
        """Clean up resources when the logger is destroyed."""
        try:
            # Final save
            self.save_data()
            
            # Stop async writer
            if hasattr(self, 'async_writer'):
                self.async_writer.stop()
        except:
            pass