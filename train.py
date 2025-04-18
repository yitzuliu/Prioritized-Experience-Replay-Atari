"""
Training script for DQN with Prioritized Experience Replay on Atari Ice Hockey.

This script performs the main training loop for the DQN agent with PER,
handling environment interactions, neural network updates, and logging.

用於 Atari 冰球遊戲的 DQN 優先經驗回放訓練腳本。

此腳本執行 DQN 智能體的主要訓練循環，處理環境交互、神經網絡更新和日誌記錄。

Pseudocode for DQN with PER Training Loop:
1. Initialize environment and agent
   - Create environment with appropriate wrappers
   - Initialize DQN agent with primary and target networks
   - Set up Prioritized Experience Replay memory

2. Training loop (for each episode):
   a. Reset environment to initial state
   
   b. For each step in episode:
      i. Select action using ε-greedy policy
      ii. Execute action, observe reward and next state
      iii. Store transition in replay memory
      iv. Optimize model if enough samples are collected
      v. Update target network periodically
   
   c. Evaluate agent performance periodically
   d. Save checkpoints and log statistics
"""

import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime
import time
import sys
import gc  # 添加垃圾回收模塊
import psutil  # 用於記憶體監控

import config
from src.environment import make_atari_env
from src.agent import DQNAgent
from src.utils.device_utils import get_device, get_system_info
from src.utils.logger import Logger
from src.utils.visualization import save_training_plots


def parse_args():
    """
    Parse command line arguments.
    解析命令行參數。
    """
    parser = argparse.ArgumentParser(description="Train DQN with PER on Atari Ice Hockey")
    
    parser.add_argument("--episodes", type=int, default=config.TRAINING_EPISODES,
                        help="Number of training episodes")
    parser.add_argument("--use-per", type=bool, default=config.USE_PER,
                        help="Whether to use Prioritized Experience Replay")
    parser.add_argument("--alpha", type=float, default=config.ALPHA,
                        help="Priority exponent for PER")
    parser.add_argument("--beta", type=float, default=config.BETA_START,
                        help="Initial importance sampling weight for PER")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during training")
    parser.add_argument("--eval-frequency", type=int, default=config.EVAL_FREQUENCY,
                        help="How often to evaluate the agent")
    parser.add_argument("--eval-episodes", type=int, default=config.EVAL_EPISODES,
                        help="Number of episodes for evaluation")
    parser.add_argument("--save-dir", type=str, default=config.CHECKPOINT_DIR,
                        help="Directory to save model checkpoints")
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint file to resume training from")
    parser.add_argument("--start-episode", type=int, default=1,
                        help="Episode to start from when resuming training")
    
    return parser.parse_args()


def create_directories():
    """Create necessary directories for saving results."""
    os.makedirs(config.RESULT_DIR, exist_ok=True)  # Create main result directory first
    os.makedirs(config.LOG_DIR, exist_ok=True)
    # os.makedirs(config.MODEL_DIR, exist_ok=True)  
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    # os.makedirs(config.DATA_DIR, exist_ok=True)


def evaluate_agent(agent, env, num_episodes=10):
    """
    Evaluate the agent performance over several episodes.
    
    Args:
        agent: The DQN agent
        env: The environment
        num_episodes: Number of evaluation episodes
    
    Returns:
        tuple: (average_reward, episode_rewards, episode_lengths)
    
    評估智能體在多個回合中的表現。
    
    參數：
        agent: DQN 智能體
        env: 環境
        num_episodes: 評估回合數
        
    返回：
        tuple: (平均獎勵, 回合獎勵列表, 回合長度列表)
    """
    print(f"\nEvaluating agent for {num_episodes} episodes...")
    episode_rewards = []
    episode_lengths = []
    
    for i in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        truncated = False
        
        while not (done or truncated):
            # Select action using greedy policy (no exploration)
            action = agent.select_action(state, evaluate=True)
            
            # Take action in environment
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Update state and statistics
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {i+1}/{num_episodes}: Reward = {episode_reward}, Length = {episode_length}")
    
    avg_reward = np.mean(episode_rewards)
    print(f"Evaluation complete. Average reward: {avg_reward:.2f}")
    
    return avg_reward, episode_rewards, episode_lengths


def train(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
        
    主訓練函數。
    
    參數：
        args: 命令行參數
    """
    # Print system information
    system_info = get_system_info()
    print("System Information:")
    print(f"OS: {system_info['os']} {system_info['os_version']}")
    print(f"CPU: {system_info['cpu_type']} ({system_info['cpu_count']} cores)")
    print(f"PyTorch: {system_info['torch_version']}")
    
    if system_info.get('cuda_available', False) and not args.cpu:
        print(f"GPU: {system_info.get('gpu_name', 'Unknown')} ({system_info.get('gpu_memory_gb', 'Unknown')} GB)")
    elif system_info.get('mps_available', False) and not args.cpu:
        print("GPU: Apple Silicon (Metal)")
    else:
        print("GPU: None detected, using CPU only")
    print("-" * 40)
    
    # Set up device (CPU or GPU)
    if args.cpu:
        device = torch.device("cpu")
        config.AUTO_DETECT_DEVICE = False
        print("Forcing CPU usage as specified")
    else:
        device = get_device()
    
    # Create training environment
    render_mode = "human" if args.render else None
    env = make_atari_env(render_mode=render_mode)
    print(f"Environment: {config.ENV_NAME}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Create evaluation environment (no rendering for speed)
    eval_env = make_atari_env(render_mode=None)
    
    # Initialize DQN agent
    agent = DQNAgent(device=device, use_per=args.use_per)
    
    # Variables for training loop
    episode_count = 0
    total_steps = 0
    best_eval_reward = float('-inf')
    
    # 初始化 start_episode
    start_episode = (args.start_episode + 1) if args.resume else 1
    
    # Memory management variables
    last_memory_check = time.time()
    memory_check_interval = config.memory_check_interval  # 
    memory_threshold_percent = config.memory_threshold_percent  # 
    peak_memory_usage = 0
    
    # Resume training if checkpoint provided
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}...")
            checkpoint = agent.load(args.resume)
            if isinstance(checkpoint, dict) and 'total_steps' in checkpoint:
                total_steps = checkpoint['total_steps']
                best_eval_reward = checkpoint.get('best_eval_reward', float('-inf'))
                print(f"Resumed from step {total_steps} with best reward {best_eval_reward:.2f}")
            else:
                print("Loaded model weights only, no training statistics available")
        else:
            print(f"Warning: Checkpoint file {args.resume} not found. Starting training from scratch.")
    else:
        print("Starting new training run")
    
    # Set up logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"dqn_{'per' if args.use_per else 'uniform'}_{timestamp}"
    if args.resume:
        # If resuming, try to use the same experiment name to continue the logs
        base_name = os.path.basename(args.resume).split('_ep')[0]
        if base_name.startswith("dqn_"):
            experiment_name = base_name
            
    logger = Logger(log_dir=config.LOG_DIR, experiment_name=experiment_name)
    
    # 訓練循環開始前更新恢復信息
    logger.update_training_progress(
        episode=start_episode-1, 
        total_steps=total_steps, 
        best_reward=best_eval_reward
    )
    
    # Training loop - main algorithm
    print("\nStarting training...\n")
    # Variables to track training progress
    training_start_time = time.time()
    progress_update_frequency = 10  # Update progress every 10 episodes

    try:
        for episode in range(start_episode, args.episodes + 1):
            # 2a. Reset environment to initial state
            episode_count = episode
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            truncated = False
            
            # Calculate and display overall progress percentage
            progress_pct = (episode - start_episode) / (args.episodes - start_episode + 1) * 100
            elapsed_time = time.time() - training_start_time
            if episode == start_episode or episode % progress_update_frequency == 0:
                est_total_time = elapsed_time / max(0.001, (episode - start_episode)) * (args.episodes - start_episode + 1)
                est_remaining = max(0, est_total_time - elapsed_time)
                hours, remainder = divmod(est_remaining, 3600)
                minutes, seconds = divmod(remainder, 60)
                print(f"\rOverall Progress: {progress_pct:.1f}% | Est. remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s", 
                      end="", flush=True)
                print()  # 添加這一行使得進度和下一行Episode輸出分開
            
            # 2b. For each step in episode:
            while not (done or truncated):
                # i. Select action using ε-greedy policy
                action = agent.select_action(state)
                
                # ii. Execute action, observe reward and next state
                next_state, reward, done, truncated, _ = env.step(action)
                
                # iii. Store transition in replay memory
                agent.store_transition(state, action, reward, next_state, done)
                
                # iv. Optimize model if enough samples are collected
                state = next_state
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                if total_steps % config.UPDATE_FREQUENCY == 0 and total_steps >= config.LEARNING_STARTS:
                    loss = agent.optimize_model()
                    
                    # Log training step
                    logger.log_training_step(total_steps, loss)
                
                # v. Update target network periodically
                if agent.should_update_target_network():
                    agent.update_target_network()
                    print(f"Updated target network at step {agent.steps_done}")
                
                # Log priority distribution occasionally
                if args.use_per and total_steps % 10000 == 0:
                    priorities = agent.memory.tree.get_leaf_values()
                    logger.log_priorities(priorities)
            
            # 修改：基於記憶體使用百分比的清理機制
            current_time = time.time()
            if current_time - last_memory_check > memory_check_interval:
                # 獲取當前記憶體使用情況
                process = psutil.Process(os.getpid())
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # 獲取系統總記憶體
                system_memory = psutil.virtual_memory()
                total_memory = system_memory.total / 1024 / 1024  # MB
                
                # 計算進程記憶體使用百分比
                memory_percent = system_memory.percent
                process_memory_percent = (current_memory / total_memory) * 100
                
                # 更新峰值記憶體使用
                if current_memory > peak_memory_usage:
                    peak_memory_usage = current_memory
                
                print(f"\nMemory check: Process using {current_memory:.1f}MB ({process_memory_percent:.1f}%), "
                      f"System memory: {memory_percent:.1f}% used.")
                
                # 如果記憶體使用率超過閾值，執行清理
                if memory_percent > memory_threshold_percent or process_memory_percent > memory_threshold_percent / 2:
                    # 強制進行垃圾回收
                    gc.collect()
                    
                    # 清空PyTorch的CUDA快取（如果使用）
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # 清空MPS快取（如果在Apple Silicon上）
                    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        # MPS沒有直接的清除函數，但可以釋放一些不使用的變數
                        torch.mps.empty_cache()
                    
                    # 記錄記憶體情況
                    post_cleanup_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_freed = max(0, current_memory - post_cleanup_memory)
                    
                    print(f"Memory cleanup performed: {memory_freed:.1f}MB freed. "
                          f"Current usage: {post_cleanup_memory:.1f}MB, Peak: {peak_memory_usage:.1f}MB")
                    
                last_memory_check = current_time
            
            # 更新恢復信息
            logger.update_training_progress(episode, total_steps, best_eval_reward)
            
            # c. Evaluate agent performance periodically
            # Log episode statistics
            # Get current epsilon from agent
            epsilon = max(config.EPSILON_END, config.EPSILON_START - 
                          (config.EPSILON_START - config.EPSILON_END) * 
                          min(1.0, agent.steps_done / config.EPSILON_DECAY))
            logger.log_episode(episode, episode_reward, episode_length, epsilon)
            # Evaluate agent periodically
            if episode % args.eval_frequency == 0:
                avg_reward, eval_rewards, eval_lengths = evaluate_agent(
                    agent, eval_env, num_episodes=args.eval_episodes
                )
                logger.log_evaluation(episode, eval_rewards, eval_lengths)
                
                # Save best model
                if avg_reward > best_eval_reward:
                    best_eval_reward = avg_reward
                    model_path = os.path.join(args.save_dir, f"{experiment_name}_best.pth")
                    agent.save(model_path)
                    print(f"New best model saved with reward: {best_eval_reward:.2f}")
            
            # d. Save checkpoints and log statistics
            # Save checkpoint periodically
            if episode % config.SAVE_FREQUENCY == 0:
                checkpoint_path = os.path.join(args.save_dir, f"{experiment_name}_ep{episode}.pth")
                agent.save(checkpoint_path)
                print(f"Checkpoint saved at episode {episode}")
                
                # Also save current statistics and generate plots
                logger.save_stats()
                # 保存恢復信息
                logger.save_recovery_info()
                save_training_plots(logger.get_stats(), run_name=experiment_name)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user. Saving final state...")
    except Exception as e:
        print(f"\n\nError occurred during training: {e}")
        import traceback
        traceback.print_exc()
        print("Attempting to save current state before exiting...")
    finally:
        # 最終記憶體使用情況報告
        if 'process' in locals():
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            print(f"\nFinal memory usage: {final_memory:.1f}MB, Peak: {peak_memory_usage:.1f}MB")
        
        # Always execute cleanup and save, even if an error occurs
        print("\nSaving final model and statistics...")
        
        # Save final model
        final_path = os.path.join(args.save_dir, f"{experiment_name}_final.pth")
        try:
            # Save with additional training statistics 
            agent.save(final_path, {
                'total_steps': total_steps,
                'best_eval_reward': best_eval_reward,
                'episode_count': episode_count
            })
            print(f"Final model saved to {final_path}")
            
            # Save statistics and generate plots
            logger.save_stats()
            # 保存最終恢復信息
            logger.save_recovery_info()
            results_dir = save_training_plots(logger.get_stats(), run_name=experiment_name, output_dir=config.PLOT_DIR)
            
            print("\nTraining summary:")
            print(f"Trained for {episode_count} episodes, {total_steps} total steps")
            if best_eval_reward > float('-inf'):
                print(f"Best evaluation reward: {best_eval_reward:.2f}")
            else:
                print("No evaluation was performed during this run")
            print(f"Results and plots saved to {results_dir}")
        except Exception as e:
            print(f"Error during final saving: {e}")
            print("Training data may be incomplete.")
        
        # Clean up
        try:
            env.close()
            eval_env.close()
        except:
            pass
            
        # Force flush stdout to ensure all messages are displayed
        sys.stdout.flush()


if __name__ == "__main__":
    # Set up directories
    create_directories()
    
    # Initialize device (it will be obtained when needed, not storing in config)
    from src.utils.device_utils import get_device
    
    # Parse command line arguments
    args = parse_args()
    
    # Start training
    train(args)