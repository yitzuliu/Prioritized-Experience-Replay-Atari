"""
Resume script for DQN with Prioritized Experience Replay.

This script loads a saved training state and continues training from a checkpoint.
It allows for seamless resume of interrupted training, with options to modify
training parameters.

DQN 優先經驗回放的恢復訓練腳本。

此腳本加載保存的訓練狀態並從檢查點繼續訓練。
它允許無縫恢復中斷的訓練，並提供修改訓練參數的選項。
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import torch
from datetime import datetime
import signal

# Add parent directory to path to import config.py
# 將父目錄添加到路徑以導入 config.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from src.env_wrappers import make_atari_env
from src.dqn_agent import DQNAgent
from src.logger import Logger
from src.visualization import Visualizer
from src.device_utils import get_device, get_system_info

# Import training functions from train.py
# 從 train.py 導入訓練函數
from train import train_one_episode, evaluate_agent, check_resources, handle_interrupt

# Global variables for interrupt handling
# 全局變量，用於中斷處理
global agent, logger, model_dir, visualizer
agent = None
logger = None
model_dir = None
visualizer = None

def parse_arguments():
    """
    Parse command line arguments for resuming training.
    
    解析命令行參數設置恢復訓練
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Resume DQN training with Prioritized Experience Replay')
    parser.add_argument('--experiment_name', type=str, required=True,
                        help='Name of the experiment to resume')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Checkpoint name to load (e.g., checkpoint_100.pt) - if empty, latest will be used')
    parser.add_argument('--additional_episodes', type=int, default=config.TRAINING_EPISODES,
                        help=f'Additional episodes to train, default: {config.TRAINING_EPISODES}')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate, if not provided, uses the one from original training')
    parser.add_argument('--no_per', action='store_true',
                        help='Disable Prioritized Experience Replay, use standard replay')
    
    return parser.parse_args()

def find_latest_checkpoint(experiment_name):
    """
    Find the latest checkpoint for the given experiment.
    
    查找給定實驗的最新檢查點
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        str: Path to the latest checkpoint, or None if no checkpoint is found
        int: Episode number of the latest checkpoint
    """
    model_dir = os.path.join(config.MODEL_DIR, experiment_name)
    if not os.path.exists(model_dir):
        print(f"Error: Experiment directory '{model_dir}' not found!")
        return None, 0
    
    # Find checkpoint files
    # 查找檢查點文件
    checkpoint_files = [f for f in os.listdir(model_dir) if f.startswith("checkpoint_") and f.endswith(".pt")]
    
    if not checkpoint_files:
        # Check for interrupted model
        # 檢查中斷的模型
        if os.path.exists(os.path.join(model_dir, "interrupted_model.pt")):
            return os.path.join(model_dir, "interrupted_model.pt"), -1
        
        print(f"Error: No checkpoints found in '{model_dir}'!")
        return None, 0
    
    # Extract episode numbers
    # 提取回合號
    episode_numbers = []
    for ckpt in checkpoint_files:
        try:
            episode = int(ckpt.replace("checkpoint_", "").replace(".pt", ""))
            episode_numbers.append((episode, ckpt))
        except ValueError:
            continue
    
    if not episode_numbers:
        print(f"Error: Couldn't parse episode numbers from checkpoint filenames!")
        return None, 0
    
    # Find the latest
    # 查找最新的
    latest_episode, latest_ckpt = max(episode_numbers, key=lambda x: x[0])
    return os.path.join(model_dir, latest_ckpt), latest_episode

def load_experiment_config(experiment_name):
    """
    Load experiment configuration from file.
    
    從文件加載實驗配置
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        dict: Experiment configuration, or None if not found
    """
    config_path = os.path.join(config.MODEL_DIR, experiment_name, "experiment_config.json")
    
    if not os.path.exists(config_path):
        print(f"Error: Experiment configuration file '{config_path}' not found!")
        return None
    
    with open(config_path, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Couldn't parse experiment configuration file!")
            return None

def main():
    """
    Main function to resume training.
    
    恢復訓練的主函數
    """
    global agent, logger, model_dir, visualizer
    
    # Parse command line arguments
    # 解析命令行參數
    args = parse_arguments()
    
    # Set experiment name
    # 設置實驗名稱
    experiment_name = args.experiment_name
    model_dir = os.path.join(config.MODEL_DIR, experiment_name)
    
    print(f"Attempting to resume training for experiment: {experiment_name}")
    
    # Load experiment configuration
    # 加載實驗配置
    exp_config = load_experiment_config(experiment_name)
    if exp_config is None:
        return
    
    # Load checkpoint
    # 加載檢查點
    checkpoint_path = None
    start_episode = 0
    
    if args.checkpoint:
        checkpoint_path = os.path.join(model_dir, args.checkpoint)
        if not os.path.exists(checkpoint_path):
            print(f"Error: Specified checkpoint '{checkpoint_path}' not found!")
            return
        try:
            start_episode = int(args.checkpoint.replace("checkpoint_", "").replace(".pt", ""))
        except ValueError:
            start_episode = 0
            if args.checkpoint == "interrupted_model.pt":
                print("Resuming from interrupted training...")
            else:
                print("Couldn't determine episode number from checkpoint filename, starting from episode 1.")
    else:
        checkpoint_path, start_episode = find_latest_checkpoint(experiment_name)
        if checkpoint_path is None:
            return
    
    print(f"Resuming from checkpoint: {checkpoint_path}")
    
    # Determine device
    # 確定設備
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize environment
    # 初始化環境
    env_name = exp_config.get("env_name", config.ENV_NAME)
    env = make_atari_env(env_name=env_name, training=True)
    
    # Initialize agent and load model
    # 初始化智能體並加載模型
    learning_rate = args.learning_rate if args.learning_rate is not None else exp_config.get("learning_rate", config.LEARNING_RATE)
    use_per = not args.no_per if args.no_per else exp_config.get("use_per", True)
    
    agent = DQNAgent(
        state_shape=(config.FRAME_STACK, config.FRAME_HEIGHT, config.FRAME_WIDTH),
        action_space_size=env.action_space.n,
        learning_rate=learning_rate,
        use_per=use_per
    )
    
    # Load model parameters
    # 加載模型參數
    episode_number = agent.load_model(checkpoint_path)
    if episode_number > 0:
        start_episode = episode_number
    
    # Initialize logger with the original experiment name
    # 使用原始實驗名稱初始化記錄器
    logger = Logger(experiment_name=experiment_name)
    logger.load_training_state()
    
    # Setup interrupt handler
    # 設置中斷處理器
    signal.signal(signal.SIGINT, handle_interrupt)
    
    # Log resume information
    # 記錄恢復信息
    logger.log_text("\n" + "="*50)
    logger.log_text(f"Resuming training for experiment: {experiment_name}")
    logger.log_text(f"Checkpoint: {os.path.basename(checkpoint_path)}")
    logger.log_text(f"Starting from episode: {start_episode + 1}")
    logger.log_text(f"Additional episodes: {args.additional_episodes}")
    logger.log_text(f"Learning rate: {learning_rate}")
    logger.log_text(f"Using PER: {use_per}")
    logger.log_text(f"Device: {device}")
    logger.log_text(f"System info: {get_system_info()}")
    logger.log_text("="*50 + "\n")
    
    # Initialize training variables
    # 初始化訓練變量
    total_episodes = start_episode + args.additional_episodes
    total_steps = logger.get_total_steps()
    best_eval_reward = logger.get_best_eval_reward()
    start_time = time.time()
    
    try:
        # Main training loop
        # 主訓練循環
        for episode in range(start_episode + 1, total_episodes + 1):
            # Train one episode
            # 訓練一個回合
            episode_stats, steps_taken = train_one_episode(env, agent, logger, episode, total_steps)
            total_steps += steps_taken
            
            # Periodically output training progress
            # 定期輸出訓練進度
            if episode % config.LOGGER_DETAILED_INTERVAL == 0:
                elapsed_time = time.time() - start_time
                hrs, rem = divmod(elapsed_time, 3600)
                mins, secs = divmod(rem, 60)
                
                logger.log_text(
                    f"Episode: {episode}/{total_episodes} "
                    f"({episode/total_episodes*100:.1f}%) | "
                    f"Reward: {episode_stats['reward']:.2f} | "
                    f"Steps: {episode_stats['steps']} | "
                    f"Epsilon: {episode_stats['epsilon']:.4f} | "
                    f"Elapsed time (since resume): {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"
                )
            
            # Periodically evaluate
            # 定期評估
            if episode % config.EVAL_FREQUENCY == 0:
                eval_reward = evaluate_agent(env, agent, logger, episode)
                
                # Save best model
                # 保存最佳模型
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_model_path = os.path.join(model_dir, "best_model.pt")
                    agent.save_model(best_model_path)
                    logger.log_text(f"New best model! Mean reward: {best_eval_reward:.2f}")
            
            # Periodically save checkpoints
            # 定期保存檢查點
            if episode % config.SAVE_FREQUENCY == 0:
                checkpoint_path = os.path.join(model_dir, f"checkpoint_{episode}.pt")
                agent.save_model(checkpoint_path, episode=episode)
                logger.save_training_state()
                logger.log_text(f"Checkpoint saved (episode {episode})")
            
            # Periodically check resources
            # 定期檢查資源
            if episode % 10 == 0:
                resources = check_resources()
                if resources['memory_percent'] > config.MEMORY_THRESHOLD_PERCENT:
                    logger.log_text(f"Warning: Memory usage {resources['memory_percent']:.1f}% exceeds threshold")
                    logger.limit_memory_usage()
        
        # Training completed
        # 訓練完成
        logger.log_text("\nResumed training completed!")
        
        # Save final model
        # 保存最終模型
        final_model_path = os.path.join(model_dir, "final_model.pt")
        agent.save_model(final_model_path, episode=total_episodes)
        logger.log_text(f"Final model saved to: {final_model_path}")
        
        # Save training state
        # 保存訓練狀態
        logger.save_training_state()
        
        # Log training summary
        # 記錄訓練摘要
        total_time = time.time() - start_time
        hrs, rem = divmod(total_time, 3600)
        mins, secs = divmod(rem, 60)
        
        logger.log_text(
            f"Resumed training summary:\n"
            f"- Original starting episode: {start_episode + 1}\n"
            f"- Total episodes trained: {total_episodes - start_episode}\n"
            f"- Final episode: {total_episodes}\n"
            f"- Total steps: {total_steps}\n"
            f"- Best evaluation reward: {best_eval_reward:.2f}\n"
            f"- Total resumed training time: {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"
        )
        
        # Generate final visualizations
        # 生成最終可視化
        logger.log_text("Generating training visualizations...")
        visualizer = Visualizer(logger_instance=logger)
        plots = visualizer.generate_all_plots(show=False)
        logger.log_text(f"Generated {len(plots)} visualization plots")
        
    except Exception as e:
        # Exception handling
        # 異常處理
        logger.log_text(f"\nException occurred during resumed training: {str(e)}")
        
        # Save model at exception
        # 保存異常時的模型
        exception_model_path = os.path.join(model_dir, "exception_model.pt")
        agent.save_model(exception_model_path)
        logger.log_text(f"Model at exception saved to: {exception_model_path}")
        
        # Save training state
        # 保存訓練狀態
        logger.save_training_state()

if __name__ == "__main__":
    main()