"""
Training script for DQN with Prioritized Experience Replay.

This script handles the complete training process for a DQN agent with 
Prioritized Experience Replay on Atari games. It manages environment setup,
agent training, evaluation, checkpointing, and visualization.

DQN 優先經驗回放的訓練腳本。

此腳本處理 DQN 智能體在 Atari 遊戲上使用優先經驗回放的完整訓練過程。
它管理環境設置、智能體訓練、評估、檢查點保存和可視化。
"""

import os
import sys
import time
import json
import argparse
import random
import numpy as np
import torch
import signal
import psutil
from datetime import datetime

# Add parent directory to path to import config.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from src.env_wrappers import make_atari_env
from src.dqn_agent import DQNAgent
from src.logger import Logger
from src.visualization import Visualizer
from src.device_utils import get_device, get_system_info

# Global variables for interrupt handling
global agent, logger, model_dir, visualizer
agent = None
logger = None
model_dir = None
visualizer = None

def parse_arguments():
    """
    Parse command line arguments for training configuration.
    
    解析命令行參數設置訓練配置
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train DQN with Prioritized Experience Replay')
    parser.add_argument('--experiment_name', type=str, default='',
                        help='Experiment name, auto-generated if empty')
    parser.add_argument('--env_name', type=str, default=config.ENV_NAME,
                        help=f'Environment name, default: {config.ENV_NAME}')
    parser.add_argument('--episodes', type=int, default=config.TRAINING_EPISODES,
                        help=f'Number of training episodes, default: {config.TRAINING_EPISODES}')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed, random if not specified')
    parser.add_argument('--no_per', action='store_true',
                        help='Disable Prioritized Experience Replay, use standard replay')
    parser.add_argument('--learning_rate', type=float, default=config.LEARNING_RATE,
                        help=f'Learning rate, default: {config.LEARNING_RATE}')
    parser.add_argument('--difficulty', type=int, default=config.DIFFICULTY, choices=range(5),
                        help=f'Game difficulty level (0-4, where 0 is easiest), default: {config.DIFFICULTY}')
    
    return parser.parse_args()

def train_one_episode(env, agent, logger, episode_num, total_steps):
    """
    Run a single training episode.
    
    執行單個訓練回合
    
    Args:
        env: Game environment
        agent: DQN agent
        logger: Training logger
        episode_num: Current episode number
        total_steps: Cumulative training steps
    
    Returns:
        dict: Episode statistics
        int: Number of steps taken in this episode
    """
    # Record episode start
    logger.log_episode_start(episode_num)
    
    # Reset environment
    obs, info = env.reset()
    done = False
    episode_reward = 0
    episode_losses = []
    step_count = 0
    
    # Episode loop
    while not done:
        # Select action
        action = agent.select_action(obs)
        
        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Store experience
        agent.store_transition(obs, action, reward, next_obs, done)
        
        # Learn
        if total_steps >= config.LEARNING_STARTS and total_steps % config.UPDATE_FREQUENCY == 0:
            loss = agent.optimize_model(logger)
            if loss is not None:
                episode_losses.append(loss)
        
        # Update state
        obs = next_obs
        episode_reward += reward
        step_count += 1
        total_steps += 1
        
        # Record epsilon change (every 1000 steps)
        if total_steps % 1000 == 0:
            logger.log_epsilon(total_steps, agent.epsilon)
        
        # Update target network (at specified frequency)
        if total_steps % config.TARGET_UPDATE_FREQUENCY == 0:
            agent.target_network.load_state_dict(agent.policy_network.state_dict())
    
    # Record episode end and statistics
    logger.log_episode_end(
        episode_num=episode_num,
        total_reward=episode_reward,
        steps=step_count,
        avg_loss=np.mean(episode_losses) if episode_losses else None,
        epsilon=agent.epsilon
    )
    
    # Return episode statistics and step count
    return {
        'reward': episode_reward,
        'steps': step_count,
        'avg_loss': np.mean(episode_losses) if episode_losses else None,
        'epsilon': agent.epsilon
    }, step_count

def evaluate_agent(env, agent, logger, episode_num, num_episodes=config.EVAL_EPISODES):
    """
    Evaluate agent performance.
    
    評估智能體的性能
    
    Args:
        env: Game environment
        agent: DQN agent
        logger: Training logger
        episode_num: Current training episode number
        num_episodes: Number of evaluation episodes
    
    Returns:
        float: Average reward across evaluation episodes
    """
    logger.log_text(f"Starting evaluation (training episode {episode_num})")
    
    # Create evaluation environment
    eval_env = make_atari_env(env_name=config.ENV_NAME, training=False)
    
    # Temporarily switch to evaluation mode
    agent.set_evaluation_mode(True)
    
    eval_rewards = []
    for eval_episode in range(1, num_episodes + 1):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        eval_rewards.append(episode_reward)
    
    # Restore training mode
    agent.set_evaluation_mode(False)
    
    # Calculate evaluation results
    mean_reward = np.mean(eval_rewards)
    std_reward = np.std(eval_rewards)
    
    # Log evaluation results
    logger.log_text(
        f"Evaluation results (training episode {episode_num}): "
        f"Mean reward = {mean_reward:.2f} ± {std_reward:.2f}, "
        f"Min/Max = {np.min(eval_rewards):.2f}/{np.max(eval_rewards):.2f}"
    )
    
    # Close evaluation environment
    eval_env.close()
    
    return mean_reward

def check_resources():
    """
    Check system resource usage.
    
    檢查系統資源使用情況
    
    Returns:
        dict: Resource usage statistics
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    resources = {
        'memory_bytes': memory_info.rss,
        'memory_percent': memory_percent,
        'cpu_percent': process.cpu_percent(interval=0.1)
    }
    
    return resources

def handle_interrupt(signum, frame):
    """
    Handle interrupt signal (Ctrl+C), save current state and generate visualizations.
    
    處理中斷信號 (Ctrl+C)，保存當前狀態並生成可視化
    
    Args:
        signum: Signal number
        frame: Current stack frame
    """
    global agent, logger, model_dir, visualizer
    
    if logger is not None:
        logger.log_text("\nTraining interrupted, saving state...")
        
        # Save model and training state
        if agent is not None and model_dir is not None:
            interrupt_model_path = os.path.join(model_dir, "interrupted_model.pt")
            agent.save_model(interrupt_model_path)
            logger.log_text(f"Model saved to: {interrupt_model_path}")
        
        logger.save_training_state()
        logger.log_text("Training state saved")
        
        # Generate visualizations
        logger.log_text("Generating visualizations...")
        if visualizer is None:
            visualizer = Visualizer(logger_instance=logger)
        plots = visualizer.generate_all_plots(show=False)
        logger.log_text(f"Generated {len(plots)} visualization plots")
        logger.log_text("Training safely interrupted. Use resume.py to continue training.")
    
    # Normal exit
    sys.exit(0)

def main():
    """
    Main training function that coordinates the entire training process.
    
    主訓練函數，協調整個訓練過程
    """
    global agent, logger, model_dir, visualizer
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set experiment name
    experiment_name = args.experiment_name
    if not experiment_name:
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed) if torch.cuda.is_available() else None
    
    # Detect device
    device = get_device()
    
    # Create directories
    for directory in [config.MODEL_DIR, config.LOG_DIR, config.DATA_DIR, config.PLOT_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    model_dir = os.path.join(config.MODEL_DIR, experiment_name)
    os.makedirs(model_dir, exist_ok=True)
    
    # Save experiment configuration
    config_path = os.path.join(model_dir, "experiment_config.json")
    with open(config_path, 'w') as f:
        config_data = {
            "experiment_name": experiment_name,
            "env_name": args.env_name,
            "episodes": args.episodes,
            "seed": args.seed,
            "use_per": not args.no_per,
            "learning_rate": args.learning_rate,
            "difficulty": args.difficulty,
            "device": str(device),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        json.dump(config_data, f, indent=4)
    
    # Initialize environment
    env = make_atari_env(env_name=args.env_name, training=True)
    
    # Initialize agent
    agent = DQNAgent(
        state_shape=(config.FRAME_STACK, config.FRAME_HEIGHT, config.FRAME_WIDTH),
        action_space_size=env.action_space.n,
        learning_rate=args.learning_rate,
        use_per=not args.no_per
    )
    
    # Initialize logger
    logger = Logger(experiment_name=experiment_name)
    
    # Log training start
    logger.log_text(f"Starting training experiment: {experiment_name}")
    logger.log_text(f"Environment: {args.env_name}")
    logger.log_text(f"Device: {device}")
    logger.log_text(f"Using PER: {not args.no_per}")
    logger.log_text(f"System info: {get_system_info()}")
    
    # Initialize training variables
    total_episodes = args.episodes
    total_steps = 0
    best_eval_reward = float('-inf')
    start_time = time.time()
    
    try:
        # Main training loop
        for episode in range(1, total_episodes + 1):
            # Train one episode
            episode_stats, steps_taken = train_one_episode(env, agent, logger, episode, total_steps)
            total_steps += steps_taken
            
            # Periodically output training progress
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
                    f"Elapsed time: {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"
                )
            
            # Periodically evaluate
            if episode % config.EVAL_FREQUENCY == 0:
                eval_reward = evaluate_agent(env, agent, logger, episode)
                
                # Save best model
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_model_path = os.path.join(model_dir, "best_model.pt")
                    agent.save_model(best_model_path)
                    logger.log_text(f"New best model! Mean reward: {best_eval_reward:.2f}")
            
            # Periodically save checkpoints
            if episode % config.SAVE_FREQUENCY == 0:
                checkpoint_path = os.path.join(model_dir, f"checkpoint_{episode}.pt")
                agent.save_model(checkpoint_path, episode=episode)
                logger.save_training_state()
                logger.log_text(f"Checkpoint saved (episode {episode})")
            
            # Periodically check resources
            if episode % 10 == 0:
                resources = check_resources()
                if resources['memory_percent'] > config.MEMORY_THRESHOLD_PERCENT:
                    logger.log_text(f"Warning: Memory usage {resources['memory_percent']:.1f}% exceeds threshold")
                    logger.limit_memory_usage()
        
        # Training completed
        logger.log_text("\nTraining completed!")
        
        # Save final model
        final_model_path = os.path.join(model_dir, "final_model.pt")
        agent.save_model(final_model_path, episode=total_episodes)
        logger.log_text(f"Final model saved to: {final_model_path}")
        
        # Save training state
        logger.save_training_state()
        
        # Log training summary
        total_time = time.time() - start_time
        hrs, rem = divmod(total_time, 3600)
        mins, secs = divmod(rem, 60)
        
        logger.log_text(
            f"Training summary:\n"
            f"- Total episodes: {total_episodes}\n"
            f"- Total steps: {total_steps}\n"
            f"- Best evaluation reward: {best_eval_reward:.2f}\n"
            f"- Total training time: {int(hrs):02d}:{int(mins):02d}:{int(secs):02d}"
        )
        
        # Generate final visualizations
        logger.log_text("Generating training visualizations...")
        visualizer = Visualizer(logger_instance=logger)
        plots = visualizer.generate_all_plots(show=False)
        logger.log_text(f"Generated {len(plots)} visualization plots")
        
    except Exception as e:
        # Exception handling
        logger.log_text(f"\nException occurred during training: {str(e)}")
        
        # Save model at exception
        exception_model_path = os.path.join(model_dir, "exception_model.pt")
        agent.save_model(exception_model_path)
        logger.log_text(f"Model at exception saved to: {exception_model_path}")
        
        # Save training state
        logger.save_training_state()
        
        # Generate visualizations at exception
        logger.log_text("Generating visualizations...")
        visualizer = Visualizer(logger_instance=logger)
        visualizer.generate_all_plots(show=False)
        
        # Re-raise exception
        raise
        
    finally:
        # Close environment
        if 'env' in locals():
            env.close()
        
        # Output completion information
        if logger is not None:
            logger.log_text("Training program exited")

if __name__ == "__main__":
    # Register signal handler
    signal.signal(signal.SIGINT, handle_interrupt)
    
    # Start training
    main()