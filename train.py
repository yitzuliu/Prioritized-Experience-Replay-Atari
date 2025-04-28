"""
Training script for DQN with Prioritized Experience Replay.

This script handles the complete training process for a DQN agent with 
Prioritized Experience Replay on Atari games. It manages environment setup,
agent training, evaluation, and visualization.

DQN 優先經驗回放的訓練腳本。

此腳本處理 DQN 智能體在 Atari 遊戲上使用優先經驗回放的完整訓練過程。
它管理環境設置、智能體訓練、評估和可視化。
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
global agent, logger, visualizer, env
agent = None
logger = None
visualizer = None
env = None

def signal_handler(sig, frame):
    """
    Handle interruption signals like SIGINT (Ctrl+C) to ensure graceful exit.
    
    處理中斷信號，如 SIGINT（Ctrl+C），以確保優雅退出
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    global agent, logger, visualizer, env
    
    print("\n\nInterrupt received. Saving progress and exiting gracefully...")
    
    # Log the interruption
    if logger is not None:
        logger.log_text("\nTraining interrupted by user or system signal.")
        
        # Save model checkpoint at interruption point
        if agent is not None:
            try:
                # Create interruption checkpoint filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                interrupt_path = os.path.join(
                    config.MODEL_DIR, 
                    logger.experiment_name, 
                    f"interrupt_checkpoint_{timestamp}.pt"
                )
                
                # Save the model with metadata about the interruption
                metadata = {
                    'interruption_time': timestamp,
                    'reason': 'User or system interruption',
                    'training_progress': f"{logger.current_episode}/{config.TRAINING_EPISODES} episodes"
                }
                
                agent.save_model(
                    path=interrupt_path, 
                    save_optimizer=True,
                    include_memory=False,
                    metadata=metadata
                )
                
                logger.log_text(f"Model checkpoint saved at interruption point: {interrupt_path}")
            except Exception as e:
                logger.log_text(f"Error saving model at interruption: {str(e)}")
        
        # Generate visualizations before exit
        logger.log_text("Generating visualizations of current progress...")
        
        try:
            visualizer = Visualizer(logger_instance=logger)
            plots = visualizer.generate_all_plots(show=False)
            logger.log_text(f"Successfully generated {len(plots)} visualization plots at interruption point")
        except Exception as e:
                        logger.log_text(f"Error generating visualizations: {str(e)}")
    
    # Close the environment if open
    if env is not None:
        try:
            env.close()
        except:
            pass
    
    # Final message
    if logger is not None:
        logger.log_text("Training program exited after interruption")
    
    print("Cleanup complete. Exiting.")
    
    # Exit with success code since this is a controlled exit
    sys.exit(0)

def parse_arguments():
    """
    Parse command line arguments for training configuration.
    
    解析命令行參數設置訓練配置
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train DQN with Prioritized Experience Replay')
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
    parser.add_argument('--enable_file_logging', action='store_true',
                        help='Enable file logging, default: disabled')
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
    
    # Get current beta value if agent uses PER
    beta = agent.memory.beta if hasattr(agent, 'memory') and hasattr(agent.memory, 'beta') else None
    
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
    
    # Update beta value at the end of episode for logging
    if hasattr(agent, 'memory') and hasattr(agent.memory, 'beta'):
        beta = agent.memory.beta
    
    # Record episode end and statistics
    logger.log_episode_end(
        episode_num=episode_num,
        total_reward=episode_reward,
        steps=step_count,
        avg_loss=np.mean(episode_losses) if episode_losses else None,
        epsilon=agent.epsilon,
        beta=beta  # Pass the beta value to the logger
    )
    
    # Return episode statistics and step count
    return {
        'reward': episode_reward,
        'steps': step_count,
        'avg_loss': np.mean(episode_losses) if episode_losses else None,
        'epsilon': agent.epsilon,
        'beta': beta  # Include beta in the returned statistics
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

def main():
    """
    Main training function that coordinates the entire training process.
    
    主訓練函數，協調整個訓練過程
    """
    global agent, logger, visualizer, env
    
    # Register signal handlers for graceful exit
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Set experiment name
    experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Override config settings with command line arguments
    if args.enable_file_logging:
        config.ENABLE_FILE_LOGGING = True
        print("File logging enabled through command line argument")
    
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
    
    # Initialize visualizer - Added this line to fix the error
    visualizer = Visualizer(logger_instance=logger)
    
    #configuration output
    visualizer.generate_training_config_markdown()

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

            # Periodically evaluate
            if episode % config.EVAL_FREQUENCY == 0:
                eval_reward = evaluate_agent(env, agent, logger, episode)
                
                # Track best evaluation reward
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    logger.update_best_eval_reward(best_eval_reward)
                    logger.log_text(f"New best result! Mean reward: {best_eval_reward:.2f}")
            
            # Periodically check resources
            if episode % 10 == 0:
                resources = check_resources()
                if resources['memory_percent'] > config.MEMORY_THRESHOLD_PERCENT:
                    logger.log_text(f"Warning: Memory usage {resources['memory_percent']:.1f}% exceeds threshold")
                    logger.limit_memory_usage()
            
            # Periodically save visualizations during training
            if episode % config.VISUALIZATION_SAVE_INTERVAL == 0:
                try:
                    visualizer = Visualizer(logger_instance=logger)
                    # Create just the overview plot to avoid too many files
                    visualizer.plot_training_overview(save=True, show=False)
                    logger.log_text(f"Saved intermediate visualization at episode {episode}")
                except Exception as e:
                    logger.log_text(f"Error generating intermediate visualization: {str(e)}")
        
            # Add periodic checkpoint saving
            if episode % config.LOGGER_SAVE_INTERVAL == 0:
                try:
                    checkpoint_path = os.path.join(
                        config.MODEL_DIR, 
                        logger.experiment_name, 
                        f"checkpoint_{episode}.pt"
                    )
                    
                    agent.save_model(
                        path=checkpoint_path,
                        save_optimizer=True,
                        include_memory=False,
                        metadata={'checkpoint_episode': episode}
                    )
                    
                    logger.log_text(f"Saved checkpoint at episode {episode}: {checkpoint_path}")
                except Exception as e:
                    logger.log_text(f"Error saving checkpoint: {str(e)}")
            
        # Training completed
        logger.log_text("\nTraining completed successfully!")
        
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
        
        # Generate visualizations at exception
        logger.log_text("Generating visualizations before exit...")
        try:
            visualizer = Visualizer(logger_instance=logger)
            plots = visualizer.generate_all_plots(show=False)
            logger.log_text(f"Successfully generated {len(plots)} visualization plots at interruption point")

        except Exception as viz_error:
            logger.log_text(f"Error generating visualizations: {str(viz_error)}")
        
        # Re-raise exception
        raise
        
    finally:
        # Close environment
        if env is not None:
            env.close()
        
        # Output completion information
        if logger is not None:
            logger.log_text("Training program exited")

if __name__ == "__main__":
    # Start training
    main()