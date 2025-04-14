"""
Training script for DQN with Prioritized Experience Replay on Atari Ice Hockey.

This script performs the main training loop for the DQN agent with PER,
handling environment interactions, neural network updates, and logging.

用於 Atari 冰球遊戲的 DQN 優先經驗回放訓練腳本。

此腳本執行 DQN 智能體的主要訓練循環，處理環境交互、神經網絡更新和日誌記錄。
"""

import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime

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
    
    return parser.parse_args()


def create_directories():
    """Create necessary directories for saving results."""
    os.makedirs(config.RESULT_DIR, exist_ok=True)  # Create main result directory first
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)  
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    os.makedirs(config.DATA_DIR, exist_ok=True)


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
    print(f"Agent initialized with {'PER' if args.use_per else 'uniform sampling'}")
    
    # Set up logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"dqn_{'per' if args.use_per else 'uniform'}_{timestamp}"
    logger = Logger(log_dir=config.LOG_DIR, experiment_name=experiment_name)
    
    # Variables for training loop
    episode_count = 0
    total_steps = 0
    best_eval_reward = float('-inf')
    
    # Training loop - main algorithm
    print("\nStarting training...\n")
    try:
        for episode in range(1, args.episodes + 1):
            episode_count = episode
            state, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            truncated = False
            
            # Episode loop
            while not (done or truncated):
                # Select action using epsilon-greedy policy
                action = agent.select_action(state)
                
                # Execute action in environment
                next_state, reward, done, truncated, _ = env.step(action)
                
                # Store transition in replay buffer
                agent.store_transition(state, action, reward, next_state, done)
                
                # Update state and statistics
                state = next_state
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # Update network
                if total_steps % config.UPDATE_FREQUENCY == 0 and total_steps >= config.LEARNING_STARTS:
                    loss = agent.optimize_model()
                    
                    # Log training step
                    logger.log_training_step(total_steps, loss)
                
                # Update target network periodically - Fixed logic
                # Use the agent's method which checks internal steps_done counter
                if agent.should_update_target_network():
                    agent.update_target_network()
                    print(f"Updated target network at step {agent.steps_done}")
                
                # Log priority distribution occasionally
                if args.use_per and total_steps % 10000 == 0:
                    priorities = agent.memory.tree.get_leaf_values()
                    logger.log_priorities(priorities)
            
            # Log episode statistics
            # Get current epsilon from agent
            epsilon = max(config.EPSILON_END, config.EPSILON_START - 
                          (config.EPSILON_START - config.EPSILON_END) * 
                          min(1.0, total_steps / config.EPSILON_DECAY))
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
            
            # Save checkpoint periodically
            if episode % config.SAVE_FREQUENCY == 0:
                checkpoint_path = os.path.join(args.save_dir, f"{experiment_name}_ep{episode}.pth")
                agent.save(checkpoint_path)
                print(f"Checkpoint saved at episode {episode}")
                
                # Also save current statistics and generate plots
                logger.save_stats()
                save_training_plots(logger.get_stats(), run_name=experiment_name)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    final_path = os.path.join(args.save_dir, f"{experiment_name}_final.pth")
    agent.save(final_path)
    print(f"Final model saved to {final_path}")
    
    # Save statistics and generate plots
    logger.save_stats()
    results_dir = save_training_plots(logger.get_stats(), run_name=experiment_name, output_dir=config.PLOT_DIR)
    
    print("\nTraining complete!")
    print(f"Trained for {episode_count} episodes, {total_steps} total steps")
    print(f"Best evaluation reward: {best_eval_reward:.2f}")
    print(f"Results and plots saved to {results_dir}")
    
    # Clean up
    env.close()
    eval_env.close()


if __name__ == "__main__":
    # Set up directories
    create_directories()
    
    # Initialize device (it will be obtained when needed, not storing in config)
    from src.utils.device_utils import get_device
    
    # Parse command line arguments
    args = parse_args()
    
    # Start training
    train(args)