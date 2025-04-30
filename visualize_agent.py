"""
Game Screen Reproduction Script (遊戲畫面重現腳本)

This script loads a pre-trained DQN model and runs the Atari game environment
in visualization mode to observe agent performance across different experiments.
Specifically used to visualize the checkpoint_6500.pt model file from the
exp_20250430_014335 experiment.

Usage:
    python visualize_agent.py                     # Use the latest checkpoint
    python visualize_agent.py --checkpoint checkpoint_1000.pt  # Specify checkpoint
    python visualize_agent.py --speed 0.5         # Slow motion viewing
    python visualize_agent.py --episodes 5        # Multi-episode observation
    python visualize_agent.py --difficulty 2      # Adjust game difficulty
"""

import os
import sys
import time
import argparse
import torch
import numpy as np

# Add parent directory to path to import config.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from src.env_wrappers import make_atari_env
from src.dqn_agent import DQNAgent
from src.device_utils import get_device

def parse_arguments():
    """
    Parse command line arguments for game reproduction configuration.
    
    Returns:
        argparse.Namespace: Parsed arguments (解析後的參數)
    """
    parser = argparse.ArgumentParser(description="Visualize trained agent playing Atari games")
    
    parser.add_argument('--exp_id', type=str, default= config.VISUALIZATION_SPECIFIC_EXPERIMENT,
                        help='Experiment ID (results/models/exp_XXXXXX)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint filename to load (e.g., checkpoint_1000.pt)')
    parser.add_argument('--latest', action='store_true', default=True,
                        help='Use the latest checkpoint in the experiment directory')
    
    #
    parser.add_argument('--difficulty', type=int, default=config.DIFFICULTY,
                        help=f'Game difficulty level (0-4) (default: {config.DIFFICULTY})')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Game speed multiplier (less than 1.0 to slow down)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of game episodes to run (default: 3)')
    
    return parser.parse_args()

def get_latest_checkpoint(exp_path):
    """
    Find the latest checkpoint file in the experiment directory.
    
    Args:
        exp_path: Experiment directory path (實驗目錄路徑)
        
    Returns:
        str: Latest checkpoint filename (最新checkpoint檔案名稱)
    """
    checkpoints = [f for f in os.listdir(exp_path) if f.startswith('checkpoint_') and f.endswith('.pt')]
    
    if not checkpoints:
        return None
    
    checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    latest = checkpoints[-1]
    print(f"Found {len(checkpoints)} checkpoint files, latest is: {latest}")
    return latest

def preprocess_observation(observation):
    """
    Preprocess observation data to meet the input requirements of PyTorch convolutional layers.
    
    This function converts observation data from shape (4,84,84,1) to a format suitable for 
    the DQN network by removing the trailing dimension and converting to PyTorch tensor.
    
    Args:
        observation: Raw observation from environment (環境返回的原始觀察)
        
    Returns:
        torch.Tensor: Processed observation data ready for network input
    """
    if isinstance(observation, np.ndarray) and observation.shape == (4, 84, 84, 1):
        observation = observation.squeeze(-1)
        
    if not isinstance(observation, torch.Tensor):
        observation = torch.FloatTensor(observation)
        
    return observation

def load_agent(exp_id, checkpoint_file=None, use_latest=True):
    """
    Load a pre-trained agent from saved model checkpoint.
    
    This function loads a DQN agent with the specified experiment ID and checkpoint file,
    or automatically uses the latest available checkpoint if not specified.
    
    Args:
        exp_id: Experiment ID (實驗ID)
        checkpoint_file: Specific checkpoint filename to load
        use_latest: Whether to use the latest checkpoint file (是否使用最新的checkpoint檔案)
        
    Returns:
        DQNAgent: Agent with loaded model ready for visualization
    """
    model_dir = os.path.join(config.MODEL_DIR, f'exp_{exp_id}')
    
    if not os.path.exists(model_dir):
        print(f"Error: Experiment directory {model_dir} does not exist!")
        sys.exit(1)
    
    if checkpoint_file is None and use_latest:
        checkpoint_file = get_latest_checkpoint(model_dir)
        print(f"Using latest checkpoint: {checkpoint_file}")
    
    if checkpoint_file is None:
        print("Error: No checkpoint file specified and --latest not set!")
        sys.exit(1)
    
    checkpoint_path = os.path.join(model_dir, checkpoint_file)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file {checkpoint_path} does not exist!")
        sys.exit(1)
    
    env = make_atari_env(render_mode=None, training=False)
    
    state_shape = env.observation_space.shape
    print(f"Observation space shape: {state_shape}")
    
    if len(state_shape) == 4 and state_shape[-1] == 1:
        state_shape = state_shape[:-1]
    
    action_space_size = env.action_space.n
    print(f"Action space: {env.action_space}")
    env.close()
    
    print(f"Creating DQN agent with state shape={state_shape}, action space size={action_space_size}")
    agent = DQNAgent(state_shape, action_space_size, use_per=config.USE_PER)
    
    print(f"Loading model from {checkpoint_path}...")
    if not agent.load_model(checkpoint_path):
        print("Error: Model loading failed!")
        sys.exit(1)
    
    print(f"Successfully loaded '{checkpoint_file}' model file from experiment '{exp_id}'")
    
    agent.set_evaluation_mode(True)
    
    return agent

def visualize_gameplay(agent, difficulty=config.DIFFICULTY, speed=1.0, num_episodes=3):
    """
    Run the agent in visualization mode to display gameplay.
    
    Args:
        agent: Trained DQN agent (已訓練的DQN智能體)
        difficulty: Game difficulty level (0-4) (遊戲難度級別)
        speed: Game speed multiplier, slower when less than 1.0
        num_episodes: Number of game episodes to run
    """
    env = make_atari_env(render_mode="human", training=False, difficulty=difficulty)
    total_rewards = []
    
    print(f"\nStarting gameplay visualization, difficulty={difficulty}, speed={speed}x, episodes={num_episodes}")
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        print(f"\nStarting episode {episode+1}/{num_episodes}")
        
        while not (done or truncated):
            processed_observation = preprocess_observation(observation)
            
            action = agent.select_action(processed_observation, evaluate=True)
            
            # Execute action
            observation, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            steps += 1
            
            # Display progress
            if steps % 10 == 0:
                print(f"Steps: {steps}, Current reward: {total_reward}", end="\r")
            
            # Control game speed - Improved implementation
            if speed < 1.0:
                # Use a larger base delay for more noticeable effect
                time.sleep((1.0 - speed) * 0.3)  # Increased from 0.1 to 0.3 for more noticeable effect
            elif speed > 1.0:
                # For faster speeds, we can't actually speed up the game, but we can skip frames
                # In this case, we'll just make the delay minimal
                pass  # No delay needed for faster speeds
        
        total_rewards.append(total_reward)
        print(f"\nEpisode {episode+1} ended after {steps} steps, total reward: {total_reward}")
    
    env.close()
    
    # Display summary statistics
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nSummary of {num_episodes} episodes:")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Minimum reward: {min(total_rewards)}")
    print(f"Maximum reward: {max(total_rewards)}")
    print(f"Standard deviation: {np.std(total_rewards):.2f}")

def main():
    """
    Main function that coordinates the entire visualization process.
    (主函數，協調整個可視化過程)
    """
    args = parse_arguments()
    
    print(f"Experiment ID: '{args.exp_id}'")
    
    agent = load_agent(args.exp_id, args.checkpoint, args.latest)
    
    visualize_gameplay(agent, args.difficulty, args.speed, args.episodes)

if __name__ == "__main__":
    main()