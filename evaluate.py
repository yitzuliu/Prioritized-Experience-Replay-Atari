"""
Evaluation script for DQN agent with Prioritized Experience Replay.

This script loads a trained DQN model and evaluates its performance on
Atari Ice Hockey, allowing visualization of agent behavior and collection
of performance metrics.

DQN 優先經驗回放智能體的評估腳本。

此腳本載入訓練好的 DQN 模型並評估其在 Atari 冰球遊戲上的表現，
允許可視化智能體行為並收集性能指標。

Pseudocode for DQN Evaluation:
1. Load trained DQN model
   - Initialize agent with appropriate device
   - Load saved model weights

2. Evaluation loop (for each episode):
   a. Reset environment to initial state
   
   b. For each step in episode:
      i. Select action using greedy policy (no exploration)
      ii. Execute action, observe reward and next state
      iii. Continue until episode ends or max steps reached
   
   c. Record episode statistics (reward, steps, duration)

3. Generate evaluation summary
   - Calculate average rewards and steps
   - Plot performance metrics
   - Save results
"""

import os
import argparse
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from datetime import datetime

import config
from src.environment import make_atari_env
from src.agent import DQNAgent
from src.utils.device_utils import get_device, get_system_info


def parse_args():
    """
    Parse command line arguments.
    解析命令行參數。
    """
    parser = argparse.ArgumentParser(description="Evaluate DQN agent on Atari Ice Hockey")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the model checkpoint file")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to evaluate")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment")
    parser.add_argument("--max-steps", type=int, default=10000,
                        help="Maximum steps per episode")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--slow", action="store_true",
                        help="Slow down rendering for better visualization")
    parser.add_argument("--record", action="store_true",
                        help="Record video of the agent playing (overrides config setting)")
    parser.add_argument("--cpu", action="store_true",
                        help="Force using CPU even if GPU is available")
    
    return parser.parse_args()


def evaluate(args):
    """
    Main evaluation function.
    
    Args:
        args: Command line arguments
        
    主評估函數。
    
    參數：
        args: 命令行參數
    """
    # Print basic info
    print("\nEvaluating DQN agent on Atari Ice Hockey")
    print(f"Model path: {args.model}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.max_steps}")
    
    # Check if model file exists
    if not os.path.isfile(args.model):
        print(f"Error: Model file {args.model} does not exist")
        return
    
    # Set random seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Random seed set to {args.seed}")
    
    # Set up device (CPU or GPU)
    if args.cpu:
        device = torch.device("cpu")
        config.AUTO_DETECT_DEVICE = False
        print("Forcing CPU usage as specified")
    else:
        device = get_device()
    
    # Determine if video recording should be enabled
    should_record = args.record or config.ENABLE_VIDEO_RECORDING
    if should_record:
        print("Video recording enabled")
        # Create video directory if it doesn't exist
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
    
    # Create environment with appropriate rendering/recording settings
    render_mode = "human" if args.render else None
    env = make_atari_env(
        render_mode=render_mode,
        record_video=should_record,
        video_dir=config.VIDEO_DIR if should_record else None
    )
    
    # /* EVALUATION - Step 1 */
    # Load trained DQN model
    # Initialize agent with appropriate device
    agent = DQNAgent(device=device)
    
    # Load saved model weights
    print(f"Loading model from {args.model}...")
    try:
        agent.load(args.model)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Prepare for evaluation
    total_rewards = []
    total_steps = []
    
    # /* EVALUATION - Step 2 */
    # Evaluation loop (for each episode)
    print("\nStarting evaluation...")
    for episode in range(1, args.episodes + 1):
        # If recording is enabled, set unique video path for this episode
        if should_record:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = os.path.join(config.VIDEO_DIR, f"episode_{episode}_{timestamp}.mp4")
            env.metadata['video_path'] = video_path
        
        # /* EVALUATION - Step 2a */
        # Reset environment to initial state
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        truncated = False
        
        start_time = time.time()
        
        # /* EVALUATION - Step 2b */
        # For each step in episode
        while not (done or truncated) and step_count < args.max_steps:
            # i. Select action using greedy policy (no exploration)
            action = agent.select_action(state, evaluate=True)
            
            # ii. Execute action, observe reward and next state
            next_state, reward, done, truncated, _ = env.step(action)
            
            # iii. Continue until episode ends or max steps reached
            # Update state and statistics
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Slow down rendering if requested
            if args.render and args.slow:
                time.sleep(0.05)  # Add a small delay for better visualization
        
        elapsed_time = time.time() - start_time
        
        # /* EVALUATION - Step 2c */
        # Record episode statistics (reward, steps, duration)
        total_rewards.append(episode_reward)
        total_steps.append(step_count)
        
        # Print episode summary
        print(f"Episode {episode}/{args.episodes}: "
              f"Reward = {episode_reward}, Steps = {step_count}, "
              f"Duration = {elapsed_time:.2f}s")
    
    # /* EVALUATION - Step 3 */
    # Generate evaluation summary
    # Calculate average rewards and steps
    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)
    avg_steps = np.mean(total_steps)
    std_steps = np.std(total_steps)
    
    print("\nEvaluation Summary:")
    print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f} ± {std_steps:.2f}")
    print(f"Total Steps Across All Episodes: {sum(total_steps)}")
    
    if sum(total_steps) >= 5000:
        print("\n✅ Successfully completed over 5000 steps across all episodes!")
        print("This demonstrates the agent's capability for extended gameplay.")
    
    # Plot performance metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot rewards
    ax1.bar(range(1, args.episodes + 1), total_rewards, color='blue')
    ax1.axhline(y=avg_reward, color='red', linestyle='--', 
                label=f'Average: {avg_reward:.2f}')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot steps
    ax2.bar(range(1, args.episodes + 1), total_steps, color='green')
    ax2.axhline(y=avg_steps, color='red', linestyle='--', 
                label=f'Average: {avg_steps:.2f}')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Steps')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"evaluation_{timestamp}.png"
    plot_path = os.path.join(config.PLOT_DIR, plot_filename)
    # Ensure directory exists
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    plt.savefig(plot_path)
    print(f"Evaluation plot saved to {plot_path}")
    
    # Show the plot
    plt.show()
    
    # Clean up
    env.close()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)