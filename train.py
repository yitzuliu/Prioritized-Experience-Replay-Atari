"""
Enhanced training script for DQN with Prioritized Experience Replay.

This script handles the complete training process for a DQN agent with 
Prioritized Experience Replay on Atari games. It includes enhanced monitoring,
error handling, and performance optimization features.

增強的 DQN 優先經驗回放訓練腳本。

此腳本處理 DQN 智能體在 Atari 遊戲上使用優先經驗回放的完整訓練過程。
包括增強的監控、錯誤處理和性能優化功能。
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
import traceback
from datetime import datetime

# Add parent directory to path to import config.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

from src.env_wrappers import make_atari_env
from src.dqn_agent import DQNAgent
from src.logger import Logger
from src.visualization import Visualizer
from src.device_utils import get_device, get_system_info

# Import new performance monitoring (with fallback if not available)
try:
    from src.performance_monitor import PerformanceMonitor, HyperparameterTuner
    PERFORMANCE_MONITORING_AVAILABLE = True
except ImportError:
    PERFORMANCE_MONITORING_AVAILABLE = False
    print("Performance monitoring not available. Continuing without advanced monitoring...")

# Global variables for interrupt handling
global agent, logger, visualizer, env, performance_monitor
agent = None
logger = None
visualizer = None
env = None
performance_monitor = None

def signal_handler(sig, frame):
    """
    Enhanced interrupt signal handler with comprehensive cleanup.
    
    增強的中斷信號處理器，具有綜合清理功能。
    
    Args:
        sig: Signal number
        frame: Current stack frame
    """
    global agent, logger, visualizer, env, performance_monitor
    
    print("\n\n🔄 Interrupt received. Performing enhanced cleanup and saving progress...")
    
    # Stop performance monitoring first
    if performance_monitor is not None:
        try:
            performance_monitor.stop_monitoring()
            
            # Save performance report
            if logger is not None:
                report_path = os.path.join(
                    config.PLOT_DIR, 
                    logger.experiment_name, 
                    f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )
                performance_monitor.save_report(report_path)
                logger.log_text(f"Performance report saved: {report_path}")
        except Exception as e:
            print(f"Error stopping performance monitor: {str(e)}")
    
    # Enhanced model saving with verification
    if logger is not None:
        logger.log_text("\n🚨 Training interrupted by user or system signal.")
        
        if agent is not None:
            try:
                # Create comprehensive interruption checkpoint
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                interrupt_path = os.path.join(
                    config.MODEL_DIR, 
                    logger.experiment_name, 
                    f"interrupt_checkpoint_{timestamp}.pt"
                )
                
                # Get training diagnostics
                try:
                    diagnostics = agent.get_training_diagnostics()
                    training_phase = diagnostics['training_progress']['learning_phase']
                except:
                    training_phase = 'unknown'
                
                # Enhanced metadata for interruption checkpoint
                metadata = {
                    'interruption_time': timestamp,
                    'reason': 'User or system interruption',
                    'training_progress': f"{logger.current_episode}/{config.TRAINING_EPISODES} episodes",
                    'training_phase': training_phase,
                    'system_info': get_system_info(),
                    'config_summary': config.get_config_summary() if hasattr(config, 'get_config_summary') else {}
                }
                
                # Save with enhanced verification
                success = agent.save_model(
                    path=interrupt_path, 
                    save_optimizer=True,
                    include_memory=False,  # Skip memory to save quickly during interrupt
                    metadata=metadata
                )
                
                if success:
                    logger.log_text(f"✅ Enhanced checkpoint saved: {interrupt_path}")
                else:
                    logger.log_text("❌ Failed to save interruption checkpoint")
                    
            except Exception as e:
                logger.log_text(f"❌ Error saving model at interruption: {str(e)}")
                logger.log_text(f"Traceback: {traceback.format_exc()}")
        
        # Generate emergency visualizations
        logger.log_text("📊 Generating emergency visualizations...")
        
        try:
            visualizer = Visualizer(logger_instance=logger)
            plots = visualizer.generate_all_plots(show=False)
            logger.log_text(f"✅ Generated {len(plots)} emergency visualization plots")
        except Exception as e:
            logger.log_text(f"❌ Error generating emergency visualizations: {str(e)}")
    
    # Enhanced environment cleanup
    if env is not None:
        try:
            env.close()
            print("✅ Environment closed successfully")
        except Exception as e:
            print(f"⚠️ Error closing environment: {str(e)}")
    
    # Final status message
    if logger is not None:
        logger.log_text("🏁 Enhanced training program exited after interruption")
        
        # Save final training state
        try:
            logger.save_training_state()
            print("✅ Training state saved successfully")
        except Exception as e:
            print(f"⚠️ Error saving training state: {str(e)}")
    
    print("🎯 Enhanced cleanup complete. Exiting gracefully.")
    sys.exit(0)

def parse_arguments():
    """
    Enhanced argument parser with additional options for reliability and performance.
    
    增強的參數解析器，包含可靠性和性能的額外選項。
    
    Returns:
        argparse.Namespace: The parsed arguments
    """
    parser = argparse.ArgumentParser(description='Enhanced DQN with Prioritized Experience Replay Training')
    
    # Core training arguments
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
    
    # Logging and monitoring arguments
    parser.add_argument('--enable_file_logging', action='store_true',
                        help='Enable file logging, default: disabled')
    parser.add_argument('--enable_performance_monitoring', action='store_true',
                        help='Enable advanced performance monitoring')
    parser.add_argument('--performance_report_interval', type=int, default=1000,
                        help='Episodes between performance reports, default: 1000')
    
    # Checkpoint and recovery arguments
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Path to checkpoint file to resume training from')
    parser.add_argument('--checkpoint_interval', type=int, default=config.LOGGER_SAVE_INTERVAL,
                        help=f'Episodes between checkpoints, default: {config.LOGGER_SAVE_INTERVAL}')
    parser.add_argument('--enable_auto_tuning', action='store_true',
                        help='Enable automatic hyperparameter tuning suggestions')
    
    # Safety and reliability arguments
    parser.add_argument('--max_memory_percent', type=float, default=90.0,
                        help='Maximum memory usage percentage before emergency cleanup, default: 90.0')
    parser.add_argument('--emergency_save_interval', type=int, default=50,
                        help='Episodes between emergency saves, default: 50')
    
    return parser.parse_args()

def setup_enhanced_training_environment(args):
    """
    Set up enhanced training environment with all reliability features.
    
    設置增強的訓練環境，包含所有可靠性功能。
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        tuple: (agent, logger, visualizer, performance_monitor, env)
    """
    # Set experiment name
    experiment_name = f"enhanced_exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Override config settings with command line arguments
    if args.enable_file_logging:
        config.ENABLE_FILE_LOGGING = True
        print("📝 File logging enabled through command line argument")
    
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed) 
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"🎲 Random seed set to: {args.seed}")
    
    # Detect and configure device
    device = get_device()
    print(f"🖥️ Using device: {device}")
    
    # Create directories with error handling
    for directory in [config.MODEL_DIR, config.LOG_DIR, config.DATA_DIR, config.PLOT_DIR]:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"⚠️ Warning: Could not create directory {directory}: {str(e)}")
    
    # Initialize environment with error handling
    try:
        env = make_atari_env(env_name=args.env_name, training=True)
        print(f"🎮 Environment initialized: {args.env_name}")
    except Exception as e:
        print(f"❌ Failed to initialize environment: {str(e)}")
        raise
    
    # Initialize enhanced DQN agent
    try:
        agent = DQNAgent(
            state_shape=(config.FRAME_STACK, config.FRAME_HEIGHT, config.FRAME_WIDTH),
            action_space_size=env.action_space.n,
            learning_rate=args.learning_rate,
            use_per=not args.no_per
        )
        print(f"🤖 Enhanced DQN agent initialized (PER: {not args.no_per})")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {str(e)}")
        raise
    
    # Initialize enhanced logger
    try:
        logger = Logger(experiment_name=experiment_name)
        print(f"📊 Enhanced logger initialized: {experiment_name}")
    except Exception as e:
        print(f"❌ Failed to initialize logger: {str(e)}")
        raise
    
    # Initialize visualizer
    try:
        visualizer = Visualizer(logger_instance=logger)
        print("📈 Visualizer initialized")
    except Exception as e:
        print(f"⚠️ Warning: Visualizer initialization failed: {str(e)}")
        visualizer = None
    
    # Initialize performance monitor if available
    performance_monitor = None
    if args.enable_performance_monitoring and PERFORMANCE_MONITORING_AVAILABLE:
        try:
            performance_monitor = PerformanceMonitor(monitor_interval=1.0, history_size=1000)
            performance_monitor.start_monitoring()
            print("📏 Performance monitoring started")
        except Exception as e:
            print(f"⚠️ Warning: Performance monitoring failed to start: {str(e)}")
    
    # Resume from checkpoint if specified
    if args.resume_from:
        try:
            load_result = agent.load_model(args.resume_from)
            if load_result['success']:
                print(f"🔄 Successfully resumed from checkpoint: {args.resume_from}")
                print(f"   Resumed at episode: {load_result.get('steps_done', 'unknown')}")
                logger.log_text(f"Training resumed from checkpoint: {args.resume_from}")
            else:
                print(f"⚠️ Failed to resume from checkpoint: {load_result.get('error', 'unknown error')}")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {str(e)}")
    
    return agent, logger, visualizer, performance_monitor, env

def enhanced_resource_check(max_memory_percent=90.0):
    """
    Enhanced resource checking with detailed reporting.
    
    增強的資源檢查，包含詳細報告。
    
    Args:
        max_memory_percent: Maximum allowed memory usage percentage
        
    Returns:
        dict: Resource status and recommendations
    """
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        cpu_percent = process.cpu_percent(interval=0.1)
        
        # Check available disk space
        disk_usage = psutil.disk_usage(os.getcwd())
        disk_free_gb = disk_usage.free / (1024**3)
        
        status = {
            'memory_bytes': memory_info.rss,
            'memory_percent': memory_percent,
            'cpu_percent': cpu_percent,
            'disk_free_gb': disk_free_gb,
            'warnings': [],
            'critical': False
        }
        
        # Check for critical conditions
        if memory_percent > max_memory_percent:
            status['warnings'].append(f"Critical memory usage: {memory_percent:.1f}%")
            status['critical'] = True
        
        if cpu_percent > 95:
            status['warnings'].append(f"Critical CPU usage: {cpu_percent:.1f}%")
        
        if disk_free_gb < 1.0:
            status['warnings'].append(f"Low disk space: {disk_free_gb:.1f} GB remaining")
            status['critical'] = True
        
        return status
        
    except Exception as e:
        return {
            'error': str(e),
            'warnings': [f"Resource check failed: {str(e)}"],
            'critical': False
        }

def main():
    """
    Enhanced main training function with comprehensive error handling and monitoring.
    
    增強的主訓練函數，具有綜合錯誤處理和監控功能。
    """
    global agent, logger, visualizer, env, performance_monitor
    
    print("🚀 Starting Enhanced DQN Training with PER")
    print("=" * 60)
    
    # Register enhanced signal handlers
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Validate configuration
        if hasattr(config, 'validate_config'):
            try:
                validation_errors = config.validate_config()
                if validation_errors:
                    print("❌ Configuration validation failed:")
                    for error in validation_errors:
                        print(f"   - {error}")
                    return
                else:
                    print("✅ Configuration validation passed")
            except Exception as e:
                print(f"⚠️ Configuration validation error: {str(e)}")
        
        # Setup enhanced training environment
        print("\n🔧 Setting up enhanced training environment...")
        agent, logger, visualizer, performance_monitor, env = setup_enhanced_training_environment(args)
        
        # Log enhanced training start
        logger.log_text("🚀 Enhanced DQN training started")
        logger.log_text(f"Environment: {args.env_name}")
        logger.log_text(f"Device: {get_device()}")
        logger.log_text(f"Using PER: {not args.no_per}")
        logger.log_text(f"System info: {get_system_info()}")
        
        if hasattr(config, 'get_config_summary'):
            logger.log_text(f"Configuration summary: {config.get_config_summary()}")
        
        # Initialize training variables
        total_episodes = args.episodes
        total_steps = 0
        best_eval_reward = float('-inf')
        start_time = time.time()
        last_performance_report = 0
        emergency_saves = 0
        
        # Initialize hyperparameter tuner if enabled
        hyperparameter_tuner = None
        if args.enable_auto_tuning and PERFORMANCE_MONITORING_AVAILABLE:
            try:
                hyperparameter_tuner = HyperparameterTuner(performance_monitor)
                logger.log_text("🎛️ Hyperparameter tuning enabled")
            except Exception as e:
                logger.log_text(f"⚠️ Hyperparameter tuning initialization failed: {str(e)}")
        
        print(f"\n🎯 Starting training for {total_episodes} episodes...")
        print("=" * 60)
        
        # Main enhanced training loop
        for episode in range(1, total_episodes + 1):
            try:
                # Enhanced resource monitoring
                if episode % 10 == 0:
                    resource_status = enhanced_resource_check(args.max_memory_percent)
                    
                    if resource_status.get('critical', False):
                        logger.log_text(f"🚨 Critical resource condition detected: {resource_status['warnings']}")
                        
                        # Trigger emergency cleanup
                        if hasattr(agent.memory, '_check_memory_and_cleanup'):
                            agent.memory._check_memory_and_cleanup()
                        
                        # Emergency save if resources are critical
                        if episode % args.emergency_save_interval == 0:
                            try:
                                emergency_path = os.path.join(
                                    config.MODEL_DIR, 
                                    logger.experiment_name, 
                                    f"emergency_save_{episode}.pt"
                                )
                                agent.save_model(emergency_path, include_memory=False)
                                emergency_saves += 1
                                logger.log_text(f"💾 Emergency save #{emergency_saves}: {emergency_path}")
                            except Exception as e:
                                logger.log_text(f"❌ Emergency save failed: {str(e)}")
                
                # Record frame for performance monitoring
                if performance_monitor:
                    performance_monitor.record_frame()
                
                # Train one episode with enhanced monitoring
                with performance_monitor.time_operation('batch_time') if performance_monitor else contextlib.nullcontext():
                    episode_stats, steps_taken = train_one_episode(env, agent, logger, episode, total_steps)
                
                total_steps += steps_taken
                
                # Enhanced periodic evaluation
                if episode % config.EVAL_FREQUENCY == 0:
                    try:
                        eval_reward = evaluate_agent(env, agent, logger, episode)
                        
                        if eval_reward > best_eval_reward:
                            best_eval_reward = eval_reward
                            logger.update_best_eval_reward(best_eval_reward)
                            logger.log_text(f"🏆 New best evaluation reward: {best_eval_reward:.2f}")
                            
                            # Save best model
                            best_model_path = os.path.join(
                                config.MODEL_DIR, 
                                logger.experiment_name, 
                                f"best_model_episode_{episode}.pt"
                            )
                            agent.save_model(best_model_path, include_memory=False)
                            
                    except Exception as e:
                        logger.log_text(f"⚠️ Evaluation failed at episode {episode}: {str(e)}")
                
                # Enhanced periodic checkpoint saving
                if episode % args.checkpoint_interval == 0:
                    try:
                        checkpoint_path = os.path.join(
                            config.MODEL_DIR, 
                            logger.experiment_name, 
                            f"checkpoint_{episode}.pt"
                        )
                        
                        # Get training diagnostics for checkpoint metadata
                        try:
                            diagnostics = agent.get_training_diagnostics()
                        except:
                            diagnostics = {}
                        
                        success = agent.save_model(
                            path=checkpoint_path,
                            save_optimizer=True,
                            include_memory=False,
                            metadata={
                                'checkpoint_episode': episode,
                                'training_diagnostics': diagnostics,
                                'performance_stats': performance_monitor.get_current_stats() if performance_monitor else {}
                            }
                        )
                        
                        if success:
                            logger.log_text(f"💾 Enhanced checkpoint saved: {checkpoint_path}")
                        
                    except Exception as e:
                        logger.log_text(f"❌ Checkpoint save failed: {str(e)}")
                
                # Performance reporting and hyperparameter tuning
                if (episode - last_performance_report) >= args.performance_report_interval:
                    last_performance_report = episode
                    
                    if performance_monitor:
                        try:
                            # Generate performance report
                            report_path = os.path.join(
                                config.PLOT_DIR, 
                                logger.experiment_name, 
                                f"performance_report_episode_{episode}.json"
                            )
                            performance_monitor.save_report(report_path)
                            logger.log_text(f"📊 Performance report generated: {report_path}")
                            
                            # Hyperparameter tuning suggestions
                            if hyperparameter_tuner and episode > 100:  # Wait for some training data
                                try:
                                    # Prepare metrics for tuning
                                    training_metrics = {
                                        'reward_trend': episode_stats.get('reward', 0),
                                        'loss_variance': 0.1,  # Placeholder - should calculate from recent losses
                                        'avg_fps': performance_monitor.get_current_stats().get('fps', {}).get('mean', 0),
                                        'memory_usage': performance_monitor.get_current_stats().get('memory_percent', {}).get('current', 0) / 100.0
                                    }
                                    
                                    tuning_recommendations = hyperparameter_tuner.get_tuning_recommendations(training_metrics)
                                    
                                    if tuning_recommendations['overall_suggestions']:
                                        logger.log_text("🎛️ Hyperparameter tuning suggestions:")
                                        for suggestion in tuning_recommendations['overall_suggestions']:
                                            logger.log_text(f"   💡 {suggestion}")
                                            
                                except Exception as e:
                                    logger.log_text(f"⚠️ Hyperparameter tuning failed: {str(e)}")
                                    
                        except Exception as e:
                            logger.log_text(f"⚠️ Performance reporting failed: {str(e)}")
                
                # Enhanced periodic visualization generation
                if episode % config.VISUALIZATION_SAVE_INTERVAL == 0:
                    try:
                        if visualizer:
                            visualizer.plot_training_overview(save=True, show=False)
                            logger.log_text(f"📈 Enhanced visualization generated at episode {episode}")
                    except Exception as e:
                        logger.log_text(f"⚠️ Visualization generation failed: {str(e)}")
                
            except Exception as e:
                logger.log_text(f"❌ Error in training loop at episode {episode}: {str(e)}")
                logger.log_text(f"Traceback: {traceback.format_exc()}")
                
                # Continue training after logging the error
                continue
        
        # Enhanced training completion
        logger.log_text("\n🎉 Enhanced training completed successfully!")
        
        # Final comprehensive report
        total_time = time.time() - start_time
        hrs, rem = divmod(total_time, 3600)
        mins, secs = divmod(rem, 60)
        
        final_report = {
            'total_episodes': total_episodes,
            'total_steps': total_steps,
            'best_evaluation_reward': best_eval_reward,
            'total_training_time': f"{int(hrs):02d}:{int(mins):02d}:{int(secs):02d}",
            'emergency_saves': emergency_saves,
            'final_epsilon': agent.epsilon
        }
        
        logger.log_text(f"📋 Final training report: {json.dumps(final_report, indent=2)}")
        
        # Generate final enhanced visualizations
        if visualizer:
            try:
                logger.log_text("📊 Generating final enhanced visualizations...")
                plots = visualizer.generate_all_plots(show=False)
                logger.log_text(f"✅ Generated {len(plots)} final visualization plots")
            except Exception as e:
                logger.log_text(f"⚠️ Final visualization generation failed: {str(e)}")
        
        # Final performance report
        if performance_monitor:
            try:
                final_performance_path = os.path.join(
                    config.PLOT_DIR, 
                    logger.experiment_name, 
                    "final_performance_report.json"
                )
                performance_monitor.save_report(final_performance_path)
                logger.log_text(f"📊 Final performance report saved: {final_performance_path}")
            except Exception as e:
                logger.log_text(f"⚠️ Final performance report failed: {str(e)}")
        
        print("\n🎯 Enhanced training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        signal_handler(signal.SIGINT, None)
        
    except Exception as e:
        print(f"\n❌ Critical error in main training loop: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        
        if logger is not None:
            logger.log_text(f"❌ Critical training error: {str(e)}")
            logger.log_text(f"Traceback: {traceback.format_exc()}")
        
        # Attempt emergency save
        if agent is not None and logger is not None:
            try:
                emergency_path = os.path.join(
                    config.MODEL_DIR, 
                    logger.experiment_name, 
                    f"emergency_error_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
                )
                agent.save_model(emergency_path, include_memory=False)
                print(f"💾 Emergency save created: {emergency_path}")
            except Exception as save_error:
                print(f"❌ Emergency save failed: {str(save_error)}")
        
        raise
        
    finally:
        # Enhanced cleanup
        print("\n🧹 Performing enhanced cleanup...")
        
        if performance_monitor:
            try:
                performance_monitor.stop_monitoring()
                print("✅ Performance monitoring stopped")
            except Exception as e:
                print(f"⚠️ Error stopping performance monitor: {str(e)}")
        
        if env is not None:
            try:
                env.close()
                print("✅ Environment closed")
            except Exception as e:
                print(f"⚠️ Error closing environment: {str(e)}")
        
        if logger is not None:
            try:
                logger.save_training_state()
                logger.log_text("🏁 Enhanced training program completed")
                print("✅ Training state saved")
            except Exception as e:
                print(f"⚠️ Error saving final training state: {str(e)}")
        
        print("🎯 Enhanced cleanup completed")


# Import missing context manager
import contextlib

# Import the training functions from the original script
def train_one_episode(env, agent, logger, episode_num, total_steps):
    """
    Run a single training episode with enhanced monitoring.
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
        beta=beta
    )
    
    return {
        'reward': episode_reward,
        'steps': step_count,
        'avg_loss': np.mean(episode_losses) if episode_losses else None,
        'epsilon': agent.epsilon,
        'beta': beta
    }, step_count

def evaluate_agent(env, agent, logger, episode_num, num_episodes=config.EVAL_EPISODES):
    """
    Evaluate agent performance with enhanced error handling.
    """
    logger.log_text(f"🧪 Starting evaluation (training episode {episode_num})")
    
    try:
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
                action = agent.select_action(obs, evaluate=True)
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
            f"📊 Evaluation results (episode {episode_num}): "
            f"Mean reward = {mean_reward:.2f} ± {std_reward:.2f}, "
            f"Min/Max = {np.min(eval_rewards):.2f}/{np.max(eval_rewards):.2f}"
        )
        
        # Close evaluation environment
        eval_env.close()
        
        return mean_reward
        
    except Exception as e:
        logger.log_text(f"❌ Evaluation failed: {str(e)}")
        # Restore training mode even if evaluation failed
        try:
            agent.set_evaluation_mode(False)
        except:
            pass
        return 0.0

if __name__ == "__main__":
    main()