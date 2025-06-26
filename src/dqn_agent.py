"""
Deep Q-Network (DQN) agent with Prioritized Experience Replay (PER).

This module implements a DQN agent that uses PER to learn how to play
Atari games. The agent combines a Q-Network for action-value approximation
with PER for efficient experience replay.

帶有優先經驗回放 (PER) 的深度 Q 網絡 (DQN) 智能體。

本模組實現了一個使用 PER 學習如何玩 Atari 遊戲的 DQN 智能體。
該智能體結合了用於動作價值近似的 Q 網絡和用於高效經驗回放的 PER。
"""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import sys
import random
import time
import datetime
from collections import deque

# Add parent directory to path to import config.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

from src.q_network import QNetwork
from src.per_memory import PERMemory
from src.device_utils import get_device


class DQNAgent:
    """
    Deep Q-Network Agent with Prioritized Experience Replay.
    
    This agent learns to play games through trial and error using
    a deep neural network to approximate the optimal action-value function.
    It uses PER to efficiently learn from past experiences.
    
    深度 Q 網絡智能體，帶有優先經驗回放。
    
    此智能體通過嘗試和錯誤學習玩遊戲，使用深度神經網絡近似最優動作價值函數。
    它使用 PER 從過去的經驗中高效學習。
    """
    
    def __init__(self, state_shape, action_space_size,
                 learning_rate=config.LEARNING_RATE,
                 gamma=config.GAMMA,
                 epsilon_start=config.EPSILON_START,
                 epsilon_end=config.EPSILON_END,
                 epsilon_decay=config.EPSILON_DECAY,
                 memory_capacity=config.MEMORY_CAPACITY,
                 batch_size=config.BATCH_SIZE,
                 target_update_frequency=config.TARGET_UPDATE_FREQUENCY,
                 use_per=config.USE_PER,
                 per_log_frequency=config.PER_LOG_FREQUENCY,
                 evaluate_mode=config.DEFAULT_EVALUATE_MODE,
                 learning_starts=config.LEARNING_STARTS):
        """
        Initialize the DQN agent.
        
        Args:
            state_shape: Shape of the state (n_channels, height, width)
            action_space_size: Number of possible actions
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Rate of epsilon decay
            memory_capacity: Capacity of the replay buffer
            batch_size: Size of the minibatch for training
            target_update_frequency: How often to update the target network
            use_per: Whether to use Prioritized Experience Replay
            evaluate_mode: Whether to run in evaluation mode (no exploration)
            learning_starts: Number of steps before learning starts
        """
        # Store parameters
        self.state_shape = state_shape
        self.action_space_size = action_space_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.use_per = use_per
        self.evaluate_mode = evaluate_mode
        self._epsilon = epsilon_start 
        self.per_log_frequency = per_log_frequency
        self.learning_starts = learning_starts
  
        # Set device (CPU, CUDA, or MPS)
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Initialize the policy and target networks
        self.policy_network = QNetwork(state_shape, action_space_size)
        self.target_network = QNetwork(state_shape, action_space_size)
        
        # Initialize the target network with the same weights as the policy network
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()  # Set target network to evaluation mode
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        # Initialize replay memory
        if use_per:
            self.memory = PERMemory(memory_capacity)
        else:
            # Use a simple deque for uniform sampling
            self.memory = deque(maxlen=memory_capacity)
        
        # Initialize counters
        self.steps_done = 0
        self.episode_rewards = []
        self.training_steps = 0
        
        # For saving training statistics
        self.loss_history = []
        self.epsilon_history = []
        self.reward_history = []
        self.priority_history = []
        self.learning_start_time = None
    
    @property
    def epsilon(self):
        """
        Get the current epsilon value.
        
        Returns:
            float: Current epsilon value
        """
        return self._epsilon
    
    def select_action(self, state, evaluate=False):
        """
        Select an action using an epsilon-greedy policy.
        
        Args:
            state: Current state observation
            evaluate: Whether to use evaluation mode (greedy policy)
            
        Returns:
            int: Selected action
        """
        # /* TRAINING LOOP - Step 4.b.i */
        # Select action a_t using ε-greedy policy based on Q(s_t; θ)
        
        # Convert state to PyTorch tensor for network forward pass
        if isinstance(state, np.ndarray):
            # Check if state needs conversion to CHW format based on shape
            if state.ndim == 3 and state.shape[-1] in [1, 3, 4]:  # If channels dimension is last (HWC format)
                # Rearrange from [H, W, C] to [C, H, W]
                state = np.transpose(state, (2, 0, 1))
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.unsqueeze(0).to(self.device) if state.dim() == 3 else state.to(self.device)

        # Calculate current epsilon
        if evaluate:
            # Use greedy policy for evaluation
            epsilon = 0
        else:
            # Epsilon-greedy exploration strategy after learning starts            
            if self.steps_done < self.learning_starts:
                epsilon = self.epsilon_start 
            else:
                # Smoother polynomial decay formula
                # The power parameter (0.5) makes the curve more gradual than exponential decay
                progress = min(1.0, (self.steps_done - self.learning_starts) / self.epsilon_decay)
                
                if progress < 0.35:
                    # Early phase - slow decay, maintain high exploration
                    epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - progress) ** 2
                elif progress < 0.65:
                    # Middle phase - steadier decay
                    epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - progress)
                else:
                    # Final phase - accelerated decay to minimum
                    epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - progress) ** 0.5
            
            # Increment step counter
            self.steps_done += 1
            
            # Record epsilon for visualization
            if self.steps_done % 1000 == 0:
                self.epsilon_history.append((self.steps_done, epsilon))
        
        # Update instance epsilon value
        self._epsilon = epsilon
        
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            # Exploit: select best action (choose a_t = argmax_a Q(s_t, a; θ))
            with torch.no_grad():
                # Forward pass through policy network
                q_values = self.policy_network(state_tensor)
                
                # Select action with highest Q-value
                action = q_values.max(1)[1].item()
        else:
            # Explore: select random action for exploration
            action = random.randrange(self.action_space_size)
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        Store a transition in the replay memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if self.use_per:
            # For PER, initially store with maximum priority
            self.memory.add(state, action, reward, next_state, done)
        else:
            # For uniform sampling, simply append to the deque
            self.memory.append((state, action, reward, next_state, done))
    
    def _calculate_td_error(self, state, action, reward, next_state, done):
        """
        Calculate the TD error for a transition.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
            
        Returns:
            float: TD error
        """
        # Convert to tensors and add batch dimension
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)
        
        # Calculate current Q-value
        current_q = self.policy_network(state).gather(1, action.unsqueeze(1)).squeeze(1)
        
        # Calculate target Q-value
        with torch.no_grad():
            next_q = self.target_network(next_state).max(1)[0]
            target_q = reward + (1 - done) * self.gamma * next_q
        
        # Calculate TD error
        td_error = target_q - current_q
        
        return td_error.item()
    
    def optimize_model(self, logger=None):
        """
        Perform a single optimization step.
        
        Args:
            logger: Optional logger to record training metrics
            
        Returns:
            float: The loss value for this batch
            
        執行單個優化步驟。
        
        參數:
            logger: 可選的日誌記錄器，用於記錄訓練指標
            
        返回：
            float：此批次的損失值
        """
        # /* TRAINING LOOP - Step 4.b.iv */
        # Learning Phase
        
        # Check if memory has enough samples
        if self.use_per and len(self.memory) < self.batch_size:
            return 0.0
        elif not self.use_per and len(self.memory) < self.batch_size:
            return 0.0
        
        # Start timing learning if not already started
        if self.learning_start_time is None:
            self.learning_start_time = time.time()
        
        if self.use_per:
            # Divide total priority sum into B segments for balanced sampling
            # Sample batch of transitions with probability proportional to priority
            # Calculate importance sampling weights w_j = (N·P(j))^(-β) to correct bias
            indices, weights, transitions = self.memory.sample(self.batch_size)
            
            # Extract batch elements
            batch_states = np.array([t.state for t in transitions])
            batch_actions = np.array([t.action for t in transitions])
            batch_rewards = np.array([t.reward for t in transitions])
            batch_next_states = np.array([t.next_state for t in transitions])
            batch_dones = np.array([t.done for t in transitions], dtype=np.float32)
            
            # Convert importance sampling weights to tensor
            batch_weights = torch.FloatTensor(weights).to(self.device)
        else:
            # Uniform sampling
            transitions = random.sample(self.memory, self.batch_size)
            
            # Extract batch elements
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = zip(*transitions)
            
            # No importance sampling weights for uniform sampling
            batch_weights = torch.ones(self.batch_size).to(self.device)
        
        # Convert to tensors
        batch_states = torch.FloatTensor(np.array(batch_states)).to(self.device)
        batch_actions = torch.LongTensor(np.array(batch_actions)).to(self.device)
        batch_rewards = torch.FloatTensor(np.array(batch_rewards)).to(self.device)
        batch_next_states = torch.FloatTensor(np.array(batch_next_states)).to(self.device)
        batch_dones = torch.FloatTensor(np.array(batch_dones)).to(self.device)
        
        # Compute current Q values: Q(s_j, a_j; θ)
        current_q_values = self.policy_network(batch_states).gather(1, batch_actions.unsqueeze(1))
        
        # Compute max Q values for next states using the target network: max_a Q'(s_{j+1}, a; θ')
        with torch.no_grad():
            next_q_values = self.target_network(batch_next_states).max(1)[0]
        
        # Compute the expected (or target) Q values
        # y_j = r_j if done (terminal state)
        #     = r_j + γ·max_a Q'(s_{j+1}, a; θ') otherwise (non-terminal)
        expected_q_values = batch_rewards + (1 - batch_dones) * self.gamma * next_q_values
        
        # Reshape for loss calculation
        expected_q_values = expected_q_values.unsqueeze(1)
        
        # Compute TD-error δ_j = y_j - Q(s_j, a_j; θ)
        # This is used for updating priorities in the memory
        td_errors = (expected_q_values - current_q_values).detach()
        
        # Update priorities in D using |TD-error|^α + ε
        # where α determines how much prioritization is used
        # and ε ensures non-zero probability
        if self.use_per:
            priorities = td_errors.abs().cpu().numpy().flatten()
            self.memory.update_priorities(indices, priorities)
            
            # Record priority statistics for visualization
            if self.training_steps % 100 == 0:
                self.priority_history.append((self.training_steps, priorities.mean()))
        
        # Calculate the weighted loss L = 1/B · Σ w_j · (δ_j)²
        # Importance sampling weights correct the bias introduced by prioritized sampling
        element_wise_loss = F.smooth_l1_loss(current_q_values, expected_q_values, reduction='none')
        
        # Ensure dimensions match, adjust batch_weights to [batch_size, 1] shape to match element_wise_loss
        batch_weights = batch_weights.view(-1, 1)
        
        # Apply weights and take mean
        weighted_loss = (batch_weights * element_wise_loss)
        loss = weighted_loss.mean()
        
        # Perform gradient descent step on L with respect to θ
        self.optimizer.zero_grad()  # Zero gradients from previous step
        loss.backward()  # Compute gradients
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), config.GRAD_CLIP_NORM)
        
        # Apply gradients to update network parameters
        self.optimizer.step()
        
        # Increment training steps
        self.training_steps += 1
        
        # Update target network if needed
        if self.training_steps % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            print(f"Target network updated at step {self.training_steps}")
        
        # Record loss for visualization
        if self.training_steps % 100 == 0:
            self.loss_history.append((self.training_steps, loss.item()))
        
        # Update beta parameter in PER memory
        if self.use_per:
            self.memory.update_beta(self.steps_done)
            
            # Only log PER metrics at specified reduced frequency to avoid excessive data
            if logger is not None and self.training_steps % self.per_log_frequency == 0:
                # Get the current beta value from memory
                current_beta = self.memory.beta
                
                # Log PER update metrics - pass only summary statistics to logger
                logger.log_per_update(
                    self.steps_done,
                    current_beta,
                    priorities,
                    td_errors.cpu().numpy().flatten(),
                    batch_weights.cpu().numpy().flatten()
                )
        
        return loss.item()
    
    def set_evaluation_mode(self, evaluate=True):
        """
        Set whether the agent is in evaluation mode.
        
        Args:
            evaluate: Whether to enable evaluation mode
        """
        self.evaluate_mode = evaluate
        if evaluate:
            self.policy_network.eval()
        else:
            self.policy_network.train()
    
    def save_model(self, path, save_optimizer=True, include_memory=False, metadata=None):
        """
        Enhanced model saving with verification and comprehensive metadata.
        
        Args:
            path: Path to save the model
            save_optimizer: Whether to save optimizer state
            include_memory: Whether to include replay memory in the save (can be large)
            metadata: Additional metadata to include in the save
            
        保存模型權重到文件（增強版本）。
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Prepare comprehensive model state
            model_state = {
                'policy_network': self.policy_network.state_dict(),
                'target_network': self.target_network.state_dict(),
                'steps_done': self.steps_done,
                'training_steps': self.training_steps,
                'epsilon': self._epsilon,
                'timestamp': datetime.datetime.now().isoformat(),
                'model_config': {
                    'state_shape': self.state_shape,
                    'action_space_size': self.action_space_size,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'gamma': self.gamma,
                    'batch_size': self.batch_size,
                    'use_per': self.use_per
                },
                'device': str(self.device),
                'pytorch_version': torch.__version__
            }
            
            # Add optimizer state if requested
            if save_optimizer:
                model_state['optimizer'] = self.optimizer.state_dict()
            
            # Add memory state if requested and using PER
            if include_memory and self.use_per and hasattr(self.memory, 'get_performance_stats'):
                try:
                    memory_stats = self.memory.get_performance_stats()
                    model_state['memory_stats'] = memory_stats
                    print(f"Including memory statistics in save: {memory_stats['samples_drawn']} samples drawn")
                except Exception as e:
                    print(f"Warning: Could not save memory stats: {str(e)}")
            
            # Add custom metadata if provided
            if metadata:
                model_state['metadata'] = metadata
            
            # Add performance metrics if available
            if hasattr(self, 'loss_history') and self.loss_history:
                model_state['metrics'] = {
                    'loss_history': self.loss_history[-100:],  # Keep last 100 entries
                    'reward_history': getattr(self, 'reward_history', [])[-100:],
                    'epsilon_history': getattr(self, 'epsilon_history', [])[-100:],
                }
            
            # Save to temporary file first, then move to final location
            temp_path = path + '.tmp'
            torch.save(model_state, temp_path)
            
            # Verify the saved file can be loaded
            try:
                verification_state = torch.load(temp_path, map_location='cpu')
                required_keys = ['policy_network', 'target_network', 'steps_done', 'training_steps']
                for key in required_keys:
                    if key not in verification_state:
                        raise ValueError(f"Missing required key: {key}")
                
                # Move temporary file to final location
                os.rename(temp_path, path)
                print(f"Model successfully saved and verified: {path}")
                return True
                
            except Exception as e:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise ValueError(f"Model verification failed: {str(e)}")
                
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, path, strict=True, device_override=None):
        """
        Enhanced model loading with validation and device handling.
        
        Args:
            path: Path to the saved model
            strict: Whether to strictly enforce state dict loading
            device_override: Override device selection
            
        Returns:
            dict: Loaded model metadata and status
            
        加載模型權重和狀態（增強版本）。
        """
        if not os.path.exists(path):
            error_msg = f"Model file {path} not found."
            print(error_msg)
            return {'success': False, 'error': error_msg}
        
        try:
            # Determine device for loading
            load_device = device_override if device_override else self.device
            
            # Load model state
            print(f"Loading model from {path} to device {load_device}")
            model_state = torch.load(path, map_location=load_device)
            
            # Validate required keys
            required_keys = ['policy_network', 'target_network']
            missing_keys = [key for key in required_keys if key not in model_state]
            if missing_keys:
                error_msg = f"Missing required keys in model file: {missing_keys}"
                print(error_msg)
                return {'success': False, 'error': error_msg}
            
            # Load network states
            try:
                self.policy_network.load_state_dict(model_state['policy_network'], strict=strict)
                self.target_network.load_state_dict(model_state['target_network'], strict=strict)
                print("Network states loaded successfully")
            except Exception as e:
                error_msg = f"Error loading network states: {str(e)}"
                print(error_msg)
                return {'success': False, 'error': error_msg}
            
            # Load training state
            if 'steps_done' in model_state:
                self.steps_done = model_state['steps_done']
            if 'training_steps' in model_state:
                self.training_steps = model_state['training_steps']
            if 'epsilon' in model_state:
                self._epsilon = model_state['epsilon']
            
            # Load optimizer state if available
            if 'optimizer' in model_state:
                try:
                    self.optimizer.load_state_dict(model_state['optimizer'])
                    print("Optimizer state loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load optimizer state: {str(e)}")
            
            # Load performance metrics if available
            if 'metrics' in model_state:
                metrics = model_state['metrics']
                self.loss_history = metrics.get('loss_history', [])
                self.reward_history = metrics.get('reward_history', [])
                self.epsilon_history = metrics.get('epsilon_history', [])
                print(f"Loaded {len(self.loss_history)} loss history entries")
            
            # Prepare return metadata
            load_info = {
                'success': True,
                'timestamp': model_state.get('timestamp', 'Unknown'),
                'steps_done': self.steps_done,
                'training_steps': self.training_steps,
                'epsilon': self._epsilon,
                'device': str(load_device),
                'model_config': model_state.get('model_config', {}),
            }
            
            # Add memory statistics if available
            if 'memory_stats' in model_state:
                load_info['memory_stats'] = model_state['memory_stats']
            
            print(f"Model loaded successfully. Steps: {self.steps_done}, Training steps: {self.training_steps}")
            return load_info
            
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            print(error_msg)
            return {'success': False, 'error': error_msg}
    
    def get_training_diagnostics(self):
        """
        Get comprehensive training diagnostics for monitoring and debugging.
        
        獲取用於監控和調試的綜合訓練診斷信息。
        
        Returns:
            dict: Training diagnostics
        """
        diagnostics = {
            'training_progress': {
                'steps_done': self.steps_done,
                'training_steps': self.training_steps,
                'epsilon': self._epsilon,
                'learning_phase': 'learning' if self.steps_done >= self.learning_starts else 'warmup'
            },
            'network_info': {
                'device': str(self.device),
                'policy_network_params': sum(p.numel() for p in self.policy_network.parameters()),
                'target_network_params': sum(p.numel() for p in self.target_network.parameters()),
                'trainable_params': sum(p.numel() for p in self.policy_network.parameters() if p.requires_grad)
            },
            'memory_info': {
                'use_per': self.use_per,
                'memory_size': len(self.memory) if hasattr(self.memory, '__len__') else 'Unknown'
            }
        }
        
        # Add PER-specific diagnostics
        if self.use_per and hasattr(self.memory, 'get_performance_stats'):
            try:
                per_stats = self.memory.get_performance_stats()
                diagnostics['per_info'] = per_stats
            except Exception as e:
                diagnostics['per_info'] = {'error': str(e)}
        
        # Add recent performance metrics
        if hasattr(self, 'loss_history') and self.loss_history:
            recent_losses = [loss for _, loss in self.loss_history[-10:]]
            diagnostics['recent_performance'] = {
                'avg_loss_last_10': np.mean(recent_losses) if recent_losses else 0,
                'loss_history_length': len(self.loss_history)
            }
        
        return diagnostics


# For testing purposes
if __name__ == "__main__":
    # Create a DQN agent
    state_shape = (4, 84, 84)  # 4 stacked frames of 84x84 pixels
    action_space_size = config.ACTION_SPACE_SIZE
    
    # Test with PER
    agent_per = DQNAgent(state_shape, action_space_size, use_per=True)
    print(f"Agent created with device: {agent_per.device}")
    
    # Test action selection
    state = np.random.rand(*state_shape)
    action = agent_per.select_action(state)
    print(f"Selected action: {action}")
    
    # Test storing transitions
    for _ in range(10):
        state = np.random.rand(*state_shape)
        action = np.random.randint(0, action_space_size)
        reward = np.random.randn()
        next_state = np.random.rand(*state_shape)
        done = np.random.random() > 0.8
        
        agent_per.store_transition(state, action, reward, next_state, done)
    
    print(f"Memory size: {len(agent_per.memory)}")