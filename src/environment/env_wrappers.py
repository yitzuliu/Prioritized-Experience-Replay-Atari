"""
Environment wrappers for Atari games processing.

This module provides a set of wrappers for preprocessing Atari game frames and
implementing essential modifications for reinforcement learning training.
Key functionalities include:
1. Frame preprocessing (grayscaling, resizing, normalization)
2. Frame stacking for temporal information
3. Action repetition and frame skipping
4. Episode termination handling
5. Reward clipping for stability

These wrappers follow best practices established in DQN literature.
"""

import gymnasium as gym
import config
import sys
import ale_py
import numpy as np
import cv2
from collections import deque
import torch
from gymnasium import spaces

def make_env(env_name=config.ENV_NAME, render_mode=config.RENDER_MODE, difficulty=config.DIFFICULTY, mode=config.MODE):
    """
    Create an Atari Ice Hockey environment with no wrappers.
    
    This function creates a raw gymnasium environment for Ice Hockey without 
    any preprocessing or wrappers. The environment will use the original frame
    rate, image dimensions, and reward structure of the Atari game.
    
    Args:
        env_name (str): Name of the Atari environment to create
            (default: value from config.ENV_NAME)
        render_mode (str): Rendering mode, None for no rendering or 'human' for
            visualization (default: value from config.RENDER_MODE)
        difficulty (int): Game difficulty level (0-3)
            (default: value from config.DIFFICULTY)
        mode (int): Game mode (0-3)
            (default: value from config.MODE)
        
    Returns:
        gym.Env: A raw gymnasium environment for DQN training
    """
    try:
        # Try to create the environment with specified difficulty and mode
        env = gym.make(
            env_name, 
            render_mode=render_mode,
            difficulty=difficulty,
            mode=mode
        )
        return env
    except gym.error.NamespaceNotFound:
        print("\nERROR: Atari Learning Environment (ALE) not found.")
        print("Please install the required packages with these commands:")
        print("pip install ale-py gymnasium[atari,accept-rom-license]")
        print("OR")
        print("pip install ale-py")
        print("pip install gymnasium[atari]")
        print("python -m ale_py.roms --install-dir <path-to-roms>\n")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: An unexpected error occurred: {e}")
        print("Please make sure all dependencies are installed correctly.\n")
        sys.exit(1)

class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    
    No-op is a no operation, meaning the agent doesn't take any action.
    This wrapper helps the agent explore different game states by introducing
    randomness at the beginning of each episode.
    """
    def __init__(self, env, noop_max=30):
        """
        Initialize the wrapper with the given environment.
        
        Args:
            env: The environment to wrap
            noop_max (int): Maximum number of random no-ops to perform at reset
        """
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0  # NOOP action is typically 0 in Atari
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'
        
    def reset(self, **kwargs):
        """Execute random number of no-ops after environment reset."""
        obs, info = self.env.reset(**kwargs)
        
        # Execute between 1 and noop_max no-ops
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        
        return obs, info

class FireResetEnv(gym.Wrapper):
    """
    Wrapper for environments where 'FIRE' action is required to start the game.
    
    Some Atari games, including Ice Hockey, require pressing the FIRE button to
    start the game or to restart after losing a life. This wrapper automatically
    presses FIRE when needed.
    """
    def __init__(self, env):
        """
        Initialize the wrapper with the given environment.
        
        Args:
            env: The environment to wrap
        """
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3
        
    def reset(self, **kwargs):
        """Reset environment and press FIRE to start the game."""
        obs, info = self.env.reset(**kwargs)
        
        # Execute 'FIRE' action to start the game
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE action
        
        # If game terminated after FIRE action, reset again
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
            
        return obs, info

class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life treated as end-of-episode, but only reset on true game over.
    
    This makes it easier for the agent to learn about life consequences, as it
    treats each life as a separate episode but preserves the game state between lives.
    """
    def __init__(self, env):
        """
        Initialize the wrapper with the given environment.
        
        Args:
            env: The environment to wrap
        """
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done = True
        
    def step(self, action):
        """
        Execute action and check for life loss.
        
        Returns 'done' signal if a life was lost but game is not truly over.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated
        
        # Check if the agent lost a life
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # Signal episode end when a life is lost
            terminated = True
            
        self.lives = lives
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """
        Reset only when lives are exhausted.
        
        When real 'done' occurs, resets the environment fully.
        When life lost but game not over, only take FIRE action.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Reset after life lost but not game over (just press FIRE)
            obs, _, _, _, info = self.env.step(1)  # FIRE action
            
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

class MaxAndSkipEnv(gym.Wrapper):
    """
    Skip frames and return max pixel values.
    
    Return only every `skip`-th frame and take maximum over the skipped frames.
    This reduces computation cost and handles flickering in some Atari games.
    """
    def __init__(self, env, skip=4):
        """
        Initialize the wrapper with the given environment.
        
        Args:
            env: The environment to wrap
            skip (int): Number of frames to skip
        """
        super(MaxAndSkipEnv, self).__init__(env)
        self._skip = skip
        
        # Most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        
    def step(self, action):
        """
        Repeat action, sum reward, and max over last observations.
        
        Executes the same action for 'skip' frames, taking the maximum pixel
        values over the last two frames to handle flickering.
        """
        total_reward = 0.0
        terminated = truncated = False
        
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            terminated = terminated or term
            truncated = truncated or trunc
            
            self._obs_buffer.append(obs)
            
            if terminated or truncated:
                break
                
        # Max pooling over the most recent observations
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        
        return max_frame, total_reward, terminated, truncated, info

class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip rewards to {+1, 0, -1}.
    
    This makes the reward signal less sparse and easier to learn from.
    It also makes the scale of rewards consistent across different games.
    """
    def __init__(self, env):
        """
        Initialize the wrapper with the given environment.
        
        Args:
            env: The environment to wrap
        """
        super(ClipRewardEnv, self).__init__(env)
        
    def reward(self, reward):
        """Clip rewards to {+1, 0, -1}."""
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    """
    Warp frames to 84x84 as done in the Nature paper.
    
    This reduces the dimensionality of the input and makes it easier to process
    with convolutional neural networks.
    """
    def __init__(self, env, width=84, height=84, grayscale=True, gpu_resize=True):
        """
        Initialize the wrapper with the given environment.
        
        Args:
            env: The environment to wrap
            width (int): Width to resize frames to
            height (int): Height to resize frames to
            grayscale (bool): Whether to convert frames to grayscale
            gpu_resize (bool): Use GPU for frame resizing if available
        """
        super(WarpFrame, self).__init__(env)
        self.width = width
        self.height = height
        self.grayscale = grayscale
        
        # Check for CUDA, MPS (Apple Silicon), or CPU
        if gpu_resize:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.gpu_resize = True
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.gpu_resize = True
            else:
                self.gpu_resize = False
        else:
            self.gpu_resize = False
        
        if self.grayscale:
            num_channels = 1
        else:
            num_channels = 3
            
        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.height, self.width, num_channels),
            dtype=np.uint8
        )
        
        # Update the observation space
        self.observation_space = new_space
        
        # Set up GPU tensors for frame processing if using GPU
        if self.gpu_resize:
            # Create reusable GPU tensors - optimize by pre-allocating memory
            # Use the exact input frame size (typically 210x160x3 for Atari)
            frame_height, frame_width = 210, 160
            if hasattr(env.observation_space, 'shape'):
                if len(env.observation_space.shape) == 3:
                    frame_height, frame_width = env.observation_space.shape[:2]
            
            self.gpu_frame = torch.zeros((1, 3, frame_height, frame_width), 
                                       dtype=torch.float32, 
                                       device=self.device)
            
    def observation(self, frame):
        """
        Process a frame by converting to grayscale and resizing.
        
        Args:
            frame: The raw frame from the environment
            
        Returns:
            The processed frame
        """
        if self.gpu_resize and torch.cuda.is_available():
            # Process on GPU
            return self._gpu_process_frame(frame)
        else:
            # Process on CPU
            return self._cpu_process_frame(frame)
    
    def _cpu_process_frame(self, frame):
        """Process frame using CPU with OpenCV"""
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
        # Resize using OpenCV
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        
        if self.grayscale:
            frame = np.expand_dims(frame, -1)  # Add channel dimension
            
        return frame
    
    def _gpu_process_frame(self, frame):
        """Process frame using GPU with PyTorch"""
        # Convert to PyTorch tensor and move to GPU
        with torch.no_grad():
            try:
                # Copy frame data to pre-allocated GPU tensor
                self.gpu_frame[0] = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
                
                # Convert to grayscale if needed
                if self.grayscale:
                    # RGB to grayscale conversion weights
                    grayscale = self.gpu_frame[0][0] * 0.299 + self.gpu_frame[0][1] * 0.587 + self.gpu_frame[0][2] * 0.114
                    grayscale = grayscale.unsqueeze(0)  # Add channel dimension
                else:
                    grayscale = self.gpu_frame[0]
                    
                # Resize using GPU
                resized = torch.nn.functional.interpolate(
                    grayscale.unsqueeze(0),  # Add batch dimension
                    size=(self.height, self.width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)  # Remove batch dimension
                
                # Convert back to numpy and rescale to [0, 255]
                if self.grayscale:
                    result = (resized * 255).byte().cpu().numpy().transpose(1, 2, 0)
                else:
                    result = (resized * 255).byte().cpu().numpy().transpose(1, 2, 0)
                
                # Ensure memory is freed
                del resized
                del grayscale
                torch.cuda.empty_cache()
                    
                return result
                
            except RuntimeError as e:
                # Handle CUDA out of memory error
                if "CUDA out of memory" in str(e):
                    print(f"GPU memory error in frame processing: {e}")
                    print("Falling back to CPU processing...")
                    return self._cpu_process_frame(frame)
                else:
                    raise e

class FrameStack(gym.Wrapper):
    """
    Stack n last frames.
    
    This wrapper stacks several consecutive frames together to provide
    temporal information, which is important for learning in Atari games.
    """
    def __init__(self, env, n_frames=4):
        """
        Initialize the wrapper with the given environment.
        
        Args:
            env: The environment to wrap
            n_frames (int): Number of frames to stack
        """
        super(FrameStack, self).__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        
        # Update observation space to account for stacked frames
        wrapped_obs_space = env.observation_space
        low = np.repeat(wrapped_obs_space.low, n_frames, axis=-1)
        high = np.repeat(wrapped_obs_space.high, n_frames, axis=-1)
        self.observation_space = spaces.Box(
            low=low, high=high, dtype=wrapped_obs_space.dtype
        )
        
    def reset(self, **kwargs):
        """
        Reset environment and initialize frame stack with duplicates of first observation.
        """
        obs, info = self.env.reset(**kwargs)
        [self.frames.append(obs) for _ in range(self.n_frames)]
        return self._get_observation(), info
    
    def step(self, action):
        """
        Execute action, add new frame to stack, and return stacked observation.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        Combine stacked frames into a single observation.
        
        Concatenates frames along the channel dimension for RGB or grayscale input.
        """
        # Stack frames along the channel dimension
        return np.concatenate(list(self.frames), axis=-1)

class PyTorchFrame(gym.ObservationWrapper):
    """
    Convert observations to PyTorch format.
    
    Transposes the dimensions from (H, W, C) to (C, H, W) as PyTorch expects
    and converts to floating point normalized to [0, 1].
    """
    def __init__(self, env):
        """
        Initialize the wrapper with the given environment.
        
        Args:
            env: The environment to wrap
        """
        super(PyTorchFrame, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0,
            shape=(obs_shape[-1], obs_shape[0], obs_shape[1]),
            dtype=np.float32
        )
        
    def observation(self, observation):
        """
        Convert observation to PyTorch-friendly format.
        
        - Converts from uint8 to float32
        - Normalizes values to [0, 1]
        - Moves channel dim from last to first position (HWC -> CHW)
        """
        # Convert to float and scale to [0, 1]
        observation = np.array(observation, dtype=np.float32) / 255.0
        
        # Move channel dim to first position (HWC -> CHW)
        observation = np.transpose(observation, (2, 0, 1))
        
        return observation

def make_atari_env(env_name=config.ENV_NAME, 
                  render_mode=config.RENDER_MODE,
                  frame_skip=config.FRAME_SKIP, 
                  frame_stack=config.FRAME_STACK,
                  difficulty=config.DIFFICULTY,
                  mode=config.MODE,
                  clip_rewards=True, 
                  episode_life=True, 
                  use_pytorch_format=True,
                  gpu_acceleration=True,
                  force_training_mode=None):
    """
    Create a wrapped Atari environment ready for DQN training.
    
    Applies a series of wrappers to make the environment suitable for
    reinforcement learning with deep neural networks.
    
    Args:
        env_name (str): Name of the Atari environment
        render_mode (str): Rendering mode, None for no rendering, 'human' for visualization
        frame_skip (int): Number of frames to skip (action repeat)
        frame_stack (int): Number of frames to stack
        difficulty (int): Game difficulty level (0-3)
        mode (int): Game mode (0-3)
        clip_rewards (bool): Whether to clip rewards to {-1, 0, 1}
        episode_life (bool): Whether to treat loss of life as episode termination
        use_pytorch_format (bool): Whether to convert observations to PyTorch format
        gpu_acceleration (bool): Whether to use GPU for frame processing if available
        force_training_mode (bool): Override config.TRAINING_MODE if provided
        
    Returns:
        gym.Env: A fully preprocessed and wrapped Atari environment
    """
    # Determine if we're in training mode
    is_training = force_training_mode if force_training_mode is not None else config.TRAINING_MODE
    
    # Determine effective render mode based on training mode and provided render_mode
    if is_training and render_mode == config.RENDER_MODE:
        effective_render_mode = None
        if config.RENDER_MODE == 'human':
            print("Warning: TRAINING_MODE=True is overriding RENDER_MODE='human'. "
                  "Set TRAINING_MODE=False in config.py if you want to see visualization.")
    else:
        effective_render_mode = render_mode
        
    # Create base environment with specified parameters
    env = make_env(env_name, effective_render_mode, difficulty, mode)
    
    # Apply appropriate wrappers in sequence
    env = NoopResetEnv(env, noop_max=config.NOOP_MAX)
    
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    
    env = MaxAndSkipEnv(env, skip=frame_skip)
    
    if episode_life:
        env = EpisodicLifeEnv(env)
    
    env = WarpFrame(
        env, 
        width=config.FRAME_WIDTH, 
        height=config.FRAME_HEIGHT, 
        grayscale=True,
        gpu_resize=gpu_acceleration and torch.cuda.is_available()
    )
    
    if clip_rewards:
        env = ClipRewardEnv(env)
    
    if frame_stack > 1:
        env = FrameStack(env, n_frames=frame_stack)
    
    if use_pytorch_format:
        env = PyTorchFrame(env)
    
    return env

if __name__ == "__main__":
    import src.utils.utils as utils
    
    # Print system information
    system_info = utils.get_system_info()
    print("System Information:")
    print(f"OS: {system_info['os']} {system_info['os_version']}")
    print(f"CPU: {system_info['cpu_type']} ({system_info['cpu_count']} cores)")
    print(f"PyTorch: {system_info['torch_version']}")
    
    if system_info.get('cuda_available', False):
        print(f"GPU: {system_info.get('gpu_name', 'Unknown')} ({system_info.get('gpu_memory_gb', 'Unknown')} GB)")
    elif system_info.get('mps_available', False):
        print("GPU: Apple Silicon (Metal)")
    else:
        print("GPU: None detected, using CPU only")
    print("-" * 40)
    
    # Test environment setup
    print("Attempting to create the Ice Hockey environment...")
    env = make_env()
    print(f"Environment: {config.ENV_NAME}")
    print(f"Difficulty: {config.DIFFICULTY}")
    print(f"Mode: {config.MODE}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Display some action meanings (first 5)
    action_meanings = env.unwrapped.get_action_meanings()
    print(f"Sample actions: {', '.join(action_meanings[:5])}...")
    env.close()
    
    # Test wrapped environment
    print("\nCreating wrapped environment for testing...")
    wrapped_env = make_atari_env(render_mode='human')
    print(f"Wrapped observation space: {wrapped_env.observation_space}")
    print(f"Wrapped action space: {wrapped_env.action_space}")
    
    # Try a few steps
    obs, info = wrapped_env.reset()
    print(f"Observation shape: {obs.shape}, dtype: {obs.dtype}")
    
    for i in range(10):
        action = wrapped_env.action_space.sample()
        next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
        print(f"Step {i+1}: Action={action}, Reward={reward}, Done={terminated or truncated}")
        if terminated or truncated:
            break
    
    wrapped_env.close()
    print("\nEnvironment test completed successfully!")

