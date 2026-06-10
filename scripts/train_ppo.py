"""PPO training and evaluation for log collection in MineRLObtainDiamondShovel-v0."""

import os
import gym
import minerl
import numpy as np
import psutil
import time
import torch
import gc
import matplotlib.pyplot as plt
import cv2
import traceback  # Added for better error reporting
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import BaseCallback
from datetime import datetime

# Configuration parameters
LOG_DIR = "results/ppo"
FRAMES_DIR = "results/ppo/frames"
ENV_ID = "MineRLObtainDiamondShovel-v0"
MAX_RAM_USAGE_GB = 24.0  # Maximum RAM usage in GB
PROGRESS_LOG_INTERVAL = 10  # Log progress every X steps
SHOWFRAMES = False  # Set True to save episode frames during training

# Frame processing parameters - Using small 64x64 for memory efficiency
TARGET_WIDTH = 64    # Target width for resized frames
TARGET_HEIGHT = 64   # Target height for resized frames

# User-configurable parameters
MAX_STEPS_PER_EPISODE = 1000  # Maximum steps per episode
TOTAL_EPISODES = 50  # Total number of episodes to run
TOTAL_TIMESTEPS = 1000  # Total timesteps for training
LEARNING_RATE = 1e-3  # Learning rate

# Create frames directory if SHOWFRAMES is enabled
if SHOWFRAMES:
    os.makedirs(FRAMES_DIR, exist_ok=True)
    print(f"Frame images will be saved to: {FRAMES_DIR}")

# === Observation Flattening Wrapper ===
# This wrapper changes the Dict observation space to Box space for the POV only
class FlattenObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # If the original observation space is a Dict, extract POV space
        if isinstance(env.observation_space, gym.spaces.Dict) and 'pov' in env.observation_space.spaces:
            # We expect the raw observation to be 720×1280×3 but will resize it to TARGET_HEIGHT×TARGET_WIDTH×3
            self.observation_space = spaces.Box(
                low=0, 
                high=255, 
                shape=(TARGET_HEIGHT, TARGET_WIDTH, 3), 
                dtype=np.uint8
            )
            print(f"Flattened observation space: {self.observation_space}")
        else:
            # If not a Dict or no 'pov', keep as is
            self.observation_space = env.observation_space
            print(f"Using original observation space: {self.observation_space}")

    def observation(self, observation):
        # Extract POV observation if the observation is a Dict
        if isinstance(observation, dict) and 'pov' in observation:
            pov = observation['pov']
            # Debug: Check the shape and values of the raw POV
            print(f"Original POV shape: {pov.shape}, dtype: {pov.dtype}, min: {np.min(pov)}, max: {np.max(pov)}")
            
            # Ensure POV has the right shape
            if pov.size == 2764800:  # 720×1280×3
                pov = pov.reshape((720, 1280, 3))
                
            # Resize to smaller dimensions for memory efficiency
            try:
                pov = cv2.resize(
                    pov, 
                    (TARGET_WIDTH, TARGET_HEIGHT),  # OpenCV expects (width, height)
                    interpolation=cv2.INTER_AREA
                )
                # Debug: Check the shape and values after resize
                print(f"Resized POV shape: {pov.shape}, dtype: {pov.dtype}, min: {np.min(pov)}, max: {np.max(pov)}")
                return pov
            except Exception as e:
                print(f"ERROR in resizing observation: {e}")
                traceback.print_exc()
                # Return fallback observation
                return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        return observation

# === Episode Termination Wrapper (to limit episode length) ===
class EpisodeLengthWrapper(gym.Wrapper):
    def __init__(self, env, max_steps=MAX_STEPS_PER_EPISODE):
        super().__init__(env)
        self.max_steps = max_steps
        self.current_step = 0
        
    def reset(self):
        self.current_step = 0
        return self.env.reset()
        
    def step(self, action):
        try:
            obs, reward, done, info = self.env.step(action)
            self.current_step += 1
            
            # End episode if max steps reached
            if self.current_step >= self.max_steps:
                done = True
                info["TimeLimit.truncated"] = True
                
            return obs, reward, done, info
        except Exception as e:
            print(f"ERROR in EpisodeLengthWrapper.step: {e}")
            traceback.print_exc()
            # Return fallback values
            return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8), 0.0, True, {}

# === Frame Processing Utility Functions ===
def save_frame_as_image(frame, episode, step, directory=FRAMES_DIR):
    """Save a frame as an image file."""
    try:
        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate a filename with episode, step, and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(directory, f"episode_{episode}_step_{step}_{timestamp}.png")
        
        # Debug: Check the shape and values of the frame
        print(f"Frame to save shape: {frame.shape}, dtype: {frame.dtype}, min: {np.min(frame)}, max: {np.max(frame)}")
        
        # Ensure the frame is in a proper format for saving as an RGB image
        if len(frame.shape) == 2:  # Grayscale
            # Convert grayscale to RGB (duplicate to all channels)
            frame = np.stack([frame, frame, frame], axis=2)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:  # Single channel with explicit dimension
            # Convert to RGB (duplicate to all channels)
            frame = np.concatenate([frame, frame, frame], axis=2)
        elif len(frame.shape) == 3 and frame.shape[0] in [1, 3]:  # Channel-first format
            # Convert from CHW to HWC
            frame = np.transpose(frame, (1, 2, 0))
            if frame.shape[2] == 1:  # Single channel
                # Duplicate to all channels
                frame = np.concatenate([frame, frame, frame], axis=2)
        
        # Save the image
        plt.imsave(filename, frame)
        print(f"Saved frame to: {filename}")
        return filename
    except Exception as e:
        print(f"ERROR in save_frame_as_image: {e}")
        traceback.print_exc()
        return None

class ResizeObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env, size=(TARGET_HEIGHT, TARGET_WIDTH)):
        super().__init__(env)
        self.size = size
        
        # Update observation space to match new size
        if isinstance(env.observation_space, gym.spaces.Box):
            # If observation space is already a Box
            if len(env.observation_space.shape) == 3:  # Image observation
                low = env.observation_space.low
                high = env.observation_space.high
                # Create a new observation space with the resized dimensions
                if low.shape[0] == 3:  # If channels first (C, H, W)
                    self.observation_space = gym.spaces.Box(
                        low=np.min(low),
                        high=np.max(high),
                        shape=(3, size[0], size[1]),  # (C, H, W)
                        dtype=env.observation_space.dtype
                    )
                else:  # If channels last (H, W, C)
                    self.observation_space = gym.spaces.Box(
                        low=np.min(low),
                        high=np.max(high),
                        shape=(size[0], size[1], 3),  # (H, W, C)
                        dtype=env.observation_space.dtype
                    )
        
    def observation(self, observation):
        try:
            # Handle direct observation (after FlattenObservationWrapper)
            if isinstance(observation, np.ndarray):
                # Debug: Check the shape and values of the observation
                print(f"ResizeObservationWrapper input shape: {observation.shape}, dtype: {observation.dtype}")
                
                resized_obs = cv2.resize(
                    observation, 
                    (self.size[1], self.size[0]),  # OpenCV expects (width, height)
                    interpolation=cv2.INTER_AREA
                )
                
                # Debug: Check the shape and values after resize
                print(f"ResizeObservationWrapper output shape: {resized_obs.shape}, dtype: {resized_obs.dtype}")
                return resized_obs
            
            # If we still have a dict (e.g., FlattenObservationWrapper not used)
            if isinstance(observation, dict) and 'pov' in observation:
                original_pov = observation['pov']
                
                # Debug: Check the shape and values of the original POV
                print(f"Original POV shape: {original_pov.shape}, dtype: {original_pov.dtype}")
                
                resized_pov = cv2.resize(
                    original_pov, 
                    (self.size[1], self.size[0]),  # OpenCV expects (width, height)
                    interpolation=cv2.INTER_AREA
                )
                
                # Debug: Check the shape and values after resize
                print(f"Resized POV shape: {resized_pov.shape}, dtype: {resized_pov.dtype}")
                
                observation['pov'] = resized_pov
            
            return observation
        except Exception as e:
            print(f"ERROR in ResizeObservationWrapper.observation: {e}")
            traceback.print_exc()
            # Return fallback observation
            if isinstance(observation, dict) and 'pov' in observation:
                observation['pov'] = np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
                return observation
            else:
                return np.zeros((self.size[0], self.size[1], 3), dtype=np.uint8)
        
# === Frame Rendering Wrapper ===
class FrameRenderWrapper(gym.Wrapper):
    def __init__(self, env, render_frames=False):
        super().__init__(env)
        self.render_frames = render_frames
        self.fig = None
        self.ax = None
        self.episode_count = 0
        
    def reset(self):
        try:
            obs = self.env.reset()
            self.episode_count += 1
            self.step_count = 0
            
            # Debug: Check the observation from reset
            if isinstance(obs, np.ndarray):
                print(f"Reset observation shape: {obs.shape}, dtype: {obs.dtype}, min: {np.min(obs)}, max: {np.max(obs)}")
            elif isinstance(obs, dict) and 'pov' in obs:
                print(f"Reset POV shape: {obs['pov'].shape}, dtype: {obs['pov'].dtype}, min: {np.min(obs['pov'])}, max: {np.max(obs['pov'])}")
            
            # Render and save the first frame if enabled
            if self.render_frames:
                self._render_frame(obs, is_first_frame=True)
            
            return obs
        except Exception as e:
            print(f"ERROR in FrameRenderWrapper.reset: {e}")
            traceback.print_exc()
            # Return fallback observation
            return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
        
    def step(self, action):
        try:
            # Debug: Print action before stepping
            print(f"FrameRenderWrapper.step: action type: {type(action)}, value: {action}")
            
            obs, reward, done, info = self.env.step(action)
            self.step_count += 1
            
            # Check for errors in the info dictionary
            if 'error' in info:
                print(f"Error in step: {info['error']}")
                # Handle the error by returning a fallback observation
                return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8), 0.0, True, info

            # Render the frame if enabled
            if self.render_frames:
                self._render_frame(obs, is_first_frame=False)
                
            return obs, reward, done, info
        except Exception as e:
            print(f"ERROR in FrameRenderWrapper.step: {e}")
            traceback.print_exc()
            # Return fallback values
            return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8), 0.0, True, {}
    
    def _render_frame(self, obs, is_first_frame=False):
        try:
            # Extract POV observation
            if isinstance(obs, dict) and 'pov' in obs:
                frame = obs['pov']
            else:
                frame = obs
            
            # Print the shape before reshaping
            print(f"POV shape before reshape: {frame.shape}, dtype: {frame.dtype}, min: {np.min(frame)}, max: {np.max(frame)}")

            # Check for NaN values
            if np.isnan(frame).any():
                print("WARNING: NaN values in frame")
                # Replace NaN values with zeros
                frame = np.nan_to_num(frame)

            # If frame is in channel-first format, transpose it back to HWC for display
            if len(frame.shape) == 3 and frame.shape[0] in [1, 2, 3]:  # Channel-first format
                # Convert from CHW to HWC format for display
                if frame.shape[0] == 3:  # Full RGB
                    # Transpose from (3, H, W) to (H, W, 3)
                    frame = frame.transpose(1, 2, 0)
                elif frame.shape[0] == 1:  # Single channel
                    # Convert single channel to RGB for display
                    channel = frame[0]
                    frame = np.stack([channel, channel, channel], axis=2)  # Grayscale representation
                elif frame.shape[0] == 2:  # Two channels (unusual)
                    # Use first two channels as R and G, set B to zero
                    r, g = frame[0], frame[1]
                    frame = np.stack([r, g, np.zeros_like(r)], axis=2)
                    
            # Ensure frame is in HWC format with 3 channels for display
            if len(frame.shape) == 2:  # Handle grayscale
                frame = np.stack([frame, frame, frame], axis=2)  # Convert to RGB
                
            # Save the first frame of each episode
            if is_first_frame:
                save_frame_as_image(frame, self.episode_count, self.step_count)
            
            # Initialize plot if needed
            if self.fig is None or self.ax is None:
                plt.ion()  # Enable interactive mode
                self.fig, self.ax = plt.subplots(figsize=(8, 8))
                self.ax.set_title(f"MineRL Observation (Episode {self.episode_count})")
                self.img = self.ax.imshow(frame)
            else:
                self.img.set_data(frame)
                self.ax.set_title(f"MineRL Observation (Episode {self.episode_count}, Step {self.step_count})")
                
            # Try to update the display, but catch the UserWarning about non-interactive backend
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(0.01)  # Small pause to update the plot
            except Exception as e:
                # Just log the error once to avoid spam
                if not hasattr(self, 'display_error_logged'):
                    print(f"Note: Could not update display in interactive mode: {e}")
                    print("Will continue saving frames to disk.")
                    self.display_error_logged = True
        except Exception as e:
            print(f"ERROR in _render_frame: {e}")
            traceback.print_exc()

# === RAM Usage Monitor ===
class RAMMonitor:
    def __init__(self, max_ram_usage_gb):
        self.max_ram_usage_bytes = max_ram_usage_gb * 1024 * 1024 * 1024
        self.process = psutil.Process(os.getpid())
    
    def check_ram_usage(self):
        # Get current RAM usage
        current_ram = self.process.memory_info().rss
        return current_ram
    
    def is_ram_exceeded(self):
        return self.check_ram_usage() > self.max_ram_usage_bytes
    
    def get_ram_usage_gb(self):
        return self.check_ram_usage() / (1024 * 1024 * 1024)

# === Progress Tracking Callback with Enhanced Logging ===
class TrainingMetricsCallback(BaseCallback):
    def __init__(self, log_interval=1, ram_monitor=None, log_dir=None, verbose=0):
        super().__init__(verbose)
        self.log_interval = log_interval  # Set to 1 to log every step
        self.ram_monitor = ram_monitor
        self.log_dir = log_dir
        self.start_time = time.time()
        self.total_timesteps = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.losses = {}
        self.episode_count = 0
        
        # Create log file
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = open(os.path.join(log_dir, "training_metrics.txt"), "w")
        self.log_file.write("Timestep,Episode,EpisodeStep,Reward,EpisodeCumulativeReward,ExplainedVariance,PolicyLoss,ValueLoss,EntropyLoss,LossWithRewards,RAM_GB,FPS\n")
        
        # Create detailed rewards log file
        self.rewards_log_file = open(os.path.join(log_dir, "detailed_rewards.txt"), "w")
        self.rewards_log_file.write("Timestep,Episode,EpisodeStep,Reward,CumulativeReward,RewardComponents\n")
        
        # Create detailed losses log file
        self.losses_log_file = open(os.path.join(log_dir, "detailed_losses.txt"), "w")
        self.losses_log_file.write("Timestep,Episode,EpisodeStep,PolicyLoss,ValueLoss,EntropyLoss,TotalLoss,LossWithRewards\n")
        
    def _init_callback(self):
        self.training_start_time = time.time()
        
    def _on_step(self):
        try:
            self.total_timesteps += 1
            self.current_episode_length += 1
            
            # Extract rewards from the most recent step
            # Note: In vectorized environments, we take the first environment's reward
            if len(self.model.ep_info_buffer) > 0 and 'r' in self.model.ep_info_buffer[-1]:
                step_reward = self.model.ep_info_buffer[-1]['r']
            else:
                step_reward = 0
                
            self.current_episode_reward += step_reward
            
            # Extract reward info if available
            reward_info = {}
            if hasattr(self.locals, 'infos') and self.locals.get('infos') is not None:
                infos = self.locals.get('infos')
                if infos and len(infos) > 0:
                    if 'rewards' in infos[0]:
                        reward_info = infos[0]['rewards']
                    if 'original_reward' in infos[0]:
                        reward_info['original_reward'] = infos[0]['original_reward']
            
            # Always log rewards at each step (not just at log_interval)
            self.rewards_log_file.write(f"{self.total_timesteps},{self.episode_count},{self.current_episode_length}," +
                                   f"{step_reward},{self.current_episode_reward}," +
                                   f"\"{str(reward_info).replace(',', ';')}\"\n")
            self.rewards_log_file.flush()
            
            # Get training metrics if available (only available during actual training updates)
            loss_info = {}
            if hasattr(self.model, "logger") and self.model.logger is not None:
                for key in ["explained_variance", "policy_loss", "value_loss", "entropy_loss", "approx_kl", "clip_fraction", "clip_range", "learning_rate"]:
                    if key in self.model.logger.name_to_value:
                        loss_info[key] = self.model.logger.name_to_value[key]
            
            # Compute loss with rewards (combining policy loss with rewards)
            loss_with_rewards = None
            total_loss = None
            
            if 'policy_loss' in loss_info and 'value_loss' in loss_info:
                policy_loss = loss_info.get('policy_loss', 0)
                value_loss = loss_info.get('value_loss', 0)
                entropy_loss = loss_info.get('entropy_loss', 0)
                
                # Approximate total loss
                total_loss = policy_loss + value_loss - 0.01 * entropy_loss  # Standard PPO formulation
                
                # Loss with rewards (regularized by rewards)
                loss_with_rewards = policy_loss - (0.01 * step_reward)  # Reward acts as a regularizer
            
            # Log losses if we have them, regardless of interval
            if 'policy_loss' in loss_info:
                self.losses_log_file.write(f"{self.total_timesteps},{self.episode_count},{self.current_episode_length}," +
                                    f"{loss_info.get('policy_loss', 'N/A')}," +
                                    f"{loss_info.get('value_loss', 'N/A')}," +
                                    f"{loss_info.get('entropy_loss', 'N/A')}," +
                                    f"{total_loss if total_loss is not None else 'N/A'}," +
                                    f"{loss_with_rewards if loss_with_rewards is not None else 'N/A'}\n")
                self.losses_log_file.flush()
            
            # Check if episode ended
            if self.locals.get("dones")[0]:
                self.episode_count += 1
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lengths.append(self.current_episode_length)
                
                # Save checkpoint at the end of each episode - make sure the directory exists
                try:
                    checkpoint_path = os.path.join(self.log_dir, f"episode_{self.episode_count}_checkpoint")
                    self.model.save(checkpoint_path)
                    print(f"Checkpoint saved at episode {self.episode_count}: {checkpoint_path}.zip")
                except Exception as e:
                    print(f"ERROR saving checkpoint: {e}")
                    traceback.print_exc()
                    # Try to create parent directory if it doesn't exist
                    os.makedirs(self.log_dir, exist_ok=True)
                    try:
                        self.model.save(checkpoint_path)
                        print(f"Retry successful: Checkpoint saved at episode {self.episode_count}: {checkpoint_path}.zip")
                    except Exception as e2:
                        print(f"FAILED to save checkpoint after retry: {e2}")
                        traceback.print_exc()
                
                # Log episode completion with reward info
                print(f"Episode completed: Length={self.current_episode_length} | Total Reward={self.current_episode_reward:.2f}")
                if reward_info:
                    print(f"Reward breakdown: {reward_info}")
                
                # Reset episode tracking
                self.current_episode_reward = 0
                self.current_episode_length = 0
                
                # Calculate loss with rewards more robustly if we have policy loss
                if 'policy_loss' in loss_info:
                    policy_loss = loss_info['policy_loss']
                    loss_with_rewards = policy_loss - (0.01 * self.current_episode_reward)
                    loss_info['loss_with_rewards'] = loss_with_rewards
                else:
                    loss_with_rewards = "N/A"
                            
            # Log progress at specified intervals
            if self.total_timesteps % self.log_interval == 0:
                elapsed_time = time.time() - self.training_start_time
                fps = int(self.total_timesteps / elapsed_time if elapsed_time > 0 else 0)
                
                # RAM usage tracking
                ram_usage = "N/A"
                if self.ram_monitor:
                    ram_usage = f"{self.ram_monitor.get_ram_usage_gb():.2f} GB"
                    if self.ram_monitor.is_ram_exceeded():
                        print(f"WARNING: RAM usage limit exceeded: {ram_usage}")
                
                # Current episode info
                episode_num = self.episode_count
                
                # Print to console
                # Fix string formatting for loss_with_rewards
                if isinstance(loss_with_rewards, str):
                    loss_with_rewards_str = loss_with_rewards
                else:
                    # Fix string formatting for log file
                    if loss_with_rewards is None:
                        loss_with_rewards_str = "N/A"
                    elif isinstance(loss_with_rewards, str):
                        loss_with_rewards_str = loss_with_rewards
                    else:
                        loss_with_rewards_str = f"{loss_with_rewards:.4f}"                
                    
                print(f"Step {self.total_timesteps} | Episode {episode_num} | " +
                    f"Step in Episode: {self.current_episode_length} | " + 
                    f"Reward: {step_reward:.2f} | " +
                    f"Cumulative Reward: {self.current_episode_reward:.2f} | " +
                    f"Losses: {loss_info} | " +
                    f"Loss with Rewards: {loss_with_rewards_str} | " +
                    f"RAM: {ram_usage} | FPS: {fps}")
                
                # Write to log file
                # Fix string formatting for log file
                if isinstance(loss_with_rewards, str):
                    loss_with_rewards_str = loss_with_rewards
                else:
                    # Fix string formatting for log file
                    if loss_with_rewards is None:
                        loss_with_rewards_str = "N/A"
                    elif isinstance(loss_with_rewards, str):
                        loss_with_rewards_str = loss_with_rewards
                    else:
                        loss_with_rewards_str = f"{loss_with_rewards:.4f}"                
                    
                self.log_file.write(f"{self.total_timesteps},{episode_num},{self.current_episode_length}," +
                                f"{step_reward},{self.current_episode_reward}," +
                                f"{loss_info.get('explained_variance', 'N/A')}," +
                                f"{loss_info.get('policy_loss', 'N/A')}," +
                                f"{loss_info.get('value_loss', 'N/A')}," +
                                f"{loss_info.get('entropy_loss', 'N/A')}," +
                                f"{loss_with_rewards_str}," +
                                f"{self.ram_monitor.get_ram_usage_gb() if self.ram_monitor else 'N/A'}," +
                                f"{fps}\n")
                self.log_file.flush()
                
                # Garbage collection to help manage memory
                gc.collect()
            
            return True
        except Exception as e:
            print(f"ERROR in _on_step: {e}")
            traceback.print_exc()
            return False
        
    def _on_training_end(self):
        try:
            # Close log files
            if hasattr(self, 'log_file') and self.log_file is not None:
                self.log_file.close()
                
            if hasattr(self, 'rewards_log_file') and self.rewards_log_file is not None:
                self.rewards_log_file.close()
                
            if hasattr(self, 'losses_log_file') and self.losses_log_file is not None:
                self.losses_log_file.close()
                
            # Write episode summary
            with open(os.path.join(self.log_dir, "episode_summary.txt"), "w") as f:
                f.write("Episode,Length,Reward\n")
                for i, (length, reward) in enumerate(zip(self.episode_lengths, self.episode_rewards)):
                    f.write(f"{i+1},{length},{reward}\n")
        except Exception as e:
            print(f"ERROR in _on_training_end: {e}")
            traceback.print_exc()

# === Action Wrapper ===
class ActionSimplifier(gym.ActionWrapper):
    def __init__(self, env, always_attack=True, angle=5):
        super().__init__(env)
        self.action_dict = {
            0: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 1, 'jump': 0, 'left': 0, 'right': 0,
                'sneak': 0, 'sprint': 0},
            1: {'attack': always_attack, 'back': 0, 'camera': [0, angle], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            2: {'attack': 1, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0, 'sneak': 0,
                'sprint': 0},
            3: {'attack': always_attack, 'back': 0, 'camera': [angle, 0], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            4: {'attack': always_attack, 'back': 0, 'camera': [-angle, 0], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            5: {'attack': always_attack, 'back': 0, 'camera': [0, -angle], 'forward': 0, 'jump': 0, 'left': 0,
                'right': 0, 'sneak': 0, 'sprint': 0},
            6: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 1, 'jump': 1, 'left': 0, 'right': 0,
                'sneak': 0, 'sprint': 0},
            7: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 1, 'right': 0,
                'sneak': 0, 'sprint': 0},
            8: {'attack': always_attack, 'back': 0, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 1,
                'sneak': 0, 'sprint': 0},
            9: {'attack': always_attack, 'back': 1, 'camera': [0, 0], 'forward': 0, 'jump': 0, 'left': 0, 'right': 0,
                'sneak': 0, 'sprint': 0}}
        self.action_space = gym.spaces.Discrete(len(self.action_dict))
        
        
    def action(self, action_idx):
        try:
            # Debug: Print the action index
            print(f"ActionSimplifier: Converting action_idx {action_idx} (type: {type(action_idx)})")
            
            # Handle numpy array/tensor type
            if isinstance(action_idx, np.ndarray):
                if action_idx.size == 1:
                    action_idx = action_idx.item()
                else:
                    print(f"WARNING: Unexpected action shape: {action_idx.shape}")
                    action_idx = 0  # Default to action 0 if unexpected
            
            # Convert the action index to the appropriate action dictionary
            base_action = self.action_dict.get(action_idx, self.action_dict[0])
            
            # Create a full action dictionary for MineRL
            full_action = {
                "forward": base_action['forward'], 
                "back": base_action['back'], 
                "left": base_action['left'], 
                "right": base_action['right'], 
                "jump": base_action['jump'], 
                "sneak": base_action['sneak'], 
                "sprint": base_action['sprint'], 
                "attack": base_action['attack'],
                "camera": base_action['camera'],
                "hotbar.1": 0, "hotbar.2": 0, "hotbar.3": 0, "hotbar.4": 0,
                "hotbar.5": 0, "hotbar.6": 0, "hotbar.7": 0, "hotbar.8": 0,
                "hotbar.9": 0, "inventory": 0, "use": 0, "drop": 0,
                "swapHands": 0, "pickItem": 0, "ESC": 0
            }
            
            # Debug: Print the converted action
            print(f"ActionSimplifier: Converted to full_action: {full_action}")
            
            return full_action
        except Exception as e:
            print(f"ERROR in ActionSimplifier.action: {e}")
            traceback.print_exc()
            # Return a default action in case of error
            return self.action_dict[0]

# === Create a custom wrapper to modify rewards ===
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        print("CustomRewardWrapper: Enabling only log collecting rewards.")
    
    def step(self, action):
        try:
            # Debug: Print the action before stepping
            print(f"CustomRewardWrapper.step: action type: {type(action)}, value: {action}")
            
            obs, reward, done, info = self.env.step(action)
            
            # Extract the specific reward components from info if available
            collected_log = False
            if 'rewards' in info:
                reward_dict = info['rewards']
                
                # Check if the agent collected a log
                for key in reward_dict:
                    if 'log' in key.lower() and reward_dict[key] > 0:
                        collected_log = True
                        break
                
                # Override the reward: 1 for log collection, 0 for everything else
                if collected_log:
                    reward = 2048.0
                else:
                    reward = 0.0
                
                # Store the original rewards for debugging
                info['original_reward'] = info['rewards']
                # Create a new rewards dict with only log collecting enabled
                info['rewards'] = {'log_collected': float(collected_log)}
            
            # Debug: Check observation after step
            if isinstance(obs, np.ndarray):
                print(f"CustomRewardWrapper.step output obs shape: {obs.shape}, dtype: {obs.dtype}")
            elif isinstance(obs, dict) and 'pov' in obs:
                print(f"CustomRewardWrapper.step output POV shape: {obs['pov'].shape}, dtype: {obs['pov'].dtype}")
                
            return obs, reward, done, info
        except Exception as e:
            print(f"ERROR in CustomRewardWrapper.step: {e}")
            traceback.print_exc()
            # Return fallback values
            return np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8), 0.0, True, {}

# === Make wrapped environment directly without vectorization ===
def make_env():
    """Create and wrap a MineRL environment."""
    try:
        print("Creating base MineRL environment...")
        env = gym.make(ENV_ID)
        print(f"Base environment created: {env}, observation_space: {env.observation_space}, action_space: {env.action_space}")
        
        # First flatten the observation space to only use POV
        print("Applying FlattenObservationWrapper...")
        env = FlattenObservationWrapper(env)
        print(f"After flatten: observation_space: {env.observation_space}")
        
        # Then resize the observation
        print("Applying ResizeObservationWrapper...")
        env = ResizeObservationWrapper(env, size=(TARGET_HEIGHT, TARGET_WIDTH))
        print(f"After resize: observation_space: {env.observation_space}")
        
        # Apply action simplifier wrapper
        print("Applying ActionSimplifier...")
        env = ActionSimplifier(env, always_attack=True, angle=5)
        print(f"After action simplifier: action_space: {env.action_space}")
        
        # Apply custom reward wrapper
        print("Applying CustomRewardWrapper...")
        env = CustomRewardWrapper(env)
        
        # Apply episode length wrapper
        print("Applying EpisodeLengthWrapper...")
        env = EpisodeLengthWrapper(env, max_steps=MAX_STEPS_PER_EPISODE)
        
        # Add frame rendering wrapper if SHOWFRAMES is True
        if SHOWFRAMES:
            print("Applying FrameRenderWrapper...")
            env = FrameRenderWrapper(env, render_frames=True)
        
        print("Environment creation completed successfully!")
        return env
        
    except Exception as e:
        print(f"ERROR in make_env: {e}")
        traceback.print_exc()
        raise

# Enhanced environment reset function with retry capability
def robust_env_reset(env, max_retries=3, retry_delay=5):
    """Attempt to reset the environment with retries on failure"""
    for attempt in range(max_retries):
        try:
            print(f"Attempting environment reset (attempt {attempt+1}/{max_retries})...")
            obs = env.reset()
            print("Environment reset successful!")
            
            # Verify that the observation is valid
            if isinstance(obs, np.ndarray):
                print(f"Reset returned observation with shape: {obs.shape}, dtype: {obs.dtype}, min: {np.min(obs)}, max: {np.max(obs)}")
                if obs.size == 0:
                    print("Warning: Empty observation array received from reset, creating fallback")
                    # Create a fallback observation with proper shape
                    if hasattr(env, 'observation_space'):
                        fallback_obs = np.zeros(env.observation_space.shape, dtype=np.uint8)
                    else:
                        fallback_obs = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
                    return fallback_obs
            elif obs is None:
                print("Warning: None observation received from reset, creating fallback")
                if hasattr(env, 'observation_space'):
                    fallback_obs = np.zeros(env.observation_space.shape, dtype=np.uint8)
                else:
                    fallback_obs = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
                return fallback_obs
                
            return obs
            
        except Exception as e:
            print(f"Reset failed with error: {e}")
            traceback.print_exc()
            if attempt < max_retries - 1:
                print(f"Waiting {retry_delay} seconds before retrying...")
                time.sleep(retry_delay)
                # Force garbage collection to clean up any lingering connections
                gc.collect()
            else:
                print("Maximum reset attempts reached. Creating new environment...")
                # Close the old environment
                try:
                    env.close()
                except Exception as close_error:
                    print(f"Error closing environment: {close_error}")
                    traceback.print_exc()
                
                # Create a new environment
                try:
                    env = make_env()
                    obs = env.reset()
                    # Verify that the observation is valid
                    if obs is None or (isinstance(obs, np.ndarray) and obs.size == 0):
                        print("Warning: Invalid observation from new environment, creating fallback")
                        fallback_obs = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
                        return fallback_obs
                    return obs
                except Exception as e2:
                    print(f"Final reset attempt failed: {e2}")
                    traceback.print_exc()
                    print("Creating emergency fallback observation")
                    fallback_obs = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
                    return fallback_obs

def main():
    print("starting...")

    # === Check for GPU and set device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Create environment ===
    print("Creating environment...")
    try:
        env = make_env()
    except Exception as e:
        print(f"CRITICAL ERROR creating environment: {e}")
        traceback.print_exc()
        print("Attempting to continue with a fallback approach...")
        try:
            env = gym.make(ENV_ID)
            env = FlattenObservationWrapper(env)
            env = ActionSimplifier(env)
        except Exception as e2:
            print(f"FATAL ERROR: Could not create environment even with fallback: {e2}")
            traceback.print_exc()
            raise RuntimeError("Environment creation failed completely")

    ram_monitor = RAMMonitor(MAX_RAM_USAGE_GB)
    os.makedirs(LOG_DIR, exist_ok=True)

    metrics_callback = TrainingMetricsCallback(
        log_interval=1,
        ram_monitor=ram_monitor,
        log_dir=LOG_DIR
    )

    model_path = os.path.join(LOG_DIR, "ppo_model")

    if os.path.exists(model_path + ".zip"):
        try:
            model = PPO.load(model_path, env=None, device=device)
            print(f"Loaded existing model from {model_path}.zip")
            vec_env = DummyVecEnv([lambda: env])
            vec_env = VecTransposeImage(vec_env)
            model.set_env(vec_env)
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            print("Training a new model instead")
            model = None
    else:
        model = None

    if model is None:
        print(f"Training new model with RAM limit of {MAX_RAM_USAGE_GB} GB")
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecTransposeImage(vec_env)
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            batch_size=32,
            n_steps=32,
            learning_rate=LEARNING_RATE,
            device=device,
            policy_kwargs={"net_arch": [16, 16]}
        )
        print(f"Beginning training for {TOTAL_TIMESTEPS} timesteps...")
        try:
            model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=metrics_callback)
        except Exception as e:
            print(f"Error during training: {e}")
            traceback.print_exc()
            print("Attempting to save model despite training error...")
        try:
            model.save(model_path)
            print(f"Trained and saved new model to {model_path}.zip")
        except Exception as e:
            print(f"Error saving model: {e}")
            traceback.print_exc()

    print(f"\nRunning {TOTAL_EPISODES} episodes with max {MAX_STEPS_PER_EPISODE} steps per episode...")

    eval_log_path = os.path.join(LOG_DIR, "evaluation_metrics.txt")
    os.makedirs(os.path.dirname(eval_log_path), exist_ok=True)

    try:
        eval_log_file = open(eval_log_path, "w")
        eval_log_file.write("Episode,Step,Reward,CumulativeReward,ValueEstimate,LogProb,Entropy,LossWithRewards,RAM_GB\n")
    except Exception as e:
        print(f"WARNING: Could not open evaluation log file: {e}")
        traceback.print_exc()
        eval_log_path = "evaluation_metrics.txt"
        try:
            eval_log_file = open(eval_log_path, "w")
            eval_log_file.write("Episode,Step,Reward,CumulativeReward,ValueEstimate,LogProb,Entropy,LossWithRewards,RAM_GB\n")
            print(f"Created fallback log file at {eval_log_path}")
        except Exception as e2:
            print(f"CRITICAL: Could not create any log file: {e2}")
            traceback.print_exc()
            from io import StringIO
            eval_log_file = StringIO()

    try:
        print("\nCreating evaluation environment...")
        eval_env = make_env()
    except Exception as e:
        print(f"ERROR creating evaluation environment: {e}")
        traceback.print_exc()
        print("Using training environment for evaluation as fallback")
        eval_env = env

    for episode in range(TOTAL_EPISODES):
        print(f"\nStarting Episode {episode+1}/{TOTAL_EPISODES}")
    
        # Use robust reset function
        try:
            obs = robust_env_reset(eval_env)
        except Exception as e:
            print(f"CRITICAL: Could not reset environment: {e}")
            traceback.print_exc()
            print("Attempting to save current model and exit...")
            # Try to save the model before exiting
            try:
                emergency_path = os.path.join(LOG_DIR, f"emergency_save_ep{episode}")
                model.save(emergency_path)
                print(f"Emergency save successful: {emergency_path}.zip")
            except Exception as save_error:
                print(f"Emergency save failed: {save_error}")
                traceback.print_exc()
            break  # Break out of the episode loop
    
        done = False
        step = 0
        episode_reward = 0
        episode_losses_with_rewards = []

        while not done and step < MAX_STEPS_PER_EPISODE:  # Add step limit as safety
            # Debugging: print the shape of the observation
            if isinstance(obs, np.ndarray):
                print(f"Step {step}: Observation shape: {obs.shape}, dtype: {obs.dtype}, min: {np.min(obs)}, max: {np.max(obs)}")
            else:
                print(f"Step {step}: Observation type: {type(obs)}")
            
            # Check for NaN values in observation
            if isinstance(obs, np.ndarray) and np.isnan(obs).any():
                print("WARNING: NaN values in observation! Replacing with zeros.")
                obs = np.nan_to_num(obs)
        
            # Get action and value/log_prob/entropy estimates from the policy
            try:
                with torch.no_grad():
                    # Ensure observation is in the right format
                    if isinstance(obs, np.ndarray):
                        # Make a copy to avoid modifying the original
                        obs_copy = obs.copy()
                    
                        # For image observations (H, W, C) -> (C, H, W)
                        if len(obs_copy.shape) == 3 and obs_copy.shape[2] == 3:
                            obs_copy = np.transpose(obs_copy, (2, 0, 1))
                    
                        # Add batch dimension
                        obs_batch = np.expand_dims(obs_copy, axis=0)
                    
                        # Convert to tensor
                        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(device)
                    
                        # Debug info
                        print(f"Policy input tensor shape: {obs_tensor.shape}, device: {obs_tensor.device}")
                    
                        # Use model's predict method for safer action selection
                        action, _, info_dict = model.policy.forward(obs_tensor)
                    
                        # Debug info
                        print(f"Policy output action: {action}, shape: {action.shape}, device: {action.device}")
                    
                        # Extract additional info if available
                        try:
                            distribution = model.policy.get_distribution(obs_tensor)
                            log_prob = distribution.log_prob(action)
                            entropy = distribution.entropy()
                            value_estimate = model.policy.predict_values(obs_tensor)
                        except Exception as e:
                            print(f"Warning: Could not get distribution info: {e}")
                            traceback.print_exc()
                            log_prob = torch.zeros(1)
                            entropy = torch.zeros(1)
                            value_estimate = torch.zeros(1)
                    
                        # Convert to numpy and remove batch dimension
                        action_np = action.cpu().numpy()
                        if len(action_np.shape) > 1:
                            action_np = action_np[0]  # Remove batch dimension
                        
                        # Debug info
                        print(f"Final action (numpy): {action_np}, shape: {action_np.shape}, type: {type(action_np)}")
                    
                        # If action is a single value array, convert to scalar for discrete action space
                        if action_np.shape == (1,):
                            action_np = action_np.item()
                            print(f"Converted action to scalar: {action_np}")
                        
                        # Final action for step
                        action = action_np
                    else:
                        print(f"WARNING: Unexpected observation type: {type(obs)}")
                        # Fallback to random action
                        action = eval_env.action_space.sample()
            except Exception as e:
                print(f"ERROR during policy evaluation: {e}")
                traceback.print_exc()
                # Fallback to random action
                action = eval_env.action_space.sample()
                log_prob = torch.zeros(1)
                entropy = torch.zeros(1)
                value_estimate = torch.zeros(1)

            # Step in the environment with error handling
            try:
                print(f"TAKING STEP with action {action} (type: {type(action)})")
            
                # Validate action space compatibility before stepping
                if isinstance(eval_env.action_space, gym.spaces.Discrete):
                    print(f"Environment has Discrete action space with {eval_env.action_space.n} actions")
                    if isinstance(action, np.ndarray):
                        if action.size == 1:
                            action = action.item()
                            print(f"Converted array action to scalar: {action}")
                        else:
                            print(f"WARNING: Action is array with size > 1: {action.shape}")
                            action = 0  # Default fallback
                    elif not isinstance(action, (int, np.int32, np.int64)):
                        print(f"WARNING: Action type {type(action)} not compatible with Discrete space, converting to int")
                        if hasattr(action, 'item'):
                            action = action.item()
                        else:
                            action = int(action)
            
                # Take the step
                obs, reward, done, info = eval_env.step(action)
            
                # Debug: Check the step outputs
                print(f"Step result - reward: {reward}, done: {done}")
                if isinstance(obs, np.ndarray):
                    print(f"New observation shape: {obs.shape}, dtype: {obs.dtype}, min: {np.min(obs)}, max: {np.max(obs)}")
                    if np.isnan(obs).any():
                        print("WARNING: NaN values in new observation! Replacing with zeros.")
                        obs = np.nan_to_num(obs)
                    
            except Exception as e:
                print(f"ERROR during environment step: {e}")
                traceback.print_exc()
                print("Terminating episode early due to environment error")
                done = True
                reward = 0.0
                break
            
            step += 1
            episode_reward += reward
        
            # Compute loss with rewards
            try:
                # Handle tensor or scalar values for log_prob
                if isinstance(log_prob, torch.Tensor):
                    if log_prob.dim() > 0:  # Multi-dimensional tensor
                        policy_loss = -log_prob.mean().item()
                    else:  # 0-dimensional tensor (scalar)
                        policy_loss = -log_prob.item()
                else:  # Already a scalar
                    policy_loss = -log_prob
            
                # Compute loss with rewards
                loss_with_rewards = policy_loss - (0.01 * reward)
                episode_losses_with_rewards.append(loss_with_rewards)
            except Exception as e:
                print(f"Warning: Could not compute loss with rewards: {e}")
                traceback.print_exc()
                loss_with_rewards = 0.0
                episode_losses_with_rewards.append(loss_with_rewards)

            # Log to file - ensure proper formatting of tensor values
            try:
                value_est_formatted = value_estimate.cpu().numpy().item() if isinstance(value_estimate, torch.Tensor) else 0.0
                log_prob_formatted = log_prob.cpu().numpy().mean() if isinstance(log_prob, torch.Tensor) else 0.0
                entropy_formatted = entropy.cpu().numpy().mean() if isinstance(entropy, torch.Tensor) else 0.0
            
                eval_log_file.write(f"{episode+1},{step},{reward:.2f},{episode_reward:.2f}," +
                                f"{value_est_formatted:.4f}," +
                                f"{log_prob_formatted:.4f}," +
                                f"{entropy_formatted:.4f}," +
                                f"{loss_with_rewards:.4f}," +
                                f"{ram_monitor.get_ram_usage_gb():.2f}\n")
                eval_log_file.flush()  # Flush after each write for real-time logging
            except Exception as e:
                print(f"WARNING: Error writing to log file: {e}")
                traceback.print_exc()

        print(f"Episode {episode+1} finished: Steps={step}, Total Reward = {episode_reward:.2f}, Avg Loss with Rewards = {np.mean(episode_losses_with_rewards):.4f}")
    
        # Save checkpoint at the end of each episode
        try:
            checkpoint_path = os.path.join(LOG_DIR, f"eval_episode_{episode+1}_checkpoint")
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            model.save(checkpoint_path)
            print(f"Evaluation checkpoint saved: {checkpoint_path}.zip")
        except Exception as e:
            print(f"ERROR saving evaluation checkpoint: {e}")
            traceback.print_exc()

    # Save final model
    final_model_path = os.path.join(LOG_DIR, "final_model")
    try:
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        model.save(final_model_path)
        print(f"Final model saved to {final_model_path}.zip")
    except Exception as e:
        print(f"ERROR saving final model: {e}")
        traceback.print_exc()

    # Clean up
    if hasattr(eval_log_file, 'close'):
        eval_log_file.close()

    # Close all environments
    if 'eval_env' in locals() and eval_env is not None:
        try:
            eval_env.close()
        except Exception as e:
            print(f"Error closing eval_env: {e}")
            traceback.print_exc()

    if 'env' in locals() and env is not None:
        try:
            env.close()
        except Exception as e:
            print(f"Error closing env: {e}")
            traceback.print_exc()

    # Close any matplotlib plots
    if SHOWFRAMES:
        try:
            plt.close('all')
        except Exception as e:
            print(f"Error closing matplotlib plots: {e}")
            traceback.print_exc()
    
    print("Done!")


if __name__ == "__main__":
    main()
