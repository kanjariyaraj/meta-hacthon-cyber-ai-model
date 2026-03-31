import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import os

class CyberGymEnv(gym.Env):
    """Gym environment that replays a real dataset for Data-Driven training."""
    def __init__(self, dataset_path='cyber_traffic_dataset.csv'):
        super().__init__()
        
        self.dataset_path = dataset_path
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset {self.dataset_path} not found. Run generate_dataset.py first.")
            
        self.df = pd.read_csv(self.dataset_path)
        self.max_steps = len(self.df) - 1
        self.current_step = 0
        self.episode_step = 0
        
        # Actions: 0 -> block_ip, 1 -> alert_admin, 2 -> isolate_system, 3 -> ignore
        self.action_space = spaces.Discrete(4)
        
        # Observations: failed_login, alert_count, ddos_flag, suspicious_flag, system_status
        self.observation_space = spaces.Box(low=0, high=100, shape=(5,), dtype=np.float32)
        
        self.system_status = 10.0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Pick a random starting point in the dataset for variety
        # Leave at least 100 steps for an episode
        self.current_step = np.random.randint(0, self.max_steps - 100)
        self.episode_step = 0
        self.system_status = 10.0
        
        obs = self._get_obs()
        return np.array(obs, dtype=np.float32), {}
        
    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row['failed_login_count'],
            row['alert_count'],
            row['ddos_flag'],
            row['suspicious_flag'],
            self.system_status
        ], dtype=np.float32)
        # Scale to max 100 to fit observation space
        return np.clip(obs, 0, 100)
        
    def step(self, action):
        row = self.df.iloc[self.current_step]
        actual_attack = int(row['attack_type']) # 0: Normal, 1: Brute Force, 2: DDoS, 3: Malware
        
        mitigated = False
        action_penalty = 0.0
        
        # Check if action was appropriate for the attack
        if action == 0: # block_ip
            action_penalty = -0.1
            if actual_attack == 1: mitigated = True
        elif action == 1: # alert_admin
            action_penalty = -0.2
            if actual_attack == 3: mitigated = True
        elif action == 2: # isolate_system
            action_penalty = -0.5 # High business cost
            if actual_attack == 2: mitigated = True
        elif action == 3: # ignore
            action_penalty = 0.0
            if actual_attack == 0: mitigated = True # Correctly ignoring normal traffic
            
        # Update system status based on mitigation
        if mitigated:
            self.system_status = min(10.0, self.system_status + 1.0)
            reward = 1.0 + action_penalty
        else:
            # Failed to mitigate an attack, or false positive on normal traffic
            if actual_attack != 0:
                self.system_status -= 2.0
                reward = -2.0 + action_penalty
            else:
                # False positive: applying action on normal traffic
                reward = -1.0 + action_penalty
                
        self.system_status = max(0.0, float(self.system_status))
        
        self.current_step += 1
        self.episode_step += 1
        
        done = self.episode_step >= 100 or self.system_status <= 0.0
        if self.system_status <= 0.0:
            reward -= 10.0 # Heavy penalty for system crash
            
        info = {
            'actual_attack': actual_attack,
            'system_health': self.system_status,
            'mitigated': mitigated
        }
        
        terminated = done
        truncated = False
        
        obs = self._get_obs()
        return np.array(obs, dtype=np.float32), float(reward), terminated, truncated, info
