import numpy as np
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from cyber_env import CyberGymEnv

def main():
    if not os.path.exists("cyber_rl_model.zip") or not os.path.exists("vec_normalize.pkl"):
        print("Model or normalization statistics not found. Run train.py first.")
        return

    # Recreate the environment
    env = CyberGymEnv()
    vec_env = DummyVecEnv([lambda: env])
    
    # Load normalization statistics
    vec_env = VecNormalize.load("vec_normalize.pkl", vec_env)
    
    # Turn off training and reward normalization for evaluation
    vec_env.training = False
    vec_env.norm_reward = False

    # Load model
    model = PPO.load("cyber_rl_model", env=vec_env)

    num_episodes = 10
    all_rewards = []
    total_steps = 0
    correct_actions = 0

    print("--- Evaluation Started ---")
    for episode in range(num_episodes):
        obs = vec_env.reset()
        done = False
        episode_reward = 0.0
        
        while not done:
            # Predict action cleanly without randomness
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = vec_env.step(action)
            episode_reward += reward[0] # reward is returned as an array in VecEnv
            done = dones[0]
            
            info = infos[0]
            actual_attack = info.get('actual_attack', 0)
            act = action[0]
            
            total_steps += 1
            if info.get('mitigated', False):
                correct_actions += 1
            elif actual_attack == 0 and act == 3:
                # Agent correctly ignored normal traffic
                correct_actions += 1
            
        all_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes} - Reward: {episode_reward:.2f}")

    avg_reward = np.mean(all_rewards)
    accuracy = (correct_actions / total_steps) * 100 if total_steps > 0 else 0
    print("--------------------------")
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.2f}")
    print(f"Model Accuracy (Correct mitigations or passes): {accuracy:.2f}% ({correct_actions}/{total_steps})")

if __name__ == "__main__":
    main()
