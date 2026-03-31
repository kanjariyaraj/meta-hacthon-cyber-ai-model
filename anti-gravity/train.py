import os
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from cyber_env import CyberGymEnv

def main():
    # Create the environment and wrap it with a Monitor for tracking rewards
    log_dir = "./ppo_logs/monitor/"
    os.makedirs(log_dir, exist_ok=True)
    
    env = CyberGymEnv()
    env = Monitor(env, log_dir)
    
    # Vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Add reward normalization
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # Setup callback for checkpoint saving
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./logs/',
        name_prefix='ppo_cyber_model'
    )

    # Initialize model
    model = PPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./ppo_logs/"
    )

    # Train model
    print("Starting training...")
    model.learn(
        total_timesteps=500000,
        callback=checkpoint_callback,
        tb_log_name="PPO_run"
    )

    # Save trained model
    model.save("cyber_rl_model")
    
    # Save normalization statistics
    vec_env.save("vec_normalize.pkl")
    print("Training complete and model saved.")
    
    # Optionally plot training rewards
    try:
        x, y = ts2xy(load_results(log_dir), 'timesteps')
        plt.figure(figsize=(10, 5))
        plt.plot(x, y)
        plt.title('Training Rewards over Timesteps')
        plt.xlabel('Timesteps')
        plt.ylabel('Rewards')
        plt.savefig('reward_plot.png')
        print("Reward curve plotted and saved as reward_plot.png")
    except Exception as e:
        print(f"Could not plot rewards: {e}")

if __name__ == "__main__":
    main()
