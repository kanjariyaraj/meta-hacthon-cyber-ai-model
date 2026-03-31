import numpy as np
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from cyber_env import CyberGymEnv

def get_manual_input():
    print("\n" + "="*30)
    print(" ENTER SYSTEM METRICS")
    print("="*30)
    try:
        failed_login = float(input("Failed Login Count (0-100): "))
        alert_count = float(input("Alert Count (0-100): "))
        ddos_flag = float(input("DDoS Flag (0.0-100.0): "))
        suspicious_flag = float(input("Suspicious Flag (0.0-100.0): "))
        system_status = float(input("System Health (0.0-10.0): "))
        
        print("\nAttack Types: 0: Normal, 1: Brute Force, 2: DDoS, 3: Malware")
        actual_attack = input("Actual Attack Type (0-3 or Enter to skip): ")
        actual_attack = int(actual_attack) if actual_attack.strip() != "" else None
        
        return np.array([failed_login, alert_count, ddos_flag, suspicious_flag, system_status], dtype=np.float32), actual_attack
    except ValueError:
        print(">> Invalid input. Please enter numbers.")
        return None, None

def calculate_baseline_accuracy(model, vec_env, episodes=5):
    """Quick evaluation to show 'Real Accuracy' from the dataset."""
    print(f"\n[Calculating Baseline Accuracy over {episodes} episodes...]")
    correct = 0
    total = 0
    attack_to_action = {0: 3, 1: 0, 2: 2, 3: 1}
    
    for _ in range(episodes):
        obs = vec_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = vec_env.step(action)
            info = infos[0]
            actual = info.get('actual_attack', 0)
            if action[0] == attack_to_action.get(actual):
                correct += 1
            total += 1
            done = dones[0]
    return (correct / total) * 100 if total > 0 else 0

def main():
    if not os.path.exists("cyber_rl_model.zip") or not os.path.exists("vec_normalize.pkl"):
        print("Error: Trained model or normalization stats not found.")
        return

    # Load environment and normalization
    env = CyberGymEnv()
    vec_env = DummyVecEnv([lambda: env])
    vec_env = VecNormalize.load("vec_normalize.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    # Load the PPO model
    model = PPO.load("cyber_rl_model", env=vec_env)
    
    actions_map = {
        0: "Block IP",
        1: "Alert Admin",
        2: "Isolate System",
        3: "Ignore"
    }
    attack_to_action = {0: 3, 1: 0, 2: 2, 3: 1}

    # Show Baseline Accuracy
    baseline_acc = calculate_baseline_accuracy(model, vec_env)
    print(f">> Dataset Baseline Accuracy: {baseline_acc:.2f}%")
    print("------------------------------------------------")

    print("\n--- Manual Model Checker (with Confidence) ---")
    
    total_checks = 0
    correct_checks = 0
    
    while True:
        obs, actual_attack = get_manual_input()
        if obs is None: continue
        
        # 1. Normalize and Predict
        norm_obs = vec_env.normalize_obs(obs.reshape(1, -1))
        
        # 2. Get Confidence (Probabilities)
        obs_tensor = torch.as_tensor(norm_obs).to(model.device)
        with torch.no_grad():
            # For PPO, we get the distribution from the policy
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs[0].cpu().numpy()
        
        action, _ = model.predict(norm_obs, deterministic=True)
        predicted_action = action[0]
        confidence = probs[predicted_action] * 100
        
        print("\n" + "-"*30)
        print(f"PREDICTION: {actions_map[predicted_action].upper()}")
        print(f"CONFIDENCE: {confidence:.2f}%")
        print("-"*30)
        
        print("Full Probability Distribution:")
        for idx, p in enumerate(probs):
            print(f" - {actions_map[idx]:<15}: {p*100:6.2f}%")
        
        if actual_attack is not None:
            total_checks += 1
            correct_action = attack_to_action.get(actual_attack)
            if predicted_action == correct_action:
                correct_checks += 1
                print("\nRESULT: [CORRECT] Match found.")
            else:
                print(f"\nRESULT: [WRONG] Expected: {actions_map[correct_action]}")
        
        cont = input("\nTest another sample? (y/n): ")
        if cont.lower() != 'y':
            break

    # Show Final Accuracy Summary
    if total_checks > 0:
        session_accuracy = (correct_checks / total_checks) * 100
        print("\n" + "="*40)
        print("        MANUAL TEST SESSION SUMMARY")
        print("="*40)
        print(f"Dataset Baseline Accuracy: {baseline_acc:.2f}%")
        print(f"Manual Session Accuracy:   {session_accuracy:.2f}%")
        print(f"Total Correct:             {correct_checks}/{total_checks}")
        print("="*40)
    else:
        print("\nNo manual accuracy data recorded.")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
