import pandas as pd
import numpy as np
import datetime
import os

def generate_dataset(num_samples=1000000, filename='cyber_traffic_dataset.csv'):
    print(f"Generating realistic dataset with {num_samples} samples...")
    np.random.seed(42)
    
    # Base timestamps
    start_time = datetime.datetime.now() - datetime.timedelta(days=30)
    timestamps = [start_time + datetime.timedelta(minutes=i) for i in range(num_samples)]
    
    # 0: Normal, 1: Brute Force, 2: DDoS, 3: Malware
    # Let's make 80% normal traffic, 20% attacks
    attack_types = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.8, 0.05, 0.1, 0.05])
    
    failed_login_count = np.zeros(num_samples)
    alert_count = np.zeros(num_samples)
    ddos_flag = np.zeros(num_samples)
    suspicious_flag = np.zeros(num_samples)
    
    for i in range(num_samples):
        if attack_types[i] == 0: # Normal
            failed_login_count[i] = np.random.poisson(0.5)
            alert_count[i] = np.random.poisson(0.1)
            ddos_flag[i] = np.random.uniform(0, 0.5)
            suspicious_flag[i] = np.random.uniform(0, 1.0)
        elif attack_types[i] == 1: # Brute Force
            failed_login_count[i] = np.random.uniform(10, 50)
            alert_count[i] = np.random.poisson(2)
            ddos_flag[i] = np.random.uniform(0, 1)
            suspicious_flag[i] = np.random.uniform(2, 4)
        elif attack_types[i] == 2: # DDoS
            failed_login_count[i] = np.random.poisson(1)
            alert_count[i] = np.random.poisson(5)
            ddos_flag[i] = np.random.uniform(8, 10)
            suspicious_flag[i] = np.random.uniform(1, 3)
        elif attack_types[i] == 3: # Malware
            failed_login_count[i] = np.random.poisson(0.2)
            alert_count[i] = np.random.poisson(3)
            ddos_flag[i] = np.random.uniform(0, 0.5)
            suspicious_flag[i] = np.random.uniform(8, 10)
            
    df = pd.DataFrame({
        'timestamp': timestamps,
        'failed_login_count': failed_login_count,
        'alert_count': alert_count,
        'ddos_flag': ddos_flag,
        'suspicious_flag': suspicious_flag,
        'attack_type': attack_types
    })
    
    df.to_csv(filename, index=False)
    print(f"Dataset generated and saved as {filename}.")

if __name__ == "__main__":
    generate_dataset()
