import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Load and Clean
df = pd.read_csv('Sabrina(12-11,25-1).csv')
categories = ['Exactly 0', 'Exactly 1', 'Exactly 2', 'Exactly 3', 'Exactly 4', 'Exactly 5', 'Exactly 6']
df['sum_p'] = df[categories].sum(axis=1)

# Discard anomalies: sum < 80 or > 120 and drop NaNs
clean_df = df[(df['sum_p'] >= 80) & (df['sum_p'] <= 120)].dropna(subset=categories)

# Outcomes: Exactly 0 = 1, others = 0
outcomes = {cat: (1 if cat == 'Exactly 0' else 0) for cat in categories}

def neg_log_likelihood(theta, p_vals, y_vals):
    p = np.clip(p_vals, 1e-6, 1 - 1e-6)
    prob = p**theta
    prob = np.clip(prob, 1e-6, 1 - 1e-6)
    return -np.sum(y_vals * np.log(prob) + (1 - y_vals) * np.log(1 - prob))

# Pool data for Market-wide Efficiency
all_p, all_y = [], []
for cat in categories:
    p_data = clean_df[cat].values / 100.0
    all_p.extend(p_data)
    all_y.extend(np.full(len(p_data), outcomes[cat]))

# Maximize
res = minimize(neg_log_likelihood, x0=[1.0], args=(np.array(all_p), np.array(all_y)), bounds=[(0.01, 10)])
print(f"MLE theta: {res.x[0]:.4f}")

