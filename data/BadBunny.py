import pandas as pd
import numpy as np
from scipy.optimize import minimize


df = pd.read_csv('BadBunny(12-11,25-1).csv')
Outcome = 'Exactly 3'
#Cleaning up data
headers = list(df.columns)
headers.remove('timestamp')
headers.sort()
df = df.dropna(subset=headers)
#Discard anomalies (greater or less than 3 standard deviations from mean)

for i in headers:
    mean = df[i].mean()
    sd = df[i].std()
    upper = mean + 3*sd
    lower = mean - 3*sd
    df = df[(df[i] >= lower) & (df[i] <= upper)]

outcomes = {cat: (1 if cat == Outcome else 0) for cat in headers}

#Transform it into log-likelihood
df_prices = df[headers] / 100.0

def negLogLikelihood(theta):
    result = 0
    for i in headers:
        p = np.clip(df_prices[i].values,1e-7, 1-1e-7)
        if i == Outcome:
            result += np.sum(np.log(p**theta))
        else:
            result += np.sum(np.log(1-p**theta))
    return -result

res = minimize(negLogLikelihood, 1.0)

print(f"MLE theta = {res.x[0]:.4f}")