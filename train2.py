import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Generate a synthetic dataset
np.random.seed(42)
n_samples = 1000

data = {
    'Age': np.random.randint(18, 70, n_samples),
    'Monthly_Charges': np.round(np.random.uniform(20.0, 120.0, n_samples), 2),
    'Subscription_Length_Months': np.random.randint(1, 60, n_samples),
    'Support_Calls': np.random.randint(0, 10, n_samples),
}

df = pd.DataFrame(data)

# Churn logic (synthetic but realistic)
churn_prob = (
    (df['Age'] > 50).astype(int) * 0.2 +
    (df['Monthly_Charges'] > 80).astype(int) * 0.3 +
    (df['Support_Calls'] > 3).astype(int) * 0.4 -
    (df['Subscription_Length_Months'] > 24).astype(int) * 0.2
)
df['Churn'] = (churn_prob + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)

# Save the dataset so they have a CSV to show
df.to_csv('customer_churn2.csv', index=False)

# Train the model
X = df[['Age', 'Monthly_Charges', 'Subscription_Length_Months', 'Support_Calls']]
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
with open('model2.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Success! Generated customer_churn2.csv and model2.pkl")
