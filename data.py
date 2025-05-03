import pandas as pd

# Load the dataset
df = pd.read_csv("malicious_phish.csv")

# Map multi-class labels to binary: 'genuine' for benign, 'fraud' for others
df['label'] = df['type'].apply(lambda x: 'genuine' if x == 'benign' else 'fraud')

# Optional: Convert to numeric labels
df['label'] = df['label'].map({'genuine': 0, 'fraud': 1})

# Check counts
print(df['label'].value_counts())  # 0 = genuine, 1 = fraud

# Save the cleaned binary-labeled dataset (optional)
df.to_csv("upi_fraud_binary.csv", index=False)
