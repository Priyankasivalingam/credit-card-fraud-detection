import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("transactions_synthetic.csv")

# Drop unwanted columns (IDs, generated data, etc.)
df = df.drop(columns=['transaction_id', 'fraud_probability'], errors='ignore')

# Separate features and target
y = df['is_fraud']
X = df.drop(columns=['is_fraud'])

# One-hot encode categorical columns (except IDs)
cat_cols = X.select_dtypes(include=['object', 'category']).columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

# Save column structure
model_columns = X.columns.tolist()

# Scale numeric columns
num_cols = X.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save all artifacts
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(model_columns, "model_columns.pkl")

print("✅ Model training completed and saved successfully!")
