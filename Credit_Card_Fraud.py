import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Create a larger synthetic dataset with both classes
np.random.seed(0)
n_samples_legitimate = 800
n_samples_fraudulent = 200

legitimate_data = {
    'Time': np.random.randint(0, 100, n_samples_legitimate),
    'V1': np.random.randn(n_samples_legitimate),
    'V2': np.random.randn(n_samples_legitimate),
    'V3': np.random.randn(n_samples_legitimate),
    'Amount': np.random.uniform(1, 500, n_samples_legitimate),
    'Class': [0] * n_samples_legitimate  # 0 represents legitimate transactions
}

fraudulent_data = {
    'Time': np.random.randint(0, 100, n_samples_fraudulent),
    'V1': np.random.randn(n_samples_fraudulent),
    'V2': np.random.randn(n_samples_fraudulent),
    'V3': np.random.randn(n_samples_fraudulent),
    'Amount': np.random.uniform(1, 500, n_samples_fraudulent),
    'Class': [1] * n_samples_fraudulent  # 1 represents fraudulent transactions
}

# Combine legitimate and fraudulent data
df = pd.concat([pd.DataFrame(legitimate_data), pd.DataFrame(fraudulent_data)])

# Explore the dataset
print(df)

# Standardize the 'Amount' column
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# Split the data into features (X) and target (y)
X = df.drop(['Class'], axis=1)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

# Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Evaluation
def evaluate_model(model_name, predictions):
    print(f"Classification Report for {model_name}:")
    print(classification_report(y_test, predictions))
    print(f"Confusion Matrix for {model_name}:")
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

evaluate_model("Logistic Regression", lr_predictions)
evaluate_model("Decision Tree", dt_predictions)
evaluate_model("Random Forest", rf_predictions)
