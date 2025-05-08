import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
print("Loading dataset...")
data = pd.read_csv('diabetes.csv')

# Display initial data information
print("\nDataset Info:")
print(data.info())
print("\nFirst few rows:")
print(data.head())
print("\nStatistical summary:")
print(data.describe())

# Handle missing values (zeros in some columns)
print("\nHandling missing values...")
for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
    data[col] = data[col].replace(0, data[col].median())

# Visualize key features
print("\nCreating visualizations...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=data)
plt.title('Relationship between Glucose and BMI')
plt.savefig('glucose_bmi_plot.png')
plt.close()

# Prepare data
print("\nPreparing data for model training...")
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
print("\nTraining Logistic Regression model...")
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Test with a sample
print("\nTesting with a sample:")
sample = X_test[0].reshape(1, -1)
print("Prediction for sample:", model.predict(sample))
print("Actual value:", y_test.iloc[0])

print("\nProject completed! Check the generated plots: glucose_bmi_plot.png and confusion_matrix.png") 