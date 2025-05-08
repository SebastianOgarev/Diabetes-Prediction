import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

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
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
sns.scatterplot(x='Glucose', y='BMI', hue='Outcome', data=data)
plt.title('Glucose vs BMI')

plt.subplot(2, 2, 2)
sns.boxplot(x='Outcome', y='Age', data=data)
plt.title('Age Distribution by Outcome')

plt.subplot(2, 2, 3)
sns.histplot(data=data, x='BMI', hue='Outcome', multiple="stack")
plt.title('BMI Distribution')

plt.subplot(2, 2, 4)
sns.histplot(data=data, x='Glucose', hue='Outcome', multiple="stack")
plt.title('Glucose Distribution')

plt.tight_layout()
plt.savefig('feature_analysis.png')
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

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42)
}

# Train and evaluate models
print("\nTraining and evaluating models...")
results = {}

for name, model in models.items():
    print(f"\n{name}:")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # Train and evaluate on test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Store results
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    print(f"Test set accuracy: {results[name]['accuracy']:.3f}")
    print("\nConfusion Matrix:")
    print(results[name]['confusion_matrix'])
    print("\nClassification Report:")
    print(results[name]['classification_report'])

# Visualize model comparison
plt.figure(figsize=(10, 6))
accuracies = [results[name]['accuracy'] for name in models.keys()]
plt.bar(models.keys(), accuracies)
plt.title('Model Comparison - Accuracy')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.savefig('model_comparison.png')
plt.close()

# Feature importance for Random Forest
rf_model = models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

print("\nProject completed! Check the generated plots:")
print("- feature_analysis.png: Analysis of key features")
print("- model_comparison.png: Comparison of model accuracies")
print("- feature_importance.png: Feature importance from Random Forest") 