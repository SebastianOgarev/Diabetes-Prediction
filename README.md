# Diabetes Prediction using Machine Learning

This is the final project for our Algorithms class. It implements a supervised learning model to predict diabetes using the Pima Indians Diabetes Dataset (https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) . The project includes two implementations:
1. A basic implementation using Logistic Regression
2. An enhanced implementation with multiple models and advanced features

## Features

### Basic Implementation (`diabetes_prediction.py`)
- Data preprocessing and cleaning
- Feature standardization
- Logistic Regression model implementation
- Basic model evaluation
- Simple data visualization

### Enhanced Implementation (`diabetes_prediction_enhanced.py`)
- All features from basic implementation
- Multiple ML models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine (SVM)
- Cross-validation for robust model evaluation
- Comprehensive data visualization:
  - Feature analysis plots
  - Model comparison visualization
  - Feature importance analysis
- Detailed performance metrics for each model

## Dataset

The project uses the Pima Indians Diabetes Dataset, which includes the following features:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- DiabetesPedigreeFunction
- Age

## Model Performance

### Basic Implementation
- Logistic Regression achieves ~77% accuracy
- Precision: 81% for non-diabetic, 68% for diabetic
- Recall: 83% for non-diabetic, 65% for diabetic

### Enhanced Implementation
The enhanced version compares multiple models:
- Logistic Regression: ~77% accuracy
- Random Forest: ~76% accuracy
- SVM: ~75% accuracy

Each model's performance is evaluated using:
- Cross-validation scores
- Confusion matrices
- Precision, recall, and F1-score
- Feature importance analysis (for Random Forest)

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/SebastianOgarev/Diabetes-Prediction.git
cd Diabetes-Prediction
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the model:
```bash
# For basic implementation:
python diabetes_prediction.py

# For enhanced version with multiple models:
python diabetes_prediction_enhanced.py
```

## Project Structure

- `diabetes_prediction.py`: Basic implementation with logistic regression
- `diabetes_prediction_enhanced.py`: Enhanced version with multiple models and cross-validation
- `requirements.txt`: Python package dependencies
- `diabetes.csv`: Dataset file
- Generated visualizations:
  - `feature_analysis.png`: Analysis of key features
  - `model_comparison.png`: Comparison of model accuracies
  - `feature_importance.png`: Feature importance from Random Forest
  - `glucose_bmi_plot.png`: Relationship between glucose and BMI
  - `confusion_matrix.png`: Model prediction accuracy visualization

## Future Improvements

Potential areas for enhancement:
1. Implement hyperparameter tuning
2. Add more advanced algorithms (XGBoost, Neural Networks)
3. Create a simple web interface for predictions
4. Add more data visualizations
5. Implement model persistence for predictions

## Author

Sebastian Ogarev

