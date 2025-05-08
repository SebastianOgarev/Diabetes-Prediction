# Diabetes Prediction using Machine Learning

This project implements a supervised learning model to predict diabetes using the Pima Indians Diabetes Dataset. The model uses logistic regression to classify whether a patient has diabetes based on various health indicators.

## Features

- Data preprocessing and cleaning
- Feature standardization
- Logistic Regression model implementation
- Model evaluation with multiple metrics
- Data visualization
- Sample prediction functionality

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

The current model achieves:
- Accuracy: ~77%
- Precision: 81% for non-diabetic, 68% for diabetic
- Recall: 83% for non-diabetic, 65% for diabetic

## Setup and Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
cd diabetes-prediction
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
python diabetes_prediction.py
```

## Project Structure

- `diabetes_prediction.py`: Main script containing the model implementation
- `requirements.txt`: Python package dependencies
- `diabetes.csv`: Dataset file
- Generated visualizations:
  - `glucose_bmi_plot.png`: Relationship between glucose and BMI
  - `confusion_matrix.png`: Model prediction accuracy visualization

## Future Improvements

Potential areas for enhancement:
1. Try different machine learning algorithms (Random Forest, SVM)
2. Implement cross-validation
3. Add feature importance analysis
4. Create a simple web interface for predictions
5. Add more data visualizations

## Author

Vsevolod Ogarev
