# Heart Disease Prediction Using Machine Learning Models
### Project Summary
This project shows a complete binary classification workflow using the UCI Heart Disease dataset.
The goal is to predict whether a patient is at risk of heart disease or not at risk based on their medical information.

The workflow includes:
- Loading and inspecting the dataset
- Visualizing the data
- Scaling features
- Splitting into train/test sets
- Training multiple machine learning models
- Evaluating them using several metrics
- Comparing results
- Predicting risk for a new patient

### Dataset Overview

The dataset contains medical details commonly used in heart disease diagnosis.

It includes:
- 13 numerical features (age, cholesterol, blood pressure, chest pain type, etc.)
- Target variable:
     - 0 → Not at risk
     - 1 → At risk
       
The dataset is already clean and ready for modeling.

### Step 1: Load Dataset

A cleaned CSV version of the UCI Heart Disease dataset is loaded from GitHub and converted into a Pandas DataFrame.

### Step 2: Data Inspection

Basic inspection includes:
- Checking structure (df.info())
- Summary statistics (df.describe())
- Missing values
- Class distribution
  
Findings:
- No missing values
- Balanced classes
- All features are numerical

### Step 3: Data Visualization

Two main visualizations are created:
- Class Distribution

Shows how many patients are at risk vs. not at risk.
- Correlation Heatmap
  
Shows relationships between medical features and helps understand which ones may influence heart disease.

### Step 4: Scaling + Train/Test Split
To prepare the data:
- Features are scaled using StandardScaler
- Data is split into 80% training and 20% testing
- Stratification keeps class balance consistent
  
This ensures fair model evaluation.

### Step 5: Train Machine Learning Models
Three classical ML models are trained:

- Logistic Regression
  
Simple, interpretable linear model.
- Random Forest Classifier
  
Ensemble of decision trees; handles non‑linear patterns well.
- XGBoost Classifier
  
Boosting‑based model; strong performance on structured data.

Each model is trained on the scaled training data and used to predict test labels.

### Step 6: Confusion Matrices
Confusion matrices are plotted for all models to show:
- Correct predictions
- Misclassifications
- Class‑wise performance
  
This helps visualize how each model behaves.

### Step 7: Evaluation Metrics
For each model, the following metrics are calculated:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC‑AUC
  
These metrics give a complete view of model performance.

### Step 8: Final Comparison Table
A comparison table is created to show which model performed best.

Example (actual results):
| Model                | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Random Forest        | 0.819672 | 0.761905  | 0.969697 | 0.853333 | 0.806277 |
| Logistic Regression  | lower    | lower     | lower  | lower    | lower    |
| XGBoost              | lower    | lower     | lower  | lower    | lower    |


Random Forest achieved the highest F1 Score.

### Step 9: Predicting Risk for a New Patient
A new patient’s medical values are entered into the model.

The model outputs a clear message:
- “The patient IS at risk of heart disease.”
  
or

- “The patient is NOT at risk of heart disease.”
  
This makes the project practical and easy to demonstrate.

### Final Conclusion
Overall Findings
- All models performed well on the dataset.
- Random Forest gave the best balance of precision and recall.
- High recall means it correctly identified most at‑risk patients, which is important in medical prediction.
- Logistic Regression and XGBoost performed reasonably but did not outperform Random Forest.

### Best Overall Model: Random Forest Classifier

This project demonstrates a complete machine learning pipeline/workflow, starting from data loading and preprocessing to model training, evaluation, and final prediction.

