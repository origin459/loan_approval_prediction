# Loan Approval Prediction Model

## Project Overview
This project focuses on building a machine learning classification model to predict whether a loan application will be approved or not. The provided dataset (`assignment_train.csv`) contains various features related to applicants, including financial details and personal information. The objective is to preprocess this data and develop a model capable of accurately classifying loan approvals.

## Dataset Description
- **Dataset File**: `assignment_train.csv`
- **Target Variable**: `Loan Status` (Approved/Not Approved)
- **Features**:
  - Applicant's financial details (e.g., income, loan amount)
  - Applicant's personal information (e.g., employment status, credit history)
  - Other relevant data points contributing to loan approval decisions

## Data Preprocessing
The following steps were applied to preprocess the dataset:
1. **Handling Missing Values**: Missing values were imputed with appropriate substitutes based on feature characteristics (e.g., median for numerical, mode for categorical).
2. **Feature Engineering**: Created new features where necessary, such as deriving age from `dob` if needed.
3. **Categorical Encoding**: Categorical variables were label encoded using sklearn's `LabelEncoder` to convert them into numerical form.
4. **Scaling**: Continuous variables were scaled using `MinMaxScaler` to ensure uniformity across features.

## Model Building
A classification model was built to predict loan approval using the following steps:
1. **Data Splitting**: The data was split into training and testing sets.
2. **Algorithms Used**:
   - **Random Forest Classifier**: Selected as the final model due to superior performance.
   - **Logistic Regression**: Also tested but performed less well than Random Forest.
3. **Model Evaluation**:
   - **Accuracy**: Evaluated using accuracy score.
   - **Classification Report**: Provided precision, recall, and F1 score for better insight.
   - **Confusion Matrix**: Used to visualize model performance.

## Final Model
- **Trained Model**: The final model was saved as `random_forest_model.pkl`.

## Usage
1. Clone this repository:
   ```bash
   git clone https://github.com/origin459/loan_approval_prediction.git
   ```
2. Ensure you have all necessary libraries installed from `requirements.txt`.
```bash
pip install -r requirements.txt
```  
3. Run the main.py file and ensure that all the paths are correctly initialised since they are hard coded and are relative to the current directory.
   ```bash
   python main.py
   ```
