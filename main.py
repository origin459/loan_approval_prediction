from preprocess import Dataset 
from model import Trainer 
import pandas as pd 
import joblib 

#Change the paths accordingly
train_path = "Assignment_Train.csv"
test_path = "Assignment_Test.csv"  
output_path = "Cleaned_data.csv"
result_path = "Predictions.csv"
model_path = "random_forest_model.pkl"  

#First we make a temporary csv file to save the preprocessed results 
Dataset(train_path) 

#We use this preprocessed csv file to train the model and the output will be saved in the model_path
Trainer(output_path) 

#Again make a temporary preprocessing csv file for the test path
Dataset(test_path)  

#Load the model
model = joblib.load(model_path)  
test_df = pd.read_csv(output_path)
X_test = test_df.drop(columns=['UID'])  # Exclude 'uid' column for predictions
predictions = model.predict(X_test)

# Create a DataFrame for the output
results_df = pd.DataFrame({
    'uid': test_df['UID'],
    'Application_Status': predictions
}) 

#Replace 0 with a True and 1 with a False
results_df.replace(0,"True",inplace=True)
results_df.replace(1,"False",inplace=True)

# Save to CSV
results_df.to_csv(result_path, index=False)

print(f"Predictions saved to {result_path}")