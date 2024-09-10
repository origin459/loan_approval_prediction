import pandas as pd
from sklearn.preprocessing import LabelEncoder

class Dataset:
    def __init__(self, path):
        # Load the dataset
        self.df = pd.read_csv(path)
        self.preprocess_data()
        
    def preprocess_data(self):
        # List of columns to drop (irrelevant or redundant for prediction)
        columns_to_drop = [
            'DEALER NAME', 'AADHAR VERIFIED', 'DEALER ID', 'APPLICATION LOGIN DATE', 'HDB BRANCH NAME', 'HDB BRANCH STATE', 
            'FIRST NAME', 'MIDDLE NAME', 'LAST NAME', 'mobile', 'Personal Email Address',
            'Pan Name', 'name', 'vpa', 'upi_name', 'EMPLOYER NAME', 'EMPLOY CONSTITUTION', 'MOBILE VERIFICATION'
        ]

        # Drop the unnecessary columns
        self.df_cleaned = self.df.drop(columns=columns_to_drop)

        # Handle missing or 'NO RESPONSE' Cibil Score
        self.df_cleaned['Cibil Score'] = self.df_cleaned['Cibil Score'].replace('NO RESPONSE', -1)
        self.df_cleaned['Cibil Score'] = self.df_cleaned['Cibil Score'].fillna(-1)  # Fill other missing values with -1
        self.df_cleaned.fillna('Unknown', inplace=True)

        # Convert the 'Cibil Score' to numeric to avoid issues with strings
        self.df_cleaned['Cibil Score'] = pd.to_numeric(self.df_cleaned['Cibil Score'], errors='coerce')

        # Extract the year from the 'DOB' column
        self.df_cleaned['DOB'] = self.df_cleaned['DOB'].astype(str).str[-4:].astype(int)

        # Calculate age by subtracting the year from 2022
        self.df_cleaned['AGE'] = 2022 - self.df_cleaned['DOB']

        # Drop the 'DOB' column if it's no longer needed
        self.df_cleaned = self.df_cleaned.drop(columns=['DOB'])

        # Initialize LabelEncoder
        label_encoder = LabelEncoder()

        # Function to identify if a column is truly categorical (i.e., contains only strings)
        def is_categorical(column):
            return self.df_cleaned[column].apply(lambda x: isinstance(x, str)).all()

        # Identify columns that are truly categorical (all string values)
        categorical_columns = [col for col in self.df_cleaned.columns if is_categorical(col)]

        # Apply LabelEncoder only to categorical columns
        for col in categorical_columns:
            print(f"Encoding column: {col}")
            self.df_cleaned[col] = label_encoder.fit_transform(self.df_cleaned[col])

        # Replace 'Unknown' with -1 for consistency
        self.df_cleaned = self.df_cleaned.replace('Unknown', -1)

        self.df_cleaned.to_csv("Cleaned_data.csv",index=False)