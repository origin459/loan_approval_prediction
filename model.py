import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

class Trainer:
    def __init__(self, data_path, target_column='Application Status'):
        # Load the cleaned dataset
        self.df_cleaned = pd.read_csv(data_path)
        self.target_column = target_column
        self.prepare_data()
        self.train_model()
        
    def prepare_data(self):
        # Features (X) and target (y)
        self.X = self.df_cleaned.drop(columns=[self.target_column])
        self.y = self.df_cleaned[self.target_column]
        
        # Split the dataset into training and testing sets (80% training, 20% testing)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
    def train_model(self):
        # Initialize the RandomForestClassifier
        self.rf_model = RandomForestClassifier(n_estimators=1000, random_state=42)
        
        # Train the model
        self.rf_model.fit(self.X_train, self.y_train)
        
        # Make predictions on the test set
        self.y_pred = self.rf_model.predict(self.X_test)
        
        # Evaluate the model
        self.evaluate_model()
        
        # Save the trained model (optional)
        joblib.dump(self.rf_model, 'random_forest_model.pkl')
        
    def evaluate_model(self):
        # Calculate accuracy
        self.accuracy = accuracy_score(self.y_test, self.y_pred)
        print(f'Accuracy: {self.accuracy * 100:.2f}%')

        # Generate a classification report
        print(classification_report(self.y_test, self.y_pred))

        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.rf_model.classes_, yticklabels=self.rf_model.classes_)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

