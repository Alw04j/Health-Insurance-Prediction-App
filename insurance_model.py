import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle

class InsuranceModel:
    def __init__(self, model=None, scaler=None, mse=None, r2=None, y_test=None, y_pred=None):
        self.model = model or GradientBoostingRegressor()
        self.scaler = scaler or StandardScaler()
        self.mse = mse
        self.r2 = r2
        self.y_test = y_test
        self.y_pred = y_pred

    def predict_insurance(self, age, sex, bmi, children, smoker, region):
        sex_encoded = 1 if sex == 'male' else 0
        smoker_encoded = 1 if smoker == 'yes' else 0
        region_encoded = {'northeast': 0, 'northwest': 1, 'southeast': 2, 'southwest': 3}[region]

        input_data = pd.DataFrame([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]],
                                  columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])
        input_data = self.scaler.transform(input_data)
        return self.model.predict(input_data)[0]

    def train_model(self, data_file='healt insurance.csv'):
        # Load and split data
        (X_train, X_test, self.y_train, self.y_test), feature_names = load_data(data_file)
        
        # Standardize the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, self.y_train)
        
        # Predict and evaluate
        self.y_pred = self.model.predict(X_test_scaled)
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.r2 = r2_score(self.y_test, self.y_pred)
        
        return self.mse, self.r2

    def retrain_model(self, data_file='health insurance.csv'):
        # Retrain the model with the updated dataset
        return self.train_model(data_file)

    def append_data(self, new_data, data_file='health insurance.csv'):
        # Load existing data
        data = pd.read_csv(data_file)

        # Append new data
        data = pd.concat([data, new_data], ignore_index=True)
    
        # Save updated data
        data.to_csv(data_file, index=False)

        # Retrain the model with updated data
        self.retrain_model(data_file)

def load_data(url):
    data = pd.read_csv(url)
    X = data.drop('charges', axis=1)
    y = data['charges']

    # Encode categorical features
    label_encoder = LabelEncoder()
    X['sex'] = label_encoder.fit_transform(X['sex'])
    X['smoker'] = label_encoder.fit_transform(X['smoker'])
    X['region'] = label_encoder.fit_transform(X['region'])

    feature_names = X.columns.tolist()
    return train_test_split(X, y, test_size=0.2, random_state=42), feature_names

def save_model(model, filename='insurance_model.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def load_model(filename='insurance_model.pkl'):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# Example of training the model and saving it

    
def main():
    # Load and split data
    (X_train, X_test, y_train, y_test), feature_names = load_data('health insurance.csv')
    print (len(X_train), len(X_test), len(y_train), len(y_test))
    # Initialize the model
    insurance_model = InsuranceModel()

    # Train and evaluate the model
    mse, r2 = insurance_model.train_model('health insurance.csv')
    
    # Store values in the InsuranceModel instance
    insurance_model.mse = mse
    insurance_model.r2 = r2
    print(len(insurance_model.y_pred))
    #print(len(insurance_model.y_test))
    # Save the model
    save_model(insurance_model)

if __name__ == "__main__":
    main()