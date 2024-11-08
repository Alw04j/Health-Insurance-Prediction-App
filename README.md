# Health Insurance Prediction App 📊

A Streamlit-based web application that predicts health insurance charges based on user inputs such as age, BMI, number of children, smoking status, and other relevant factors. This app leverages machine learning algorithms to make accurate predictions and provides a user-friendly interface for easy interaction.

## 🌟 Features
- **Prediction**: Enter information like age, BMI, smoking status, etc., to get an estimated insurance charge.
- **Data Visualization**: View graphs displaying feature importance and actual vs. predicted charges.
- **Contribute Data**: Submit new data entries to improve model performance over time.

## 🚀 Demo
You can access the live demo of the app here: [Live Demo Link](https://health-insurance-prediction-app-1.streamlit.app/) *(Replace this link once the app is deployed)*

## 📂 Project Structure
- `app.py`: Main script containing the Streamlit app code.
- `model.pkl`: Serialized model file for loading the pre-trained Gradient Boosting model.
- `data/`: Contains the dataset and any other data files.
- `requirements.txt`: Lists all the necessary packages for running the app.

## 🧠 Model Details
This application uses a **Gradient Boosting Regressor** for predicting health insurance charges. The model was trained using the following features:
- **age**: Age of the individual.
- **sex**: Gender of the individual.
- **bmi**: Body Mass Index.
- **children**: Number of children covered by insurance.
- **smoker**: Smoking status of the individual.
- **region**: Residential region.

After trying several models, Gradient Boosting was chosen due to its high performance based on metrics like **R-squared** and **Mean Squared Error**.

## 📊 Data Visualization
- **Feature Importance**: Visualizes the impact of each feature on predictions.
- **Actual vs Predicted Charges**: Compares the model’s predicted values with actual charges for evaluation.

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Alw04j/Health-Insurance-Prediction-App.git
   
2. Change directory to the project folder:
   ```bash
   cd Health-Insurance-Prediction-App
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the Streamlit app:
5. ```bash
   streamlit run app.py

## 📝 Usage
1. Go to the Prediction page and enter the required details to get an insurance charge estimate.
2. Visit the Data Visualization section to see how each feature impacts insurance charges.
3. Use the Contribute Data page to submit new information, which will be stored for model retraining.

## 🤝 Contributing
Feel free to submit issues or pull requests. Contributions are welcome to enhance the app!

## 👤 Author
Alwin Jojy

Developed with ❤️ using Streamlit and Scikit-learn.
