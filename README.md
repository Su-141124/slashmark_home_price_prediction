House Price Prediction using XGBoost:-
This project is a machine learning implementation that predicts house prices based on various
property features using the XGBoost Regressor.
Overview:-
The model uses the King County housing dataset, which includes attributes like number of
bedrooms, square footage, location, and more, to predict housing prices. The dataset is processed,
scaled, and used to train an XGBoost regression model.
Features Used:-
- bedrooms, bathrooms, sqft_living, sqft_lot, floors
- waterfront, view, condition, grade
- yr_built, yr_renovated, zipcode, lat, long
- sqft_living15, sqft_lot15
Workflow:-
1. Load the dataset (kc_house_data1.csv)
2. Preprocess the data: drop missing values, select features
3. Split into training and test sets
4. Scale features using StandardScaler
5. Train an XGBoost Regressor
6. Evaluate using:
 - Mean Absolute Error (MAE)
 - Mean Squared Error (MSE)
 - R² Score
Model Evaluation:-
After training, the model evaluates its performance on the test data using standard regression
metrics to ensure accuracy and generalization.
Requirements:-
- Python 3.x
- pandas
- numpy
- scikit-learn
- xgboost
Install dependencies via:
 pip install pandas numpy scikit-learn xgboost
