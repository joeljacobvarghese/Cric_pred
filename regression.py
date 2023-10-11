# Import the necessary libraries

import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("odi.csv")

# Select the features and target variable
X = data[["runs", "wickets", "overs"]]
y = data["total"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Regressor model
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(X_train, y_train)

# Create and train the SVM Regression model
svm_model = SVR(kernel="linear")
svm_model.fit(X_train, y_train)
#svm_model = random_forest_model

# Create and train the Polynomial Regression model
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)

# Calculate errors on the test set
y_test_pred_rf = random_forest_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_test_pred_rf)
r2_rf = r2_score(y_test, y_test_pred_rf)

y_test_pred_svm = svm_model.predict(X_test)
mse_svm = mean_squared_error(y_test, y_test_pred_svm)
r2_svm = r2_score(y_test, y_test_pred_svm)

y_test_pred_poly = poly_model.predict(X_test)
mse_poly = mean_squared_error(y_test, y_test_pred_poly)
r2_poly = r2_score(y_test, y_test_pred_poly)

def get_errors():
    return {
        "RandomForest": {"Mean Squared Error (MSE)": mse_rf, "R-squared (R2)": r2_rf},
        "SVM": {"Mean Squared Error (MSE)": mse_svm, "R-squared (R2)": r2_svm},
        "Polynomial": {"Mean Squared Error (MSE)": mse_poly, "R-squared (R2)": r2_poly}
    }


# Function to predict total based on the selected model
def predict_total(runs, wickets, overs, model_type="RandomForest"):
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({"runs": [runs], "wickets": [wickets], "overs": [overs]})

    # Select the model based on the specified model_type
    if model_type == "RandomForest":
        model = random_forest_model
    elif model_type == "SVM":
        model = svm_model
    elif model_type == "Polynomial":
        model = poly_model
    else:
        raise ValueError("Invalid model_type. Supported values are 'RandomForest', 'SVM', and 'Polynomial'.")

    # Make predictions using the selected model
    predicted_total = model.predict(input_data)
    errors = get_errors()
    return predicted_total[0], errors
