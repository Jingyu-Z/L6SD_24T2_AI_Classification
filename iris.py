import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_iris
from joblib import dump, load

# Load the dataset
iris = load_iris()
iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Display basic information about the dataset
print(iris_df.head())
print(iris_df.tail())
print(iris_df.shape)
print(iris_df.info())

# Check for null values in the DataFrame
print(iris_df.isnull().sum())

# Create input dataset from original dataset by selecting only features (drop the target)
X = iris_df.drop(['target'], axis=1)
Y = iris_df['target']

# Standardize the dataset
sc = StandardScaler()
x_scaled = sc.fit_transform(X)

# Standardize the target as well
sc1 = StandardScaler()
y_reshape = Y.values.reshape(-1, 1)
y_scaled = sc1.fit_transform(y_reshape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)

# Initialize the models
linear_model = LinearRegression()
ridge = Ridge()
en = ElasticNet()

# Train the models using the training set
linear_model.fit(X_train, y_train)
ridge.fit(X_train, y_train)
en.fit(X_train, y_train)

# Prediction on the validation/test data
linear_model_preds = linear_model.predict(X_test)
ridge_preds = ridge.predict(X_test)
en_preds = en.predict(X_test)

# Evaluate model performance
linear_model_rmse = mean_squared_error(y_test, linear_model_preds, squared=False)
ridge_rmse = mean_squared_error(y_test, ridge_preds, squared=False)
en_rmse = mean_squared_error(y_test, en_preds, squared=False)

# Display the evaluation results
print(f"Linear Regression RMSE: {linear_model_rmse}")
print(f"Ridge Regression RMSE: {ridge_rmse}")
print(f"Elastic Net RMSE: {en_rmse}")

# Choose the best model
model_objects = [linear_model, ridge, en]
rmse_value = [linear_model_rmse, ridge_rmse, en_rmse]

best_model_index = rmse_value.index(min(rmse_value))
best_model_object = model_objects[best_model_index]

# Visualize the model results
models = ['Linear Regression', 'Ridge Regression', 'Elastic Net']

plt.figure(figsize=(8, 6))
bars = plt.bar(models, rmse_value, color=['blue', 'green', 'orange'])

# Add RMSE values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.0001, round(yval, 5), ha='center', va='bottom', fontsize=10)

plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Model RMSE Comparison')
plt.xticks(rotation=0)  # Rotate model names for better visibility
plt.tight_layout()
plt.show()

# Retrain the best model on the entire dataset
best_model_object.fit(x_scaled, y_scaled)

# Save the best model
dump(best_model_object, "iris_model.joblib")

# Load the model
loaded_model = load('iris_model.joblib')

# Gathering user inputs for prediction
try:
    sepal_length = float(input("Enter sepal length: "))
    sepal_width = float(input("Enter sepal width: "))
    petal_length = float(input("Enter petal length: "))
    petal_width = float(input("Enter petal width: "))
except ValueError as e:
    print(f"Invalid input: {e}")
    exit()

# Ensure the new input data has the same feature names as the original data
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=iris['feature_names'])

# Transform the new input data
X_test_new = sc.transform(input_data)

# Predict on new test data
preds_value = loaded_model.predict(X_test_new)
predicted_target = sc1.inverse_transform(preds_value)
print("Predicted target value based on input: ", predicted_target[0][0])
