# Importing required libraries
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
iris = datasets.load_iris()

# # Create a DataFrame
# iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
# print(iris_df.head())

# # Describe the dataset 
# print(iris_df.describe())

# # Split into X and y
# X = iris_df.iloc[:, :-1]
# y = iris_df.iloc[:, -1]
# print(X.head())
# print(y.head())

# # Split into training and testing 
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0)

# X_train = np.asarray(X_train)
# y_train = np.asarray(y_train)
# X_test = np.asarray(X_test)
# y_test = np.asarray(y_test)

# print(f"Training set size: {X_train.shape[0]} samples \nTest set size: {X_test.shape[0]} samples")

# # Normalize the dataset
# scaler = StandardScaler().fit(X_train)  # Use StandardScaler instead of Normalizer
# normalized_X_train = scaler.transform(X_train)  # Apply scaler to the training set
# normalized_X_test = scaler.transform(X_test)  # Apply scaler to the test set

# print("X train before Normalization")
# print(X_train[0:5])
# print("\nX train after Normalization")
# print(normalized_X_train[0:5])

# # Visualize the dataset before normalization
# di = {0.0: "Setosa", 1.0: "Versicolor", 2.0: "Virginica"}
# before = sns.pairplot(iris_df.replace({"target": di}), hue='target')
# before.fig.suptitle("Pair Plot of the dataset Before normalization", y=1.08)
# plt.show()

# # Visualize the dataset after normalization
# normalized_df = pd.DataFrame(data=np.c_[normalized_X_test, y_test], columns=iris['feature_names'] + ['target'])
# after = sns.pairplot(normalized_df.replace({"target": di}), hue='target')
# after.fig.suptitle("Pair Plot of the dataset After normalization", y=1.08)
# plt.show()

X = iris.data[:, :3]  # Using Sepal Length, Sepal Width, Petal Length as features
y = iris.data[:, 3]   # Using Petal Width as the target

# Preprocess the data (standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
svr = SVR()

# Train models
lr.fit(X_train, y_train)
dt.fit(X_train, y_train)
svr.fit(X_train, y_train)

# Test models
y_pred_lr = lr.predict(X_test)
y_pred_dt = dt.predict(X_test)
y_pred_svr = svr.predict(X_test)

# Evaluate model performance
mse_lr = mean_squared_error(y_test, y_pred_lr)
mse_dt = mean_squared_error(y_test, y_pred_dt)
mse_svr = mean_squared_error(y_test, y_pred_svr)

r2_lr = r2_score(y_test, y_pred_lr)
r2_dt = r2_score(y_test, y_pred_dt)
r2_svr = r2_score(y_test, y_pred_svr)

print("Linear Regression Performance:")
print(f"Mean Squared Error: {mse_lr:.2f}")
print(f"R^2 Score: {r2_lr:.2f}\n")

print("Decision Tree Regression Performance:")
print(f"Mean Squared Error: {mse_dt:.2f}")
print(f"R^2 Score: {r2_dt:.2f}\n")

print("Support Vector Regression Performance:")
print(f"Mean Squared Error: {mse_svr:.2f}")
print(f"R^2 Score: {r2_svr:.2f}")


