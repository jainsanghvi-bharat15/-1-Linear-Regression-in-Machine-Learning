import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
# 1. Load the dataset (Insurance price prediction data)
df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(1) Linear Regresion/Insurance-Price-Predication_LR/insurance.csv")

# 2. Clean the data by converting categorical values into numbers
    # Convert gender to numeric: female=0, male=1
df['sex'] = df['sex'].map({'female': 0, 'male': 1})

# Convert region to numeric categories
df['region'] = df['region'].map({'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3})

# Convert smoker status to numeric: yes=1, no=0
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

# 3. Check for missing values
print(df.isnull().sum())        # Expect all zeros (no missing values)
print(df)

# 4. Separate independent (X) and dependent (Y) variables
x = df.drop('charges', axis='columns')   # Features
y = df.charges                            # Target label

# 5. Split the dataset into training and testing sets
    # test_size=0.3 → 30% test data, 70% training data
    # random_state=0 → Ensures the same split every run
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# 6. Train the Linear Regression model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)
print(model)

# 7. Predict values for the training data
y_pred = model.predict(x_train)
print(y_pred)

# 8. Compare actual vs predicted (training data)
plt.scatter(y_train, y_pred, marker='D', color='blue')
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Training Data: Actual vs Predicted")
plt.show()

# 9. Calculate R² score for training data
    # R² close to 1 means accurate prediction, close to 0 means weak prediction
print("R2 Score (Training Data):", r2_score(y_train, y_pred))

# 10. Predict values for the test dataset
y_pred = model.predict(x_test)
print(y_pred)

# 11. Compare actual vs predicted (testing data)
plt.scatter(y_test, y_pred, marker='^', color='red')
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Testing Data: Actual vs Predicted")
plt.show()

# 12. Calculate R² score for testing data
print   ("R2 Score (Testing Data):", r2_score(y_test, y_pred))