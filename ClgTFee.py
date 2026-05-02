# 1. Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# 2. Load the dataset (College data for MLR)
df = pd.read_excel("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(1) Linear Regresion/College_Tution_Fee/College_Fee_MLR.xlsx")
print("Dataset:\n", df)

# 3. Define independent (features) and dependent (target) variables
X = df[['Ranking', 'Student_Satisfaction', 'Placement_Rate (%)']]  # Features: Ranking, Student Satisfaction, Placement Rate
y = df['Tuition_Fee ($)']   # Target: Tuition Fee

# 4. Train the Multiple Linear Regression model
model = linear_model.LinearRegression()
model.fit(X, y)
print("\nModel trained successfully")

# 5. Take user input for prediction
ranking = int(input("Enter College Ranking (1 = Best): "))
satisfaction = float(input("Enter Student Satisfaction (0–10): "))
placement = float(input("Enter Placement Rate (%): "))

# Model expects input in 2D array format
input_data = np.array([[ranking, satisfaction, placement]])

# Predict tuition fee
predicted_fee = model.predict(input_data)
print("\nPredicted Tuition Fee: $", round(predicted_fee[0], 2))

# -------------------------------------------------------------
# 6. Extract model parameters (for manual formula calculation)
alpha = model.intercept_   # Intercept (α)
beta = model.coef_         # Coefficients (β1, β2, β3)
print("\nModel Coefficients:")
print("β1 (Ranking):", beta[0])
print("β2 (Satisfaction):", beta[1])
print("β3 (Placement):", beta[2])
print("α (Intercept):", alpha)

# 7. Manual calculation using regression formula
# Formula: y = α + β1*x1 + β2*x2 + β3*x3
formula_fee = ( alpha
                + beta[0] * ranking
                + beta[1] * satisfaction
                + beta[2] * placement
              )
print("\nPredicted Tuition Fee (Manual Formula): $", round(formula_fee, 2))