# 1. Import required libraries
import pandas as pd
from sklearn import linear_model

# 2. Load the dataset
df = pd.read_excel("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(1) Linear Regresion/Salary Predication MLR/employee.xlsx")
print("Original Data:\n", df)

# 3. Data Cleaning and Preprocessing
# Rename columns for easier access (remove spaces and special characters)
df = df.rename(columns=
               {
                  'test_score(out of 10)': 'TestScore',
                  'interview_score(out of 10)': 'InterviewScore',
                  'salary($)': 'Salary'
               })
# Convert experience (text → numeric)
exp_map = { 'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
          }
df['experience'] = df['experience'].map(exp_map)

# Handle missing values
df['experience'] = df['experience'].fillna(2) # Replace missing experience with 2
df['TestScore'] = df['TestScore'].fillna(8)   # Replace missing test score with 8
print("\nCleaned Data:\n", df)

# 4. Train the Multiple Linear Regression model
model = linear_model.LinearRegression()

# Features → experience, TestScore, InterviewScore
# Target → Salary
model.fit(df.drop('Salary', axis='columns'), df['Salary'])
print("\nModel Trained Successfully")
print(model)

# 5. Predict Salary for new input values
# Example: 3 years experience, 9 test score, 6 interview score
predicted_salary = model.predict([[3, 9, 6]])
print("\nPredicted Salary:", predicted_salary)

# 6. Display model parameters (for manual calculation)
coef = model.coef_        # Coefficients (β values)
intercept = model.intercept_   # Intercept (α)
print("\nModel Coefficients (β):", coef)
print("Model Intercept (α):", intercept)

# 7. Manual calculation using regression equation
# Formula: y = α + β1*x1 + β2*x2 + β3*x3
salary = (
    intercept
    + coef[0] * 3   # experience
    + coef[1] * 9   # TestScore
    + coef[2] * 6   # InterviewScore
        )
print("\nManual Calculated Salary:", salary)