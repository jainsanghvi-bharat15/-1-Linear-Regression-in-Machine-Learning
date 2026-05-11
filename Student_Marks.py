import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# 1. Read the training dataset (Marks vs Hours)
# This file contains two columns: hours, marks
mydf = pd.read_excel("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(1) Linear Regresion/Student_Marks LR_ML/result.xlsx")
print(mydf)
# 2. Plot the data to visualize how salary changes with experience
plt.xlabel('Hours', color='k')
plt.ylabel('Marks', color='k')
plt.grid(True, linestyle='--', alpha=0.8, color='black')
plt.plot(mydf.hours, mydf.marks, marker='o', color='red',markeredgecolor='black',markerfacecolor='yellow')
plt.show()

# 3. Create and train the Linear Regression model
mymodel = linear_model.LinearRegression()

# Drop marks column so only 'hours' is used as the dependency
mymodel.fit(mydf.drop('marks', axis='columns'), mydf.marks)
print(mymodel)

# 4. Predict salary for a single experience value (user input)
hr = int(input("Enter your study hours: "))
# For prediction, scikit-learn always expects a 2D array
print("Predicted marks:", mymodel.predict([[hr]]))

# 5. Manual verification using formula: y = α + βx
alpha = mymodel.intercept_     # Intercept term (α)
beta = mymodel.coef_           # Slope/Coefficient (β)
print("Manual calculation of ML model:", alpha + beta * hr)

# 6. Batch prediction using another Excel file
mydf_hr = pd.read_excel("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/Student_Marks LR_ML/result.xlsx",sheet_name='Sheet2')

# Predict marks for the entire column of experience values
pred = mymodel.predict(mydf_hr)
print("Predicted Salary via bulk experience format is:", pred)
print("Prediction done!")