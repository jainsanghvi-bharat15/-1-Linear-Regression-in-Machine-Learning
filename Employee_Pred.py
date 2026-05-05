import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
# 1. Read the training dataset (Experience vs Salary)
# This file contains two columns: experience, salary
mydf = pd.read_excel("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(1) Linear Regresion/Employee LR_ML/Employee.xlsx")
print(mydf)

# 2. Plot the data to visualize how salary changes with experience
plt.xlabel('Experience', color='blue')
plt.ylabel('Salary', color='blue')

# Scatter plot of experience vs salary
plt.scatter( mydf.experience,mydf.salary,color='red',edgecolors='black',marker='D')
plt.grid(True, linestyle=':', alpha=0.2, color='blue')
plt.show()

# 3. Create and train the Linear Regression model
mymodel = linear_model.LinearRegression()

# Drop salary column so only 'experience' is used as the dependency
mymodel.fit(mydf.drop('salary', axis='columns'), mydf.salary)
print(mymodel)

# 4. Predict salary for a single experience value (user input)
ex = int(input("Enter your experience: "))
# For prediction, scikit-learn always expects a 2D array
print("Predicted salary is:", mymodel.predict([[ex]]))

# 5. Manual verification using formula: y = α + βx
alpha = mymodel.intercept_     # Intercept term (α)
beta = mymodel.coef_           # Slope/Coefficient (β)
print("Manual calculation of ML model:", alpha + beta * ex)

# 6. Batch prediction using another Excel file
mydf_exp = pd.read_excel("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/Employee LR_ML/Experience.xlsx")

# Predict salary for the entire column of experience values
pred = mymodel.predict(mydf_exp)
print("Predicted Salary via bulk experience format is:", pred)

# Add predictions to the DataFrame
mydf_exp['Salary'] = pred

# Save updated sheet with predictions included
mydf_exp.to_excel("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/Employee LR_ML/Experience.xlsx",sheet_name="Sheet2")
print("Prediction done!")