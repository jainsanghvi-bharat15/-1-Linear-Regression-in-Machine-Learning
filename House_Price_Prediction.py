# (1) Home Price Prediction Application in Machine Learning via LR
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

# 1. Read the training dataset (Area vs Price)
mydf = pd.read_excel("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(1) Linear Regresion/House Prediction LR_ML/homeprice.xlsx")# DataFrame contains columns: area, price
print(mydf)

# 2. Plot the data to visualize the relationship
plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(mydf.area, mydf.price, color='blue', marker='*')# Scatter plot
plt.show()

# 3. Create and train the Linear Regression model
mymodel = linear_model.LinearRegression()   #Returns object of linear regression.
mynewdf = mydf.drop('price', axis='columns')#Drop the price column to get only the area
mymodel.fit(mynewdf, mydf.price)# Train the model using area as input and price as output

# 4. Read new area values to predict prices
area_df = pd.read_excel("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(1) Linear Regresion/House Prediction LR_ML/homearea.xlsx")   # Contains only 'area' column
print(area_df)

# 5. Predict prices for the new area values
pred = mymodel.predict(area_df)
print(pred)

# Add predictions to the DataFrame
area_df['Price'] = pred
print(area_df)

# 6. Save the predictions to a new Excel file
area_df.to_excel("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(1) Linear Regresion/House Prediction LR_ML/newprediction.xlsx")

# OPTIONAL: Taking user input for prediction
ar = int(input("Enter area for Price prediction"))
print("Predicted Price is:", mymodel.predict([[ar]]))

# OPTIONAL: Display model parameters (Coefficient and Intercept)
print("Coefficient (β):", mymodel.coef_)
print("Intercept (α):", mymodel.intercept_)
print("Manual Price for 5000 sqft:", mymodel.intercept_ + mymodel.coef_ * 5000)
print("Model is trained now")
print(mymodel)
