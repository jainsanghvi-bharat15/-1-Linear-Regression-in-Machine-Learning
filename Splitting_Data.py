# Import required libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("C:/Users/HP/3D Objects/Desktop/DA_using_Python/ML/(1) Linear Regresion/Splitting Data/Spilt_Data_CarP.csv")
df.head(5)  # Display first 5 rows

# Rename columns for easier usage (remove special characters)
df = df.rename(columns={'Age(yrs)': 'Age'})
df = df.rename(columns={'Sell Price($)': 'SellPrice'})
df.head(5)  # Check updated data

#Plot relationship between Mileage and Selling Price
plt.scatter(df['Mileage'], df['SellPrice'], color='red')

#Plot relationship between Age and Selling Price
plt.scatter(df['Age'], df['SellPrice'], color='red')

x = df[['Mileage', 'Age']]  # Define independent variables (features)
y = df[['SellPrice']]       # Define dependent variable (target)

# Split data into training and testing sets (70% train, 30% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Check size of each dataset
print(len(x_train), len(y_train), len(x_test), len(y_test))

# View datasets (optional for understanding)
print(x_train, y_train, x_test, y_test)

model = linear_model.LinearRegression()
model.fit(x_train, y_train) # Train model using training data
model.predict(x_test)       # Predict values for test data
model.score(x_test, y_test) # Check accuracy using R² score
y_pred = model.predict(x_train) # Predict values for training data

# Plot actual vs predicted values (training data)
plt.scatter(y_train, y_pred, color='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

# Calculate R² score for training data
print("R2 Score for Training data:", r2_score(y_train, y_pred))
y_pred = model.predict(x_test)  # Predict values for test data

# Plot actual vs predicted values (testing data)
plt.scatter(y_test, y_pred, color='blue', marker='+')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.show()

# Calculate R² score for testing data
print("R2 Score for Testing data:", r2_score(y_test, y_pred))