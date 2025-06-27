import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

Advertising_data = pd.read_csv('Advertising.csv')

# print(Advertising_data.head(5))

# print(Advertising_data.info())

# Advertising_data = Advertising_data.drop(columns=["Unnamed: 0"])

# print(Advertising_data.describe())

X = Advertising_data[['TV','Radio','Newspaper']]
y = Advertising_data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

Sales_predict = model.predict(X_test)

mse = mean_squared_error(y_test, Sales_predict)
mae = mean_absolute_error(y_test, Sales_predict)
r2 = r2_score(y_test, Sales_predict)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

plt.figure(figsize=(8,5))
plt.scatter(y_test, Sales_predict, color='Blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='yellow',linewidth=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs. Predicted Sales")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize = (8,5))
features = ['TV', 'Radio', 'Newspaper']
coeffs = model.coef_

plt.bar(features, coeffs, color=['skyblue', 'orange', 'lightgrey'])
plt.title('Impact of Advertising Channels on Sales')
plt.xlabel('Channel')
plt.ylabel('Impact (Coefficient)')


for i in range(len(coeffs)):
    plt.text(i, coeffs[i] + 0.01, f"{coeffs[i]:.2f}", ha='center')

plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
