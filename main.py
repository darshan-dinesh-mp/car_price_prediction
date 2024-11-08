from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv("CarPrice.csv")

df = df.drop(columns=['car_ID', 'CarName'])

df.fillna(df.mean(), inplace=True)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')


# wheelbase = float(input("Enter wheelbase: "))
# carlength = float(input("Enter car length: "))
# carwidth = float(input("Enter car width: "))
# carheight = float(input("Enter car height: "))
# curbweight = float(input("Enter curb weight: "))
# enginesize = float(input("Enter engine size: "))
# horsepower = float(input("Enter horsepower: "))
# peakrpm = float(input("Enter peak RPM: "))
# citympg = float(input("Enter city MPG: "))
# highwaympg = float(input("Enter highway MPG: "))

# new_car_features = pd.DataFrame({
#     'wheelbase': [wheelbase],
#     'carlength': [carlength],
#     'carwidth': [carwidth],
#     'carheight': [carheight],
#     'curbweight': [curbweight],
#     'enginesize': [enginesize],
#     'horsepower': [horsepower],
#     'peakrpm': [peakrpm],
#     'citympg': [citympg],
#     'highwaympg': [highwaympg]
# })


new_car_features = {
    'wheelbase': [88.6],
    'carlength': [168.8],
    'carwidth': [64.1],
    'carheight': [48.8],
    'curbweight': [2548],
    'enginesize': [130],
    'horsepower': [111],
    'peakrpm': [5000],
    'citympg': [21],
    'highwaympg': [27]
}

new_cars = pd.DataFrame(new_car_features)
new_cars_scaled = scaler.transform(new_cars)
predicted_price = model.predict(new_cars_scaled)
print(f'Predicted Price: {predicted_price[0]}')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)],
         color="green", linestyle="--", linewidth=2, label="Perfect Prediction")

plt.scatter(y_test, y_pred, color="blue", label="Predicted vs. Actual Price")

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.legend()
plt.show()
