import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("Housing.csv")
LR = LinearRegression()
LE = LabelEncoder()
		 
data['mainroad'] = LE.fit_transform(data['mainroad'])
data['guestroom'] = LE.fit_transform(data['guestroom'])
data['basement'] = LE.fit_transform(data['basement'])
data['hotwaterheating'] = LE.fit_transform(data['hotwaterheating'])
data['airconditioning'] = LE.fit_transform(data['airconditioning'])
data['prefarea'] = LE.fit_transform(data['prefarea'])
data['furnishingstatus'] = LE.fit_transform(data['furnishingstatus'])

dataf = pd.DataFrame(data)

X = dataf.drop("price",axis=1)
Y = dataf["price"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LR.fit(X_train, Y_train)

# Predict House Prices

pred = model.predict(X_test)

# Evaluate R square value

rsq_score = r2_score(Y_test, pred)

print(f"R-Square Score: {(rsq_score * 100):.2f} / 100")

# Plot predicted vs actual values
plt.scatter(range(len(Y_test)), Y_test, color='blue', label='Real Value')
plt.scatter(range(len(pred)), pred, color='red', label='Prediction')

plt.legend()
plt.show()