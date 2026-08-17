import pandas as pd
import numpy as np

data = pd.read_csv("Titanic-Dataset.csv")

mean_Age = data['Age'].mean()
mean_Age = np.int64(mean_Age)

mode_Cabin = data['Cabin'].mode()[0].split()[0]

mode_Embarked = data['Embarked'].mode()[0]

data['Age'] = data['Age'].fillna(mean_Age)
data['Cabin'] = data['Cabin'].fillna(mode_Cabin)
data['Embarked'] = data['Embarked'].fillna(mode_Embarked)

dataFrame = pd.DataFrame(data)
print(dataFrame)