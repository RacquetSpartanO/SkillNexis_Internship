import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv("Titanic-Dataset.csv")

#cleaning dataset
mean_Age = data['Age'].mean()
mean_Age = np.int64(mean_Age)

mode_Cabin = data['Cabin'].mode()[0].split()[0]

mode_Embarked = data['Embarked'].mode()[0]

data['Age'] = data['Age'].fillna(mean_Age)
data['Cabin'] = data['Cabin'].fillna(mode_Cabin)
data['Embarked'] = data['Embarked'].fillna(mode_Embarked)

dataFrame = pd.DataFrame(data)

#Encode "Sex" and "Embarked"

OHE = OneHotEncoder(sparse_output=False, drop='first')
LE = LabelEncoder()
"""encoded_sex = OHE.fit_transform(dataFrame[['Sex']])
encoded_sex_columns = OHE.get_feature_names_out(['Sex'])
encoded_sex_df = pd.DataFrame(encoded_sex, columns=encoded_sex_columns)
dataFrame = pd.concat([dataFrame.drop(columns=['Sex']), encoded_sex_df], axis=1)"""

dataFrame["Sex"] = LE.fit_transform(dataFrame['Sex'])

OHE = OneHotEncoder(sparse_output=False, drop='first')
encoded_cabin = OHE.fit_transform(dataFrame[['Cabin']])
encoded_cabin_columns = OHE.get_feature_names_out(['Cabin'])
encoded_cabin_df = pd.DataFrame(encoded_cabin, columns=encoded_cabin_columns)
dataFrame = pd.concat([dataFrame.drop(columns=['Cabin']), encoded_cabin_df], axis=1)

encoded_embarked = OHE.fit_transform(dataFrame[['Embarked']])
encoded_embarked_columns = OHE.get_feature_names_out(['Embarked'])
encoded_embarked_df = pd.DataFrame(encoded_embarked, columns=encoded_embarked_columns)
dataFrame = pd.concat([dataFrame.drop(columns=['Embarked']), encoded_embarked_df], axis=1)

#build model
LR = LogisticRegression(max_iter=1000)

X = dataFrame.drop(["Survived","PassengerId","Name","Ticket"],axis=1)
Y = dataFrame["Survived"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

model = LR.fit(X_train,Y_train)

accuracy = model.score(X_test,Y_test)

print(f"Accuracy: {(accuracy * 100):.2f}%")