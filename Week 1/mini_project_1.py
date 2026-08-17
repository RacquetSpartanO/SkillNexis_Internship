import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

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

#print(dataFrame)

#Encode "Sex" and "Embarked"

OHE = OneHotEncoder(sparse_output=False, drop='first')
encoded_sex = OHE.fit_transform(dataFrame[['Sex']])
encoded_sex_columns = OHE.get_feature_names_out(['Sex'])
encoded_sex_df = pd.DataFrame(encoded_sex, columns=encoded_sex_columns)
dataFrame = pd.concat([dataFrame.drop(columns=['Sex']), encoded_sex_df], axis=1)

encoded_embarked = OHE.fit_transform(dataFrame[['Embarked']])
encoded_embarked_columns = OHE.get_feature_names_out(['Embarked'])
encoded_embarked_df = pd.DataFrame(encoded_embarked, columns=encoded_embarked_columns)
dataFrame = pd.concat([dataFrame.drop(columns=['Embarked']), encoded_embarked_df], axis=1)

#Visualising Age distribution

age_counts = dataFrame['Age'].value_counts().sort_index()

plt.plot(age_counts.index, age_counts.values)
plt.title('Age Distribution (Line Plot)', fontsize=14)
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True)
plt.show()

#exporting csv file

dataFrame.to_csv("new_titanic_file.csv", index=False)