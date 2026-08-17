import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data = pd.read_csv("Titanic-Dataset.csv")
dataFrame = pd.DataFrame(data)

OHE = OneHotEncoder(sparse_output=False, drop='first')
LE = LabelEncoder()

dataFrame['encoded_sex'] =  LE.fit_transform(dataFrame['Sex'])
dataFrame.drop(["Sex"],axis=1)

encoded_embarked = OHE.fit_transform(dataFrame[['Embarked']])
encoded_embarked_columns = OHE.get_feature_names_out(['Embarked'])
encoded_embarked_df = pd.DataFrame(encoded_embarked, columns=encoded_embarked_columns)
dataFrame = pd.concat([dataFrame.drop(columns=['Embarked']), encoded_embarked_df], axis=1)

print(dataFrame)