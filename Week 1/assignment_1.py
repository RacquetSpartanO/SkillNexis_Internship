import pandas as pd

data = pd.read_csv("Titanic-Dataset.csv")
dataFrame = pd.DataFrame(data)
Shape = dataFrame.shape
information = dataFrame.info()
description = dataFrame.describe()

print(dataFrame)
print(f"\nDATASET INFO: {information}\nDATASET SHAPE: {Shape}")
print(f"\nDATASET DECRIPTION: {description}")