import pandas as pd

df = pd.read_csv('data/raw/train.csv')

features_to_keep = ['Survived', 'Age', 'Pclass', 'Sex', 'SibSp', 'Fare', 'Embarked']

df[features_to_keep].to_csv('data/features/train.csv', index=False)

