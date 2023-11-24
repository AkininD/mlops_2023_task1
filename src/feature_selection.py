import pandas as pd

files = ['train.csv', 'test.csv']

for file in files:
    df = pd.read_csv(f'data/raw/{file}')
    features_to_keep = ['Survived', 'Age', 'Pclass', 'Sex', 'SibSp', 'Fare', 'Embarked']
    df[features_to_keep].to_csv(f'data/features/{file}', index=False)

