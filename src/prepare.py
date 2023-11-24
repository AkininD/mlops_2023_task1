import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def prepare_data(df):
    imputer = SimpleImputer(strategy='mean')
    df['Age'] = imputer.fit_transform(df[['Age']])
    df['Fare'] = imputer.fit_transform(df[['Fare']])

    label_encoder = LabelEncoder()
    for column in ['Sex', 'Embarked']:
        df[column] = label_encoder.fit_transform(df[column].astype(str))

    return df

df = pd.read_csv('data/features/train.csv')
prepared_df = prepare_data(df)
prepared_df.to_csv('data/prepared/train.csv')

