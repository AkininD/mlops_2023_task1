import pandas as pd
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_params(file_path):
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params

params = load_params('model/params.yaml')

prepared_features = pd.read_csv('data/prepared/train.csv')

X = prepared_features.drop('Survived', axis=1)
y = prepared_features['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_params = params.get('random_forest', {})
model = RandomForestClassifier(**rf_params)
model.fit(X_train, y_train)

with open('model/model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

