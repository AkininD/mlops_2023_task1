import json
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score

with open('model/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

prepared_features = pd.read_csv('data/prepared/test.csv')
X_test = prepared_features.drop('Survived', axis=1)
y_test = prepared_features['Survived']

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

with open('model/evaluate.json', 'w') as eval_file:
    json.dump({'accuracy': accuracy}, eval_file)

