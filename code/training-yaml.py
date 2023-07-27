# import important packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import yaml

# Load config file
config = yaml.safe_load(open("../config/config.yaml"))

# path to the dataset
filename = config['dataset']['path']

# load data
data = pd.read_csv(filename, header=None)

# add column names
data.columns = config['dataset']['columns']

# replace "?" with -99999
data = data.replace('?', -99999)

# drop id column
data = data.drop(config['dataset']['unnecessary_columns'], axis=1)

# Define X (independent variables) and y (target variable)
X = data.drop([config['dataset']['target_column']], axis=1)
y = data[config['dataset']['target_column']]

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config['dataset']['test_size'], random_state=config['dataset']['random_state'])

# call our model and fit to our data
model = KNeighborsClassifier(
                            n_neighbors=config['model']['params']['n_neighbors'], 
                            weights=config['model']['params']['weights'],
                            algorithm=config['model']['params']['algorithm'],
                            leaf_size=config['model']['params']['leaf_size'],
                            p=config['model']['params']['p'],
                            metric=config['model']['params']['metric'],
                            n_jobs=config['model']['params']['n_jobs']
                        )

# training the model
model.fit(X_train, y_train)

# test our model
result = model.score(X_test, y_test)
print("Accuracy score is {:.1f} %".format(result*100))

# save our classifier in the model directory
model_name = config['model']['output']['name']
joblib.dump(model, f'../model/{model_name}')
print(f"Model saved as {model_name}")
    