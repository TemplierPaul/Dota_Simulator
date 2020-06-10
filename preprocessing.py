import warnings

warnings.filterwarnings("ignore")

from constant_features import *

# Utils
import pandas as pd
import numpy as np
import time
import random as rd
import glob
from joblib import dump, load
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_squared_log_error, r2_score

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor

"""
Toggle to remove constant features
"""
remove_constant_features = True

"""
Cast an action into a 1-hot vector.
Movements are only translated into North and East movements (can take -1, 0, or 1 as values).
"""


def action_to_vect(action):
    l = np.zeros(30)
    l[action] = 1
    north = l[[1, 2, 8]].sum(axis=0) - l[[4, 5, 6]].sum(axis=0)
    east = l[[2, 3, 4]].sum(axis=0) - l[[6, 7, 8]].sum(axis=0)
    vect = np.concatenate([[north, east], l[9:]])
    return vect


"""
Cast an action or a pandas Series into a &-hot vector or a DataFrame.
"""


def actions_to_vect(a):
    if type(a) == pd.Series:
        columns = ['North', "East",
                   'Top Bounty Rune', 'Bot Bounty Rune', 'Top Powerup  Rune', 'Bot Powerup  Rune',
                   "Attack Enemy Hero", "Attack Enemy Tower",
                   "Attack Creep 0 (Nearest Enemy)", "Attack Creep 1", "Attack Creep 2", "Attack Creep 3",
                   "Attack Creep 4",
                   "Attack Creep 5 (Nearest Friendly)", "Attack Creep 6", "Attack Creep 7", "Attack Creep 8",
                   "Attack Creep 9",
                   "Cast Shadowraze 1 (Short)", "Cast Shadowraze 2 (Medium)", "Cast Shadowraze 3 (Long)",
                   "Cast Requiem of Souls (Ultimate)", "Use Healing Salve"
                   ]
        df = pd.DataFrame(list(a.apply(action_to_vect)), columns=columns)
        return df

    else:
        return action_to_vect(a)


"""
Transforms and scales data into input and output DataFrames, and also returns the scaler used
"""


def transform(df, scaler=None):
    action = actions_to_vect(df['action'])
    state = df.drop(columns=['action'])

    # Remove constant features
    if remove_constant_features:
        state = remove_constants(state)

    # Scale
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(state)
    state = pd.DataFrame(scaler.transform(state))

    #     print(state.head())

    X = pd.concat([state, action], axis=1, ignore_index=False)
    X = X.drop(index=len(X) - 1)

    y = state.drop(index=0)
    return X, y, scaler


"""
Pipeline to import and transform data from a list of file paths.
"""


def data_pipeline(files, verbose=True):
    scaler = None
    X_list, y_list = [], []
    for f in files:
        df = pd.read_csv(f, index_col=0)
        X, y, scaler = transform(df, scaler=scaler)
        X_list.append(X)
        y_list.append(y)
        if verbose: print('%s   \tImported: %d lines' % (f, len(df)), end='\n')
    X = pd.concat(X_list, axis=0, ignore_index=True)
    y = pd.concat(y_list, axis=0, ignore_index=True)
    return X, y, scaler


"""
Trains one or many estimators on the same dataset. 
Computes multiple scores for each.
"""


def train_model(model, X, y):
    print("X shape:", X.shape, "\ny shape:", y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    print("Train_test shapes", X_train.shape, X_test.shape, y_train.shape, y_test.shape, '\n')

    if type(model) != list:
        model = [model]

    for m in model:
        print("-- Training %s --\n" % str(type(m)).split('.')[-1][:-2])
        t0 = time.time()
        print('Training...', end='\n')
        m.fit(X_train, y_train)
        print('Training time %ds\n' % int(time.time() - t0))
        y_pred = m.predict(X_test).cpu()

        print("Scores:")
        score_expl_var = explained_variance_score(y_test, y_pred)
        print("Explained variance: \t{:4.4f} (Best: 1.0)".format(score_expl_var))

        try:
            score_mse = mean_squared_error(y_test, y_pred)
            print("Mean Square Error: \t{:4.4f} (Best: 0.0)".format(score_mse))
        except:
            pass
        try:
            score_msle = mean_squared_log_error(y_test, y_pred)
            print("Mean Square Log Error: \t{:4.4f} (Best: 0.0)".format(score_msle))
        except:
            pass
        try:
            score_r2 = r2_score(y_test, y_pred)
            print("R2 score: \t\t{:4.4f} (Best: 1.0)\n\n".format(score_r2))
        except:
            pass
    return model
