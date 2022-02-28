from collections import defaultdict
from tkinter import W
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def linear_model_descs_selective(pkl_file1, pkl_file2, logger):
    df1 = pd.read_pickle(pkl_file1)
    df2 = pd.read_pickle(pkl_file2)

    X_train, y_train = df1.loc[:, df1.columns != 'DG_TS'], df1[['DG_TS']]
    X_test, y_test = df2.loc[:, df2.columns != 'DG_TS'], df2[['DG_TS']]

    reg = LinearRegression()

    reg.fit(X_train, y_train.values.ravel())

    X_test['predicted'] = df2.apply(lambda x: reg.predict(np.array(x[df2.columns != 'DG_TS']).reshape(1, -1)), axis=1)

    mae = mean_absolute_error(X_test['predicted'], y_test['DG_TS'])
    rmse = np.sqrt(mean_squared_error(X_test['predicted'], y_test['DG_TS']))

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')


def linear_model_descs(pkl_file, logger, n_train=0.8): 
    df = pd.read_pickle(pkl_file) 
    X, y = df.loc[:, df.columns != 'DG_TS'], df[['DG_TS']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train)

    reg = LinearRegression()

    reg.fit(X_train, y_train.values.ravel())

    X_test['predicted'] = df.apply(lambda x: reg.predict(np.array(x[df.columns != 'DG_TS']).reshape(1, -1)), axis=1)

    mae = mean_absolute_error(X_test['predicted'], y_test['DG_TS'])
    rmse = np.sqrt(mean_squared_error(X_test['predicted'], y_test['DG_TS']))

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')
