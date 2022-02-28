from collections import defaultdict
from tkinter import W
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from rdkit.Chem import AllChem


def random_forest_fp_selective(csv_file1, csv_file2, logger):
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    df1['fingerprints_reactants'] = df1['smiles'].apply(lambda x: 
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x.split('>')[0]), radius=2, nBits=1024))
    df2['fingerprints_reactants'] = df2['smiles'].apply(lambda x: 
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x.split('>')[0]), radius=2, nBits=1024))
    fps1, fps2 = [], []

    for fp in df1['fingerprints_reactants'].values.tolist():
        fps1.append(fp)

    for fp in df2['fingerprints_reactants'].values.tolist():
        fps2.append(fp)

    reg = GridSearchCV(RandomForestRegressor(), cv=8, param_grid={
        "n_estimators": np.linspace(50, 250, 5).astype('int'), "max_features": np.linspace(0.2, 1.0, 4)},
        scoring='neg_mean_absolute_error', n_jobs=-1)

    reg.fit(fps1, df1['DG_TS'])

    logger.info(f'Best parameters: {reg.best_params_}')

    y_pred = reg.predict(fps2)

    mae = mean_absolute_error(df2['DG_TS'], y_pred)
    rmse = np.sqrt(mean_squared_error(df2['DG_TS'], y_pred)) 

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')


def random_forest_fp(csv_file1, logger, n_train=0.8):
    df = pd.read_csv(csv_file1)
    df['fingerprints_reactants'] = df['smiles'].apply(lambda x: 
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x.split('>')[0]), radius=2, nBits=1024))
    fps = []

    for fp in df['fingerprints_reactants'].values.tolist():
        fps.append(fp)
    X_train, X_test, y_train, y_test = train_test_split(fps, df['DG_TS'], train_size=n_train)

    reg = GridSearchCV(RandomForestRegressor(), cv=8, param_grid={
        "n_estimators": np.linspace(50, 250, 5).astype('int'), "max_features": np.linspace(0.2, 1.0, 4)},
        scoring='neg_mean_absolute_error', n_jobs=-1)

    reg.fit(X_train, y_train)

    logger.info(f'Best parameters: {reg.best_params_}')

    y_pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')


def random_forest_descs_selective(pkl_file1, pkl_file2, logger):
    df1 = pd.read_pickle(pkl_file1)
    df2 = pd.read_pickle(pkl_file2)

    X_train, y_train = df1.loc[:, df1.columns != 'DG_TS'], df1[['DG_TS']]
    X_test, y_test = df2.loc[:, df2.columns != 'DG_TS'], df2[['DG_TS']]

    reg = GridSearchCV(RandomForestRegressor(), cv=8, param_grid={
        'n_estimators': np.linspace(50, 250, 5).astype('int'), 'max_features': np.linspace(0.2, 1.0, 4)},
        scoring='neg_mean_absolute_error', n_jobs=-1)

    reg.fit(X_train, y_train.values.ravel())
    logger.info(f'Best parameters: {reg.best_params_}')

    X_test['predicted'] = df2.apply(lambda x: reg.predict(np.array(x[df2.columns != 'DG_TS']).reshape(1,-1)),axis=1)

    mae = mean_absolute_error(X_test['predicted'], y_test['DG_TS'])
    rmse = np.sqrt(mean_squared_error(X_test['predicted'], y_test['DG_TS']))

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')


def random_forest_descs(pkl_file, logger, n_train=0.8): 
    df = pd.read_pickle(pkl_file) 
    X, y = df.loc[:, df.columns != 'DG_TS'], df[['DG_TS']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train)

    reg = GridSearchCV(RandomForestRegressor(), cv=8, param_grid={
        'n_estimators': np.linspace(50, 250, 5).astype('int'), 'max_features': np.linspace(0.2, 1.0, 4)},
        scoring='neg_mean_absolute_error', n_jobs=-1)

    reg.fit(X_train, y_train.values.ravel())
    logger.info(f'Best parameters: {reg.best_params_}')

    X_test['predicted'] = df.apply(lambda x: reg.predict(np.array(x[df.columns != 'DG_TS']).reshape(1,-1)),axis=1)

    mae = mean_absolute_error(X_test['predicted'], y_test['DG_TS'])
    rmse = np.sqrt(mean_squared_error(X_test['predicted'], y_test['DG_TS']))

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')
