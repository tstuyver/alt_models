from collections import defaultdict
from tkinter import W
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVR
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from rdkit.Chem import AllChem
from functools import partial
from types import SimpleNamespace

from hyperopt import fmin, hp, tpe, Trials

def SVR_fp_selective(csv_file1, csv_file2, logger, rad=2, nbits=1024):
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)
    df1['fingerprints_reactants'] = df1['smiles'].apply(lambda x: 
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x.split('>')[0]), radius=rad, nBits=nbits))
    df2['fingerprints_reactants'] = df2['smiles'].apply(lambda x: 
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x.split('>')[0]), radius=rad, nBits=nbits))
    fps1, fps2 = [], []

    for fp in df1['fingerprints_reactants'].values.tolist():
        fps1.append(fp)

    for fp in df2['fingerprints_reactants'].values.tolist():
        fps2.append(fp)

    parameters = {'kernel': ('linear', 'rbf','poly'), 'C':np.linspace(1.0, 10.0, 4),'gamma': [1e-7, 1e-4],'epsilon':np.linspace(0.1, 0.7, 4)}
    svr = SVR()
    reg = GridSearchCV(svr, parameters)

    reg.fit(fps1, df1['DG_TS'])

    logger.info(f'Best parameters: {reg.best_params_}')

    y_pred = reg.predict(fps2)

    mae = mean_absolute_error(df2['DG_TS'], y_pred)
    rmse = np.sqrt(mean_squared_error(df2['DG_TS'], y_pred)) 

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')


def SVR_fp(csv_file1, logger, n_train=0.8, rad=2, nbits=1024):
    df = pd.read_csv(csv_file1)
    df['fingerprints_reactants'] = df['smiles'].apply(lambda x: 
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x.split('>')[0]), radius=rad, nBits=nbits))
    fps = []

    for fp in df['fingerprints_reactants'].values.tolist():
        fps.append(fp)
    X_train, X_test, y_train, y_test = train_test_split(fps, df['DG_TS'], train_size=n_train)

    parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C':np.linspace(1.0, 10.0, 4),'gamma': [1e-7, 1e-4],'epsilon':np.linspace(0.1, 0.7, 4)}
    svr = SVR()
    reg = GridSearchCV(svr, parameters)

    reg.fit(X_train, y_train)

    logger.info(f'Best parameters: {reg.best_params_}')

    y_pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')


def SVR_descs_selective(pkl_file1, pkl_file2, logger):
    df1 = pd.read_pickle(pkl_file1)
    df2 = pd.read_pickle(pkl_file2)

    X_train, y_train = df1.loc[:, df1.columns != 'DG_TS'], df1[['DG_TS']]
    X_test, y_test = df2.loc[:, df2.columns != 'DG_TS'], df2[['DG_TS']]

    parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C':np.linspace(1.0, 10.0, 4),'gamma': [1e-7, 1e-4],'epsilon':np.linspace(0.1, 0.7, 4)}
    svr = SVR()
    reg = GridSearchCV(svr, parameters)

    reg.fit(X_train, y_train.values.ravel())
    logger.info(f'Best parameters: {reg.best_params_}')

    X_test['predicted'] = df2.apply(lambda x: reg.predict(np.array(x[df2.columns != 'DG_TS']).reshape(1,-1)),axis=1)

    mae = mean_absolute_error(X_test['predicted'], y_test['DG_TS'])
    rmse = np.sqrt(mean_squared_error(X_test['predicted'], y_test['DG_TS']))

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')


def SVR_descs(pkl_file, logger, n_train=0.8): 
    df = pd.read_pickle(pkl_file) 
    X, y = df.loc[:, df.columns != 'DG_TS'], df[['DG_TS']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train)

    parameters = {'kernel': ('linear', 'rbf','poly'), 'C':np.linspace(1.0, 10.0, 4),'gamma': [1e-7, 1e-4],'epsilon':np.linspace(0.1, 0.7, 4)}
    svr = SVR()
    reg = GridSearchCV(svr, parameters)

    reg.fit(X_train, y_train.values.ravel())
    logger.info(f'Best parameters: {reg.best_params_}')

    X_test['predicted'] = df.apply(lambda x: reg.predict(np.array(x[df.columns != 'DG_TS']).reshape(1,-1)),axis=1)

    mae = mean_absolute_error(X_test['predicted'], y_test['DG_TS'])
    rmse = np.sqrt(mean_squared_error(X_test['predicted'], y_test['DG_TS']))

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')


def objective(args0, data, targets):
    args = SimpleNamespace(**args0)
    kernel_dict = {1: 'linear', 2: 'rbf', 3: 'poly'}

    estimator = SVR(kernel=kernel_dict[int(args.kernel)], C=args.C, gamma=args.gamma, epsilon=args.epsilon)

    cval = cross_val_score(estimator, data, targets, scoring='neg_root_mean_squared_error', cv=4)

    return (cval.mean() * (-1))


def svr_descs_bayesian(pkl_file, logger, n_train=0.8):
    df = pd.read_pickle(pkl_file) 
    X, y = df.loc[:, df.columns != 'DG_TS'], df[['DG_TS']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train)

    space = {
        'kernel': hp.quniform('kernel', low=1, high=3, q=1),
        'C': hp.loguniform('C', low=0, high=100, q=0.1),
        'gamma': hp.loguniform('gamma', low=1e-8, high=1e-2),
        'epsilon': hp.loguniform('epsilon', low=0, high=1)
    }

    fmin_objective = partial(objective, data=X_train, targets=y_train.values.ravel())    

    best = fmin(fmin_objective, space, algo=tpe.suggest, max_evals=8)

    logger.info(f"optimal parameters: {best}")

    model = SVR(n_estimators=int(best['n_estimators']), max_features=best['max_features'])

    model.fit(X_train, y_train.values.ravel())

    X_test['predicted'] = df.apply(lambda x: model.predict(np.array(x[df.columns != 'DG_TS']).reshape(1,-1)),axis=1)

    mae = mean_absolute_error(X_test['predicted'], y_test['DG_TS'])
    rmse = np.sqrt(mean_squared_error(X_test['predicted'], y_test['DG_TS']))

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')
