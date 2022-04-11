from collections import defaultdict
from tkinter import W
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, DMatrix, train, plot_importance
from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from rdkit.Chem import AllChem
from functools import partial
from types import SimpleNamespace

from hyperopt import fmin, hp, tpe, Trials


def xgboost_fp_selective(csv_file1, csv_file2, logger, rad=2, nbits=1024):
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

    parameters = {'max_depth': np.linspace(3, 10, 4).astype(int), 'gamma': np.linspace(1,9, 2),
                'reg_alpha': np.linspace(40, 180, 3), 'reg_lambda': np.linspace(0, 1, 2), 
                'n_estimators': np.linspace(10, 250, 3).astype(int)}
    xgbr = XGBRegressor()
    reg = GridSearchCV(xgbr, parameters)

    reg.fit(fps1, df1['DG_TS'])

    logger.info(f'Best parameters: {reg.best_params_}')

    y_pred = reg.predict(fps2)

    mae = mean_absolute_error(df2['DG_TS'], y_pred)
    rmse = np.sqrt(mean_squared_error(df2['DG_TS'], y_pred)) 

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')


def xgboost_fp(csv_file1, logger, n_train=0.8, rad=2, nbits=1024):
    df = pd.read_csv(csv_file1)
    df['fingerprints_reactants'] = df['smiles'].apply(lambda x: 
            AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x.split('>')[0]), radius=rad, nBits=nbits))
    fps = []

    for fp in df['fingerprints_reactants'].values.tolist():
        fps.append(fp)
    X_train, X_test, y_train, y_test = train_test_split(fps, df['DG_TS'], train_size=n_train)

    parameters = {'max_depth': np.linspace(3, 10, 4).astype(int), 'gamma': np.linspace(1,9, 2),
                'reg_alpha': np.linspace(40, 180, 3), 'reg_lambda': np.linspace(0, 1, 2), 
                'n_estimators': np.linspace(10, 250, 3).astype(int)}
    xgbr = XGBRegressor()
    reg = GridSearchCV(xgbr, parameters)

    reg.fit(X_train, y_train)

    logger.info(f'Best parameters: {reg.best_params_}')

    y_pred = reg.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) 

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')


def xgboost_descs_selective(pkl_file1, pkl_file2, logger):
    df1 = pd.read_pickle(pkl_file1)
    df2 = pd.read_pickle(pkl_file2)

    X_train, y_train = df1.loc[:, df1.columns != 'DG_TS'], df1[['DG_TS']]
    X_test, y_test = df2.loc[:, df2.columns != 'DG_TS'], df2[['DG_TS']]

    parameters = {'max_depth': np.linspace(3, 10, 4).astype(int), 'gamma': np.linspace(1,9, 2),
                'reg_alpha': np.linspace(40, 180, 3), 'reg_lambda': np.linspace(0, 1, 2), 
                'n_estimators': np.linspace(10, 250, 3).astype(int)}
    xgbr = XGBRegressor()
    reg = GridSearchCV(xgbr, parameters)

    reg.fit(X_train, y_train.values.ravel())
    logger.info(f'Best parameters: {reg.best_params_}')

    X_test['predicted'] = df2.apply(lambda x: reg.predict(np.array(x[df2.columns != 'DG_TS']).reshape(1,-1)),axis=1)

    mae = mean_absolute_error(X_test['predicted'], y_test['DG_TS'])
    rmse = np.sqrt(mean_squared_error(X_test['predicted'], y_test['DG_TS']))

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')


def xgboost_descs(pkl_file, logger, n_train=0.8): 
    df = pd.read_pickle(pkl_file) 
    X, y = df.loc[:, df.columns != 'DG_TS'], df[['DG_TS']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train)

    parameters = {'max_depth': np.linspace(1, 5, 5).astype(int), 'gamma': np.linspace(0.1, 6, 4),
                'reg_alpha': np.linspace(10, 100, 4), 'reg_lambda': np.linspace(0, 0.1, 5), 
                'n_estimators': np.linspace(20, 140, 4).astype(int)}
    xgbr = XGBRegressor()
    reg = GridSearchCV(xgbr, parameters)
    reg.fit(X_train, y_train.values.ravel())

    X_test['predicted'] = df.apply(lambda x: reg.predict(np.array(x[df.columns != 'DG_TS']).reshape(1,-1)),axis=1)

    mae = mean_absolute_error(X_test['predicted'], y_test['DG_TS'])
    rmse = np.sqrt(mean_squared_error(X_test['predicted'], y_test['DG_TS']))

    print(reg.best_params_, mae, rmse)
    logger.info(f'Best parameters: {reg.best_params_}')


def objective(args0, data, targets):
    args = SimpleNamespace(**args0)

    estimator = XGBRegressor(max_depth=int(args.max_depth), 
                                    gamma=args.gamma,
                                    reg_alpha=args.reg_alpha,
                                    reg_lambda=args.reg_lambda,
                                    n_estimators=int(args.n_estimators))

    cval = cross_val_score(estimator, data, targets, scoring='neg_root_mean_squared_error', cv=4)

    return (cval.mean() * (-1))


def xgboost_descs_bayesian(pkl_file, logger, n_train=0.8):
    df = pd.read_pickle(pkl_file) 
    X, y = df.loc[:, df.columns != 'DG_TS'], df[['DG_TS']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train)

    space = {
        'max_depth': hp.quniform('max_depth', low=1, high=6, q=1),
        'gamma': hp.loguniform('gamma', low=0.0, high=10),
        'reg_alpha': hp.uniform('reg_alpha', low=0.0, high=100),
        'reg_lambda': hp.uniform('reg_lambda', low=0.0, high=1.0),
        'n_estimators': hp.quniform('n_estimators', low=10, high=300, q=1)
    }

    fmin_objective = partial(objective, data=X_train, targets=y_train.values.ravel())    

    best = fmin(fmin_objective, space, algo=tpe.suggest, max_evals=32)

    logger.info(f"optimal parameters: {best}")

    model = XGBRegressor(max_depth=int(best['max_depth']), gamma=best['gamma'], reg_alpha=best['reg_alpha'],
                        reg_lambda=best['reg_lambda'], n_estimators=int(best['n_estimators']))

    model.fit(X_train, y_train.values.ravel())

    X_test['predicted'] = df.apply(lambda x: model.predict(np.array(x[df.columns != 'DG_TS']).reshape(1,-1)),axis=1)

    mae = mean_absolute_error(X_test['predicted'], y_test['DG_TS'])
    rmse = np.sqrt(mean_squared_error(X_test['predicted'], y_test['DG_TS']))

    logger.info(f'MAE = {mae}')
    logger.info(f'RMSE = {rmse}')

    plt.figure(figsize = (16, 12))
    plot_importance(model) 
    plt.savefig('test.png')   