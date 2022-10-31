import pandas as pd
from .cross_val import cross_val, cross_val_fp
from sklearn.linear_model import LinearRegression
from hyperopt import hp
from .bayesian_opt import bayesian_opt, objective_rf, objective_rf_fp
from .bayesian_opt import objective_xgboost, objective_xgboost_fp
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from .bayesian_opt import objective_knn_fp
from .k_nearest_neighbors import KNN


def get_rmse_linear_regression(df, logger, n_fold):
    model = LinearRegression()
    rmse, mae = cross_val(df, model, n_fold)

    logger.info(f'{n_fold}-fold CV RMSE and MAE for linear regression: {rmse} {mae}')


def get_rmse_xgboost_descriptors(df, logger, n_fold, max_eval=32):
    space = {
        'max_depth': hp.quniform('max_depth', low=1, high=6, q=1),
        'gamma': hp.loguniform('gamma', low=0.0, high=10),
        'reg_alpha': hp.uniform('reg_alpha', low=0.0, high=100),
        'reg_lambda': hp.uniform('reg_lambda', low=0.0, high=1.0),
        'n_estimators': hp.quniform('n_estimators', low=10, high=300, q=1)
    }
    optimal_parameters = bayesian_opt(df, space, objective_xgboost, XGBRegressor, max_eval=max_eval)
    model = XGBRegressor(max_depth=int(optimal_parameters['max_depth']), 
                        gamma=optimal_parameters['gamma'], 
                        reg_alpha=optimal_parameters['reg_alpha'],
                        reg_lambda=optimal_parameters['reg_lambda'], 
                        n_estimators=int(optimal_parameters['n_estimators']))
    rmse, mae = cross_val(df, model, n_fold)
    logger.info(f'{n_fold}-fold CV RMSE for xgboost: {rmse} {mae}')
    logger.info(f'Optimal parameters used: {optimal_parameters}')


def get_rmse_xgboost_fp(df_fp, logger, n_fold, max_eval=32):
    space = {
        'max_depth': hp.quniform('max_depth', low=1, high=6, q=1),
        'gamma': hp.loguniform('gamma', low=0.0, high=10),
        'reg_alpha': hp.uniform('reg_alpha', low=0.0, high=100),
        'reg_lambda': hp.uniform('reg_lambda', low=0.0, high=1.0),
        'n_estimators': hp.quniform('n_estimators', low=10, high=300, q=1)
    }
    optimal_parameters = bayesian_opt(df_fp, space, objective_xgboost_fp, XGBRegressor, max_eval=max_eval)
    model = XGBRegressor(max_depth=int(optimal_parameters['max_depth']), 
                        gamma=optimal_parameters['gamma'], 
                        reg_alpha=optimal_parameters['reg_alpha'],
                        reg_lambda=optimal_parameters['reg_lambda'], 
                        n_estimators=int(optimal_parameters['n_estimators']))
    rmse, mae = cross_val_fp(df_fp, model, n_fold)
    logger.info(f'{n_fold}-fold CV RMSE for xgboost: {rmse} {mae}')
    logger.info(f'Optimal parameters used: {optimal_parameters}')


def get_rmse_SVR_descriptors(df, logger, n_fold):
    kernel_dict = {1: 'linear', 2: 'rbf', 3: 'poly'}
    space = {
        'kernel': hp.quniform('kernel', low=1, high=3, q=1),
        'C': hp.loguniform('C', low=0, high=100, q=0.1),
        'gamma': hp.loguniform('gamma', low=1e-8, high=1e-2),
        'epsilon': hp.loguniform('epsilon', low=0, high=1)
    }
    optimal_parameters = bayesian_opt(df, space, objective_xgboost, SVR)
    model = SVR(kernel=kernel_dict[int(optimal_parameters['kernel'])], C=optimal_parameters['C'], 
                gamma=optimal_parameters['gamma'], epsilon=optimal_parameters['epsilon'])
    rmse, mae = cross_val(df, model, n_fold)
    logger.info(f'{n_fold}-fold CV RMSE and MAE for SVR: {rmse} {mae}')
    logger.info(f'Optimal parameters used: {optimal_parameters}')


def get_rmse_random_forest_fp(df_fp, logger, n_fold):
    space = {
        'n_estimators': hp.quniform('n_estimators', low=1, high=300, q=1),
        'max_features': hp.quniform('max_features', low=0.1, high=1, q=0.1)
    }
    optimal_parameters = bayesian_opt(df_fp, space, objective_rf_fp, RandomForestRegressor)
    model = RandomForestRegressor(n_estimators=int(optimal_parameters['n_estimators']), max_features=optimal_parameters['max_features'])
    rmse, mae = cross_val_fp(df_fp, model, n_fold)
    logger.info(f'{n_fold}-fold CV RMSE and MAE for random forest (fp): {rmse} {mae}')
    logger.info(f'Optimal parameters used: {optimal_parameters}')


def get_rmse_random_forest_descriptors(df, logger, n_fold):
    space = {
        'n_estimators': hp.quniform('n_estimators', low=1, high=300, q=1),
        'max_features': hp.quniform('max_features', low=0.1, high=1, q=0.1)
    }
    optimal_parameters = bayesian_opt(df, space, objective_rf, RandomForestRegressor)
    model = RandomForestRegressor(n_estimators=int(optimal_parameters['n_estimators']), max_features=optimal_parameters['max_features'])
    rmse, mae = cross_val(df, model, n_fold)
    logger.info(f'{n_fold}-fold CV RMSE and MAE for random forest (descriptors): {rmse} {mae}')
    logger.info(f'Optimal parameters used: {optimal_parameters}')


def get_rmse_knn_fp(df, logger, n_fold):
    space = {
        'n': hp.quniform('n', low=3, high=7, q=2),
        'lam': hp.quniform('lam', low=0, high=1, q=0.1),
        'mu': hp.quniform('mu', low=0, high=1, q=0.1)
    }
    optimal_parameters = bayesian_opt(df, space, objective_knn_fp, KNN, max_eval=2)
    model = KNN(n=int(optimal_parameters['n']), 
                dipole_dist_weight=optimal_parameters['lam'],
                dipolarophile_dist_weight=(1 - optimal_parameters['lam']) * optimal_parameters['mu'],
                product_dist_weight=(1 - optimal_parameters['lam']) * (1 - optimal_parameters['mu'])
            )
    rmse, mae = cross_val_fp(df, model, n_fold, knn=True)
    logger.info(f'{n_fold}-fold CV RMSE and MAE for k-nearest neighbors (fingerprints): {rmse} {mae}')
    logger.info(f'Optimal parameters used: {optimal_parameters}')
