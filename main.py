import pandas as pd
from sklearn.linear_model import LinearRegression
from hyperopt import hp
from lib import get_rmse_linear_regression, get_rmse_xgboost_descriptors, get_rmse_xgboost_fp
from lib import get_rmse_random_forest_descriptors, get_rmse_random_forest_fp, get_rmse_knn_fp
from lib import create_logger
from lib import get_df_fingerprints, get_df_fingerprints_rp


if __name__ == '__main__':
    # set up
    logger = create_logger('final')
    df = pd.read_pickle('data_files/input_alt_models.pkl')
    df_rxn_smiles = pd.read_csv('data_files/full_data.csv')
    #df_fp = get_df_fingerprints(df_rxn_smiles,2,2048)
    n_fold = 10

    # linear regression
    get_rmse_linear_regression(df, logger, n_fold)

    # KNN - fingerprints
    df_fp = get_df_fingerprints_rp(df_rxn_smiles,2,2048)
    get_rmse_knn_fp(df_fp, logger, n_fold)

    # random_forest - fingerprints
    #get_rmse_random_forest_fp(df_fp, logger, n_fold)

    # random forest - descriptors
    #get_rmse_random_forest_descriptors(df, logger, n_fold)

    # xgboost - descriptors
    #get_rmse_xgboost_descriptors(df, logger, n_fold, max_eval=128)

    # xgboost - fingerprints
    #get_rmse_xgboost_fp(df_fp, logger, n_fold)