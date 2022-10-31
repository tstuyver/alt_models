from types import SimpleNamespace
from rdkit.Chem import AllChem
from rdkit import Chem
from .cross_val import cross_val, cross_val_fp   
from hyperopt import fmin, tpe
from functools import partial
import numpy as np


def bayesian_opt(df, space, objective, model_class, n_train=0.8, max_eval=32):
    """
    Overarching function for Bayesian optimization

    Args:
        df (pd.DataFrame): dataframe containing the data points
        space (dict): dictionary containing the parameters for the selected regressor
        objective (function): specific objective function to be used
        model_class (Model): the abstract model class to initialize in every iteration
        n_train (float, optional): fraction of the training data to use. Defaults to 0.8.
        max_eval (int, optional): number of iterations to perform. Defaults to 32

    Returns:
        dict: optimal parameters for the selected regressor
    """
    df_sample = df.sample(frac=n_train)
    fmin_objective = partial(objective, data=df_sample, model_class=model_class)    
    best = fmin(fmin_objective, space, algo=tpe.suggest, max_evals=max_eval)

    return best


def get_difference_fingerprint(rxn_smiles, rad, nbits):
    """
    Get difference Morgan fingerprint between reactants and products

    Args:
        rxn_smiles (str): the full reaction SMILES
        rad (int): the radii of the fingerprints
        nbits (int): the number of bits for the fingerprints

    Returns:
        np.array: the difference fingerprint
    """
    reactants = rxn_smiles.split('>>')[0]
    products = rxn_smiles.split('>>')[-1]

    reactants_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactants), radius=rad, nBits=nbits))
    products_fp = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(products), radius=rad, nBits=nbits))

    return reactants_fp - products_fp


def get_fingerprint(rxn_smiles, rad, nbits):
    """
    Get Morgan fingerprint of the reactants

    Args:
        rxn_smiles (str): the full reaction SMILES
        rad (int): the radii of the fingerprints
        nbits (int): the number of bits for the fingerprints

    Returns:
        np.array: the reactant fingerprint
    """
    reactants = rxn_smiles.split('>>')[0]
    return AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(reactants), radius=rad, nBits=nbits)


def get_df_fingerprints_rp(df, rad, nbits):
    """
    Get the dataframe with all the fingerprints of reactants and products

    Args:
        df (pd.DataFrame): the dataframe containing the reaction SMILES
        rad (int): the radii of the fingerprints
        nbits (int): the number of bits for the fingerprints

    Returns:
        pd.DataFrame: the update dataframe
    """
    df['dipole'] = df['smiles'].apply(lambda x: [smi for smi in x.split('>')[0].split('.') if '+' in smi and '-' in smi][0])
    df['dipolarophile'] = df['smiles'].apply(lambda x: [smi for smi in x.split('>')[0].split('.') if not ('+' in smi and '-' in smi)][0])
    df['fingerprint_dipole'] = df['dipole'].apply(lambda x: 
            get_fingerprint(x, rad, nbits))
    df['fingerprint_dipolarophile'] = df['dipolarophile'].apply(lambda x: 
            get_fingerprint(x, rad, nbits))
    df['fingerprint_product'] = df['smiles'].apply(lambda x: get_fingerprint(x.split('>')[-1], rad, nbits))
    
    return df[['fingerprint_dipole', 'fingerprint_dipolarophile', 'fingerprint_product','DG_TS']]


def get_df_fingerprints(df, rad, nbits):
    """
    Get the dataframe with all the difference fingerprints

    Args:
        df (pd.DataFrame): the dataframe containing the reaction SMILES
        rad (int): the radii of the fingerprints
        nbits (int): the number of bits for the fingerprints

    Returns:
        pd.DataFrame: the update dataframe
    """
    df['fingerprint'] = df['smiles'].apply(lambda x: 
            get_difference_fingerprint(x, rad, nbits))

    print(df['fingerprint'].head())
    return df[['fingerprint','DG_TS']]


def objective_rf_fp(args_dict, data, model_class):
    """
    Objective function for random forest Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the RF regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(n_estimators=int(args.n_estimators), 
                                    max_features=args.max_features,
                                    random_state=2)
    cval,_ = cross_val_fp(data, estimator, 4)

    return cval.mean() 


def objective_rf(args_dict, data, model_class):
    """
    Objective function for random forest Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the RF regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(n_estimators=int(args.n_estimators), 
                                    max_features=args.max_features,
                                    random_state=2)
    cval,_ = cross_val(data, estimator, 4)


    return cval.mean() 


def objective_xgboost(args_dict, data, model_class):
    """
    Objective function for xgboost Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the xgboost regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(max_depth=int(args.max_depth), 
                                    gamma=args.gamma,
                                    reg_alpha=args.reg_alpha,
                                    reg_lambda=args.reg_lambda,
                                    n_estimators=int(args.n_estimators))
    cval,_ = cross_val(data, estimator, 4)

    return cval.mean()     


def objective_xgboost_fp(args_dict, data, model_class):
    """
    Objective function for xgboost Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the RF regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(max_depth=int(args.max_depth), 
                                    gamma=args.gamma,
                                    reg_alpha=args.reg_alpha,
                                    reg_lambda=args.reg_lambda,
                                    n_estimators=int(args.n_estimators))
    cval,_ = cross_val_fp(data, estimator, 4)

    return cval.mean()


def objective_knn_fp(args_dict, data, model_class):
    """
    Objective function for knn Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the RF regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(n=int(args.n),
                    dipole_dist_weight=args.lam,
                    dipolarophile_dist_weight=(1-args.lam)*args.mu,
                    product_dist_weight=(1-args.lam)*(1-args.mu))

    cval,_ = cross_val_fp(data, estimator, 4, knn=True)

    return cval.mean() 


def objective_SVR(args_dict, data, model_class):
    """
    Objective function for SVR Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the SVR regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    kernel_dict = {1: 'linear', 2: 'rbf', 3: 'poly'}
    estimator = model_class(kernel=kernel_dict[int(args.kernel)], C=args.C, gamma=args.gamma, epsilon=args.epsilon)
    cval,_ = cross_val(data, estimator, 4)

    return cval.mean()
