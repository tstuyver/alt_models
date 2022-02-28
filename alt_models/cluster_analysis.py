from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def cluster_analysis(pkl_file, rf_subset, logger, n_train=0.8):
    df = pd.read_pickle(pkl_file)

    X, y = df.loc[:, df.columns != 'DG_TS'], df[['DG_TS']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X).correlation

    # Ensure the correlation matrix is symmetric
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # We convert the correlation matrix to a distance matrix before performing
    # hierarchical clustering using Ward's linkage.
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=X.columns.tolist(), ax=ax1, leaf_rotation=90)
    dendro_idx = np.arange(0, len(dendro["ivl"]))

    ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
    ax2.set_yticklabels(dendro["ivl"])
    fig.tight_layout()
    plt.savefig('cluster_analysis.png')

    if rf_subset:
        cluster_ids = hierarchy.fcluster(dist_linkage, 0.5, criterion="distance")
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_features = [df.columns[i[0]] for i in cluster_id_to_feature_ids.values()]
        logger.info(list(selected_features))

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=n_train)

        reg = GridSearchCV(RandomForestRegressor(), cv=8, param_grid={
            "n_estimators": np.linspace(50, 250, 5).astype('int'), "max_features": np.linspace(0.2, 1.0, 4)},
                               scoring='neg_mean_absolute_error', n_jobs=-1)

        reg.fit(X_train, y_train.values.ravel())

        logger.info(f'Best parameters for full model: {reg.best_params_}')

        y_pred = reg.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        logger.info(f'MAE = {mae}')
        logger.info(f'RMSE = {rmse}')

        X_train_sel = X_train[list(selected_features)]
        X_test_sel = X_test[list(selected_features)]

        reg_sel = RandomForestRegressor()

        reg_sel.set_params(**reg.best_params_)
        reg_sel.fit(X_train_sel, y_train.values.ravel())

        X_test_sel['predicted'] = df.apply(lambda x: reg_sel.predict(np.array(x[list(selected_features)]).reshape(1, -1)),
                                           axis=1)

        mae = mean_absolute_error(X_test_sel['predicted'], y_test['DG_TS'])
        rmse = np.sqrt(mean_squared_error(X_test_sel['predicted'], y_test['DG_TS']))

        logger.info(f'Results for the pruned model; same settings as full model:')
        logger.info(f'MAE = {mae}')
        logger.info(f'RMSE = {rmse}')

        result = permutation_importance(reg_sel, X_train_sel, y_train, n_repeats=10, random_state=42)
        perm_sorted_idx = result.importances_mean.argsort()

        tree_importance_sorted_idx = np.argsort(reg_sel.feature_importances_)
        tree_indices = np.arange(0, len(reg_sel.feature_importances_)) + 0.5

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        ax1.barh(tree_indices, reg_sel.feature_importances_[tree_importance_sorted_idx], height=0.7)
        ax1.set_yticks(tree_indices)
        ax1.set_yticklabels(X_train_sel.columns[tree_importance_sorted_idx])
        ax1.set_ylim((0, len(reg_sel.feature_importances_)))
        ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                    labels=X_train_sel.columns[perm_sorted_idx],)
        fig.tight_layout()
        plt.savefig('permutation_importance_pruned_features.png')
