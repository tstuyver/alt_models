from argparse import ArgumentParser, Namespace
from alt_models.cluster_analysis import cluster_analysis
from utils import create_logger
from alt_models import random_forest_fp_selective, random_forest_fp, random_forest_descs_selective, random_forest_descs
from alt_models import SVR_fp_selective, SVR_fp, SVR_descs_selective, SVR_descs 
from alt_models import linear_model_descs_selective, linear_model_descs
from alt_models import xgboost_descs
from alt_models import random_forest_descs_bayesian, svr_descs_bayesian, xgboost_descs_bayesian
from alt_models import random_forest_fp_bayesian

parser = ArgumentParser()
parser.add_argument('--selective', dest='selective', action='store_true',
                    help='whether selective sampling has been done')
parser.add_argument('--csv-file', type=str, required=False,
                    help='path to file containing the rxn-smiles')
parser.add_argument('--csv-file-train', type=str, required=False,
                    help='path to file containing training/validation rxn-smiles')
parser.add_argument('--csv-file-test', type=str, required=False,
                    help='path to file containing testing rxn-smiles')
parser.add_argument('--pkl-file', type=str, required=False,
                    help='path to pkl containing ordered descriptor values')
parser.add_argument('--pkl-file-train', type=str, required=False,
                    help='path to file containing training/validation descriptor values')
parser.add_argument('--pkl-file-test', type=str, required=False,
                    help='path to file containing testing descriptor values')
parser.add_argument('--n-train', default=None, required=False,
                    help='number (or fraction) of training points')
parser.add_argument('--clusterin-analysis', dest='clustering_analysis', action='store_true',
                    help='whether a clustering analysis needs to be performed')


if __name__ == '__main__':
    args = parser.parse_args()

    if args.clustering_analysis:
        logger = create_logger(args.csv_file)
        cluster_analysis(args.csv_file, True, logger, args.n_train)
    else:
        if args.selective:
            logger = create_logger(args.csv_file_train)
            logger.info('Random Forest based on fingerprints (2,1024) -- results:')
            random_forest_fp_selective(args.csv_file_train, args.csv_file_test, logger)
            logger.info('Random Forest based on fingerprints (3,2048) -- results:')
            random_forest_fp_selective(args.csv_file_train, args.csv_file_test, logger, 3, 2048)
            logger.info('Random Forest based on descriptors -- results:')
            random_forest_descs_selective(args.pkl_file_train, args.pkl_file_test, logger)
            logger.info('Linear model based on descriptors -- results:')
            linear_model_descs_selective(args.pkl_file_train, args.pkl_file_test, logger)
            logger.info('SVR based on fingerprints -- results:')
            SVR_fp_selective(args.csv_file_train, args.csv_file_test, logger)
            #logger.info('SVR based on descriptors -- results:')
            #SVR_descs_selective(args.pkl_file_train, args.pkl_file_test, logger)

        else:
            logger = create_logger(args.csv_file)
            if args.n_train:
                if '.' in args.n_train:
                    n_train = float(args.n_train)
                else:
                    n_train = int(args.n_train)
            else:
                n_train=0.8
            logger.info('Linear model based on descriptors -- results:')
            linear_model_descs(args.pkl_file, logger, n_train)
            logger.info('Random Forest based on fingerprints (2,1024) -- results:')
            random_forest_fp_bayesian(args.csv_file, logger, n_train) 
            logger.info('Random Forest based on fingerprints (3,2048) -- results:')
            random_forest_fp_bayesian(args.csv_file, logger, n_train, 3, 2048)
            logger.info('Random Forest based on descriptors -- results:')
            random_forest_descs_bayesian(args.pkl_file, logger, n_train)
            logger.info('xgboost based on descriptors -- results:') 
            xgboost_descs_bayesian(args.pkl_file, logger, n_train)
