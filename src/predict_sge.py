'''
   Predict Spatial Gene Expression from H&E

'''

import argparse
import datetime
import glob
import os
import numpy as np
import pandas as pd
import time
import yaml

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


from visualize import visualise_feature_vector_heatmap, viz_side_by_side, visualise_feature_matrix_heatmap


from utils import set_seed, save_features_with_names, load_features_from_csv
from feature_extractor import huggingfaceLogin, getFeatureMap, transformImage, extractFeatures
from feature_extractor import extractFeatureVectorUsingUNI, extractFeatureVectorUsingGigaPath


from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, GroupKFold, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import f1_score, roc_curve, roc_auc_score,recall_score,precision_score,accuracy_score, log_loss

from scipy.stats import loguniform, uniform, randint, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests

from huggingface_hub import login


SCORE_STAT = 'neg_mean_squared_error'

MODEL_MAP = {
    "elasticnet": ElasticNet,
    "randomforest": RandomForestRegressor,
    "gradboost": HistGradientBoostingRegressor,
    "svm": SVR
}

def variance_ratio(y_true, y_pred, ddof=0):
    """
    Compute the ratio of the variance of predictions to that of true values.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Model predictions.
    ddof : int, default=1
        Delta degrees of freedom for variance (as in np.var).

    Returns
    -------
    float
        Variance ratio = Var(y_pred) / Var(y_true)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    var_true = np.var(y_true, ddof=ddof)
    var_pred = np.var(y_pred, ddof=ddof)

    return var_pred / var_true

def concordance_correlation_coeff(y_true, y_pred):
    """Concordance Correlation Coefficient (CCC) between true and predicted values."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true, ddof=0)
    var_pred = np.var(y_pred, ddof=0)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    #cov = np.cov(y_true, y_pred, ddof=1)[0, 1] # sample rather than population

    ccc = (2 * cov) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    return ccc


#### CONFIG READER FUNCTIONS ###########


def init_model(ml_config_entry):
    seed=77
    name = ml_config_entry["name"]
    model_type = name.lower()
    params = ml_config_entry["params"]

    if model_type not in MODEL_MAP:
        raise ValueError(f"Unknown model type '{model_type}' in config.yaml")

    model = MODEL_MAP[model_type](random_state=seed) if "random_state" in MODEL_MAP[model_type]().__init__.__code__.co_varnames else MODEL_MAP[model_type]()
    
    return (name, model, params)



def read_model_configs(config):
    model_configs = None

    if 'ml_models' in config:

        if isinstance(config['ml_models'], dict):
            print("It's a dictionary!")
            # make it the first element of a list
            model_configs = [init_model(config['ml_models'])]

        elif isinstance(config['ml_models'], list):
            print("It's a list!")

            model_configs = []
            for entry in config["ml_models"]:
                model_configs.append(init_model(entry))
        else:
            raise ValueError(f"models incorrectly specified in the config")
    print(model_configs) 
    return model_configs


######################################

def plot_roc_curve(fpr,tpr,auc,title='ROC Curve'):
    plt.figure(figsize=(5, 5))
    # Plot tpr against fpr
    plt.plot(fpr, tpr,label="auc="+str(auc))
    #control line
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.show()

def get_test_scores(y_test, y_pred,y_pred_probs,train_score=0.0,label=""):
    f1 = f1_score(y_test, y_pred , average="macro")
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_probs)

    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    auc = roc_auc_score(y_test,y_pred_probs)
    #plot_roc_curve(fpr,tpr,auc, title=f"ROC Curve for {label}")

    data = [(recall, precision, f1, accuracy, auc,train_score,loss)]
    df = pd.DataFrame(data, columns=['Recall', 'Precision', 'F1 Score', 'Accuracy','AUC', 'Train F1','Loss'])
    df.insert(0, 'Run', label)

    return df

def evaluate_model(model, X_test, y_test, label=''):
    y_pred = model.predict(X_test)
    y_pred_probs = model.predict_proba(X_test)[:,1]
    scores = get_test_scores(y_test, y_pred, y_pred_probs,train_score=model.best_score_,label=label )


    return scores


def get_regression_scores_single(y_test, y_pred, train_score=0.0, label=""):
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    vr = variance_ratio(y_test, y_pred)
    ccc = concordance_correlation_coeff(y_test, y_pred)

    # Plot actual vs predicted
    #plt.figure(figsize=(6, 6))
    #sns.scatterplot(x=y_test, y=y_pred)
    #plt.xlabel("True Expression")
    #plt.ylabel("Predicted Expression")
    #plt.title(f"Actual vs Predicted for {label}")
    #plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    #plt.grid(True)
    #plt.show()

    data = [(mse, rmse, mae, r2, pearson_corr, spearman_corr, train_score,ccc,vr)]
    df = pd.DataFrame(data, columns=[
        'MSE', 'RMSE', 'MAE', 'R2',
        'PearsonCorr', 'SpearmanCorr',
        'Train_CV_Score',
        'ConcordanceCorr','VarianceRatio',
    ])
    df.insert(0, 'Run', label)
    return df

def evaluate_regression_model_single(model, X_test, y_test, label=''):
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten() if hasattr(y_pred, 'shape') and len(y_pred.shape) > 1 else y_pred
    scores = get_regression_scores_single(y_test, y_pred, train_score=model.best_score_, label=label)
    return scores

"""updated for mean of multioutput"""

def get_regression_scores(y_test, y_pred, train_score=0.0, label="",output_path=''):
    gene_names = list(y_test.columns)
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)

    #for i, gene in enumerate(gene_names):
    #    print(f"SCORING {gene}",flush=True)
    #    #print(f"y_test: {y_test[:,i]}")
    #    print(f"y_pred: {y_pred[:,i]}",flush=True)

    # Compute per-gene (column) metrics
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')

    # Correlations per gene
    pearson_results = [pearsonr(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    spearman_results = [spearmanr(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    vr = [variance_ratio(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
    ccc = [concordance_correlation_coeff(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]

    pearson_corrs, pearson_pvals = zip(*pearson_results)
    spearman_corrs, spearman_pvals = zip(*spearman_results)

    # FDR correction
    _, pearson_fdr, _, _ = multipletests(pearson_pvals, method='fdr_bh')
    _, spearman_fdr, _, _ = multipletests(spearman_pvals, method='fdr_bh')

    # Per-gene DataFrame
    n_genes = y_test.shape[1]
    per_gene_df = pd.DataFrame({
        'Gene': gene_names, #[f'Gene_{i}' for i in range(n_genes)],
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'PearsonCorr': pearson_corrs,
        'SpearmanCorr': spearman_corrs,
        'PearsonPVals': pearson_pvals,
        'SpearmanPVals': spearman_pvals,
        'PearsonFDR': pearson_fdr,
        'SpearmanFDR': spearman_fdr,
        'ConcordanceCorr': ccc,
        'VarianceRatio':vr,
    })
    per_gene_df.insert(0, 'Run', label)

    # Summary row
    # Aggregate (mean)
    mean_metrics = {
        'MSE': mse.mean(),
        'RMSE': rmse.mean(),
        'MAE': mae.mean(),
        'R2': r2.mean(),
        'PearsonCorr': np.nanmean(pearson_corrs),
        'SpearmanCorr': np.nanmean(spearman_corrs),
        'PearsonPVals': np.median(pearson_pvals),
        'SpearmanPVals': np.median(spearman_pvals),
        'PearsonFDR': np.median(pearson_fdr),
        'SpearmanFDR': np.median(spearman_fdr),
        'Train_CV_Score': train_score,
        'ConcordanceCorr': np.nanmean(ccc),
        'VarianceRatio': np.nanmean(vr),
    }

    summary_df = pd.DataFrame([mean_metrics])
    summary_df.insert(0, 'Run', label)


    curr_date=str(datetime.date.today())
    curr_time=datetime.datetime.now().strftime('%H%M')
    filename=f'summaryplot-gene0-{label}-{curr_date}_{curr_time}.png'

    # Optional: Plot one gene as example (e.g., gene 0)
    #plt.figure(figsize=(6, 6))
    #sns.scatterplot(x=y_test[:, 0], y=y_pred[:, 0])
    #plt.xlabel("True Expression (Gene 0)")
    #plt.ylabel("Predicted Expression")
    #plt.title(f"Actual vs Predicted for {label} (Gene 0)")
    #plt.plot([y_test[:, 0].min(), y_test[:, 0].max()],
    #         [y_test[:, 0].min(), y_test[:, 0].max()], 'r--')
    #plt.grid(True)
    #plotname = f"scatter_{label}_{gene_names[0]}.png".replace(" ", "_")
    #print(f"saving plot to {output_path}/{filename}")
    #print("os.path.exists:", os.path.exists(os.path.join(output_path, filename)))
    #print("Can write to output_path:", os.access(output_path, os.W_OK))

    #try:
    #    plt.savefig(os.path.join(output_path,filename), dpi=300, bbox_inches='tight')
    #    print(f"Saved plot to: {output_path}")
    #except Exception as e:
    #    print(f"❌ Failed to save plot ")
    #plt.show()

    return summary_df, per_gene_df



def evaluate_regression_model(models, X_test, y_test, label='', output_path='',segment='CD45+',save=True):
    # Predict each gene separately
    y_pred = np.column_stack([model.predict(X_test) for model in models])

    # If you want to average the best CV score across models
    train_scores = [getattr(model, 'best_score_', np.nan) for model in models]
    #print(f"train scores: {train_scores}")
    #for model in models:
    #    print(model.best_score_)

    mean_train_score = np.nanmean(train_scores)

    # Compute metrics
    summary_scores, per_gene_scores = get_regression_scores(
        y_test,
        y_pred,
        train_score=mean_train_score,
        label=label,
        output_path=output_path
    )
    # Collect and add best_params_ as string to the per_gene_scores DataFrame
    best_params = [str(getattr(model, 'best_params_', {})) for model in models]
    per_gene_scores['best_params'] = best_params

    if save:
        # Save per-gene metrics
        #print(f"saving to: {output_path}")
        #per_gene_scores.to_csv(os.path.join(output_path,f"per_gene_scores-{label}-{segment}.csv"), index=False)
        #summary_scores.to_csv(os.path.join(output_path,f"summary_scores-{label}-{segment}.csv"), index=False)
        try:
            per_gene_scores.to_csv(os.path.join(output_path,f"per_gene_scores-{label}-{segment}.csv"), index=False)
            summary_scores.to_csv(os.path.join(output_path,f"summary_scores-{label}-{segment}.csv"), index=False)
            print(f"saved scores to {output_path}")
        except Exception as e:
            print(f"failed to save to {output_path}")
            tmp_path=os.path.join("/home/hrafique/sge_tmp",os.path.basename(output_path))
            os.makedirs(tmp_path, exist_ok=True)
            print(f"saving to {tmp_path} instead")
            per_gene_scores.to_csv(os.path.join(tmp_path,f"per_gene_scores-{label}-{segment}.csv"), index=False)
            summary_scores.to_csv(os.path.join(tmp_path,f"summary_scores-{label}-{segment}.csv"), index=False)

    return summary_scores, per_gene_scores


def train_multioutput_regressor(base_model, params, label, X_train, y_train, X_test, y_test, gkf, groups,output_path='',segment='CD45+'):
    tuned_models = []
    print(f"saving to {output_path}")
    print(f"{label.upper()}: Start training")
    print(f"segment: {segment}",flush=True)
    for i in range(y_train.shape[1]):
        print(f"Fitting gene {i}")
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=params,
            cv=gkf,
            scoring=SCORE_STAT,
            n_jobs=-1,
            verbose=1
        )
        grid.fit(X_train, y_train.iloc[:, i], groups=groups)
        #tuned_models.append(grid.best_estimator_)
        tuned_models.append(grid)
        #print(grid.best_params_)
        #print(grid.best_score_)
        #print("Best estimator:", grid.best_estimator_)
    #print(tuned_models)

    #Evaluate on devel dataset
    summary_scores, per_gene_scores = evaluate_regression_model(models=tuned_models, 
                                                                X_test=X_test, y_test=y_test, 
                                                                label=label, output_path=output_path,segment=segment)
    print(f"{label.upper()}: Finished")
    return grid, summary_scores


#######################################################################################################

def train(feature_matrix, 
          merged_df, 
          outcome_cols, 
          output_path='',
          segment='CD45',
          config=None):
    
    #### READ CONFIG #####
    model_configs = None
    if config:
        print("reading config")
        model_configs = read_model_configs(config)
 



    #######################

    """
    Need to keep LEAPIDs together in train/dev split as well as in folds to avoid data leakage
    """

    gss = GroupShuffleSplit(
        n_splits=1,          # just one train/test split
        test_size=0.2,       # 20 % of the groups → test set
        random_state=77)     # reproducible

    train_idx, test_idx = next( gss.split(feature_matrix_clean,              # X
                  y=None,                 # y (optional)
                  groups=merged_df['LEAP_ID']))  # the grouping key

    """Check LEAP_IDs are only in either train or test - should pass the assertion with no output"""
    train_LEAPIDs = set(merged_df['LEAP_ID'].iloc[train_idx].unique())
    test_LEAPIDs = set(merged_df['LEAP_ID'].iloc[test_idx].unique())
    assert train_LEAPIDs.isdisjoint(test_LEAPIDs), (
        f"Data-leakage: these LEAP_IDs are in both splits → "
        f"{train_LEAPIDs & test_LEAPIDs}"
    )

    print("\n********** TRAIN LEAP IDs *****************")
    print(sorted(list(train_LEAPIDs)))
    print("\n********** TEST LEAP IDs *****************")
    print(sorted(list(test_LEAPIDs)))

    #return

    X_train = feature_matrix_clean[train_idx]
    X_dev  = feature_matrix_clean[test_idx]

    y_train = merged_df[outcome_cols].iloc[train_idx]     # if you have a target vector/series
    y_dev  = merged_df[outcome_cols].iloc[test_idx]

    ################### MULTI-OUTPUT REGRESSION ###################
    seed = 77
    kf_n=5
    ## have to ensure we don't split LEAPIDs between train and validation sets in each fold
    gkf = GroupKFold(n_splits=kf_n) 
    # `groups` must line up 1-for-1 with the rows of X_train / y_train
    groups = merged_df['LEAP_ID'].iloc[train_idx].to_numpy()


    roc_scores = []
    predictions = []
    best_models = {}

    if model_configs:
        print("using models defined in config")
    else:
        ######### DEFINE MODELS TO TRAIN #################################
        # ( label, model, params )
        model_configs = [
        #    ("elasticnet", ElasticNet(random_state=seed), {
        #        'alpha': [0.1, 0.5, 1, 5, 10],
        #        'l1_ratio': [0, 0.1, 0.5, 0.9, 1],
        #        'tol': [0.0001, 0.0005, 0.001],
        #        'max_iter': [100, 200, 500, 1000]
        #    }),
        #    ("randomforest", RandomForestRegressor(random_state=seed), {
        #        'n_estimators': [100, 200],
        #        'max_depth': [10, 20, 30, None],
        #        'min_samples_split': [2, 5, 10, 20],
        #        'min_samples_leaf': [3, 4]
        #    }),
        #    ("gradboost", HistGradientBoostingRegressor(random_state=seed), {
        #        'learning_rate': [0.01, 0.05, 0.1],
        #        'max_iter': [200,300],
        #        'max_depth': [5, 10, None],
        #        'min_samples_leaf': [10, 20, 50],
        #        'l2_regularization': [0, 0.1, 1.0]
        #    }),
            ("SVM", SVR(), [
                {
                    'kernel': ['rbf'],
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 0.01, 0.1],
                    'epsilon': [ 0.01, 0.02, 0.05]
                },
        #        {
        #            'kernel': ['poly'],
        #            'C': [0.1, 1, 10, 100],
        #            'gamma': ['scale', 0.001, 0.01, 0.1],
        #            'degree': [2, 3],
        #            'coef0': [0.0, 0.05, 1],
        #            'epsilon': [0.01, 0.05, 0.1, 0.2]
        #        },
        #        {
        #            'kernel': ['sigmoid'],
        #            'C': [0.1, 1, 10, 100],
        #            'gamma': ['scale', 0.001, 0.01, 0.1],
        #            'coef0': [0.0, 0.05, 1],
        #            'epsilon': [0.01, 0.05, 0.1, 0.2]
        #        }
            ])
        ] 

    print("starting training loop")
    ######### TRAIN MODELS ###########################################
    for label, model, params in model_configs:
        grid, scores = train_multioutput_regressor(model, params, label, X_train, y_train, X_dev, y_dev, gkf, groups,output_path,segment)
        predictions.append(scores)
        #print(scores)
        best_models[label] = grid

    predictions_c = pd.concat(predictions, ignore_index=True, sort=False)
    predictions_c['feature_extractor'] = foundation_model
    return best_models, predictions_c



if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument('-ip', '--input_path', required=True, help='path to h&e patches or path to feature file')
    ap.add_argument('-tp', '--truth_path', required=True, help='path and filename of label csv')
    ap.add_argument('-op', '--output_path', required=True, help='path to save output')
    ap.add_argument('-tf', '--truth_file',  required=True, help='name of csv')
    ap.add_argument('-ff', '--feature_file',  help='name of csv')
    ap.add_argument('-s', '--segment', help='PanCK+ or CD45+')
    ap.add_argument('-fm', '--foundation_model', help='one of uni, gigapath, virchow')
    ap.add_argument('-gf', '--get_features', action='store_true')
    ap.add_argument('-cp', '--config_path', help='full path to config file')


    args = ap.parse_args()

    #get current date and time for model name
    curr_date=str(datetime.date.today())
    curr_time=datetime.datetime.now().strftime('%H%M')

    # set default
    segment = 'PanCK+'
    foundation_model = 'uni'
    config = None

    # READ CONFIG
    if args.config_path:
        with open(args.config_path, "r") as file:
            config = yaml.safe_load(file)

        if 'segment' in config:
            segment=config['segment']

        if 'foundation_model' in config:
            foundation_model = config['foundation_model']
        

    ## give command line args priority over config file
    segment = args.segment if args.segment is not None else segment
    foundation_model = args.foundation_model if args.foundation_model is not None else foundation_model


    geneset=args.truth_file.split('_')[0]
    name=f"run-{segment}_{foundation_model}_{geneset}_{curr_date}_{curr_time}"
    #name=f"run-{segment}_{foundation_model}_{curr_date}_{curr_time}"
    print(name)

    #set up paths for models, training curves and predictions
    save_path = os.path.join(args.output_path,name)
    os.makedirs(save_path,exist_ok=True)
   

    print(f"ip: {args.input_path}")
    print(f"op: {save_path}")
    print(f"segment: {segment}")
    print(f"foundation model: {foundation_model}")
    if args.get_features:
        print("getting features from hugging face")


    # 1. read in outcomes
    print(f"labels: {args.truth_file}")
    outcome_df = pd.read_csv(os.path.join(args.truth_path,args.truth_file))
    outcome_df = outcome_df.set_index('ROI_ID')
    #print(outcome_df.head())    
    sig_name = outcome_df.columns[0]

    # 2. get features
    GET_FEATURES = args.get_features

    if GET_FEATURES:
        # load the image patches
        patch_paths = glob.glob(os.path.join(args.input_path,'patches',segment,'*.png'))
        image_names = [os.path.basename(path) for path in patch_paths]
        patch_metadata = pd.DataFrame([x.replace('.png','').split('_') for x in image_names],columns=['name', 'LEAP_ID', 'Segment','ROI_num', 'x', 'y'])
        patches = [Image.open(path) for path in patch_paths]

        # login to hugging face
        login(token="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", add_to_git_credential=False)

        # get features from hugging face
        feature_matrix = extractFeatures(foundation_model, patches)

        # save the features
        output_path = os.path.join(root_path,'features')
        save_features_with_names(feature_matrix, patch_paths, output_path, f"{foundation_model}-{segment}", format="csv")

    else:
        if args.feature_file:
            feat_file = args.feature_file
        else:
            feat_file= f'{foundation_model}-{segment}_features.csv'

        print(f"feature file: {feat_file}")
        image_names, feature_matrix = load_features_from_csv(os.path.join(args.input_path,feat_file))

        """if i'm reading features from a file then have to get metadata from image names"""
        patch_metadata = pd.DataFrame([x.replace('.png','').split('_') for x in image_names],columns=['name', 'LEAP_ID', 'Segment','ROI_num', 'x', 'y'])

    # remove patches that we don't have both features and outcomes
    patch_metadata = patch_metadata.set_index('name')
    merged_df = patch_metadata.join(outcome_df, how='left')
    mask = ~merged_df[sig_name].isna()
    feature_matrix_clean = feature_matrix[mask.values]
    merged_df = merged_df.dropna(subset=[sig_name])

    set_seed()

    print("Feature Matrix Shape:", feature_matrix_clean.shape) 
    print("Outcome Shape:", len(merged_df))  # Should match the number of images


    print("about to train")

    # TRAIN
    trained_models, scores = train(feature_matrix, merged_df, outcome_df.columns, save_path,segment, config)

    print("BEST PARAMS:")
    for label, model in trained_models.items():
        print(f"{label}: {model.best_params_}")

    print("now concat scores")
    # EVALUATE
    allscores = pd.DataFrame({
        'Run': pd.Series(dtype='str'),
        'MSE': pd.Series(dtype='float'),
        'RMSE': pd.Series(dtype='float'),
        'MAE': pd.Series(dtype='float'),
        'R2': pd.Series(dtype='float'),
        'PearsonCorr': pd.Series(dtype='float'),
        'SpearmanCorr': pd.Series(dtype='float'),
        'PearsonPVals': pd.Series(dtype='float'),
        'SpearmanPVals': pd.Series(dtype='float'),
        'PearsonFDR': pd.Series(dtype='float'),
        'SpearmanFDR': pd.Series(dtype='float'),
        'Train_CV_Score': pd.Series(dtype='float'),
        'ConcordanceCorr': pd.Series(dtype='float'),
        'VarianceRatio': pd.Series(dtype='float'),
        'feature_extractor': pd.Series(dtype='str')
    })
    allscores = pd.concat([allscores, scores], ignore_index=True)
    #print(allscores.sort_values(by='SpearmanCorr', ascending=False))
    #allscores.to_csv(os.path.join(save_path,f'allscores-{segment}.csv'),index=False)

    try:
        allscores.to_csv(os.path.join(save_path,f'allscores-{segment}.csv'),index=False)
        print(f"saved scores to {save_path}")
    except Exception as e:
        print(f"failed to save to {save_path}")
        tmp_path="/home/hrafique/sge_tmp"
        os.makedirs(tmp_path, exist_ok=True)
        print(f"saving to {tmp_path} instead")
        allscores.to_csv(os.path.join(tmp_path,f'allscores-{segment}.csv'),index=False)


    print("FINISHED RUNNING")
    
