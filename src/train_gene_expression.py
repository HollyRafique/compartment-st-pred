"""Patient-grouped gene-expression model selection.

Existing predict_sge.py arguments and ml_models YAML grids are supported.
Optional YAML sections (defaults shown)::

    cross_validation:
      strategy: group_kfold  # or leave_one_group_out
      n_splits: 5
      group_column: LEAP_ID
      selection_metric: neg_mean_squared_error
    evaluation:
      metrics: [mse, rmse, nrmse, mae, r2, pearson, spearman, concordance, variance_ratio]
      save_oof_predictions: true
    execution:
      n_workers: 1
      seed: 77
      test_size: 0.2

Use --resume RUN_DIRECTORY to resume an identical experiment. Joblib models
must only be loaded from trusted sources. OOF scores are post-selection
diagnostics; held-out patients provide the independent evaluation.
"""

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
import platform
import re
import warnings

import joblib
from joblib import Parallel, delayed, parallel_config
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, get_scorer
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit, LeaveOneGroupOut, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.stats.multitest import multipletests
from threadpoolctl import threadpool_limits
import yaml

DEBUG = True
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'config' / 'config_genetraining.yaml'
DEFAULT_METRICS = ['mse', 'rmse', 'nrmse', 'mae', 'r2', 'pearson', 'spearman',
                   'concordance', 'variance_ratio']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for short, name in [('ip', 'input_path'), ('tp', 'truth_path'),
                        ('op', 'output_path'), ('tf', 'truth_file')]:
        parser.add_argument('-' + short, '--' + name, required=True)
    for short, name in [('ff', 'feature_file'), ('s', 'segment'),
                        ('fm', 'foundation_model'), ('cp', 'config_path')]:
        parser.add_argument('-' + short, '--' + name)
    parser.add_argument('-gf', '--get_features', action='store_true')
    parser.add_argument('--n_workers', type=int, help='Override execution.n_workers')
    parser.add_argument('--resume', type=Path, help='Existing matching run directory')
    return parser.parse_args(argv)


def resolve_config(args):
    with DEFAULT_CONFIG.open() as handle:
        config = yaml.safe_load(handle)
    if args.config_path:
        with open(args.config_path) as handle:
            supplied = yaml.safe_load(handle) or {}
        config.update(supplied)
    config['segment'] = args.segment or config.get('segment', 'PanCK+')
    config['foundation_model'] = args.foundation_model or config.get('foundation_model', 'uni')
    for section, defaults in {
        'cross_validation': dict(strategy='group_kfold', n_splits=5,
                                 group_column='LEAP_ID', selection_metric='neg_mean_squared_error'),
        'evaluation': dict(metrics=DEFAULT_METRICS, save_oof_predictions=True),
        'execution': dict(n_workers=1, seed=77, test_size=0.2),
    }.items():
        config[section] = {**defaults, **config.get(section, {})}
    if args.n_workers is not None:
        config['execution']['n_workers'] = args.n_workers
    if config['execution']['n_workers'] == 0:
        raise ValueError('n_workers cannot be zero')
    entries = config['ml_models']
    config['ml_models'] = [entries] if isinstance(entries, dict) else entries
    if not config['ml_models']:
        raise ValueError('ml_models must contain at least one model')
    names = [entry['name'].lower() for entry in config['ml_models']]
    if len(names) != len(set(names)):
        raise ValueError('Model names must be unique')
    for entry in config['ml_models']:
        build_model_pipeline(entry['name'], config['execution']['seed'])
    unknown = set(config['evaluation']['metrics']) - METRIC_REGISTRY.keys()
    if unknown:
        raise ValueError(f'Unknown metrics: {sorted(unknown)}. Register a metric function first.')
    get_scorer(config['cross_validation']['selection_metric'])
    return config


def load_features(args, config):
    if not args.get_features:
        filename = args.feature_file or f"{config['foundation_model']}-{config['segment']}_features.csv"
        path = Path(args.input_path) / filename
        frame = pd.read_csv(path)
        return frame['image_name'].astype(str).tolist(), frame.drop(columns='image_name').to_numpy(dtype=float)

    # Image/torch/Hugging Face dependencies are needed only on this route.
    from PIL import Image
    from huggingface_hub import login
    from feature_extractor import extractFeatures
    token = os.environ.get('HF_TOKEN')
    if token:
        login(token=token, add_to_git_credential=False)
    paths = sorted((Path(args.input_path) / 'patches' / config['segment']).glob('*.png'))
    if not paths:
        raise ValueError('No image patches found')
    patches = []
    for path in paths:
        with Image.open(path) as image:
            patches.append(image.convert('RGB'))
    try:
        features = extractFeatures(config['foundation_model'], patches)
    finally:
        for patch in patches:
            patch.close()
    if features is None:
        raise ValueError('Unsupported foundation model for feature extraction')
    cache = Path(args.output_path) / 'features'
    cache.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(features)
    frame.insert(0, 'image_name', [path.name for path in paths])
    frame.to_csv(cache / f"{config['foundation_model']}-{config['segment']}_features-{datetime.now():%Y%m%d-%H%M%S%f}.csv", index=False)
    return frame['image_name'].tolist(), np.asarray(features, dtype=float)


def align_and_validate_data(names, features, truth_path):
    truth = pd.read_csv(truth_path, dtype={'ROI_ID': str}).set_index('ROI_ID')
    if not truth.index.is_unique or truth.index.hasnans:
        raise ValueError('Expression ROI_IDs must be nonmissing and unique')
    if truth.shape[1] == 0:
        raise ValueError('Expression file has no genes')
    rows = [Path(name).stem.split('_') for name in names]
    if any(len(row) != 6 for row in rows):
        raise ValueError('Patch names must contain name_LEAP_ID_Segment_ROI_num_x_y (six fields)')
    metadata = pd.DataFrame(rows, columns=['ROI_ID', 'LEAP_ID', 'Segment', 'ROI_num', 'x', 'y'])
    metadata.insert(0, 'image_name', names)
    if features.ndim != 2 or len(features) != len(metadata):
        raise ValueError('Feature dimensions do not match patch metadata')
    targets = truth.reindex(metadata['ROI_ID']).reset_index(drop=True)
    keep = targets.iloc[:, 0].notna().to_numpy()
    metadata = metadata.loc[keep].reset_index(drop=True)
    targets = targets.loc[keep].reset_index(drop=True).apply(pd.to_numeric, errors='raise')
    features = np.ascontiguousarray(features[keep], dtype=float)
    if not len(targets) or not features.shape[1]:
        raise ValueError('No matched samples or no feature columns')
    if not np.isfinite(features).all() or not np.isfinite(targets.to_numpy()).all():
        raise ValueError('Matched features and expression must be finite numeric values')
    metadata.insert(0, 'sample_id', np.arange(len(metadata)))
    return features, targets, metadata


def make_grouped_splits(features, metadata, config):
    cv = config['cross_validation']
    groups = metadata[cv['group_column']].to_numpy()
    execution = config['execution']
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=execution['test_size'],
                                         random_state=execution['seed']).split(features, groups=groups))
    if len(test) < 2:
        raise ValueError('At least two held-out samples are needed for correlation metrics')
    if cv['strategy'] == 'group_kfold':
        splitter = GroupKFold(n_splits=cv['n_splits'])
    elif cv['strategy'] == 'leave_one_group_out':
        splitter = LeaveOneGroupOut()
    else:
        raise ValueError('CV strategy must be group_kfold or leave_one_group_out')
    folds = list(splitter.split(features[train], groups=groups[train]))
    if len(folds) < 2:
        raise ValueError('At least two CV folds are required')
    if not set(groups[train]).isdisjoint(groups[test]):
        raise ValueError('Patient leakage in holdout split')
    for fit, valid in folds:
        if not set(groups[train][fit]).isdisjoint(groups[train][valid]):
            raise ValueError('Patient leakage in CV split')
    return train, test, folds


def build_model_pipeline(name, seed):
    factories = {
        'svm': lambda: SVR(),
        'elasticnet': lambda: ElasticNet(random_state=seed),
        'randomforest': lambda: RandomForestRegressor(random_state=seed, n_jobs=1),
        # Avoid an internal patch-random validation split for early stopping.
        'gradboost': lambda: HistGradientBoostingRegressor(random_state=seed, early_stopping=False),
    }
    key = name.lower()
    if key not in factories:
        raise ValueError(f'Unsupported model: {name}')
    return Pipeline([('scale', StandardScaler() if key in ('svm', 'elasticnet') else 'passthrough'),
                     ('model', factories[key]())])


def parameter_grid(params):
    grids = params if isinstance(params, list) else [params]
    result = []
    for grid in grids:
        converted = {}
        for key, value in grid.items():
            if key in ('n_jobs', 'model__n_jobs') and value != [1]:
                raise ValueError('Use estimator n_jobs: [1]; parallelism is across genes')
            if key in ('early_stopping', 'model__early_stopping') and value != [False]:
                raise ValueError('Disable internal random early stopping to preserve patient separation')
            converted[key if '__' in key else 'model__' + key] = value
        result.append(converted)
    return result


def correlation(y, pred, function):
    if len(y) < 2 or np.ptp(y) == 0 or np.ptp(pred) == 0:
        return np.nan, np.nan
    return function(y, pred)


def concordance(y, pred):
    denominator = np.var(y) + np.var(pred) + (np.mean(y) - np.mean(pred)) ** 2
    return 2 * np.mean((y - np.mean(y)) * (pred - np.mean(pred))) / denominator if denominator else np.nan


def normalized_rmse(y, pred):
    """Normalize by observed sample SD within the evaluated partition.

    Rows are patches, including repeated observations from the same ROI.
    Preserve the denominator for interpreting holdout, OOF and fold metrics.
    """
    std = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
    score = np.sqrt(mean_squared_error(y, pred)) / std if np.isfinite(std) and std > 0 else np.nan
    return {'nRMSE': score, 'ExpressionSD': std}


# Extend this registry with vector metrics. Spatial metrics need a separate,
# explicitly defined map reconstruction using the saved coordinates/identifiers.
METRIC_REGISTRY = {
    'mse': lambda y, p: {'MSE': mean_squared_error(y, p)},
    'rmse': lambda y, p: {'RMSE': np.sqrt(mean_squared_error(y, p))},
    'nrmse': normalized_rmse,
    'mae': lambda y, p: {'MAE': mean_absolute_error(y, p)},
    'r2': lambda y, p: {'R2': r2_score(y, p) if len(y) > 1 and np.var(y) > 0 else np.nan},
    'pearson': lambda y, p: dict(zip(['PearsonCorr', 'PearsonPVals'], correlation(y, p, pearsonr))),
    'spearman': lambda y, p: dict(zip(['SpearmanCorr', 'SpearmanPVals'], correlation(y, p, spearmanr))),
    'concordance': lambda y, p: {'ConcordanceCorr': concordance(y, p)},
    'variance_ratio': lambda y, p: {'VarianceRatio': np.var(p) / np.var(y) if np.var(y) else np.nan},
}


def evaluate_predictions(y, predicted, metrics):
    result = {}
    for name in metrics:
        result.update(METRIC_REGISTRY[name](np.asarray(y), np.asarray(predicted)))
    return result


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def gene_directory(run, index, gene):
    """Keep gene folders readable, filesystem-safe and unique by index."""
    safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', str(gene))
    return run / 'genes' / f'{index:04d}_{safe_name}'


def train_one_gene(index, gene, features, y, train, test, folds, metadata, config, run):
    if DEBUG: print(f"Training gene: {gene}", flush=True)
    directory = gene_directory(run, index, gene)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'complete.json').exists():
        print(f'Resuming: skipping completed {gene}', flush=True)
        return
    comparisons, best_search, winner = [], None, None
    caught_messages = []
    metrics = config['evaluation']['metrics']
    scorer = config['cross_validation']['selection_metric']
    # This also constrains threads for serial execution (no loky worker).
    with threadpool_limits(limits=1):
        for entry in config['ml_models']:
            name = entry['name']
            if DEBUG: print(f'{gene}:\tmodel: {name}',flush=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                search = GridSearchCV(build_model_pipeline(name, config['execution']['seed']),
                                      parameter_grid(entry['params']), cv=folds, scoring=scorer,
                                      n_jobs=1, refit=False, error_score=np.nan)
                try:
                    search.fit(features[train], y[train])
                except ValueError as exc:
                    comparisons.append(dict(Gene=gene, ML_Model=name, status='failed', error=str(exc)))
                    caught_messages.extend(str(w.message) for w in caught)
                    continue
            caught_messages.extend(str(w.message) for w in caught)
            results = pd.DataFrame(search.cv_results_)
            results.to_csv(directory / f'cv_results-{name.lower()}.csv', index=False)
            finite = np.isfinite(results['mean_test_score'].to_numpy())
            if not finite.any():
                comparisons.append(dict(Gene=gene, ML_Model=name, status='failed', error='No finite CV scores'))
                continue
            selected = int(np.argmax(np.where(finite, results['mean_test_score'], -np.inf)))
            row = dict(Gene=gene, ML_Model=name, status='ok',
                       best_params=json.dumps(search.cv_results_['params'][selected], sort_keys=True),
                       CV_Mean=float(results.iloc[selected]['mean_test_score']),
                       CV_SD=float(results.iloc[selected]['std_test_score']),
                       Failed_Candidates=int((~finite).sum()), Selected_Candidate=selected)
            comparisons.append(row)
            if winner is None or row['CV_Mean'] > winner['CV_Mean']:
                winner = row.copy()
                best_search = search
        pd.DataFrame(comparisons).to_csv(directory / 'comparison.csv', index=False)
        if winner is None:
            write_json(directory / 'warnings.json', sorted(set(caught_messages)))
            raise RuntimeError(f'All model searches failed for {gene}; see {directory}')
        params = json.loads(winner['best_params'])
        estimator = clone(best_search.estimator).set_params(**params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            estimator.fit(features[train], y[train])
            predicted = estimator.predict(features[test])
            winner.update(evaluate_predictions(y[test], predicted, metrics))
            frames = []
            held = metadata.iloc[test].copy()
            held['partition'], held['fold'] = 'holdout', -1
            held['observed'], held['predicted'] = y[test], predicted
            frames.append(held)
            if config['evaluation']['save_oof_predictions']:
                oof = np.full(len(train), np.nan)
                fold_numbers = np.full(len(train), -1)
                fold_metrics = []
                for number, (fit, valid) in enumerate(folds):
                    model = clone(estimator).fit(features[train[fit]], y[train[fit]])
                    oof[valid] = model.predict(features[train[valid]])
                    fold_numbers[valid] = number
                    fold_metrics.append(dict(fold=number, n_samples=len(valid),
                                             **evaluate_predictions(y[train[valid]], oof[valid], metrics)))
                frame = metadata.iloc[train].copy()
                frame['partition'], frame['fold'] = 'oof', fold_numbers
                frame['observed'], frame['predicted'] = y[train], oof
                frames.append(frame)
                winner.update({'OOF_' + key: value for key, value in evaluate_predictions(y[train], oof, metrics).items()})
                pd.DataFrame(fold_metrics).to_csv(directory / 'oof_fold_metrics.csv', index=False)
        caught_messages.extend(str(w.message) for w in caught)
        winner['Warning_Count'] = len(caught_messages)
        winner['feature_extractor'] = config['foundation_model']
        winner['Segment'] = config['segment']
        pd.DataFrame([winner]).to_csv(directory / 'best.csv', index=False)
        predictions = pd.concat(frames, ignore_index=True)
        predictions.insert(0, 'Gene', gene)
        predictions.to_csv(directory / 'predictions.csv', index=False)
        joblib.dump(estimator, directory / 'model.joblib')
        write_json(directory / 'warnings.json', sorted(set(caught_messages)))
        # Marker is written last; an interrupted gene is recomputed on resume.
        write_json(directory / 'complete.json', {'gene': gene})
    print(f"Finished {gene}: {winner['ML_Model']} CV={winner['CV_Mean']:.5g}", flush=True)


def dataset_signature(features, targets, metadata, config):
    digest = hashlib.sha256()
    digest.update(json.dumps(config, sort_keys=True).encode())
    digest.update(str(features.shape).encode())
    digest.update(features.tobytes())
    digest.update(targets.to_csv(index=False).encode())
    digest.update(metadata.to_csv(index=False).encode())
    digest.update(Path(__file__).read_bytes())
    digest.update(f'{sklearn.__version__}|{np.__version__}|{pd.__version__}'.encode())
    return digest.hexdigest()


def aggregate_results(run, genes):
    directories = [gene_directory(run, i, gene) for i, gene in enumerate(genes)]
    best = pd.concat([pd.read_csv(d / 'best.csv') for d in directories], ignore_index=True)
    for prefix in ['', 'OOF_']:
        for method in ['Pearson', 'Spearman']:
            column = prefix + method + 'PVals'
            if column in best:
                values = best[column].to_numpy(dtype=float)
                valid = np.isfinite(values)
                adjusted = np.full(len(values), np.nan)
                if valid.any():
                    adjusted[valid] = multipletests(values[valid], method='fdr_bh')[1]
                best[prefix + method + 'FDR'] = adjusted
    best.to_csv(run / 'best_models.csv', index=False)
    pd.concat([pd.read_csv(d / 'comparison.csv') for d in directories], ignore_index=True).to_csv(
        run / 'model_comparison.csv', index=False)
    # Stream predictions to avoid accumulating all genes in RAM.
    with (run / 'predictions.csv').open('w', newline='', encoding='utf-8') as handle:
        for index, directory in enumerate(directories):
            pd.read_csv(directory / 'predictions.csv', dtype={'ROI_ID': str, 'LEAP_ID': str}).to_csv(
                handle, index=False, header=index == 0)


def main(argv=None):
    args = parse_args(argv)
    config = resolve_config(args)
    if DEBUG:
        total_candidates = sum(
            len(ParameterGrid(parameter_grid(entry['params'])))
            for entry in config['ml_models']
        )
        print(f'Total parameter candidates per gene (all model families): {total_candidates}', flush=True)
    names, features = load_features(args, config)
    features, targets, metadata = align_and_validate_data(
        names, features, Path(args.truth_path) / args.truth_file)
    train, test, folds = make_grouped_splits(features, metadata, config)
    signature = dataset_signature(features, targets, metadata, config)
    if args.resume:
        run = args.resume.resolve()
        manifest = json.loads((run / 'run_manifest.json').read_text())
        if manifest['signature'] != signature:
            raise ValueError('Resume rejected: data, configuration, code, or versions changed')
    else:
        run = Path(args.output_path) / f"run-{config['segment']}_{config['foundation_model']}_{datetime.now():%Y%m%d_%H%M%S%f}"
        run.mkdir(parents=True, exist_ok=False)
        write_json(run / 'run_manifest.json', {
            'signature': signature, 'config': config, 'genes': targets.columns.tolist(),
            'versions': dict(python=platform.python_version(), sklearn=sklearn.__version__,
                             numpy=np.__version__, pandas=pd.__version__, joblib=joblib.__version__),
            'inputs': vars(args), 'train_indices': train.tolist(), 'test_indices': test.tolist(),
            'cv_folds': [{'train': train[a].tolist(), 'validation': train[b].tolist()} for a, b in folds],
            'interpretation': 'CV selects parameters/family; OOF is post-selection. Holdout is independent. '
                              'Selection averages folds equally. Metrics pool patches; p-values do not account for clustering.',
        })
        metadata.to_csv(run / 'samples.csv', index=False)
    print(f'{len(targets.columns)} genes, {len(train)} training / {len(test)} held-out patches, '
          f'{len(folds)} CV folds. Output: {run}', flush=True)
    with parallel_config(backend='loky', inner_max_num_threads=1,
                         max_nbytes='10M', mmap_mode='r'):
        Parallel(n_jobs=config['execution']['n_workers'])(
            delayed(train_one_gene)(i, gene, features, targets[gene].to_numpy(), train, test,
                                    folds, metadata, config, run)
            for i, gene in enumerate(targets.columns))
    aggregate_results(run, targets.columns)
    print(f'Completed. Selected parameters and metrics: {run / "best_models.csv"}', flush=True)


if __name__ == '__main__':
    main()
