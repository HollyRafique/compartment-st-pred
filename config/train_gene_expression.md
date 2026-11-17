Run the new trainer with the same arguments as `predict_sge.py`:

```powershell
python src/train_gene_expression.py -ip path/to/features -ff features.csv -tp path/to/expression -tf expression.csv -op path/to/results -cp config/config_genetraining.yaml --n_workers 4
```

The existing `ml_models` grids are used unchanged. Optional sections can be
added to that YAML (omitting them uses these defaults):

```yaml
cross_validation:
  strategy: group_kfold
  n_splits: 5
  group_column: LEAP_ID
  selection_metric: neg_mean_squared_error
evaluation:
  metrics: [mse, rmse, mae, r2, pearson, spearman, concordance, variance_ratio]
  save_oof_predictions: true
execution:
  n_workers: 1
  seed: 77
  test_size: 0.2
```

Use `n_splits: 8` for eight-fold grouped CV, or
`strategy: leave_one_group_out` for leave-one-patient-out (`n_splits` is then
ignored). Both run within the training patients, retaining an independent
20% patient holdout. Selection scores average folds equally; pooled metrics
weight patches equally. Patient IDs and exact folds are saved in the manifest.

Genes run in parallel; estimator/search threads are limited to one. Start with
a modest worker count and measure RAM use before increasing it. Scaling is
fitted inside CV for SVM and Elastic Net. Histogram boosting uses its full
iteration budget because its internal random early-stopping split is disabled
to preserve patient grouping. These changes can alter results and runtime
relative to the original script.

`best_models.csv` contains the winning family and parameters for each gene,
mean/SD CV score and held-out metrics. `model_comparison.csv` contains the best
CV result per family. `predictions.csv` contains held-out and optional OOF
predictions with sample, patient, ROI and coordinate metadata. `genes/0000/`
(gene order is in the manifest) contains the fitted winning pipeline,
candidate/fold CV scores, warning messages, fold metrics and completion marker.
Failed candidates have nonfinite CV scores; warning messages are recorded per
gene, not mapped to individual candidates.

OOF predictions require an extra fit per fold for each winning configuration.
They are post-selection diagnostics, not an unbiased estimate after tuning.
Held-out metrics provide the independent assessment. Correlation p-values do
not account for patient/ROI clustering. BH FDR is applied across finite p-values
within each run separately for held-out and OOF correlations.

Add `--resume path/to/existing/run` to the same command to skip completed genes.
Data, resolved config, code and core library versions must match the saved
signature. Interrupted genes restart; model-family searches within a gene do
not resume individually. Models remain fitted on the training patients, not
refitted on the held-out data. Only load joblib models from trusted sources.

The rarely used `-gf` route uses `feature_extractor.py`, cached Hugging Face
authentication or `HF_TOKEN`, and writes a timestamped feature CSV under the
output `features` directory. It retains the existing extractor's in-memory
batch behavior and may need substantial memory. No GPU regression backend is
introduced.

Additional vector metrics can be registered in `METRIC_REGISTRY`. SSIM and NMI
are deliberately not assigned implicit definitions: define spatial map
reconstruction/data range for SSIM and continuous-value discretization and
normalization for NMI first. Saved predictions support calculating these later
without rerunning training.
