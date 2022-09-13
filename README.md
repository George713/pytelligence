# pytelligence
pycaret clone aimed at simplicity and production-ready code within half the usual development time.

## Features
- Automation of preprocessing
- Comparison of various algorithms
- Bayesian hyperparameter tuning
- Exhaustive feature reduction
- Single-artefact export

## Installation
Install pytelligence via pip either directly from pypi.org 
```
pip install pytelligence
```
or from its github repository
```
pip install @git....
```

## Usage
First import pytelligence:

`import pytelligence as pt`

### Configuration
`pytelligence` is focused around configuration from a single `yaml` file typially called `config.yml`. The format is as follows for a classification use case:

```yaml
modelling:
  target_clf: 'class'
  numeric_cols:
    - 'deg-malig'
  categorical_cols:
    - 'age'
    - 'menopause'
    - 'tumor-size'
    - 'inv-nodes'
    - 'node-caps'
    - 'breast'
    - 'breast-quad'
    - 'irradiat'
  feature_scaling: False
```

This configuration is loaded once during the data preparation step.

`target_clf` specifies the column name of the target label for classification use cases.

`numeric_cols` & `categorical_cols` inform pytelligence about the types of feature columns, that are to be used during data preparation and subsequent training.

`feature_scaling` serves as boolean flag for activating feature scaling during data preparation. (not implemented yet)

### Data Preparation
Assuming a dataframe `df` that fits the above config file the data can be prepared. Preparation includes 
- column verification
- one-hot-encoding of categorical columns
- encoding of the target label in case of classification (if required)
- scaling of numeric features if turned on (not implemented yet)

```python
setup, X_sample, y_sample = pt.modelling.prepare_data(
    train_data=df,
    config_path="./config.yml",
)
```

The returned `setup` is a container object, that holds the prepared data and configuration. It is handed to subsequent processing steps.

`X_sample` and `y_sample` are samples from the prepared dataframe and target column.

### Algorithm Comparison
Various algorithms can be compared via

```python
compare_df, algo_list, model_list = pt.modelling.compare_algorithms(
    setup=setup,
    include=[
        "lr",
        "nb",
    ],
    sort="f1",
    return_models=True,
)
```

Implemented algorithms include:
- lr - LogisticRegression
- nb - Gaussian Naive Bayes

If no algorithm list is provided to the `include` parameter, all available algorithms are compared.

`compare_df` returns a pandas dataframe which holds the results of the different algorithms sorted by the metric provided with the `sort` parameter.

`algo_list` is a list comprised of the different algorithm abbreviations sorted by `sort`. It serves as an easy means to referencing the best algorithms in later processing steps.

`model_list` holds model instances trained on the entire training dataset sorted by `sort`. Only computed if `return_models` is set to `True`.

### Hyperparameter Tuning
Hyperparameter tuning is automated using Bayesian optimization.
```python
compare_df_tune, model_list, opt_history_dict = (
        pt.modelling.tune_hyperparams(
               setup=setup,
               include=["lr", "nb"],
               optimize="f1",
               n_trials=5,
               return_models=True,
        )
)
```
`optimize` determines the metric to optimize for.

`n_trials` specifies the number of hyperparameter sets to check for each algorithm.

`compare_df_tune` holds an overview of the achieved target metric for each algorithm along with the best set of hyperparameters.

`model_list` holds trained model instances. Only returned when `return_models=True`.

`opt_history_dict` holds plots of the optimization process, which can be accessed by the key of the algorithms, e.g. `opt_history_dict["lr"].show()`

### Exhaustive Feature Reduction (EFR)
The full list of features used during model training might not be desirable for two reasons:
1. Training/tuning or inference time might be unnecceraly large, when only a few features hold the key information.
2. Surplus features might actually decrease an algorithm's metrics.

`EFR` tackles this problem by subsequently finding the weakest feature and removing it from training. This process runs until an acceptable loss is reached.

```python
best_feature_list, metric_feature_df = pt.modelling.reduce_feature_space(
    setup=setup,
    algorithm= "nb",
    metric="f1",
    reference_metric=compare_df_tune.iloc[0]["metric"],
    acceptable_loss=0.99,
    hyperparams=compare_df_tune.iloc[0]["hyperparams"]
)
```

`reference_metric` is the so far achieved metric score.

`acceptable_loss` is a percentage value, which is used to calculated a threshold metric: `threshold = acceptable_loss * reference_metric`. Once the set of features does not reach this threshold anymore, `EFR` is stopped.

(Optional) `hyperparams` is a dictionary holding the set of hyperparameters to use for the algorithm.

`best_feature_list` holds an unsorted list of the best feature combination found.

`metric_feature_df` is a pandas dataframe with the columns `metric`, `features` and `feature_count` giving an overview of the correlation between feature number and metric. The column `features` contains lists of the features used during a specific `EFR` cycle.

### Single-Artefact Export
The preprocessing pipeline is stored within the setup container and can be accessed via `setup.prep_pipe`.

It is automatically added to a model instance during export:

```python
pt.modelling.export_model(
        setup=setup,
        model=model_list[0],
        target_dir="./",
    )
```

The naming format of the artefact is
```model_{date}_{algo}_#{number}.joblib```,
where `#{number}` is used when several models are exported during the same day.