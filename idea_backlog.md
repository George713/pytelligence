what it does under the hood:


LOADING
-------
- replacing alternativ Nones

FEATURE ANALYSIS
----------------
- feature analysis for single feature
- feature correlation

PREPROCESSING
-------------
- categorical encoding
- date encoding
- filling in NAs
- outlier removal

TRAINING
--------
- caching under the hood!
- automatic splitting into train, val, test
- comparing different algorithms
- with different metrics
- using cross validation
- validating on test set if asked
- finalizing model
- smote
- undersampling
- pca
- opt: calibration

MODEL ANALYSIS
--------------
- feature importance
- shap
- calibration plot

OPTIMIZATION
------------
- minimizing feature space w/ export functionality
- tuning

DEPLOYMENT
----------
- export to onnx
- usage of onnx interface

MONITORING
----------
???



What it could look like
=======================
import daktools as dkt

# Loading
dkt.loading.load_from_dwh(source="table_name_from_config")
<!-- Loads table X from dwh using connection specified
in env file limited to columns given in config.
Additionally casts dtypes and replaces NAs as given
in config and downcasts. -->

dkt.loading.load_from_local(source="table_name_from_config")
<!-- Loads from local file, which has been loaded
from DWH before. -->

# Feature Analysis
dkt.feat_analysis.get_distribution(table[[collist]])
<!-- Analyses columns of table via
- inferring dtype
- value_count
- nan, None, np.nan occurences
- plotting a distribution (bar plot?) -->

dkt.feat_analysis.get_distribution(table[[collist]], target_class=target_col)
<!-- Analyses columns of table given a certain target class.
Includes
- value_count
- plotting distribution (2d, blobs) -->

dkt.feat_analysis.plot_correlations(table[[collist]])
<!-- Plots 2D correlations for any two column combination. -->

# Training
setup, X_sample, y_sample = dkt.modelling.prepare_data(
    train_data
    test_data
    clf_target
    reg_target
    cat_cols
    date_cols
    ignore_cols
    cvgenerator
    outlier_removal
    custom_transformers
    na_strategy
    ignore_cache: bool = False
    sample_size: int or pct = 10
)
<!-- Applies custom_transformers in a pipeline,
followed by the encoding of cat and date cols
on the different folds, storing them withing the
setup.
X_samples and y_samples are returned for manual
inspection.
outlier_removal employes isolationforest after applying
custom_transformers and encoding. after removing outliers,
the setup is run once again on the reduces data.-->

setup_enh, X_sample, y_sample = dkt.modelling.enhance_distributions(
    setup
    undersampling
    smote
    pca
    sample_size: int or pct
)
<!-- Optional step: Applies undersampling, smote and/or pca can be
used at the end of the pipeline. -->

setup_enh.add_metric(custom_metric)
<!-- Adds custom metric to evaluation when
comparing algorithms. Needs to follow sklearn
standard. -->

compare_df, model_dict = dkt.modelling.compare_algorithms(
    setup_enh
    include=[]
    sort
    opt: return_models
    opt: test_data
)
<!-- Uses the prefitted folds to evaluate algorithms
on different metrics.
Optionally, return models training on entire train_data.
This option is False by default to save computational
time.
When using test_data, only a single model per algorithm
is trained on the entire train_data set.-->

compare_df, model_dict, hyperparams_dict = dkt.modelling.tune_algorithms(
    setup_enc
    include=[]
    optimize_for
    tune_iterations
    tune_time
    tune_early_stopping
    opt: return_models
    opt: return_hyperparams
)
<!-- Tunes algorithms -->

list_of_sufficient_features = dkt.modelling.reduces_feature_space(
    setup_enc
    model_dict.model
    acceptable_loss: float
    opt: export_to_config
)
<!-- Reruns training dropping one feature at a time. Once the
weakest feature is found, it is removed and the process repeated.
This continues until the acceptable_loss value (percentage of
current model metric) is reached. -->

dkt.modelling.finalize(
    model_dict.model
)
<!-- Trains model on entire dataset. -->

dkt.modelling.save_pipeline(model_dict.model, path)
<!-- Saves pipeline as joblib file. -->

dkt.deployment.convert_to_onnx(model_dict.model, path)
<!-- Converts pipeline to onnx format. -->

# Model Analysis
dkt.model_analysis.plot_calibration(model_dict.model, on_test: bool, opt: path)
<!-- plots current model calibration -->

dkt.model_analysis.plot_feature_importance(model_dict.model, shap: bool, opt: path)
<!-- Plots some form of feature_importance. -->