"""The supervised baselines the LLMs are measured against.

``ml/`` holds the three ensemble members and the stages around them:

    gbt.py          LightGBM+GFP and XGBoost+GFP, detection and typology
    gcpal.py        GCPAL, line graph over transactions, GIN, contrastive
    gcpal_infer.py  re-inference of the GCPAL checkpoints on the file-order split
    ensemble.py     the two-booster evaluation and frozen construction scores
    tuning.py       the Optuna search behind data/tuned_params/

``metrics.py`` holds the threshold sweep every one of them uses.

Nothing here is imported at package load, because each stage needs a different
heavy dependency: lightgbm and xgboost for the boosted trees, torch and
torch-geometric for GCPAL, optuna for the search. Import the module you need.
"""
