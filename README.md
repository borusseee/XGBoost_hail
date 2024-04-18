# XGBoost_hail
This repository contains the code of the XGBoost large hail probability algorithm.
New_Hail_proxies_extraction.py is the code used to derive the values of the 11 hail predictors from ERA-5.
create_annual_predictors_files.py is used to save these predictors in yearly files, making it ready to use by the model.
predictions_AU.py, predictions_EU.py, predictions_US.py, and predictions_world.py contain the code that is used to calculate large hail probabilities using XGBoost. These models are trained on different subsets of data (i.e. training with data from Australia, Europe, CONUS, and a combination of the 3)
