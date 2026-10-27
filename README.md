# Time Series Forecasting with XGBoost for Energy Consumption

## Project model

This implementation uses the **AEP hourly energy-consumption series** and builds an end-to-end forecasting model with:

1. Chronological data cleaning
2. Time/calendar features
3. Lag features: 1h, 2h, 3h, 24h, 48h, 72h, 168h
4. Leakage-safe rolling statistics
5. Chronological train/test split
6. Seasonal-naive 168-hour baseline
7. XGBoost regression
8. MAE, RMSE and R² evaluation
9. Actual-vs-predicted plot
10. XGBoost feature importance
11. Recursive next-24-hour forecasting
12. Saved model artifact

## Dataset

Use `AEP_hourly.csv` with columns:

- `Datetime`
- `AEP_MW`

Dataset source:
https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

A public copy of the AEP CSV is also available at:
https://github.com/khsieh18/Time-Series/blob/master/AEP_hourly.csv

## Main references used

1. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining, 2016. DOI: https://doi.org/10.1145/2939672.2939785

2. R. Mulla (Robikscube), "Time Series Forecasting with Machine Learning [YT]," Kaggle:
https://www.kaggle.com/code/robikscube/time-series-forecasting-with-machine-learning-yt

3. H.-A. Dang et al., "Deep learning and ensemble learning models for individual appliance level short-term load forecasting in buildings," 2023 Asia Meeting on Environment and Electrical Engineering (EEE-AM), DOI: https://doi.org/10.1109/EEE-AM58328.2023.10395410

4. F. Ahmed et al., "Comparative Analysis of Time Series Forecasting of a Household Load Consumption with LSTM Neural Networks and XGBoost Model," 2024 International Conference on Engineering & Computing Technologies (ICECT), DOI: https://doi.org/10.1109/ICECT61618.2024.10581123

5. "Comparative Study on Forecasting of Schedule Generation in Delhi Region for the Resilient Power Grid Using Machine Learning," IEEE Transactions on Industry Applications, 2024, DOI: https://doi.org/10.1109/TIA.2023.3316646

## Why this design

The model follows the cited XGBoost/time-series workflow while making the evaluation appropriate for temporal data: observations are ordered chronologically and the test period is later than the training period. Lag and rolling features are shifted so the target observation is not used to construct its own predictors. A seasonal-naive 168-hour baseline is included so the XGBoost result is compared against a simple weekly-seasonality benchmark.

The cited load-forecasting studies motivate comparing tree ensembles with other forecasting approaches; this project focuses on XGBoost as the selected model rather than claiming that the cited papers' reported metrics were reproduced.

## Run

```bash
pip install -r requirements.txt
python energy_forecasting_xgboost.py
```

Put `AEP_hourly.csv` in the same folder as the script.

Outputs are written to `outputs/`:
- `metrics.csv`
- `actual_vs_predicted.png`
- `feature_importance.png`
- `feature_importance_top20.csv`
- `next_24h_forecast.csv`
- `next_24h_forecast.png`
- `xgboost_energy_model.joblib`
