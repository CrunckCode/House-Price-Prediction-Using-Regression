# House Price Prediction Using Regression

An end-to-end regression workflow in scikit-learn that predicts median house value (`MEDV`) from 13 neighbourhood features: data cleaning, stratified splitting, correlation analysis, a preprocessing pipeline, and comparison of three models with cross-validation.

## Data
`data.csv`: 511 rows with CRIM (crime rate), ZN, INDUS, CHAS (river dummy), NOX, RM (rooms), AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT and the target MEDV (the Boston housing layout). Five RM values are missing and are filled with the mean.

## Method
1. Explore: `info()`, `describe()`, histograms, correlations with MEDV (strongest positive: RM 0.67; strongest negative: LSTAT -0.56) and scatter plots.
2. Split 80/20 with `StratifiedShuffleSplit` on the CHAS column, so the rare river-adjacent houses appear in both sets.
3. Engineer a TAX-per-room feature (`TAXRM`) and check its correlation (-0.53 with MEDV).
4. Preprocessing pipeline: mean imputation then standard scaling.
5. Fit and compare Linear Regression, Decision Tree and Random Forest, using 10-fold cross-validated RMSE.
6. Evaluate a final model on the held-out test set.

## Results (re-run, RMSE in the units of MEDV, thousands of dollars)
| Model | Training RMSE | 10-fold CV RMSE | Test RMSE |
|---|---|---|---|
| Linear Regression (final model in the script) | 5.07 | 5.19 +/- 1.40 | 9.70 |
| Decision Tree | 0.00 | 4.54 +/- 1.15 | 5.84 |
| Random Forest | 1.45 | 3.80 +/- 1.26 | 4.77 |

The decision tree's zero training error is overfitting (it memorises the training data); cross-validation exposes it. The random forest generalises best. The script's final evaluation uses the linear model, which has the weakest test score here; switching the final step to the random forest improves the held-out RMSE from about 9.7 to about 4.8.

## How to run
```
pip install numpy pandas matplotlib scikit-learn
python House_price_prediction.py
```
The file was exported from a Colab notebook and contains notebook-style single expressions; run it cell by cell in Jupyter for the printed outputs and plots.

## Limitations
Small dataset, a single split, no hyperparameter tuning, and the TAXRM feature is explored but not included in the final model matrix. The Boston dataset has known ethical and data-quality concerns (the `B` feature), so it is best treated as a teaching dataset.
