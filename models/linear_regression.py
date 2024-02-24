import pandas as pd
import numpy as np
import pickle
import os

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
import sklearn.metrics as metrics

def get_data(company):
    train = pd.read_parquet(f"data/{company.lower()}/{company.lower()}_train.parquet")
    test = pd.read_parquet(f"data/{company.lower()}/{company.lower()}_test.parquet")
    
    return train.drop(['price'], axis=1).to_numpy(), train[['price']].to_numpy(), test.drop(['price'], axis=1).to_numpy(), test[['price']].to_numpy()

def fit_ols_regression(train_x, train_y, test_x, test_y, company, save_model=False):
    model = LinearRegression().fit(train_x, train_y)
    # y_train_pred = model.predict(train_x)
    y_test_pred = model.predict(test_x)
    
    # Test metrics
    explained_variance = metrics.explained_variance_score(test_y, y_test_pred)
    mean_absolute_error = metrics.mean_absolute_error(test_y, y_test_pred) 
    mse = metrics.mean_squared_error(test_y, y_test_pred) 
    mean_squared_log_error = metrics.mean_squared_log_error(test_y, y_test_pred)
    median_absolute_error = metrics.median_absolute_error(test_y, y_test_pred)
    r2 = metrics.r2_score(test_y, y_test_pred)

    print()
    print("*"*5, f"{company.upper()} OLS REGRESSION", "*"*5)
    print('explained_variance: ', round(explained_variance, 4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('Median AE: ', round(median_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse),4 ))
    
    if save_model:
        filename = f'models/{company.lower()}_reg_models/{company.lower()}_ols_regression.sav'
        pickle.dump(model, open(filename, 'wb'))
    
    return model
    
def fit_lasso_regression(train_x, train_y, test_x, test_y, company, save_model=False):
    model = LassoCV(cv=5).fit(train_x, train_y)
    # y_train_pred = model.predict(train_x)
    y_test_pred = model.predict(test_x)
    
    # Test metrics
    explained_variance = metrics.explained_variance_score(test_y, y_test_pred)
    mean_absolute_error = metrics.mean_absolute_error(test_y, y_test_pred) 
    mse = metrics.mean_squared_error(test_y, y_test_pred) 
    mean_squared_log_error = metrics.mean_squared_log_error(test_y, y_test_pred)
    median_absolute_error = metrics.median_absolute_error(test_y, y_test_pred)
    r2 = metrics.r2_score(test_y, y_test_pred)

    print()
    print("*"*5, f"{company.upper()} LASSO REGRESSION", "*"*5)
    print('explained_variance: ', round(explained_variance, 4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('Median AE: ', round(median_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse),4 ))
    
    if save_model:
        filename = f'models/{company.lower()}_reg_models/{company.lower()}_lasso_regression.sav'
        pickle.dump(model, open(filename, 'wb'))
    
    return model

def fit_ridge_regression(train_x, train_y, test_x, test_y, company, save_model=False):
    model = RidgeCV(cv=5).fit(train_x, train_y)
    # y_train_pred = model.predict(train_x)
    y_test_pred = model.predict(test_x)
    
    # Test metrics
    explained_variance = metrics.explained_variance_score(test_y, y_test_pred)
    mean_absolute_error = metrics.mean_absolute_error(test_y, y_test_pred) 
    mse = metrics.mean_squared_error(test_y, y_test_pred) 
    mean_squared_log_error = metrics.mean_squared_log_error(test_y, y_test_pred)
    median_absolute_error = metrics.median_absolute_error(test_y, y_test_pred)
    r2 = metrics.r2_score(test_y, y_test_pred)

    print()
    print("*"*5, f"{company.upper()} RIDGE REGRESSION", "*"*5)
    print('explained_variance: ', round(explained_variance, 4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('Median AE: ', round(median_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse),4 ))
    
    if save_model:
        filename = f'models/{company.lower()}_reg_models/{company.lower()}_ridge_regression.sav'
        pickle.dump(model, open(filename, 'wb'))
    
    return model

def fit_regressions(train_x, train_y, test_x, test_y, company, save_model=False):
    train_y = train_y.ravel()
    test_y = test_y.ravel()
    fit_ols_regression(train_x, train_y, test_x, test_y, company, save_model)
    fit_lasso_regression(train_x, train_y, test_x, test_y, company, save_model)
    fit_ridge_regression(train_x, train_y, test_x, test_y, company, save_model)

lyft_train_x, lyft_train_y, lyft_test_x, lyft_test_y = get_data('lyft')
print("lyft train:", len(lyft_train_x), "lyft test:", len(lyft_test_x))

uber_train_x, uber_train_y, uber_test_x, uber_test_y = get_data('uber')
print("uber train:", len(uber_train_x), "uber test:", len(uber_test_x))

fit_regressions(lyft_train_x, lyft_train_y, lyft_test_x, lyft_test_y, "lyft", save_model=True)
fit_regressions(uber_train_x, uber_train_y, uber_test_x, uber_test_y, "uber", save_model=True)