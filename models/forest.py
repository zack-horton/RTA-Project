import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics

def get_data(company):
    train = pd.read_parquet(f"data/{company.lower()}/{company.lower()}_train.parquet")
    test = pd.read_parquet(f"data/{company.lower()}/{company.lower()}_test.parquet")
    
    return train.drop(['price'], axis=1).to_numpy(), train[['price']].to_numpy(), test.drop(['price'], axis=1).to_numpy(), test[['price']].to_numpy()

def fit_RF(train_x, train_y, test_x, test_y, company, save_model=False):
    model = RandomForestRegressor(random_state=2024)
    param_grid = {'n_estimators': [64, 128, 256],
                  'max_depth': [4, 8, 16]}
    
    grid_rf = GridSearchCV(model, param_grid, cv=10, verbose=4)
    grid_rf = grid_rf.fit(train_x, train_y)

    # y_train_pred = model.predict(train_x)
    y_test_pred = grid_rf.predict(test_x)
    
    # Test metrics
    explained_variance = metrics.explained_variance_score(test_y, y_test_pred)
    mean_absolute_error = metrics.mean_absolute_error(test_y, y_test_pred) 
    mse = metrics.mean_squared_error(test_y, y_test_pred) 
    mean_squared_log_error = metrics.mean_squared_log_error(test_y, y_test_pred)
    median_absolute_error = metrics.median_absolute_error(test_y, y_test_pred)
    r2 = metrics.r2_score(test_y, y_test_pred)

    print()
    print("*"*5, f"{company.upper()} RANDOM FOREST", "*"*5)
    print('explained_variance: ', round(explained_variance, 4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('Median AE: ', round(median_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse),4 ))
    
    model = grid_rf.best_estimator_
    
    if save_model:
        filename = f'models/{company.lower()}_tree_models/{company.lower()}_randomforest.sav'
        pickle.dump(model, open(filename, 'wb'))
    
    return model
    
def fit_XGB(train_x, train_y, test_x, test_y, company, save_model=False):
    model = GradientBoostingRegressor(random_state=2024)
    param_grid = {'max_depth': [2, 4, 8],
                  'n_estimators': [64, 128, 256]}
    
    grid_xgb = GridSearchCV(model, param_grid, cv=10, verbose=4)
    grid_xgb = grid_xgb.fit(train_x, train_y)
    
    # y_train_pred = model.predict(train_x)
    y_test_pred = grid_xgb.predict(test_x)
    
    # Test metrics
    explained_variance = metrics.explained_variance_score(test_y, y_test_pred)
    mean_absolute_error = metrics.mean_absolute_error(test_y, y_test_pred) 
    mse = metrics.mean_squared_error(test_y, y_test_pred) 
    mean_squared_log_error = metrics.mean_squared_log_error(test_y, y_test_pred)
    median_absolute_error = metrics.median_absolute_error(test_y, y_test_pred)
    r2 = metrics.r2_score(test_y, y_test_pred)

    print()
    print("*"*5, f"{company.upper()} XGBoost", "*"*5)
    print('explained_variance: ', round(explained_variance, 4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('Median AE: ', round(median_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse),4 ))
    
    model = grid_xgb.best_estimator_
    
    if save_model:
        filename = f'models/{company.lower()}_tree_models/{company.lower()}_xgboost.sav'
        pickle.dump(model, open(filename, 'wb'))
    
    return model

def fit_DT(train_x, train_y, test_x, test_y, company, save_model=False):
    model = DecisionTreeRegressor(random_state=2024)
    param_grid = {'max_depth': [2, 4, 8],
                  'min_samples_leaf': [2, 4, 8, 16],
                  'ccp_alpha': [0.01]}
    
    grid_dt = GridSearchCV(model, param_grid, cv=5, verbose=4)
    grid_dt = grid_dt.fit(train_x, train_y)
    
    # y_train_pred = model.predict(train_x)
    y_test_pred = grid_dt.predict(test_x)
    
    # Test metrics
    explained_variance = metrics.explained_variance_score(test_y, y_test_pred)
    mean_absolute_error = metrics.mean_absolute_error(test_y, y_test_pred) 
    mse = metrics.mean_squared_error(test_y, y_test_pred) 
    mean_squared_log_error = metrics.mean_squared_log_error(test_y, y_test_pred)
    median_absolute_error = metrics.median_absolute_error(test_y, y_test_pred)
    r2 = metrics.r2_score(test_y, y_test_pred)

    print()
    print("*"*5, f"{company.upper()} DECISION TREE", "*"*5)
    print('explained_variance: ', round(explained_variance, 4))    
    print('mean_squared_log_error: ', round(mean_squared_log_error, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mean_absolute_error, 4))
    print('Median AE: ', round(median_absolute_error, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse),4 ))
    
    model = grid_dt.best_estimator_
    
    if save_model:
        filename = f'models/{company.lower()}_tree_models/{company.lower()}_decision_tree.sav'
        pickle.dump(model, open(filename, 'wb'))
    
    return model

def fit_forest(train_x, train_y, test_x, test_y, company, save_model=False):
    train_y = train_y.ravel()
    test_y = test_y.ravel()
    fit_DT(train_x, train_y, test_x, test_y, company, save_model)
    # fit_RF(train_x, train_y, test_x, test_y, company, save_model)
    # fit_XGB(train_x, train_y, test_x, test_y, company, save_model)

lyft_train_x, lyft_train_y, lyft_test_x, lyft_test_y = get_data('lyft')
print("lyft train:", len(lyft_train_x), "lyft test:", len(lyft_test_x))

uber_train_x, uber_train_y, uber_test_x, uber_test_y = get_data('uber')
print("uber train:", len(uber_train_x), "uber test:", len(uber_test_x))

fit_forest(lyft_train_x, lyft_train_y, lyft_test_x, lyft_test_y, "lyft", save_model=True)
fit_forest(uber_train_x, uber_train_y, uber_test_x, uber_test_y, "uber", save_model=True)