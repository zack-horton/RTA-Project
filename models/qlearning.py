### Code Structure: 
# Define all locations, the times and the weather we are interested in 
# Use models to predict the price of each of these trips. 
    # Or do we just use the real data that we know -- no because an Uber driver does not knopw what the other trips are costing at a time
# Normalize price by the time taken for each trip
# Initialize Q values 
# Run Q-Learning algorithm to get reward on each route.


import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics



# Define all locations, the times and the weather we are interested in 
company = 'uber'
uber_data =  pd.read_parquet(f"data/{company.lower()}/{company.lower()}_full_data.parquet")
uber_data_limited = uber_data[uber_data['month'] == 11]
uber_data_limited = uber_data_limited[(uber_data_limited['hour'] == 6) | (uber_data_limited['hour'] == 17)]
uber_data_limited = uber_data_limited[(uber_data_limited['weather_rain'] == 1) | (uber_data_limited['weather_partly_cloudy'] == 1)]

# Time to make predictions
print(uber_data_limited.head())



