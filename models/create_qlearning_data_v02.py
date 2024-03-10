### Code Structure: 
# Define all locations, the times and the weather we are interested in 
# Use models to predict the price of each of these trips. 
    # Or do we just use the real data that we know -- no because an Uber driver does not knopw what the other trips are costing at a time
# Normalize price by the time taken for each trip
# Initialize Q values 
# Run Q-Learning algorithm to get reward on each route.


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pickle
from zipfile import ZipFile
from datetime import datetime

data = pd.read_csv('uber_lyft_file.zip')
data_f = pd.read_parquet('data/lyft/lyft_full_data.parquet')
ordered_columns = [x for x in data_f.columns if x != "price"]

now = datetime.now()
current_hour = datetime.strptime(now.strftime("%H"), "%H")
current_day = datetime.strptime(now.strftime("%d"), "%d")


destination_options = {"Back Bay": ['Boston University', 'Fenway', 'Haymarket Square', 'North End', 
                                    'Northeastern University', 'South Station'],
                       "Beacon Hill": ['Boston University', 'Fenway', 'Haymarket Square', 'North End', 
                                       'Northeastern University', 'South Station'],
                       "Boston University": ['Back Bay', 'Beacon Hill', 'Financial District', 
                                             'North Station', 'Theatre District', 'West End'],
                       "Fenway": ['Back Bay', 'Beacon Hill', 'Financial District', 
                                  'North Station', 'Theatre District', 'West End'],
                       "Financial District": ['Boston University', 'Fenway', 'Haymarket Square', 'North End', 
                                              'Northeastern University', 'South Station'],
                       "Haymarket Square": ['Back Bay', 'Beacon Hill', 'Financial District', 
                                            'North Station', 'Theatre District', 'West End'],
                       "North End": ['Back Bay', 'Beacon Hill', 'Financial District', 
                                     'North Station', 'Theatre District', 'West End'],
                       "North Station": ['Boston University', 'Fenway', 'Haymarket Square', 'North End', 
                                         'Northeastern University', 'South Station'],
                       "Northeastern University": ['Back Bay', 'Beacon Hill', 'Financial District', 
                                                   'North Station', 'Theatre District', 'West End'],
                       "South Station": ['Back Bay', 'Beacon Hill', 'Financial District', 
                                         'North Station', 'Theatre District', 'West End'],
                       "Theatre District": ['Boston University', 'Fenway', 'Haymarket Square', 'North End', 
                                            'Northeastern University', 'South Station'],
                       "West End": ['Boston University', 'Fenway', 'Haymarket Square', 'North End', 
                                    'Northeastern University', 'South Station']}


data_list = []
hours = [6, 7, 8, 9, 10, 11, 17, 18, 19, 20, 21, 22]
days = [1,2,3,4,5,6,7]
short_summaries = ['Clear','Rain']
cab = 'Uber'
name = 'Standard'
month = 12

for source, destinations in destination_options.items():
    for destination in destinations:
        for hour in hours:
            for short_summary in short_summaries:
                for day in days:
                    ride_df = data.loc[(data['source'] == source) &
                        (data['destination'] == destination)]
                    distance = ride_df[['distance']].median().values[0]
                    surge_df = data.loc[(data['hour'] == int(hour) ) &
                                (data['source'] == source)]
                    surge_multiplier = surge_df[['surge_multiplier']].mean().values[0]
                    temp_df = data.loc[(data['hour'] == hour) &
                                        (data['month'] == month)]
                    temperature = temp_df[['temperature']].median().values[0]
                    apparentTemperature = temp_df[['apparentTemperature']].median().values[0]
                    precipIntensity = temp_df[['precipIntensity']].median().values[0]
                    precipProbability = temp_df[['precipProbability']].median().values[0]
                    humidity = temp_df[['humidity']].median().values[0]
                    windSpeed = temp_df[['windSpeed']].median().values[0]
                    windGust = temp_df[['windGust']].median().values[0]
                    visibility = temp_df[['visibility']].median().values[0]
                    temperatureHigh = temp_df[['temperatureHigh']].median().values[0]
                    temperatureLow = temp_df[['temperatureLow']].median().values[0]
                    apparentTemperatureHigh = temp_df[['apparentTemperatureHigh']].median().values[0]
                    apparentTemperatureLow = temp_df[['apparentTemperatureLow']].median().values[0]

                    dewPoint = temp_df[['dewPoint']].median().values[0]
                    pressure = temp_df[['pressure']].median().values[0]
                    windBearing = temp_df[['windBearing']].median().values[0]
                    cloudCover = temp_df[['cloudCover']].median().values[0]
                    uvIndex = temp_df[['uvIndex']].median().values[0]
                    ozone = temp_df[['ozone']].median().values[0]
                    moonPhase = temp_df[['moonPhase']].median().values[0]
                    precipIntensityMax = temp_df[['precipIntensityMax']].median().values[0]
                    temperatureMin = temp_df[['temperatureMin']].median().values[0]
                    temperatureMax = temp_df[['temperatureMax']].median().values[0]
                    apparentTemperatureMin = temp_df[['apparentTemperatureMin']].median().values[0]
                    apparentTemperatureMax = temp_df[['apparentTemperatureMax']].median().values[0]

                    test_dict = {'hour': hour,
                    'day': int(day),
                    'month': int(month),
                    'distance': float(distance),
                    'surge_multiplier': float(surge_multiplier),
                    'temperature': float(temperature),
                    'apparenttemperature': float(apparentTemperature),
                    'precipintensity': precipIntensity,
                    'precipprobability': precipProbability,
                    'humidity': humidity,
                    'windspeed': windSpeed,
                    'windgust': windGust,
                    'visibility': visibility,
                    'temperaturehigh': temperatureHigh,
                    'temperaturelow': temperatureLow,
                    'apparenttemperaturehigh': apparentTemperatureHigh,
                    'apparenttemperaturelow': apparentTemperatureLow,
                    'dewpoint': dewPoint,
                    'pressure': pressure,
                    'windbearing': windBearing,
                    'cloudcover': cloudCover,
                    'uvindex': uvIndex,
                    'ozone': ozone,
                    'moonphase': moonPhase,
                    'precipintensitymax': precipIntensityMax,
                    'temperaturemin': temperatureMin,
                    'temperaturemax': temperatureMax,
                    'apparenttemperaturemin': apparentTemperatureMin,
                    'apparenttemperaturemax': apparentTemperatureMax,
                    'source_back_bay': 1 if source == "Back Bay" else 0,
                    'source_beacon_hill': 1 if source == "Beacon Hill" else 0,
                    'source_boston_university': 1 if source == "Boston University" else 0,
                    'source_fenway': 1 if source == "Fenway" else 0,
                    'source_financial_district': 1 if source == "Financial District" else 0,
                    'source_haymarket_square': 1 if source == "Haymarket Square" else 0,
                    'source_north_end': 1 if source == "North End" else 0,
                    'source_north_station': 1 if source == "North Station" else 0,
                    'source_northeastern_university': 1 if source == "Northeastern University" else 0,
                    'source_south_station': 1 if source == "South Station" else 0,
                    'source_theatre_district': 1 if source == "Theatre District" else 0,
                    'source_west_end': 1 if source == "West End" else 0,
                    'destination_back_bay': 1 if destination == "Back Bay" else 0,
                    'destination_beacon_hill': 1 if destination == "Beacon Hill" else 0,
                    'destination_boston_university': 1 if destination == "Boston University" else 0,
                    'destination_fenway': 1 if destination == "Fenway" else 0,
                    'destination_financial_district': 1 if destination == "Financial District" else 0,
                    'destination_haymarket_square': 1 if destination == "Haymarket Square" else 0,
                    'destination_north_end': 1 if destination == "North End" else 0,
                    'destination_north_station': 1 if destination == "North Station" else 0,
                    'destination_northeastern_university': 1 if destination == "Northeastern University" else 0,
                    'destination_south_station': 1 if destination == "South Station" else 0,
                    'destination_theatre_district': 1 if destination == "Theatre District" else 0,
                    'destination_west_end': 1 if destination == "West End" else 0,
                    'cab_type_lyft': 1 if cab == 'Lyft' else 0,
                    'cab_type_uber': 1 if cab == 'Uber' else 0,
                    'name_black': 1 if (cab == 'Uber') and ('Standard Luxury' == name) else 0,
                    'name_black_suv': 1 if (cab == 'Uber') and ('XL Luxury' == name) else 0,
                    'name_lux': 0,
                    'name_lux_black': 1 if (cab == 'Lyft') and ('Standard Luxury' == name) else 0,
                    'name_lux_black_xl': 1 if (cab == 'Lyft') and ('XL Luxury' == name) else 0,
                    'name_lyft': 1 if (cab == 'Lyft') and ('Standard' == name) else 0,
                    'name_lyft_xl': 1 if (cab == 'Lyft') and ('XL' == name) else 0,
                    'name_shared': 1 if (cab == 'Lyft') and ('Shared' == name) else 0,
                    'name_uberpool': 1 if (cab == 'Uber') and ('Shared' == name) else 0,
                    'name_uberx': 1 if (cab == 'Uber') and ('Standard' == name) else 0,
                    'name_uberxl': 1 if (cab == 'Uber') and ('XL' == name) else 0,
                    'name_wav': 1 if (cab == 'Uber') and ('Accessible' == name) else 0,
                    'weather_clear': 1 if short_summary == 'Clear' else 0,
                    'weather_drizzle': 1 if short_summary == 'Drizzle' else 0,
                    'weather_foggy': 1 if short_summary == 'Foggy' else 0,
                    'weather_light_rain': 1 if short_summary == 'Light Rain' else 0,
                    'weather_mostly_cloudy': 1 if short_summary == 'Mostly Cloudy' else 0,
                    'weather_overcast': 1 if short_summary == 'Overcast' else 0,
                    'weather_partly_cloudy': 1 if short_summary == 'Partly Cloudy' else 0,
                    'weather_possible_drizzle': 1 if short_summary == 'Possible Drizzle' else 0,
                    'weather_rain': 1 if short_summary == 'Rain' else 0}

                    data_list.append(test_dict)
# Convert list of dictionaries to DataFrame
test_data = pd.DataFrame(data_list)

uber_df = test_data.loc[:, ordered_columns]
uber_x = uber_df.to_numpy()

print(uber_x) 

#Okay now we predict on this test data! 
#### Accessing all stored models
model_zips = ['models/lyft_deep_models.zip',
                'models/lyft_reg_models.zip',
                'models/lyft_tree_models.zip',
                'models/uber_deep_models.zip',
                'models/uber_reg_models.zip',
                'models/uber_tree_models.zip']
lyft_models = ['lyft_ols_regression.sav', 'lyft_ridge_regression.sav', 'lyft_lasso_regression.sav',
                'lyft_decision_tree.sav', 'lyft_randomforest.sav', 'lyft_xgboost.sav',
                'lyft_neural_net.keras', 'lyft_deep_neural_net.keras']
uber_models = ['uber_ols_regression.sav', 'uber_ridge_regression.sav', 'uber_lasso_regression.sav',
                'uber_decision_tree.sav', 'uber_randomforest.sav', 'uber_xgboost.sav',
                'uber_neural_net.keras', 'uber_deep_neural_net.keras']
for model in model_zips:
    with ZipFile(model, 'r') as zip:
        zip.extractall()


uber_models_obj = {}
for u_model in uber_models:
    if os.path.exists(u_model):
        if ".sav" in u_model:  #pickled model
            model = pickle.load(open(u_model, 'rb'))
            
        model_name = " ".join([x.capitalize() for x in u_model.split(".")[0].split("_")])
        uber_models_obj[model_name] = model
        os.remove(u_model) 
    else:
        print(f"{u_model} does not exist")
print(uber_models_obj)

if os.path.exists("keep.txt"):
    os.remove("keep.txt")


uber_predictions = {}
for model in uber_models_obj:
    prediction = uber_models_obj[model].predict(uber_x)
    uber_predictions[model] = prediction
uber_pred_df = pd.DataFrame.from_dict(data=uber_predictions,
                                        orient='index').reset_index().rename(columns={'index': 'Model',
                                                                                    0: 'Price'})
# Drop last two rows and first column
df_trimmed = uber_pred_df.iloc[:-2, 1:]

# Calculate the average of each column
average_predicted_prices = df_trimmed.mean().tolist()
uber_df['predicted_price'] = average_predicted_prices 
uber_df['predicted_price_per_mile'] = uber_df['predicted_price'] / uber_df['distance']
uber_df.to_csv('q_learning/uber_qlearning_data.csv', index=False)  # Set index=False to exclude row indices from the output

print(uber_df)
