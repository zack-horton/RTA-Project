import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import pickle
from zipfile import ZipFile
from datetime import datetime

##### Init certain variables at the beginning #####
data = pd.read_csv('uber_lyft_file.zip')
data_f = pd.read_parquet('data/lyft/lyft_full_data.parquet')

now = datetime.now()
current_hour = datetime.strptime(now.strftime("%H"), "%H")
current_day = datetime.strptime(now.strftime("%d"), "%d")

location_options = ["Back Bay", "Beacon Hill", "Boston University", "Fenway", "Financial District", 
             "Haymarket Square", "North End", "North Station", "Northeastern University",
             "South Station", "Theatre District", "West End"]
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
weather_options = ["Clear", "Partly Cloudy", "Mostly Cloudy", "Overcast", "Possible Drizzle", 
                   "Drizzle", "Foggy", "Light Rain", "Rain"]
name_options = {"Shared": {'Lyft': ['Shared'],
                           'Uber': ['UberPool']},
                "Standard": {'Lyft': ['Lyft'],
                             'Uber': ['UberX']},
                "XL": {'Lyft': ['Lyft XL'],
                       'Uber': ['UberXL']},
                "Standard Luxury": {'Lyft': ['Lux', 'Lux Black'],
                                    'Uber': ['Black']},
                "XL Luxury": {'Lyft': ['Lux Black XL'],
                              'Uber': ['BlackSUV']},
                "Accessible": {'Uber': ['WAV']}}
month_options = {"November": 11,
                 "December": 12}
###################################################

# Title
st.title("Title")

# Header (not bolded)
st.header("Second line")

# Markdown
st.markdown("Hi there!")

# Success
st.success("Yay!, that was successful")

# Failure
st.error("Oops that was an error")

# Warning
st.warning("Warning, be aware")

# Info
st.info("Some info about the app")

# Write
a = 10
st.write(a)

page = st.sidebar.selectbox(label="Choose Page:",
                            options=["Riders", "Drivers"],
                            key="page_select",
                            placeholder="-")

if page == "Riders":
    st.sidebar.title("Rider Information")

    source = st.sidebar.selectbox(label="Pick-up",
                                options=location_options,
                                index=0,
                                key="source",
                                placeholder="-")
    destination = st.sidebar.selectbox(label="Drop-off",
                                    options=destination_options[source],
                                    index=0,
                                    key="destination",
                                    placeholder="-")

    month = st.sidebar.selectbox(label="Current Month",
                                options=month_options.keys(),
                                index=0,
                                key="month")

    start_time = st.sidebar.time_input(label="Pick-up Time", value=current_hour, step=60*60)
    short_summary = st.sidebar.selectbox(label="Current Weather",
                        options=weather_options,
                        index=0,
                        key="short_summary",
                        placeholder="-")
    name = st.sidebar.selectbox(label="Type of Ride",
                        options=name_options.keys(),
                        index=list(name_options.keys()).index("Standard"))

    check_prices = st.sidebar.button("Check Prices", type="primary", use_container_width=True)

    if check_prices:
        ############### Data Processing ###################
        ride_df = data.loc[(data['source'] == source) &
                        (data['destination'] == destination)]
        distance = ride_df[['distance']].median().values[0]
        surge_df = data.loc[(data['hour'] == int(start_time.strftime("%H")))]
        surge_multiplier = surge_df[['surge_multiplier']].mean().values[0]
        temp_df = data.loc[(data['hour'] == int(start_time.strftime("%H"))) &
                            (data['month'] == month_options[month])]
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
        print('Data loaded')

        cab_types = ['Lyft', 'Uber']
        ordered_columns = [x for x in data_f.columns if x != "price"]
        input_data = []
        for cab in cab_types:
            data = {'hour': int(start_time.strftime("%H")),
                    'day': int(current_day.strftime("%d")),
                    'month': int(month_options[month]),
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
            if cab == 'Lyft':
                lyft_df = pd.DataFrame.from_records(data, index=[0])
                lyft_df = lyft_df.loc[:, ordered_columns]
            else:
                uber_df = pd.DataFrame.from_records(data, index=[0])
                uber_df = uber_df.loc[:, ordered_columns]
            
        print("\nLyft data loaded:")
        print(len(lyft_df.columns))
        print("\nUber data loaded:")
        print(len(uber_df.columns))
        
        lyft_x = lyft_df.to_numpy()
        print(lyft_df[['cab_type_lyft', 'cab_type_uber']])
        
        uber_x = uber_df.to_numpy()
        print(uber_df[['cab_type_lyft', 'cab_type_uber']])
        
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
        
        lyft_models_obj = {}
        for l_model in lyft_models:
            if os.path.exists(l_model):
                if ".sav" in l_model:  #pickled model
                    model = pickle.load(open(l_model, 'rb'))
                else:  #keras model
                    model = keras.models.load_model(l_model)
                model_name = " ".join([x.capitalize() for x in l_model.split(".")[0].split("_")])
                lyft_models_obj[model_name] = model
                os.remove(l_model)
            else:
                print(f"{l_model} does not exist")
        print(lyft_models_obj)
        
        uber_models_obj = {}
        for u_model in uber_models:
            if os.path.exists(u_model):
                if ".sav" in u_model:  #pickled model
                    model = pickle.load(open(u_model, 'rb'))
                else:  #keras model
                    model = keras.models.load_model(u_model)
                    
                model_name = " ".join([x.capitalize() for x in u_model.split(".")[0].split("_")])
                uber_models_obj[model_name] = model
                os.remove(u_model) 
            else:
                print(f"{u_model} does not exist")
        print(uber_models_obj)
        
        if os.path.exists("keep.txt"):
            os.remove("keep.txt")
        
        #####
        
        # Dataframes to make predictions on are `lyft_x` and `uber_x`
        lyft_predictions = {}
        for model in lyft_models_obj:
            prediction = f"${round(float(lyft_models_obj[model].predict(lyft_x)), ndigits=2)}"
            lyft_predictions[model] = prediction
        lyft_pred_df = pd.DataFrame.from_dict(data=lyft_predictions,
                                            orient='index').reset_index().rename(columns={'index': 'Model',
                                                                                            0: 'Price'})
        
        uber_predictions = {}
        for model in uber_models_obj:
            prediction = f"${round(float(uber_models_obj[model].predict(uber_x)), ndigits=2)}"
            uber_predictions[model] = prediction
        uber_pred_df = pd.DataFrame.from_dict(data=uber_predictions,
                                            orient='index').reset_index().rename(columns={'index': 'Model',
                                                                                            0: 'Price'})
        
        lyft_col, uber_col = st.columns(2, gap="medium")
        lyft_col.header("Lyft Pricing:")
        lyft_col.data_editor(lyft_pred_df.sort_values(by='Price', ascending=False),
                            hide_index=True,
                            use_container_width=True,
                            disabled=lyft_pred_df.columns) 
        
        lyft_pred_df_floats = lyft_pred_df.copy(deep=True)
        lyft_pred_df_floats['Price'] = lyft_pred_df_floats['Price'].str.replace("$", "").astype(float)
        lyft_pred_df_floats = lyft_pred_df_floats.sort_values(by='Price', ascending=True)
        lyft_prices_plot = plt.figure()
        plt.barh(y=lyft_pred_df_floats['Model'],
                width=lyft_pred_df_floats['Price'])
        plt.xlim(0, lyft_pred_df_floats['Price'].max() + 0.25)
        plt.ylabel("Model")
        plt.xlabel("Price ($)")
        plt.title("Estimated Lyft Prices for Ride")
        plt.tight_layout()
        lyft_price_chart = lyft_col.pyplot(fig=lyft_prices_plot)

        uber_col.header("Uber Pricing:")
        uber_col.data_editor(uber_pred_df.sort_values(by='Price', ascending=False),
                            hide_index=True,
                            use_container_width=True,
                            disabled=uber_pred_df.columns)
        
        
        uber_pred_df_floats = uber_pred_df.copy(deep=True)
        uber_pred_df_floats['Price'] = uber_pred_df_floats['Price'].str.replace("$", "").astype(float)
        uber_pred_df_floats = uber_pred_df_floats.sort_values(by='Price', ascending=True)
        uber_prices_plot = plt.figure()
        plt.barh(y=uber_pred_df_floats['Model'],
                width=uber_pred_df_floats['Price'])
        plt.xlim(0, uber_pred_df_floats['Price'].max() + 0.25)
        plt.ylabel("Model")
        plt.xlabel("Price ($)")
        plt.title("Estimated Uber Prices for Ride")
        plt.tight_layout()
        uber_price_chart = uber_col.pyplot(fig=uber_prices_plot)

        
        lyft_col.write("Lyft Dataframe")
        lyft_col.data_editor(lyft_df.T.reset_index().rename(columns={'index':'Column',
                                                                    0: 'Value'}),
                    hide_index=True,
                    use_container_width=True,
                    disabled=['Columns', 'Value'])
        
        uber_col.write("Uber Dataframe")
        uber_col.data_editor(uber_df.T.reset_index().rename(columns={'index':'Column',
                                                                    0: 'Value'}),
                    hide_index=True,
                    use_container_width=True,
                    disabled=['Columns', 'Value'])
else:
    st.sidebar.title("Driver Information")
    