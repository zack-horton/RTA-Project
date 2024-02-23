import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# read in data
df = pd.read_csv('uber_lyft_file.zip')

# drop certain columns
df = df.drop(['id', 'timestamp', 'datetime', 'timezone', 'product_id', 'latitude', 'longitude', 
              'long_summary', 'windGustTime', 'temperatureHighTime', 'temperatureLowTime',
              'apparentTemperatureHighTime', 'apparentTemperatureLowTime', 'icon', 'sunriseTime',
              'sunsetTime', 'uvIndexTime', 'temperatureMinTime', 'temperatureMaxTime',
              'apparentTemperatureMinTime', 'apparentTemperatureMaxTime'],
             axis=1)

# begin one-hot-encoding certain columns
ohe_columns = ['source', 'destination', 'cab_type', 'name', 'short_summary']
for col in ohe_columns:
    df.loc[:, col] = df[col].str.lower().str.strip()  #adjust column values to be lower case and w/o whitespace

df = df.rename(columns={'short_summary': 'weather',
                        'visibility.1': 'visibility'})  #short_summary doesn't make as much sense; remove ".1" from visibility
agg_df = df.groupby('name', as_index=False) \
           .agg(count_price=('price', 'count'))

plt.bar(x=agg_df['name'], height=agg_df['count_price'])
plt.show()

# confirmed that name == 'taxi' is the only time that `price` is NA