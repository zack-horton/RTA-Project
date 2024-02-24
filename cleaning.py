import pandas as pd

# read in data
df = pd.read_csv('uber_lyft_file.zip')
df = df.loc[df['name'].str.lower().str.strip() != 'taxi']  #taxi rides don't include price

# drop certain columns
df = df.drop(['id', 'timestamp', 'datetime', 'timezone', 'product_id', 'latitude', 'longitude', 
              'long_summary', 'windGustTime', 'temperatureHighTime', 'temperatureLowTime',
              'apparentTemperatureHighTime', 'apparentTemperatureLowTime', 'icon', 'visibility.1', 'sunriseTime',
              'sunsetTime', 'uvIndexTime', 'temperatureMinTime', 'temperatureMaxTime',
              'apparentTemperatureMinTime', 'apparentTemperatureMaxTime'],
             axis=1)

# begin one-hot-encoding certain columns
ohe_columns = ['source', 'destination', 'cab_type', 'name', 'short_summary']
for col in ohe_columns:
    df.loc[:, col] = df[col].str.lower().str.strip()  #adjust column values to be lower case and w/o whitespace

df = df.rename(columns={'short_summary': 'weather'})  #short_summary doesn't make as much sense
df_full = pd.get_dummies(df, columns=['source', 'destination', 'cab_type', 'name', 'weather'], dtype=int)  #ohe variable

# rename columns to be lowercase and not have spaces
for i in range(0, len(df_full.columns)):
    df_full = df_full.rename(columns={df_full.columns[i]: df_full.columns[i].strip().lower().replace(" ", "_")})

# ensure column names are good and no null values
print(df_full.info())

# write data to parquet files (one for lyft, one for uber)
lyft_df = df_full.loc[df_full['cab_type_lyft'] == 1]
lyft_df.to_parquet("data/lyft/lyft_full_data.parquet")

uber_df = df_full.loc[df_full['cab_type_uber'] == 1]
uber_df.to_parquet("data/uber/uber_full_data.parquet")

"""
Notes:
    - Appears that all rides occur in 2018
    - When `name` is 'taxi', price is missing so we filter them out

Columns that we should OHE:
    1. `source`: beginning location of ride
    2. `destination`: stopping location of ride
    3. `cab_type`: Uber or Lyft
    4. `name`: type of ride/service used for the ride
    5. `short_summary`: description of weather at time of ride
    
Columns that we should change datatype for:
    Everything should be int / float based on how pandas wants it

Columns that we can drop:
    1. `id`: identifer of ride/rider/driver (not sure)
    2. `timestamp`: we have the hour, day, month, and year for it
    3. `datetime`: we have the hour, day, month, and year for it
    4. `timezone`: America/New York for all rows
    5. `product_id`: same info as name column
    6. `latitude`: baked into source/destination columns
    7. `longitude`: baked into source/destination columns
    8. `long_summary`: same as short_summary
    9. `windGustTime`: weird time of weather??
    10. `temperatureHighTime`: weird time of weather??
    11. `temperatureLowTime`: weird time of weather??
    12. `apparentTemperatureHighTime`: weird time of weather??
    13. `apparentTemperatureLowTime`: weird time of weather??
    14. `icon`: same as short_summary
    15. `visibility.1`: same as visibility
    16. `sunriseTime`: shouldn't matter
    17. `sunsetTime`: shouldn't matter
    18. `uvIndexTime`: shouldn't matter
    19. `temperatureMinTime`: shouldn't matter
    20. `temperatureMaxTime`: shouldn't matter
    21. `apparentTemperatureMinTime`: shouldn't matter
    22. `apparentTemperatureMaxTime`: shouldn't matter
"""
