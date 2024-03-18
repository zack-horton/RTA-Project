import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from zipfile import ZipFile
from datetime import datetime

q_data_raw = pd.read_csv('q_learning/uber_qlearning_data.csv')

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
                    test_dict = {'source': source,
                                'destination':destination,
                                'hour':hour,
                                'weather':short_summary,
                                'day': day
                            }
                    data_list.append(test_dict)
# Convert list of dictionaries to DataFrame
q_data = pd.DataFrame(data_list)
q_data['price_per_mile'] = q_data_raw['predicted_price_per_mile']


all_q_values = pd.DataFrame()



for hour in hours:
    for short_summary in short_summaries:
        for day in days:
            q_data_limited = q_data[(q_data['hour'] == hour) & (q_data['weather'] == short_summary) & (q_data['day'] == day) ]



            pivot_table = q_data_limited.pivot(index='source', columns='destination', values='price_per_mile')
            # Replace NaN values with 0
            pivot_table.fillna(0, inplace=True)
            trip_fare = pivot_table.values

            #number of episode we will run
            n_episodes = 10000

            n_states = 12
            n_actions = 12

            #maximum of iteration per episode
            max_iter_episode = 100

            #initialize the exploration probability to 1
            exploration_proba = 1

            #exploartion decreasing decay for exponential decreasing
            exploration_decreasing_decay = 0.001

            # minimum of exploration proba
            min_exploration_proba = 0.01

            #discounted factor
            gamma = 0.5

            #learning rate
            lr = 0.001

            # Define the state transition 
            def move_state(trip_fare, current_state, action):
                reward = trip_fare[current_state, action] + np.random.normal()
                return action, reward

            # Initialize the Q-table
            Q_table = np.ones((n_states, n_actions)) * 10
            # will not gain anything if stay at the same place
            for i in range(12):
                for j in range(12):
                    if trip_fare[i][j] == 0:
                        Q_table[i][j] = 0
            

            rewards_per_episode = []
            states_per_episode = []

            np.random.seed(100)

            # We iterate over episodes
            for e in range(n_episodes):
                
                # We initialize the first state of the episode
                current_state = np.random.randint(3)
                states_per_episode.append([current_state])
                
                # Sum the rewards that the agent gets from the environment
                total_episode_reward = 0
                
                for i in range(max_iter_episode): 
                    
                    # Determine if we would like to explore or exploit
                    if np.random.uniform(0,1) < exploration_proba:
                        action = np.random.choice([state for state in range(n_states) if state != current_state])
                    else:
                        action = np.argmax(Q_table[current_state,:])

                    # The environment runs the chosen action and returns the next state and a reward
                    next_state, reward = move_state(trip_fare, current_state, action)
                    
                    # We update our Q-table using the Q-learning iteration
                    target_Q = reward + gamma*max(Q_table[next_state,:])
                    error = Q_table[current_state, action] - target_Q
                    Q_table[current_state, action] = Q_table[current_state, action] - lr * error                            
                    
                    total_episode_reward = total_episode_reward + reward
                    
                    current_state = next_state
                    
                    states_per_episode[-1].append(current_state)
                    
                # We update the exploration proba using exponential decay formula 
                exploration_proba = max(min_exploration_proba, np.exp(-exploration_decreasing_decay*e))
                rewards_per_episode.append(total_episode_reward)

            labels = ["Back Bay",
                    "Beacon Hill",
                    "Boston University",
                    "Fenway",
                    "Financial District",
                    "Haymarket Square",
                    "North End",
                    "North Station",
                    "Northeastern University",
                    "South Station",
                    "Theatre District",
                    "West End"
                                ]


            Q_matrix = Q_table.reshape(12, 12)

            for i in range(12):
                for j in range(12):
                    if trip_fare[i][j] == 0:
                        Q_matrix[i][j] = 0

            final_df = pd.DataFrame(Q_matrix, columns=labels)
            final_df.insert(0,'Source', labels)
            #final_df.set_index(final_df.columns[0], inplace=True)
            
            melted_df = final_df.melt(id_vars='Source', var_name='Destination', value_name='Q')
            melted_df['hour'] = hour
            melted_df['weather'] = short_summary
            melted_df['day'] = day
            all_q_values = pd.concat([all_q_values, melted_df], ignore_index=True)




long_lat = [
    {'Location': "Back Bay", 'Start_Latitude' : 42.34950, 'Start_Longitude': -71.07950},
    {'Location': "Beacon Hill" , 'Start_Latitude': 42.35620, 'Start_Longitude':	-71.06940},
    {'Location': "Boston University", 'Start_Latitude': 42.35050,'Start_Longitude':	-71.10540},
    {'Location': "Fenway" , 'Start_Latitude': 42.34680,'Start_Longitude':	-71.09230},
    {'Location': "Financial District", 'Start_Latitude': 42.35590,'Start_Longitude':	-71.05500},
    {'Location': "Haymarket Square" , 'Start_Latitude': 42.36380,'Start_Longitude':	-71.05850},
    {'Location': "North End", 'Start_Latitude': 42.36520,'Start_Longitude':-71.05550},
    {'Location': "North Station", 'Start_Latitude': 42.36640,	'Start_Longitude':-71.06200},
    {'Location': "Northeastern University", 'Start_Latitude': 42.33990,'Start_Longitude':	-71.08990},
    {'Location': "South Station" , 'Start_Latitude': 42.35190,'Start_Longitude':	-71.05520},
    {'Location': "Theatre District", 'Start_Latitude': 42.35190,'Start_Longitude':	-71.06430},
    {'Location': "West End", 'Start_Latitude': 42.36520,'Start_Longitude':	-71.06410}
]
start_longlat_df = pd.DataFrame(long_lat)

long_lat = [
    {'Location': "Back Bay", 'End_Latitude' : 42.34950, 'End_Longitude': -71.07950},
    {'Location': "Beacon Hill" , 'End_Latitude': 42.35620, 'End_Longitude':	-71.06940},
    {'Location': "Boston University", 'End_Latitude': 42.35050,'End_Longitude':	-71.10540},
    {'Location': "Fenway" , 'End_Latitude': 42.34680,'End_Longitude':	-71.09230},
    {'Location': "Financial District", 'End_Latitude': 42.35590,'End_Longitude':	-71.05500},
    {'Location': "Haymarket Square" , 'End_Latitude': 42.36380,'End_Longitude':	-71.05850},
    {'Location': "North End", 'End_Latitude': 42.36520,'End_Longitude':-71.05550},
    {'Location': "North Station", 'End_Latitude': 42.36640,	'End_Longitude':-71.06200},
    {'Location': "Northeastern University", 'End_Latitude': 42.33990,'End_Longitude':	-71.08990},
    {'Location': "South Station" , 'End_Latitude': 42.35190,'End_Longitude':	-71.05520},
    {'Location': "Theatre District", 'End_Latitude': 42.35190,'End_Longitude':	-71.06430},
    {'Location': "West End", 'End_Latitude': 42.36520,'End_Longitude':	-71.06410}
]

end_longlat_df = pd.DataFrame(long_lat)


all_q_values = all_q_values.merge(start_longlat_df, left_on = 'Source', right_on='Location', how = 'left')
all_q_values = all_q_values.merge(end_longlat_df, left_on = 'Destination', right_on='Location', how = 'left')
all_q_values = all_q_values.drop(columns=['Location_x', 'Location_y'])
#print(all_q_values)
all_q_values.to_csv('data/final_q_values.csv', index=False)


