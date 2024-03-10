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
            all_q_values = pd.concat([all_q_values, melted_df], ignore_index=True)




long_lat_dict = {
    "Back Bay" : [42.34950,	-71.07950],
    "Beacon Hill" : [42.35620,	-71.06940],
    "Boston University": [42.35050,	-71.10540],
    "Fenway" : [42.34680,	-71.09230],
    "Financial District": [42.35590,	-71.05500],
    "Haymarket Square": [42.36380,	-71.05850],
    "North End": [42.36520,-71.05550],
    "North Station": [42.36640,	-71.06200],
    "Northeastern University": [42.33990,	-71.08990],
    "South Station": [42.35190,	-71.05520],
    "Theatre District": [42.35190,	-71.06430],
    "West End": [42.36520,	-71.06410]
}

print(long_lat_dict.map('Back Bay'))
#all_q_values.to_csv('final_q_values.csv', index=False)











            '''
            import pandas as pd
            import networkx as nx
            import matplotlib.pyplot as plt

            # Assuming df is your DataFrame with 12 sources and destinations
            # Assuming the columns represent destinations and rows represent sources
            # Also, assuming the first column is already set as the index

            # Create an empty graph

            # Create an empty graph
            G = nx.Graph()

            # Add nodes (sources and destinations) to the graph
            nodes = df.index.tolist() + df.columns.tolist()  # All sources and destinations
            G.add_nodes_from(nodes)

            # Iterate through each combination of source and destination
            for source in df.index:
                for destination in df.columns:
                    # Add an edge between source and destination with weight equal to the value in the DataFrame
                    weight = df.loc[source, destination]
                    if not pd.isnull(weight) and weight >= 0:  # Skip if weight is NaN or less than 0
                        G.add_edge(source, destination, weight=weight)

            # Position nodes using a spring layout
            pos = nx.spring_layout(G)

            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=500)

            # Draw edges
            nx.draw_networkx_edges(G, pos)

            # Draw labels for edges with weights (rounded to one decimal place)
            edge_labels = {(u, v): f"{d['weight']:.1f}" for u, v, d in G.edges(data=True) if d['weight'] >= 0}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

            # Draw labels for nodes
            nx.draw_networkx_labels(G, pos)

            # Display the plot
            plt.axis('off')
            plt.show()
'''
