import os
import pandas as pd
import pickle
from zipfile import ZipFile
import matplotlib.pyplot as plt
from sklearn import tree

df = pd.read_parquet("data/lyft/lyft_train.parquet").drop(['price'], axis=1)
model_zips = ['models/lyft_tree_models.zip', 'models/uber_tree_models.zip']

for model in model_zips:
        with ZipFile(model, 'r') as zip:
            zip.extractall()
            
lyft_dt = 'lyft_decision_tree.sav'
uber_dt = 'uber_decision_tree.sav'

lyft_tree = pickle.load(open(lyft_dt, 'rb'))
uber_tree = pickle.load(open(uber_dt, 'rb'))


text_representation = tree.export_text(decision_tree=lyft_tree,
                                       feature_names = df.columns)
print(text_representation)

text_representation = tree.export_text(decision_tree=uber_tree,
                                       feature_names=df.columns)
print(text_representation)