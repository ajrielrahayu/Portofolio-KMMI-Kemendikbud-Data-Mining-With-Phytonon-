
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Membaca data
df = pd.read_csv('Documents/Data/drug200.csv')

#memeriksa Missing value
#print(df.isna().sum())

#menginisialisasi data pada atribut karena berbentuk string
df['BP'] = pd.factorize(df.BP)[0]
df['Sex'] = pd.factorize(df.Sex)[0]
df['Cholesterol'] = pd.factorize(df.Cholesterol)[0]

#memisahkan atribut dan juga labelnya
features = df.iloc[:,:-1].values
target = df.iloc[:,-1].values

#melakukan pemodelan menggunakan algoritma C4.5
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
model=tree.DecisionTreeClassifier(random_state=0, max_depth=None, min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0,max_leaf_nodes=None,min_impurity_decrease=0)
clf = model.fit(features, target)

#Melakukan visualisasi data
import graphviz
dot_data = tree.export_graphviz(model, out_file=None,filled = True)
graph = graphviz.Source(dot_data)
graph.view()
