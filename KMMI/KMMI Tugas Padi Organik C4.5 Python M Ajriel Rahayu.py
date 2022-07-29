#Menyiapkan liblary yang dibutuhkan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Menyiapkan data yang dipakai
df = pd.read_excel('Documents/KMMI/Data Mutu Padi Organik.xls', header= 2).dropna()

#Menginisiliasi data karena masih menggunakan string
df = df.apply(lambda x: pd.factorize(x)[0])

#memisahkan atribut dan juga labelnya
features = df.iloc[:,:-1].values
target = df.iloc[:,-1].values

#Membagi data training dan juga data testing
from sklearn .model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(features, target , test_size=0.25, random_state=0)
#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

#melakukan pemodelan menggunakan algoritma C4.5
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
model=tree.DecisionTreeClassifier(random_state=0, max_depth=None, min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0,max_leaf_nodes=None,min_impurity_decrease=0)
clf = model.fit(x_train, y_train)

#confution matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score 

y_pred = model.predict(x_test)
cm =confusion_matrix(y_test,y_pred)
print("confusion matrix")
print(cm)
akurasi=classification_report(y_test,y_pred)
print("tingkat akurasi algoritma C4.5")
print("Akurasi :", akurasi)
akurasi=accuracy_score(y_test,y_pred) 
print("Tingkat Akurasi :%d persen"%(akurasi*100))

#Melakukan visualisasi data
import graphviz
dot_data = tree.export_graphviz(model, out_file=None,filled = True)
graph = graphviz.Source(dot_data)
graph.view()

