# Pada tugas kali ini saya akan menggunakan pemodelan data dengan menggunakan  naive bayes
# yang hasil outputnya adalah klasifikasi  dari data penipuan BPJS 

#Menyiapkan liblary yang akan dipakai
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Documents/KMMI/fraud_detection_train.csv')
#print(df.head())
#print(df.dtypes)

#Mengubah data yang masih menggunakan string menjadi float atau menginisialisasi data
df['jkpst'] = pd.factorize(df.jkpst)[0]
df['typeppk'] = pd.factorize(df.typeppk)[0]
df['cmg'] = pd.factorize(df.cmg)[0]
df['diagprimer'] = pd.factorize(df.diagprimer)[0]

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

#melakukan scaling pada data supaya tingkat akurasi meningkat 
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x = sc.fit_transform(x)
#print(x)

#memisahkan data training dan juga data test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state= 1)

#melakukan pemodelan Gaussian Naive Bayes pada data
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)


#confution matrix
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score 

y_pred = classifier.predict(x_test)
cm =confusion_matrix(y_test,y_pred)
print("confusion matrix")
print(cm)
akurasi=classification_report(y_test,y_pred)
print("tingkat akurasi algoritma Naive bayes")
print("Akurasi :", akurasi)
akurasi=accuracy_score(y_test,y_pred) 
print("Tingkat Akurasi :%d persen"%(akurasi*100))
