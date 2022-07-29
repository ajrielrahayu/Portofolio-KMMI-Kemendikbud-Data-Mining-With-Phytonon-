#Pada tugas kali ini saya membuat data mengenai sistem irigasi otomatis, dengan melihat faktor suhu dan juga kelembaban, 
#hasil akhir yang akan didapat yaitu dapat memprediksi pada suhu dan kelembaban berapa irigasi akan membuka(1) atau menutup(0)

#Menyiapkan liblary yang dibutuhkan 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

#membuka dataset
dataset = pd.read_csv('Documents/KMMI/INTELLIGENTIRRIGATIONSYSTEM.csv')
#print(dataset)
x = dataset.iloc[:,[1,2]].values
y = dataset.iloc[:,-1].values

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
print("tingkat akurasi algoritma naive bayes")
print("Akurasi :", akurasi)
akurasi=accuracy_score(y_test,y_pred) 
print("Tingkat Akurasi :%d persen"%(akurasi*100))

x_set, y_set = x_train, y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop =x_set[:,0].max()+1,step=0.01),
                    np.arange(start = x_set[:,1].min()-1,stop =x_set[:,0].max()+1,step=0.01))

#melakukan visualisasi pada data

plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))      
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())

for i,j in enumerate (np.unique(y_test)):
    plt.scatter(x_set[y_set==j ,0], x_set[y_set==j,1], c = ListedColormap(('red','green'))(i),label=j)
plt.title('Klasifikasi Data dengan Naivebayes')
plt.xlabel('Kelembaban')
plt.ylabel('suhu')
plt.legend()
plt.show()
