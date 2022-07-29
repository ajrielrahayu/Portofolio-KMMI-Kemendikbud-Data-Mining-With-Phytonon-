# Pada tugas kali ini saya mencoba memlakukan klasterisasi senior dan junior dengan atribut tinggi badan dan ukuran sepatu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import data

#melakukan pemanggilan data
dataset = pd.read_csv('Documents/KMMI/seniorsandjunior.csv')
#print(dataset.head())
dataset['kelas'] = pd.factorize(dataset.kelas)[0]
x = dataset.iloc[:, [0,1]].values
y = dataset.iloc[:,-1].values

#Mennyederhanakan nilai atribut
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

#melakukan pembagian data train dan test
from sklearn .model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x, y , test_size=0.25, random_state=0)

#melakukan pemodelan dengan klasifikasi KNN
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski',p=2)
classifier.fit(x_train,y_train)

from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(x_test)
cm =confusion_matrix(y_test,y_pred)

#melakukan visualisasi data
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop =x_set[:,0].max()+1,step=0.01),
                    np.arange(start = x_set[:,1].min()-1,stop =x_set[:,0].max()+1,step=0.01))

plt.contourf(x1,x2,classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))      
plt.xlim(x1.min(),x1.max())
plt.ylim(x2.min(),x2.max())
for i,j in enumerate (np.unique(y_test)):
    plt.scatter(x_set[y_set==j ,0], x_set[y_set==j,1], c = ListedColormap(('red','green'))(i),label=j)
plt.title('Klasterisasi penentuan senior dan junior dengan KNN')
plt.xlabel('ukuran sepatu')
plt.ylabel('tinggi badan')
plt.legend()
plt.show()