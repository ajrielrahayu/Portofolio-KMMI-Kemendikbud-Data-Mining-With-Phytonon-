#menyiapkan liblary yang dibutuhkan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import data

#membuka dataset 
dataset = pd.read_csv('Documents/KMMI/INTELLIGENTIRRIGATIONSYSTEM.csv')
#print(dataset.head())

#pisahkan features dan target
x = dataset.iloc[:,[1,2]].values
y = dataset.iloc[:,-1].values

# melakukan scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
x = sc.fit_transform(x)
#print(x)

#memisahkan data train dan data test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25,random_state= 1)
#print("data features train:", x_train)
#print("data features test:", x_test)

#melakukan pemodelan Gaussian NB
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

classifier = GaussianNB()
classifier.fit(x_train, y_train)

#confussion matrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

y_pred = classifier.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("confunssion matriks")
print(cm)
akurasi = classification_report(y_test, y_pred)
print("tingkat akurasi naive bayes")
print("akurasi: ", akurasi)
akurasi = accuracy_score(y_test, y_pred)
print("tingkat akurasi dalam persen", (akurasi*100))

#melakukan visualisasi data
from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1,x2 = np.meshgrid(np.arange(start = x_set[:,0].min()-1,stop = x_set[:,0].max()+1, step = 0.01),
                    np.arange(start = x_set[:,1].min()-1,stop = x_set[:,0].max()+1, step = 0.01))

plt.contour(x1,x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape), alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim(x1.min(), x1.max())
plt.xlim(x2.min(), x2.max())

for i,j in enumerate(np.unique(y_test)):
    plt.scatter(x_set[y_set==j, 0], x_set[y_set==j, 1], c = ListedColormap(('red', 'green'))(1), label = j)
plt.title("klasifikasi dengan naive bayes")
plt.xlabel("kelembaban")
plt.ylabel("suhu")
plt.legend()
plt.show()