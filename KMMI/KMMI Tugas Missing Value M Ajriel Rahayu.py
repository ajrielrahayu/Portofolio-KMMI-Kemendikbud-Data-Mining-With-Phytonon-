#Tugas mmembersihkan data dengan python

#menginput liblary yang akan dipakai
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from sklearn import impute

#Membaca data dalam file
dataset = pd.read_csv('Documents/KMMI/Data_Depresi_untuk_preprosesing.csv', delimiter=';')
#memisahkan label dan atribut
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
#mengisi data kosong dengan nilai rata-rata pada atribut
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
#Mengembalikan format data matriks menjadi Dataframe
a = pd.DataFrame(x,columns=['Pendidikan', 'Pendapatan', 'Umur',])
b = pd.DataFrame(y,columns=['Kemungkinan depresi'])
data1 = pd.merge(a,b, left_index=True, right_index= True)
data_clean = data1.astype({"Pendidikan": str, "Pendapatan": int, 'Umur': int, 'Kemungkinan depresi': str})
print(data_clean)