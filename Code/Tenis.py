# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:05:09 2024

@author: sarib
"""

import pandas as pd
import numpy as np

veriler = pd.read_csv("odev_tenis.csv")
print(veriler)

# sayısal olmayan verileri sayılsal veriye çevirmemiz lazım 
# eğer bir veri sutununda sadece 2 olasılık varsa labelencoder yeterli olur bu (1 sutunu sadece 1 ve 0 a çevirir)
# eğer 2 en fazla olasılık varsa onehotencoder kullanmamız lazım o da olması gereken sutunu 1 yapar diğer ilgili sutunları 0 yapar

# bu veri listesinde windy ve play sutunları 2 olasılıklı olduğu için labelencoder yapacağız

play = veriler.iloc[:,-1:].values
print(play)

from sklearn import preprocessing

# play sutununu labelencoder ile sayısal veriye dönüştürdük
le = preprocessing.LabelEncoder()
play[:,-1] = le.fit_transform(veriler.iloc[:,-1])
print(play)

# hemen alt kısımda ise direkt kısa yoldan bütün veri sutunlarını labelencoder uygulayabileceğimiz bir kısa yol var
veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)
# bu şekilde bütün sutunlara labelencoder uyguladık ama bütün satırlara labelencoder uygulamamız lazım
# o yüzden sadece ihtiyacımız olan sutunları alacağız veriler2 kısmından

# onehotencoder uyguladık outlook sutununa
c = veriler2.iloc[:,:1]
ohe = preprocessing.OneHotEncoder()
c = ohe.fit_transform(c).toarray()

# şimdi ise verileri birleştirme yapmamız lazım
havadurumu = pd.DataFrame(data = c , index=range(14), columns=['overcast','rainy','sunny'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]], axis=1)

sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1) # sodan 2. sutundan son sutuna kadar al demek -2 li kısım

# model eğitimi

# buradaki verilerde humidity bağımlı değişken , diğerleri bağımsız değişken oalrak alıyoruz ve ona göre işlem yapıyoruz
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test) # tahmin
# buradaki tahmin çok tutarlı olmayabilir o yüzden backward elimation ile alakasız sutunları çıkaracağoz
print(y_pred)

import statsmodels.api as sm # ifadesi, Python'da istatistiksel modeller oluşturmak ve istatistiksel veri analizi yapmak için kullanılan

X = np.append(arr = np.ones((14,1)).astype(int),values=sonveriler.iloc[:,:-1],axis=1)

x_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
x_l = np.array(x_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],x_l).fit()
print(model.summary()) # istatistik özetini çıkarma

# p değeri en büyük olan 1. sutunu çıkartıyoruz
sonveriler = sonveriler.iloc[:,1:]

x_l = sonveriler.iloc[:,[0,1,2,3,4]].values
x_l = np.array(x_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],x_l).fit()
print(model.summary()) # istatistik özetini çıkarma


# yeni tahminde bulunmak için x test ve train kısmındaki gereksiz olan windy sutununu çıkartıyoruz
x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test) # yeni tahmin değerleri
