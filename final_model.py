#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer

print "loading data"
AC_data = pd.read_excel("../PI/AC_transposed.xlsx")
orp_data = pd.read_excel("../PI/orp.xlsx")

orp_data['timestamp'] = pd.to_datetime(orp_data['timestamp'])

#'Autoclave Feed',
dep = ['Autoclave Feed S',
       'Autoclave utilisation','Autoclave feed density',
       'Sulphur Feed','Oxidation rate','Oxygen Used',
       'Oxygen : Sulphur ratio','Ave autoclave vent opening'
       ]

X = AC_data[dep].iloc[:]
y = pd.DataFrame()
y['Autoclave throughput '] = AC_data['Autoclave throughput ']

d = {}
for day in range(1,367):
    d[day] = []

for i in range(orp_data.shape[0]):
    day = orp_data['timestamp'][i].dayofyear
    d[day].append(orp_data['PPO:L-ACD_ORP_12HR'][i])

day_orp = []
for i in d.itervalues():
    day_orp.append(np.array(i).mean())
    
y['PPO:L-ACD_ORP_12HR'] = day_orp



"""
#Check for NaN
columns = X.columns
for column in columns:
#    print X[column].index[X[column].apply(np.nan)]
    print column,pd.isnull(X[column]).nonzero()[0]

columns = y.columns
for column in columns:
    print column,pd.isnull(y[column]).nonzero()[0]
"""

imputer = Imputer()
X = imputer.fit_transform(X)
imputer = Imputer()
y = imputer.fit_transform(y)





all_data = pd.read_excel("../PI/cleaned/good_columns_edited.xlsx")
columns = all_data.columns
good_columns = []

for column in columns:
    try:
        c = all_data[column].astype(float)
        good_columns.append(column)
    except:
        pass
    
from sklearn.decomposition import PCA

imputer = Imputer(missing_values = 'NaN',
                  strategy = 'mean',
                  axis = 0)

imputer = imputer.fit(all_data[good_columns])

good_cols = imputer.transform(all_data[good_columns])

print "Performing PCA"
n_comp = 3
pca = PCA(n_components = n_comp)
pca.fit(good_cols)
p = pca.transform(good_cols)

date_5min = pd.to_datetime(all_data['timestamp'])
"""
i = 1
ii = 1
i2 = 0
tmp = []

pca_daily = np.zeros((5,366))


while True:
    if date_5min[ii].dayofyear == i:
        tmp.append(ii)
        ii+=1
    else:
        pca_daily[tmp] = p[i2:i].mean(axis=0)
        i2 = ii
        if i == 366:
            break
        i+=1
"""

pca_daily = []
day = 1
i5min = 0
iold = 0

while i5min<p.shape[0]:
    if date_5min[i5min].dayofyear == day+1:
        pca_daily.append(list(p[iold:i5min].mean(axis=0)))
        iold = i5min
        day+=1 
    else:
        i5min += 1
print "PCA done"
pca_daily.append(list(p[iold:].mean(axis=0)))

pca_daily = np.array(pca_daily)
X_new = np.column_stack((X,pca_daily))


"""
for i in range(n_comp):
    X_new['pca'+str(i+1)] = pca_daily[:,i]
    dep.append('pca'+str(i+1))
"""


imputer = Imputer()
X_new = imputer.fit_transform(X_new)
imputer = Imputer()
y = imputer.fit_transform(y)

mask = y[:,0] != 0

X_new = X_new[mask]
y = y[mask]

print "Fitting with random forest"

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new,y,test_size=0.2)

regressor = RandomForestRegressor(n_estimators=500,n_jobs=4,oob_score=True,
                                  max_features=0.33)
regressor.fit(X_train,y_train)

f = -regressor.feature_importances_
ranks = f.argsort()

for i in range(n_comp):
    dep.append("PCA"+str(i+1))

for i in ranks:
    print np.array(dep)[i],-f[i]

print regressor.oob_score_


save_xtest = np.hstack((X_test,y_test,regressor.predict(X_test)))
np.savetxt("../doc/xtest",save_xtest)


plt.figure(1)
plt.plot(y_test[:,0],"-o",label = "Actual Throughput")
plt.plot(regressor.predict(X_test)[:,0],"o-", label = "Predicted Throughput")
plt.legend()
plt.xlabel("Timestamp (days)",size=18)
plt.ylabel("Autoclave Throughput (t/h/ac)",size=18)

plt.figure(2)
plt.plot(y_test[:,1],"-o",label = "Actual ORP")
plt.plot(regressor.predict(X_test)[:,1],"o-", label = "Predicted ORP")
plt.legend()
plt.xlabel("Timestamp (days)",size=18)
plt.ylabel("Oxidation Reduction Potential (mV)",size=18)
plt.show()