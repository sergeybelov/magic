# -*- coding: utf-8 -*-
import pandas as pd

from sklearn.preprocessing import StandardScaler
import time
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import hstack
import datetime as dt

import numpy as np
#from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn import metrics
import matplotlib.pyplot as plt

#https://habrahabr.ru/company/ods/blog/327242/

df=pd.read_pickle('MG_Sales_customer.pickle',compression='gzip')

#выбираем покупателей для дальнейшего анализа
sales_sum=df.groupby('Покупатель')['Количество'].sum()
#выкидываем со слишком большими продажами (сводные карты) и тех кто купил один раз
sales_sum.drop(sales_sum[(sales_sum>133)|(sales_sum==1)].index, inplace=True)
customers_name=list(sales_sum.index)
del sales_sum

#делаем выборку
select=df.loc[(df['Покупатель'].isin(customers_name))&(df['Дата']>=(dt.datetime(2014,1,1))),['Покупатель','ПокупательПол','ПокупательДатаРождения','ВидИзделия','ПодвидИзделия','СтильДизайна','ВидДизайна','ОсновнойКамень','ГруппаТовара','Коллекция','ЦветМеталла','ТоварСреднийВес','Размер','Вес']]
del customers_name
del df

#Подготовка датасета
#ЦветМеталла=list(map(lambda xx: xx,list(select['ЦветМеталла'].unique())))
def codeMetall(_str):    
    for str_split in _str.lower().split():
        if str_split=='серебро': return 0
        if str_split=='золото': return 10
        if str_split=='зол.': return 11
        if str_split=='платина': return 20
        if str_split=='сплав': return -10
    return -20

select['ПокупательПолКод']=select['ПокупательПол'].map(lambda xx: {'Ж':0, 'М':1, '<Неопределено>':None}[xx])
select['ЦветМеталлаКод']=select['ЦветМеталла'].map(lambda xx: codeMetall(xx))
select['ПокупательПолКод'].fillna(select['ПокупательПолКод'].median(),inplace=True)
select['ПокупательГодРождения']=select['ПокупательДатаРождения'].dt.year
select['ПокупательГодРождения']=select['ПокупательГодРождения'].map(lambda xx: None if xx<1900 else xx)
select['ПокупательГодРождения']=select['ПокупательГодРождения'].map(lambda xx: None if xx>2010 else xx)
select['ПокупательГодРождения'].fillna(select['ПокупательГодРождения'].median(),inplace=True)


sel=select.drop(['ПокупательДатаРождения','ПокупательПол','ЦветМеталла'],  axis=1)
#выборка колонок
numerical_columns = [c for c in sel.columns if sel[c].dtype.name != 'object']
categorial_columns = [c for c in sel.columns if sel[c].dtype.name == 'object']


#Dummy-кодирование и шкалируем
lb_style = LabelBinarizer(sparse_output=True)
concList=[]
for col in categorial_columns:
    concList.append(lb_style.fit_transform(sel[col]))    
concList.append(StandardScaler().fit_transform(sel[numerical_columns]))#добавляем шклированные значения числовых переменных
X=hstack(concList)

del sel
del concList
print('shape ',X.shape)
print('Prepare finished')


##############################################################################
#кластеризуем
#==============================================================================
# https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
#==============================================================================
# Compute K-Means
def compKMeans(max_clusters,batch_size):
    rg=range(3, max_clusters)
    if __name__ == '__main__':
        inertia = []
        for k in rg:
            hdb_t1 = time.time()
            hdb = MiniBatchKMeans(n_clusters=k,max_iter=500,init_size=10 * k, n_init=15,max_no_improvement=50,batch_size=batch_size,random_state=17).fit(X)
            #hdb = KMeans(n_clusters=k,n_jobs=15,max_iter=100,n_init=2,precompute_distances=True,verbose=3,random_state=17).fit(X)
            hdb_labels = hdb.labels_
            hdb_elapsed_time = time.time() - hdb_t1
            
            inertia.append(np.sqrt(hdb.inertia_))
            
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)
    
            print('\n\n++ KMeans Results')
            print('n_clusters: %d' % k)
            print('Estimated number of clusters: %d' % n_clusters_hdb_)
            print('Elapsed time to cluster: %.4f s' % hdb_elapsed_time)
            #print('Silhouette Coefficient: %0.3f'
             #     % metrics.silhouette_score(X, hdb_labels,random_state=17))
    
    from pylab import rcParams
    rcParams['figure.figsize'] = 14, 8
    plt.plot(rg, inertia, marker='s');
    plt.xlabel('$k$')
    plt.ylabel('$J(C_k)$')
    
    return hdb
        
    
hdb=compKMeans(20,15)

0/0

hdb_t1 = time.time()
hdb = MiniBatchKMeans(n_clusters=8,max_iter=500,n_init=20,max_no_improvement=50,batch_size=15,random_state=17).fit(X)
#hdb = KMeans(n_clusters=k,n_jobs=15,max_iter=100,n_init=2,precompute_distances=True,verbose=3,random_state=17).fit(X)
hdb_labels = hdb.labels_
hdb_elapsed_time = time.time() - hdb_t1


# Number of clusters in labels, ignoring noise if present.
n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)

print('Estimated number of clusters: %d' % n_clusters_hdb_)
print('Elapsed time to cluster: %.4f s' % hdb_elapsed_time)

cluster0=select.loc[hdb_labels==0]