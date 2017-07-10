# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 16:16:40 2017

@author: tehn-11
"""

import pandas as pd

from sklearn.preprocessing import StandardScaler
import time
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import hstack
import datetime as dt

import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import ShuffleSplit

from pylab import rcParams
rcParams['figure.figsize'] = 35, 40

import matplotlib.cm as cm


df=pd.read_pickle('MG_Sales_customer.pickle',compression='gzip')
#---------------------------
#выбираем покупателей для дальнейшего анализа
sales_sum=df.groupby('Покупатель')['Количество'].sum()
#выкидываем со слишком большими продажами (сводные карты) и тех кто купил один раз
sales_sum.drop(sales_sum[(sales_sum>133)|(sales_sum==1)].index, inplace=True)
customers_name=list(sales_sum.index)
del sales_sum

#делаем выборку
select=df.loc[(df['Покупатель'].isin(customers_name))&(df['Дата']>=(dt.datetime(2014,1,1))),['Покупатель','ПокупательПол','ПокупательДатаРождения','ВидИзделия','ПодвидИзделия','СтильДизайна','ВидДизайна','ОсновнойКамень','ГруппаТовара','Коллекция','ЦветМеталла','ТоварСреднийВес','Размер','Вес','Количество']]
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
select['ПокупательГодРождения']=select['ПокупательГодРождения'].map(lambda xx: None if xx<1917 else xx)
select['ПокупательГодРождения']=select['ПокупательГодРождения'].map(lambda xx: None if xx>2010 else xx)
select['ПокупательГодРождения'].fillna(select['ПокупательГодРождения'].median(),inplace=True)
select.drop(['ПокупательДатаРождения','ПокупательПол','ЦветМеталла','ПокупательПолКод','ПокупательГодРождения'],  axis=1, inplace=True)
#выборка колонок
numerical_columns = [c for c in select.columns if select[c].dtype.name != 'object']
categorial_columns = [c for c in select.columns if select[c].dtype.name == 'object']


#Dummy-кодирование и шкалируем
lb_style = LabelBinarizer(sparse_output=True)
concList=[]
for col in categorial_columns:
    concList.append(lb_style.fit_transform(select[col]))    
concList.append(StandardScaler().fit_transform(select[numerical_columns]))#добавляем шклированные значения числовых переменных
X=hstack(concList)

del concList
print('shape ',X.shape)
print('Prepare finished')

#---------------------------

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=150, n_iter=5)
svd_representation = svd.fit_transform(X)
var1=np.cumsum(np.round(svd.explained_variance_ratio_, decimals=5)*100)
#plt.plot(var1[-50:])

#расчитываем оптимальное количество компонент
#более 90% дисперсии и шаг приращения каждой следующей компоненты <10^-4
optimal_n=np.intersect1d(np.argwhere(var1>90.),np.argwhere(svd.explained_variance_ratio_<=10**-4))[0]
print(optimal_n)#171

if optimal_n==None:
    raise 'Not enough n_components!'

svd = TruncatedSVD(n_components=optimal_n, n_iter=7)
svd_representation = svd.fit_transform(X)
print('reduced')
del var1

#0/0
#---------------------------
del X
del select

#---------------------------
#from sklearn.model_selection import ShuffleSplit
X=svd_representation

def getPlt(start, finish, folds):
    #подбор оптимального количества точек
    ss = ShuffleSplit(n_splits=folds, train_size=int(svd_representation.shape[0]*.2))
    subs=list(ss.split(svd_representation))
    
    print('MiniBatchKMeans starts.')
    for n_clusters in np.arange(start,finish,1):    
        print('n_clusters=',n_clusters)
            
        hdb_t1 = time.time()
        hdb = MiniBatchKMeans(n_clusters=int(n_clusters),max_iter=125,max_no_improvement=25,batch_size=200,n_init=10,random_state=17).fit(svd_representation)
        
        
        # Number of clusters in labels, ignoring noise if present.
        #n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    
        hdb_elapsed_time = time.time() - hdb_t1
        print('MiniBatchKMeans Elapsed time to cluster: %4.1f m' % (hdb_elapsed_time/60))
        #print('Clusters = ',n_clusters)
        
        for index in subs:            
            X=svd_representation[index[0]]
            cluster_labels=hdb.labels_[index[0]]
                
            hdb_t1 = time.time()
    
            fig, (ax1) = plt.subplots(1, 1)
            fig.set_size_inches(12, 8)
    
            # The 1st subplot is the silhouette plot
            # The silhouette coefficient can range from -1, 1 
            ax1.set_xlim([-.6, .6])
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(X) + (n_clusters + 1) * 12])
    
            # The silhouette_score gives the average value for all the samples.
            # This gives a perspective into the density and separation of the formed
            # clusters
            silhouette_avg = silhouette_score(X, cluster_labels)
            print('Elapsed time to cluster: %6.1f m' % ((time.time()-hdb_t1)/60))
            print("For n_clusters =", n_clusters,
                  "The average silhouette_score is :", silhouette_avg)
    
            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
            y_lower = 12
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]
    
                ith_cluster_silhouette_values.sort()
    
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i
    
                color = cm.spectral(float(i) / n_clusters)
                ax1.fill_betweenx(np.arange(y_lower, y_upper),
                                  0, ith_cluster_silhouette_values,
                                  facecolor=color, edgecolor=color, alpha=0.7)
    
                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.5, y_lower + 0.5 * size_cluster_i, str(i))
    
                # Compute the new y_lower for next plot
                y_lower = y_upper + 12  # 10 for the 0 samples
    
            set_xlabel="The avg silhouette coefficient values = "+str(round(silhouette_avg,3))
            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel(set_xlabel)
            ax1.set_ylabel("Cluster label")
    
            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    
            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks(list(np.arange(-.6,1,.2)))
    
            plt.suptitle(("Silhouette analysis for MiniBatchKMeans clustering on sample data "
                          "with n_clusters = %d" % n_clusters),
                         fontsize=14, fontweight='bold')
    
            plt.show()
            
        
        print('Next')    
        print('----')
        del ith_cluster_silhouette_values
        del cluster_labels
        del sample_silhouette_values
        del X       
    
    
getPlt(7, 8, 5)
print('Done')