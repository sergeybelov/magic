# -*- coding: utf-8 -*-
import pandas as pd

from sklearn.preprocessing import StandardScaler
import time
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import hstack


from hdbscan import HDBSCAN
from sklearn import metrics

#https://habrahabr.ru/company/ods/blog/327242/

df=pd.read_pickle('MG_Sales_customer.pickle',compression='gzip')

#выбираем покупателей для дальнейшего анализа
sales_sum=df.groupby('Покупатель')['Количество'].sum()
#выкидываем со слишком большими продажами (сводные карты) и тех кто купил один раз
sales_sum.drop(sales_sum[(sales_sum>133)|(sales_sum==1)].index, inplace=True)
customers_name=list(sales_sum.index)
del sales_sum

#делаем выборку
select=df.loc[df['Покупатель'].isin(customers_name),['Покупатель','ВидИзделия','ПодвидИзделия','СтильДизайна','ВидДизайна','ОсновнойКамень','ГруппаТовара','Коллекция','ЦветМеталла','ТоварСреднийВес','Размер','Вес']]
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

select['ЦветМеталлаКод']=select['ЦветМеталла'].map(lambda xx: codeMetall(xx))
select.drop('ЦветМеталла', inplace=True, axis=1)


#select['ПокупательДатаРождения']=select['ПокупательДатаРождения'].fillna(dt.datetime(999,1,1))


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

del select
del concList
print('Prepare finished')

##############################################################################
#кластеризуем
#==============================================================================
# https://hdbscan.readthedocs.io/en/latest/basic_hdbscan.html
#==============================================================================
# Compute DBSCAN
if __name__ == '__main__':
    hdb_t1 = time.time()
    hdb = HDBSCAN(min_cluster_size=1000,core_dist_n_jobs=14).fit(X)
    hdb_labels = hdb.labels_
    hdb_elapsed_time = time.time() - hdb_t1

# Number of clusters in labels, ignoring noise if present.
n_clusters_hdb_ = len(set(hdb_labels)) - (1 if -1 in hdb_labels else 0)

print('\n\n++ HDBSCAN Results')
print('Estimated number of clusters: %d' % n_clusters_hdb_)
print('Elapsed time to cluster: %.4f s' % hdb_elapsed_time)
#print('Silhouette Coefficient: %0.3f'
 #     % metrics.silhouette_score(X, hdb_labels))