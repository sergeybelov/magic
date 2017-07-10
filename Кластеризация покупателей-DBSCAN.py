# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import time
from sklearn.preprocessing import LabelBinarizer
from scipy.sparse import hstack

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
sparse_select=hstack(concList)

#кластеризуем
tstart = time.time()
print('Start: ',time.strftime('%X %x %Z'))
if __name__ == '__main__':
    db = DBSCAN(eps=0.2,min_samples=50, leaf_size=40,n_jobs=14).fit(sparse_select)
    print('time: ',str(round((time.time() - tstart)/60,2))) 
    
labels=db.labels_
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('clusters=',n_clusters_)

