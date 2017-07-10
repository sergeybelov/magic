import pandas as pd
#import datetime as dt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pylab import rcParams
rcParams['figure.figsize'] = 12, 8
#import numpy as np
#https://habrahabr.ru/company/ods/blog/327242/


#from sklearn.metrics import mean_absolute_error
#from sklearn.preprocessing import StandardScaler

#Читаем данные
df=pd.read_pickle('MG_Sales_customer.pickle',compression='gzip')

#выбираем покупателей для дальнейшего анализа
sales_sum=df.groupby('Покупатель')['Количество'].sum()#[df['Дата'>=dt.datetime(2016,1,1)]]
#выкидываем со слишком большими продажами (сводные карты) и тех кто купил один раз
sales_sum.drop(sales_sum[(sales_sum>133)|(sales_sum==1)].index, inplace=True)
customers_name=list(sales_sum.index)
del sales_sum

#делаем выборку
select=df.loc[df['Покупатель'].isin(customers_name),['Покупатель','АртикулБезКачества','ВидИзделия','ПодвидИзделия','СтильДизайна','ВидДизайна','ПодвидДизайна','ОсновнойКамень','ГруппаТовара','Коллекция','МаркетинговаяЛинейка','ЦветМеталла','ЦветПокрытия','ТоварСреднийВес','Тип','Размер','Вес','ПокупательПол']]
#select['ПокупательДатаРождения']=select['ПокупательДатаРождения'].fillna(dt.datetime(999,1,1))
del customers_name

#шкалируем
categorical_columns = [c for c in select.columns if select[c].dtype.name == 'object']
select_fact=select.copy()
for col in categorical_columns:
    select_fact[col]=pd.factorize(select_fact[col])[0]
    
scaler = StandardScaler()
X_scaled = scaler.fit_transform(select_fact)
#del select
del select_fact
del df


tsne = TSNE()
tsne_representation = tsne.fit_transform(X_scaled[:20000])

ЦветМеталла=list(select['ЦветМеталла'].unique())
colors = list(iter(cm.rainbow(np.linspace(0, 1, len(ЦветМеталла)))))

fig, ax = plt.subplots()
ax.scatter(tsne_representation[:, 0], tsne_representation[:, 1], label=ЦветМеталла,
            c=select['ЦветМеталла'].map(lambda xx: colors[ЦветМеталла.index(xx)]))
ax.legend()
ax.grid(True)

plt.show()