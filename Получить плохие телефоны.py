# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:37:38 2017


"""

import pandas as pd

data=pd.read_csv('1028258_2016-12-01_00_00_00_2017-01-31_23_59_59_58068.csv',
                 delimiter=';',encoding='windows-1251',
                 usecols=['phone','submission_date','gate_status'],
                 dtype={'phone': str})#Читаем файло

#предобработка
data.send_date=data.submission_date.map(lambda arg: pd.to_datetime(arg, format='%d.%m.%Y %H:%M', errors='raise'))#Преобразуем к типу дата
data.gate_status=data.gate_status.map(
        lambda val: 1 if val in ['Доставлено','DELIVRD'] else 0).astype(int) #преобразуем к бинарному значению. Кому доставлено это статус "Доставлено" и Deliverd. Остальные это не доставленные.

print('Размер исходной таблицы ',data.shape)
data.drop_duplicates(inplace=True)#убираем дубликаты по всем колонкам
print('Размер таблицы после исключения повторений',data.shape)
data.sort_values('submission_date',inplace=True, ascending=True)#сортировка по дате

data['count'] = data.groupby('phone').cumcount() + 1#Нумеруем телефоны в порядке возрастания
phones=data[data['count']<=3].groupby('phone')['gate_status'].sum()#берем первые три записи по телефонам
bad_phones=phones[phones!=3]#отбираем те у которых были пропуски в доставке

