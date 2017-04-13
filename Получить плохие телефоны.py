# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:37:38 2017


"""

import pandas as pd

#ghjwtlehf чтения файла нужного формата
def read_csv_sms(file_name):
    return pd.read_csv(file_name,
                 delimiter='\t',encoding='windows-1251',
                 usecols=['phone','submission_date','gate_status'],
                 dtype={'phone': str})#Читаем файло

data=read_csv_sms('1028258_2017-01-01_00_00_00_2017-03-03_23_59_59_59888.csv')


#предобработка
data['datetime']=data.submission_date.map(pd.to_datetime).dt.date# lambda arg: pd.to_datetime(arg, format='%d.%m.%Y %H:%M', errors='raise')).dt.date#Преобразуем к типу дата
data.drop('submission_date',inplace=True, axis=1)#удаляем ненужную колонку
data.gate_status=data.gate_status.map(
        lambda val: 1 if val in ['Доставлено','DELIVRD'] else 0).astype(int) #преобразуем к бинарному значению. Кому доставлено это статус "Доставлено" и Deliverd. Остальные это не доставленные.
tmp=data.groupby(['phone','datetime'])['gate_status'].max()#считаем статус с точностью до дня, с приоритетом доставлено
data=pd.DataFrame(data=tmp.reset_index().values)#переводим индекс обратно в колонки
data.columns=['phone','datetime','gate_status']
del tmp

#обработка
data.sort_values('datetime',inplace=True, ascending=False)#сортировка по дате
data['count'] = data.groupby('phone')['datetime'].cumcount()#Нумеруем телефоны в порядке возрастания записей

#выборка плохих телефонов по размеченным данным
phones_count=data[data['count']<3].groupby('phone')['phone'].count()#отправок должно быть не менее 3х, поэтому считаем отправки
phones=data[data['count']<3].groupby('phone')['gate_status'].sum()#берем первые три записи по телефонам
total=pd.concat([phones_count,phones], axis=1)#соединяем таблицы
del phones_count,phones

bad_phones=total[(total['phone']==3) & (total['gate_status']==0)].index#отбираем те у которых три раза подряд статус отправки был недоставлено
print('Количество плохих номеров',len(bad_phones))

#пишем в файл
with open('bad_phones.csv', 'w') as file_handler:
    file_handler.write("Phones\n")
    for item in bad_phones:
        file_handler.write("{}\n".format(item))

