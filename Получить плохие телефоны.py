# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:37:38 2017


"""
import multiprocessing.dummy as mp
import pandas as pd

#Параллельный процесс обработки
def parallelCalc(proc,var):
    pool = mp.Pool(processes=4)
    if __name__ == '__main__':
        result=pool.map(proc,var)
        pool.terminate()
        pool.join()
    return result


#ghjwtlehf чтения файла нужного формата
def read_csv_sms(file_name):
    return pd.read_csv(file_name,
                 delimiter='\t',encoding='windows-1251',
                 usecols=['phone','submission_date','gate_status'],
                 dtype={'phone': str})#Читаем файло

file_name = input("Имя файла: ")
data=read_csv_sms(file_name)


#предобработка
#data['datetime']=data.submission_date.map(pd.to_datetime).dt.date# lambda arg: pd.to_datetime(arg, format='%d.%m.%Y %H:%M', errors='raise')).dt.date#Преобразуем к типу дата
data['datetime']=pd.to_datetime(data.submission_date)#data.submission_date.map(pd.to_datetime).dt.date# lambda arg: pd.to_datetime(arg, format='%d.%m.%Y %H:%M', errors='raise')).dt.date#Преобразуем к типу дата
data['datetime']=data['datetime'].dt.date
data.drop('submission_date',inplace=True, axis=1)#удаляем ненужную колонку
data.gate_status=parallelCalc(lambda val: 1 if val in ['Доставлено','DELIVRD'] else 0,data.gate_status)#преобразуем к бинарному значению. Кому доставлено это статус "Доставлено" и Deliverd. Остальные это не доставленные.
data.gate_status=data.gate_status.astype(int)

#raise SystemExit

tmp=data.groupby(['phone','datetime'])['gate_status'].max()#считаем статус с точностью до дня, с приоритетом доставлено
data=pd.DataFrame(data=tmp.reset_index().values)#переводим индекс обратно в колонки
data.columns=['phone','datetime','gate_status']
del tmp

#обработка
data.sort_values('datetime',inplace=True, ascending=False)#сортировка по дате
data['count'] = data.groupby('phone')['datetime'].cumcount()#Нумеруем телефоны в порядке возрастания записей

#выборка плохих телефонов по размеченным данным
data_count_3=data['count']<3
phones_count=data[data_count_3].groupby('phone')['phone'].count()#отправок должно быть не менее 3х, поэтому считаем отправки
phones=data[data_count_3].groupby('phone')['gate_status'].sum()#берем первые три записи по телефонам
total=pd.concat([phones_count,phones], axis=1)#соединяем таблицы
del phones_count,phones

bad_phones=total[(total['phone']==3) & (total['gate_status']==0)].index#отбираем те у которых три раза подряд статус отправки был недоставлено
print('Количество плохих номеров',len(bad_phones))

#пишем в файл
with open('bad_phones.csv', 'w') as file_handler:
    #file_handler.write("Phones\n")
    for item in bad_phones:
        file_handler.write("{},".format(item))

print('OK')