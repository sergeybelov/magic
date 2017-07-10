# -*- coding: utf-8 -*-
"""
Created on Mon May 29 12:00:22 2017

@author: tehn-11
"""

import pandas as pd
import numpy as np
import datetime as dt
import xgboost as xgb
from itertools import product
from sklearn.model_selection import TimeSeriesSplit
import multiprocessing
import time
import math
import random
from sklearn.preprocessing import StandardScaler
import os

#==============================================================================
# #Вспомогательный  блок журналирования операций
#==============================================================================
import logging
import logging.handlers
import sys, traceback

# Because you'll want to define the logging configurations for listener and workers, the
# listener and worker process functions take a configurer parameter which is a callable
# for configuring logging for that process. These functions are also passed the queue,
# which they use for communication.
#
# In practice, you can configure the listener however you want, but note that in this
# simple example, the listener does not apply level or filter logic to received records.
# In practice, you would probably want to do this logic in the worker processes, to avoid
# sending events which would be filtered out between processes.
def listener_configurer():
    root = logging.getLogger()
    h = logging.handlers.RotatingFileHandler('xgboost_forecast.log', 'a', 500*1024, 10)
    f = logging.Formatter('%(asctime)s %(processName)-10s %(levelname)-8s %(message)s')
    h.setFormatter(f)
    root.addHandler(h)
    
# This is the listener process top-level loop: wait for logging events
# (LogRecords)on the queue and handle them, quit when you get a None for a
# LogRecord.
def listener_process(queue, configurer):
    configurer()
    while True:
        try:
            record = queue.get()
            if record is None:  # We send this as a sentinel to tell the listener to quit.
                break
            logger = logging.getLogger(record.name)
            logger.handle(record)  # No level or filter logic applied - just do it!
        except Exception:
            print('Whoops! Problem:', file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            
# The worker configuration is done at the start of the worker process run.
# Note that on Windows you can't rely on fork semantics, so each process
# will run the logging configuration code when it starts.
def worker_configurer(queue):
    h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
    root = logging.getLogger()
    root.addHandler(h)
    # send all messages, for demo; no other level or filter logic applied.
    root.setLevel(logging.DEBUG)    
#------------------------------------------------------------------------------

def uniq_vals(df,col):
    return df[col].unique()

#вычисляем год назад
def yearsago(years, from_date):
    try:
        return from_date.replace(year=from_date.year - years)
    except ValueError:        
        return from_date.replace(month=2, day=28,
                                 year=from_date.year-years)

#временные характеристики
def setNewValues(time_series):
    time_series['День недели'] = time_series.index.weekday
    time_series['Неделя'] = time_series.index.week
    time_series['Год'] = time_series.index.year
    time_series['День месяца'] = time_series.index.day
    time_series['День года'] = time_series.index.dayofyear    
    time_series['Праздник'] =df.groupby('Дата')['Праздник'].max()    
    return time_series

def weekseason(time_series,week_d):
    time_series['Недельная сезонность']=time_series['День недели'].map(lambda cell: week_d.loc[cell,'Недельная сезонность'])
    return time_series

#Сдвигаем период на год вперед
def setTimebasedValues(time_series_forecast,time_series,cols,first_day_past_year,last_day_past_year):
    #вычленяем данные с колонками
    time_series_copy=pd.DataFrame(data=time_series.loc[first_day_past_year:last_day_past_year,cols].copy())
    try:
        time_series_copy.loc[dt.datetime(first_day_past_year.year,2,28)]=time_series_copy.loc[dt.datetime(first_day_past_year.year,2,28):dt.datetime(first_day_past_year.year,2,29)].mean()
    except:
        pass
    
    #Если високосный год
    try:            
        time_series_forecast[cols]=pd.concat([
                time_series_copy.loc[:dt.datetime(first_day_past_year.year,2,28),[cols]].shift(366,'D'),
                time_series_copy.loc[dt.datetime(first_day_past_year.year,3,1):,[cols]].shift(365,'D')    
                ], axis=0, join='outer')
    except:
        time_series_forecast[cols]=time_series_copy[cols].shift(365,'D')
        #TODO Тут дополнительно отработать 29 февраля текущего года
    return time_series_forecast
    
#------------------------------------------------------------------------------

#Читаем данные
df=pd.read_pickle('MG_Sales.pickle',compression='gzip')

#Формируем выборку
begin_period=dt.datetime(2015,1,1)#отбор данных
prediction_period=dt.datetime(2017,6,1)#прогноз начиная с
day_forecast=30#на сколько дней вперед строим прогноз


#Пустой период в днях обучающей выборки
delta=prediction_period-begin_period
dummy_train = pd.DataFrame(index=np.array([begin_period + dt.timedelta(days=x) for x in range(0, delta.days)]).astype('datetime64[D]'))
dummy_train.index.name='ds'



#список открытых магазинов торговавших 2 года
retail_df=df[(df['Дата']>=begin_period)].groupby(['Магазин','Дата'])['Количество'].sum().reset_index()
retail_df['Месяц']=retail_df['Дата'].dt.month
retail_df['Год']=retail_df['Дата'].dt.year
retail_df=retail_df.groupby(['Месяц','Год','Магазин'])['Количество'].sum().reset_index()
retail_df=pd.concat([retail_df[(retail_df['Год']==begin_period.year)  &(retail_df['Месяц']==1)].groupby(['Магазин'])['Количество'].sum(),
                     retail_df[(retail_df['Год']==begin_period.year+1)&(retail_df['Месяц']==12)].groupby(['Магазин'])['Количество'].sum()], axis=1).dropna(axis=0, how='any')

#Сетка прогнозируемых параметров
features_names=['ТоварЦеноваяГруппа','ЦветМеталла','Коллекция','Магазин']
parameters = product(uniq_vals(df,features_names[0]), 
                     uniq_vals(df,features_names[1]),
                     uniq_vals(df,features_names[2]),
                     retail_df.index.values)
parameters_list=list(parameters)
random.shuffle(parameters_list)






def mp_worker(parameters_list, df, out_q,log_queue, configurer):
    configurer(log_queue)
    name = multiprocessing.current_process().name
    """ The worker function, invoked in a process. 'nums' is a
        list of numbers to factor. The results are placed in
        a dictionary that's pushed to a queue.
    """
    
    logger = logging.getLogger(name)
    level = logging.INFO    
    logger.log(level, '>>Start process')
    

    tss = TimeSeriesSplit(n_splits=10)
    #Вычленяем временной ряд по сетке характеристик
    
    #outdict = {}
    res_array=[]
    count=-1
    queryStart="Дата>="+begin_period.strftime('%Y%m%d')+' and Дата<'+prediction_period.strftime('%Y%m%d')
    try:        
        for features in parameters_list:
            tstart = time.time()            
            count+=1
            
            if count%1000==0:
                logger.log(level, 'pass: '+str(count+1)+' of '+str(len(parameters_list)))
            #------
        
            
            query=queryStart
            for i in range(len(features_names)):
                query+=" and "+features_names[i]+"=='"+features[i]+"'"
            
            #Временной ряд с группировкой по дате
            query_res=df.query(query)
            if query_res.shape[0]==0:
                #logger.log(level, 'pass: '+str(count+1)+' empty query in features: '+str(features))
                continue
            
            logger.log(level, 'pass: '+str(count+1)+' len='+str(query_res.shape[0])+' in features: '+str(features))
            
            time_series=pd.DataFrame(data=query_res.groupby('Дата')['Количество'].sum())
            time_series.index.name='ds'
            time_series.columns=['y']    
            #сливаем обучающую выборку и пустой период чтобы избежать пропусков дат, пропуски заполняем нулями
            time_series=dummy_train.merge(time_series,left_index=True, right_index=True,how='outer').fillna(0)    
            
        #==============================================================================
        #     #Фомируем характеристики модели
        #==============================================================================
            #Фиксируем аномально низкие и высокие продажи
            ul=5#Персентиль высоких продаж 2
            ll=5#7#Персентиль низких продаж 10
            md=15#ширина медианы
        
            ulim=np.percentile(time_series['y'], 100.-ul)
            llim=np.percentile(time_series['y'], ll)
            med=np.percentile(time_series['y'], [50-md,50+md])
        
        
            time_series['Квантили']=0
            time_series.loc[time_series.y<med[0],'Квантили']=-1
            time_series.loc[time_series.y>med[1],'Квантили']=1
            time_series.loc[time_series.y<llim,'Квантили']=-2
            time_series.loc[time_series.y>ulim,'Квантили']=2
        
                
            time_series=setNewValues(time_series)
        
            #порядок дней в сезонности недельной продажи за исключением аномалий
            week_d=pd.DataFrame(data=time_series[time_series['Квантили']==0].groupby('День недели')['y'].sum().sort_values())
            week_d.insert(0,'Недельная сезонность',list(range(week_d.shape[0])))
            for i in list(set(range(7))-set(week_d.index.values)):
                week_d.loc[i,'Недельная сезонность']=-1
            
            time_series=weekseason(time_series,week_d)
        
            mean_dict=dict(time_series.groupby(['Год','Неделя'])['y'].mean())
            time_series['Среднее по неделе']=time_series.apply(lambda row: mean_dict[row['Год'],row['Неделя']] , axis=1)
            mean_dict=dict(time_series.groupby(['День года'])['y'].mean())
            time_series['Среднее по дню года']=time_series.apply(lambda row: mean_dict[row['День года']] , axis=1)
            time_series['Недельный тренд']=time_series['Среднее по неделе'].diff(7).fillna(0)
        
            #Вычленяем целевую переменную
            y=time_series.y
            time_series.drop(['y'], axis=1, inplace=True)
            
        #==============================================================================
        #     #Обучение
        #==============================================================================
            #Нормализуем обучающую выборку с помощью класса StandardScaler
            scaler = StandardScaler(with_mean=True,with_std=True)
            
            dtrain = xgb.DMatrix(scaler.fit_transform(time_series), label=y)    
            # задаём параметры
            params = {
                    'objective': 'reg:linear',
                    'booster':'gblinear',
                    'tree_method': 'exact',                
                    'eta': 0.05,#коэффициент обучения
                    'alpha': 10,
                    'lambda_bias': 10
                    }
            trees = 1000#y.shape[0]
            
            #фолды кросс-валидации    
            tss_cv=list(tss.split(time_series,y))
            
            # прогоняем на кросс-валидации с метрикой rmse
            cv = xgb.cv(params, dtrain, metrics = ('rmse'), early_stopping_rounds=True,verbose_eval=False, folds=tss_cv, show_stdv=False, num_boost_round=trees)
        
            # обучаем xgboost с оптимальным числом деревьев, подобранным на кросс-валидации
            mod_n=cv['test-rmse-mean'].argmin()
            bst = xgb.train(params, dtrain, num_boost_round=mod_n)
            
            # запоминаем ошибку на кросс-валидации
            deviation = cv.loc[mod_n]["test-rmse-mean"]
            
        #==============================================================================
        #     #подготавливаем выборку для прогноза
        #==============================================================================
            date_list = np.array([prediction_period + dt.timedelta(days=x) for x in range(0, day_forecast)]).astype('datetime64[D]')
            time_series_forecast=pd.DataFrame(index=date_list)
            time_series_forecast.index.name='ds'
            
            time_series_forecast=weekseason(setNewValues(time_series_forecast),week_d)
            st_day=time_series_forecast.iloc[0].name
            first_day_past_year=yearsago(1, st_day)
            #вычисляем период которым мы должны взять из прошлого года
            last_day_past_year=dt.datetime(first_day_past_year.year,12,31)
            time_series[first_day_past_year:last_day_past_year]
        
                
            time_series_forecast=setTimebasedValues(time_series_forecast,time_series,['Недельный тренд','Квантили','Среднее по неделе','Среднее по дню года'],first_day_past_year,last_day_past_year)
            time_series_forecast=time_series_forecast[time_series.columns]
            del week_d
            
        #==============================================================================
        #     #Прогноз
        #==============================================================================            
            
            prediction_y = pd.DataFrame(data=bst.predict(xgb.DMatrix(scaler.transform(time_series_forecast))),index=time_series_forecast.index, columns=['Прогноз'])
            prediction_y['Прогноз']=prediction_y['Прогноз'].map(lambda val: 0 if val<0.01 else round(val,3))#Отрицательный прогноз зануляем, остальное округляем
            prediction_y.index.names=['Дата']
            
            #дозаполняем колонки прогноза характеристиками
            for i in range(len(features_names)):
                prediction_y[features_names[i]]=features[i]        
            
            #-----------------------
            #outdict[name+'_'+str(count)]=prediction_y.reset_index()
            res_array.append(prediction_y.reset_index())
            logger.log(level, 'pass: '+str(count+1)+' RMSE: '+str(deviation) +', boost rounds:'+str(mod_n)+', in features: '+str(features)+'. iteration time (sec): '+str(round((time.time() - tstart),2)))
    except Exception as e:            
        logger.log(logging.ERROR, 'error happens: '+str(e)+', in '+str(sys.exc_info()[-1].tb_lineno))
    except Warning as e:            
        logger.log(logging.ERROR, 'warning happens: '+str(e)+', in '+str(sys.exc_info()[-1].tb_lineno))
        
    #записываем в очередь потока    
    logger.log(level, 'store queue')
    total_forecast=pd.concat(res_array,ignore_index=True)
    
    #сохраняем в файл
    fn='forecast '+name
    total_forecast.to_csv(fn+'.csv',index=False,encoding='utf-8',sep=';',compression='gzip',decimal=',',date_format='%d.%m.%Y')
    os.rename(name,fn+'.gzip')
    
    
    #outdict[name]=outdict
    #out_q.put(name)
    logger.log(level, '<<finish')
            
        
parallel_tasks=15#Одновременное количество задач
#запускаем процессы
if __name__ == '__main__':
    multiprocessing.util.log_to_stderr(10)
    tstart = time.time()
    
    #Сервис логгирования
    log_queue = multiprocessing.Queue(-1)
    listener = multiprocessing.Process(target=listener_process,
                                       args=(log_queue, listener_configurer))
    listener.start()
    
    
    out_q = multiprocessing.Queue()
    chunksize = int(math.ceil(len(parameters_list ) / float(parallel_tasks)))
    procs = []

    print('Process: ',len(parameters_list ),'/',chunksize)
    for i in range(parallel_tasks):
        params=parameters_list [chunksize * i:chunksize * (i + 1)]
        if len(params)==0: continue
        print('start ',(i+1),' process...')
    
        p = multiprocessing.Process(name='chunk '+str(i+1),
                target=mp_worker,
                args=(params,df,out_q,log_queue, worker_configurer))
        procs.append(p)
        p.start()

    #print('get queue...')
    # Collect all results into a single result dict. We know how many dicts
    # with results to expect.    
    #resultdict = {}
    #for _ in procs:
     #   resultdict.update(out_q.get())

    print('time: ',str(round((time.time() - tstart)/60,2)))   
    #print('store values...')
    #соединяем список параметров
    #res_array=[]
    #for i in range(parallel_tasks):
     #   key1='chunk '+str(i+1)
      #  if key1 not in resultdict: continue
       # rs=resultdict[key1]
        #for j in range(len(parameters_list)):
         #   key2=key1+'_'+str(j)
          #  if key2 not in rs: break
           # res_array.append(rs[key2])
    #del resultdict
    
    print('wait processes finish...')
    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    print('terminate logger...')
    listener.terminate()
    print('Done.')
