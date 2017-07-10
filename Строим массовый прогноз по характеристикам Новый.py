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
import fnmatch

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
    h = logging.handlers.RotatingFileHandler('xgboost_forecast.log', 'a', 1024*1024, 10)
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
#процедуры формирования модели
class modelForecast:
    def __init__(self, celebrate):
        self.cel=celebrate
        
        
    def setPercentil(self,row,percentiles):
        if row<percentiles['llim']: return -2
        if row>percentiles['ulim']: return 2
        if row<percentiles['med'][0]: return -1
        if row>percentiles['med'][1]: return 1
        return 0

    def getPercentil(self,ts):
        #Фиксируем аномально низкие и высокие продажи
        ul=5#Персентиль высоких продаж 2
        ll=5#7#Персентиль низких продаж 10
        md=25#ширина медианы
        percentiles={}

        percentiles['ulim']=np.percentile(ts['y'], 100.-ul)
        percentiles['llim']=np.percentile(ts['y'], ll)
        percentiles['med']=np.percentile(ts['y'], [50-md,50+md])
        return percentiles


    #временные характеристики
    def setNewValues(self,_ts):
        ts=_ts.copy()
        y=None
        
        ts['День недели'] = ts.index.weekday
        ts['Выходной'] = ((ts.index.weekday.isin([5,6]))|(ts.index.isin(self.cel['Праздник'])))*1
        ts['Неделя'] = ts.index.week
        ts['Год'] = ts.index.year
        ts['Месяц'] = ts.index.month
        ts['День месяца'] = ts.index.day


        if 'y' in ts.columns:#ТОЛЬКО для обучающей выборки
            #Фомируем характеристики модели 
            percentiles=self.getPercentil(ts)#получаем распределение персентилей


            ts['Среднее по дню года']=0
            ts['Максимальное по дню года']=0
            ts['Минимальное по дню года']=0
            #Среднее по дню года, нарастающим
            for year in ts['Год'].unique():    
                _mean_day_year=ts.loc[ts['Год']<=year,['Месяц','День месяца','y']].reset_index()#.drop('ds',axis=1)

                mean2829=np.mean(_mean_day_year.loc[(_mean_day_year['Месяц']==2)&(_mean_day_year['День месяца'].isin([28,29])),'y'])
                mean2801=np.mean(_mean_day_year.loc[(_mean_day_year['Месяц']==2)&(_mean_day_year['День месяца']==28)|(_mean_day_year['Месяц']==3)&(_mean_day_year['День месяца']==1),'y'])

                _mean_day_year.loc[(_mean_day_year['Месяц']==2)&(_mean_day_year['День месяца']==28),'y']=mean2829
                _mean_day_year.loc[(_mean_day_year['Месяц']==2)&(_mean_day_year['День месяца']==29),'y']=mean2801   

                mean_day_year=dict(_mean_day_year.groupby(['Месяц','День месяца'])['y'].mean())
                ts['Среднее по дню года']=ts.apply(lambda row: mean_day_year[row['Месяц'],row['День месяца']] if row['Год']==year else row['Среднее по дню года'] , axis=1)             
                max_day_year=dict(_mean_day_year.groupby(['Месяц','День месяца'])['y'].max())
                ts['Максимальное по дню года']=ts.apply(lambda row: max_day_year[row['Месяц'],row['День месяца']] if row['Год']==year else row['Максимальное по дню года'] , axis=1)
                min_day_year=dict(_mean_day_year.groupby(['Месяц','День месяца'])['y'].min())
                ts['Минимальное по дню года']=ts.apply(lambda row: min_day_year[row['Месяц'],row['День месяца']] if row['Год']==year else row['Минимальное по дню года'] , axis=1)
                
            ts['Квантили']=ts.apply(lambda row: self.setPercentil(row['Среднее по дню года'],percentiles), axis=1)


            #порядок дней в сезонности недельной продажи за исключением аномалий
            week_d=pd.DataFrame(data=ts[ts['Квантили']==0].groupby('День недели')['y'].max().sort_values())
            week_d.insert(0,'Недельная сезонность',list(range(week_d.shape[0])))
            for i in list(set(range(7))-set(week_d.index.values)):
                week_d.loc[i,'Недельная сезонность']=-1
            ts=self.weekseason(ts,week_d)

            #Среднее и тренды                        
            mean_dict=dict(ts.groupby(['Год'])['y'].mean())
            ts['Среднее за год']=ts.apply(lambda row: mean_dict[row['Год']] , axis=1)        

            #РАССЧИТЫВАЕМ коэффициент роста
            mean_month=ts.groupby(['Год','Месяц'])['y'].mean()
            mean_month_shift=mean_month.shift(12).fillna(0)
            inc_temp=(mean_month/mean_month_shift-1)#*mean_month.shift(11).fillna(0)        
            last_tempo=inc_temp.tail(1).values[0]

            #Вычленяем целевую переменную
            y=ts.y
            ts.drop(['y'], axis=1, inplace=True)

            #заглушки        
            ts.drop(['День недели'], axis=1, inplace=True)                

            self.week_d=week_d
            self.mean_day_year=mean_day_year
            self.percentiles=percentiles
            self.last_tempo=last_tempo            
            self.last_vals=ts.tail(1)
            self.columns=ts.columns
            self.max_day_year=max_day_year
            self.min_day_year=min_day_year
        return ts,y


    def weekseason(self,ts,week_d):
        ts['Недельная сезонность']=ts['День недели'].map(lambda cell: week_d.loc[cell,'Недельная сезонность'])
        return ts

    #Сдвигаем период на год вперед
    def fillTimeSeriesForecast(self,_tsf):
        tsf=self.weekseason(_tsf,self.week_d)
        tsf.drop(['День недели'], axis=1, inplace=True)

        tsf['Среднее по дню года']=tsf.apply(lambda row: self.mean_day_year[row['Месяц'],row['День месяца']], axis=1)
        tsf['Квантили']=tsf.apply(lambda row: self.setPercentil(row['Среднее по дню года'],self.percentiles), axis=1)

        tsf['Максимальное по дню года']=tsf.apply(lambda row: self.max_day_year[row['Месяц'],row['День месяца']], axis=1)
        tsf['Минимальное по дню года']=tsf.apply(lambda row: self.min_day_year[row['Месяц'],row['День месяца']], axis=1)
        
        
        #Мы знаем средние данные за год, учитываем как тренд        
        for col in ['Среднее за год']:
            tsf[col]=self.last_vals[col].values[0]
        return tsf[self.columns]#Колонки в правильном порядке


    def createTimeSeriesForecast(self,begin, end):
        date_list = pd.date_range(begin, end).tolist()
        time_series_forecast=pd.DataFrame(index=date_list)
        time_series_forecast.index.name='ds'    

        time_series_forecast,_=self.setNewValues(time_series_forecast)

        return self.fillTimeSeriesForecast(time_series_forecast)        
#-------------------------------------------------------------
def uniq_vals(df,col):
    return df[col].unique()

#------------------------------------------------------------
begin_period=dt.datetime(2015,1,1)
prediction_period=dt.datetime(2017,6,1)
day_forecast=29#на сколько дней вперед строим прогноз



#Пустой период в днях обучающей выборки
delta=prediction_period+dt.timedelta(days=day_forecast)-begin_period
dummy_train = pd.DataFrame(index=np.array([begin_period + dt.timedelta(days=x) for x in range(0, delta.days)]).astype('datetime64[D]'))

#Читаем данные
df=pd.read_pickle('MG_Sales.pickle',compression='gzip')
celebrate=pd.read_pickle('celebrate.pickle')


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





#параллельная работа прогноза
def mp_worker(parameters_list, df, celebrate,out_q,log_queue, configurer):
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
    new_model=modelForecast(celebrate)
    
    #outdict = {}
    res_array=[]
    count=0
    queryStart="Дата>="+begin_period.strftime('%Y%m%d')+' and Дата<'+prediction_period.strftime('%Y%m%d')
    try:        
        for features in parameters_list:
            tstart = time.time()            
            count+=1
            
            if count%1000==0:
                logger.log(level, 'pass: '+str(count+1)+' of '+str(len(parameters_list)))
                
                if count>=1000: break
            #------
        
                        
            query=queryStart
            for i in range(len(features_names)):
                query+=" and "+features_names[i]+"=='"+features[i]+"'"
            
            #Временной ряд с группировкой по дате
            query_res=df.query(query)
            if query_res.shape[0]==0:
                #logger.log(level, 'pass: '+str(count+1)+' empty query in features: '+str(features))
                continue
            
            #logger.log(level, 'pass: '+str(count+1)+' len='+str(query_res.shape[0])+' in query: '+str(query))
            
            time_series=pd.DataFrame(data=query_res.groupby('Дата')['Количество'].sum())
            time_series.index.name='ds'
            time_series.columns=['y']    
            
            
            #сливаем обучающую выборку и пустой период чтобы избежать пропусков дат, пропуски заполняем нулями
            time_series=dummy_train.merge(time_series,left_index=True, right_index=True,how='outer').fillna(0)    
            
            #==============================================================================
            #     #Фомируем характеристики модели
            #==============================================================================            
            time_series_train, y_train=new_model.setNewValues(time_series)            
    
            tss = TimeSeriesSplit(n_splits=10)
            tss_cv=list(tss.split(time_series_train,y_train))

            # задаём параметры
            params = {
                    'objective': 'reg:linear',
                    'booster':'gblinear',                
                    'eta': 0.23,#коэффициент обучения
                    'alpha': 0.1,                    
                    'eval_metric': 'rmse'
                    }

            #Нормализуйте обучающую выборку с помощью класса StandardScaler
            scaler = StandardScaler(with_mean=True,with_std=True)
            dtrain = xgb.DMatrix(scaler.fit_transform(time_series_train), label=y_train)#

            # прогоняем на кросс-валидации с метрикой rmse
            cv = xgb.cv(params, dtrain, metrics = ('rmse'), early_stopping_rounds=25,verbose_eval=False, show_stdv=False, num_boost_round=1000,folds=tss_cv)#,nfold=10

            best_iteration=cv['test-rmse-mean'].argmin()            
            bst = xgb.train(params, dtrain, best_iteration,verbose_eval=False)

            deviation = cv.loc[best_iteration]["test-rmse-mean"]
            #==============================================================================
            #     #Предсказание
            #==============================================================================                        
            time_series_forecast=new_model.createTimeSeriesForecast(prediction_period, prediction_period+dt.timedelta(days=day_forecast))
            
            
            prediction_y = pd.DataFrame(data=bst.predict(xgb.DMatrix(scaler.transform(time_series_forecast))),index=time_series_forecast.index, columns=['Прогноз'])            
            prediction_y['Прогноз']=prediction_y['Прогноз'].map(lambda val: 0 if val<0.01 else round(val,0))#Отрицательный прогноз зануляем, остальное округляем
            prediction_y.index.names=['Дата']
            
            #нулевой прогноз не нужен
            #if prediction_y['Прогноз'].sum()==0: continue
            
            #СВОРАЧИВАЕМ по году и месяцу
            prediction_y['Год'] = prediction_y.index.year
            prediction_y['Месяц'] = prediction_y.index.month
            prediction_y=pd.DataFrame(data=prediction_y.groupby(['Год','Месяц'])['Прогноз'].sum())
            
            #дозаполняем колонки прогноза характеристиками
            for i in range(len(features_names)):
                prediction_y[features_names[i]]=features[i]        

            
            #-----------------------
            #outdict[name+'_'+str(count)]=prediction_y.reset_index()
            res_array.append(prediction_y.reset_index())
            logger.log(level, 'pass: '+str(count+1)+' RMSE: '+str(deviation) +', boost rounds: '+str(best_iteration)+', in query: '+query+'. iteration time (sec): '+str(round((time.time() - tstart),2)))
    except Exception as e:            
        logger.log(logging.ERROR, 'error happens: '+str(e)+', in '+str(sys.exc_info()[-1].tb_lineno))
    except Warning as e:            
        logger.log(logging.ERROR, 'warning happens: '+str(e)+', in '+str(sys.exc_info()[-1].tb_lineno))
        
    #записываем в очередь потока    
    logger.log(level, 'store queue')
    total_forecast=pd.concat(res_array,ignore_index=True)
    
    #сохраняем в файл
    fn='forecast '+name
    total_forecast.to_csv(fn+'.csv',index=False,encoding='utf-8',sep=';',decimal=',',date_format='%d.%m.%Y')#,compression='gzip'
    #os.rename(fn+'.csv',fn+'.gzip')
    
    
    #outdict[name]=outdict
    #out_q.put(name)
    logger.log(level, '<<finish')
            
        
parallel_tasks=4#Одновременное количество задач
task_name_mask='chunk goods forecast '
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
    
        p = multiprocessing.Process(name=task_name_mask+str(i+1),
                target=mp_worker,
                args=(params,df,celebrate,out_q,log_queue, worker_configurer))
        procs.append(p)
        p.start()

    
    print('wait processes finish...')
    # Wait for all worker processes to finish
    for p in procs:
        p.join()

        
    print('terminate logger...')
    listener.terminate()
    
    
    #пишем все в один файл
    print('Обработка файлов csv')    
    files=os.listdir()    
    total_forecast=None
    for file in files:
        if fnmatch.fnmatch(file, task_name_mask+'*.csv'):
            _df=pd.read_csv(file,encoding='utf-8',sep=';',decimal=',')
            if pd==None:
                total_forecast=_df
            else:
                total_forecast=pd.concat([total_forecast,_df])
            os.remove(file)
        
    if total_forecast==None:
        print('Nothing to concatenate :(')    
    else:
        total_forecast.to_csv('total_forecast.csv',index=False,encoding='utf-8',sep=';',decimal=',',date_format='%d.%m.%Y')#,compression='gzip'
    
    print('time: ',str(round((time.time() - tstart)/60,2))) 
    print('Done.')
