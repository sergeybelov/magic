from fbprophet import Prophet
import numpy as np
import math
import multiprocessing
from itertools import product
import pandas as pd
import datetime as dt
import time


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
    h = logging.handlers.RotatingFileHandler('facebook_proph.log', 'a', 500*1024, 10)
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
#---------------------------------    
    
#Читаем данные
df=pd.read_pickle('MG_Sales.pickle',compression='gzip')

#df.groupby('ТоварЦеноваяГруппа')['Количество'].sum().sort_values(ascending=False)
#Формируем выборку
time_series=pd.DataFrame(data=df[(df['Коллекция']=='С фианитом(с-ги,к-цо,подв,бр-т,колье)')&(df['Дата']>=dt.date(2015,1,1))].groupby('Дата')['Количество'].sum())
time_series.index.name='ds'
time_series.columns=['y']
time_series['y'] = np.log(time_series['y'])



def mp_worker(params_grid, df,pediods,changepoints, out_q,log_queue, configurer):
    configurer(log_queue)
    name = multiprocessing.current_process().name
    """ The worker function, invoked in a process. 'nums' is a
        list of numbers to factor. The results are placed in
        a dictionary that's pushed to a queue.
    """
    
    logger = logging.getLogger(name)
    level = logging.INFO    
    logger.log(level, '>>Start process')
        
    try:
        outdict = {}
        count=-1
        for params in params_grid:
            tstart = time.time()            
            count+=1
            logger.log(level, 'pass: '+str(count+1)+' of '+str(len(params_grid)))
            #------
            
            lower_window,upper_window,changepoint_prior_scale,seasonality_prior_scale,holidays_prior_scale=params     
            
        
            #Праздники        
            selebrate=pd.DataFrame({
                'holiday': 'holiday',
                'ds': changepoints,
                'lower_window': int(lower_window),
                'upper_window': int(upper_window),
                })
    
    
            #changepoint_prior_scale - гиперпараметр процента точек смены тренда (по-умолчанию: 0.05)
            #changepoints  - массив для ручной установки дат смены тренда
            #interval_width - предсказательный интервал
            #mcmc_samples - количество шагов для расчета неопределенности в сезонности для байесовской модели
            #holidays - датасет с праздникам
        
            #параметр роста
            #time_series['cap'] = 150
        
            
            mape=[]
            mae=[]
            #делим выборку на обучающую и валидационную
            for begin,end in reversed(pediods):
                date_div_past=begin-dt.timedelta(days=1)
        
                time_series_train=time_series.loc[:date_div_past]
                time_series_test=time_series.loc[begin:end]
        
                m = Prophet(holidays = selebrate,# growth = 'logistic',
                            #n_changepoints=n_changepoints,
                            changepoint_prior_scale=changepoint_prior_scale,
                            seasonality_prior_scale=seasonality_prior_scale,
                            holidays_prior_scale=holidays_prior_scale,
                            yearly_seasonality=True,
                            weekly_seasonality=True)#,mcmc_samples=75
                #обучение
                m.fit(time_series_train.reset_index())
            
                #Подготовка календаря будущего
                future = time_series_test.reset_index()
        
                #Предсказание
                forecast = m.predict(future)   
            
            
                cmp_df = np.exp(forecast.set_index('ds')[['yhat']].join(time_series_test))
                cmp_df['e'] = cmp_df['y'] - cmp_df['yhat']
                cmp_df['p'] = 100*cmp_df['e']/cmp_df['y']
                mape.append(np.mean(abs(cmp_df['p'])))
                mae.append(np.mean(abs(cmp_df['e'])))
            
            #сохраняем значения
            mape=np.mean(mape)
            outdict[name+'_'+str(count)]=[params,mape,np.mean(mae)] 
            logger.log(level, 'pass: '+str(count)+', got MAPE='+str(mape)+' in params: '+str(params)+'. iteration time (sec): '+str(round((time.time() - tstart),2)))
            
    except Exception as e:            
        logger.log(logging.ERROR, 'error happens: '+str(e)+', in '+str(sys.exc_info()[-1].tb_lineno))
    except Warning as e:            
        logger.log(logging.ERROR, 'warning happens: '+str(e)+', in '+str(sys.exc_info()[-1].tb_lineno))
            
    #записываем в очередь потока    
    logger.log(level, 'store queue')
    outdict[name]=outdict
    out_q.put(outdict)
    logger.log(level, '<<finish')
    

#==============================================================================
# Подготовка к валидации    
#==============================================================================
#для валидации используем хвост временного ряда
last=time_series.iloc[-1].name
startDt=dt.datetime(last.year,last.month,last.day)
lastDay=dt.datetime(last.year,last.month,1)-dt.timedelta(seconds=1)
startmonth=dt.datetime(lastDay.year,lastDay.month,1)

folds=5
print('Running process...')
changepoints=df[df['Праздник']==1]['Дата'].unique()
#готовим период деления валидации
pediods=[]
for i in range(folds):
    pediods.append([startmonth,startDt])    
    startDt=startmonth-dt.timedelta(seconds=1)
    startmonth=dt.datetime(startDt.year,startDt.month,1)       
   

#==============================================================================
# #перебор гиперпараметров
#==============================================================================
parallel_tasks=14#Одновременное количество задач

lower_windows=[-12,-11,-10]#list(range(-14,0,1))
upper_windows=[6,7,8,9]##list(range(0,7,1))
seasonality_prior_scale=[100,110,130]#[10,40,90]#list(range(10,110,3))
holidays_prior_scale=[60,100]#[5,55,70]#list(range(5,100,3))
#n_changepoints=list(range(10,110,3))
changepoint_prior_scales=np.logspace(-.5, -.3, num=10, endpoint=False)


#MAPE=42.9877295256 in params: (-11, 6, 0.34902548789595794, 100, 15)
#MAPE=42.7812078859 in params: (-11, 7, 0.34902548789595794, 110, 30)
#MAPE=42.9492515503 in params: (-10, 6, 0.33419504002611428, 110, 15)
#MAPE=42.7989754954 in params: (-11, 6, 0.31045595881283555, 100, 45)
#
#MAPE=37.8770311618 in params: (-11, 6, 0.19952623149688797, 110, 45)
#Качество модели (%), чем выше тем лучше:  56.96
#Абсолютное значение ошибки, чем ниже тем лучше:  55.05
#Параметры:  (-11, 9, 0.33113112148259111, 100, 100)

#Качество модели (%), чем выше тем лучше:  62.27
#Абсолютное значение ошибки, чем ниже тем лучше:  19.25
#Параметры:  (-11, 9, 0.16788040181225602, 110, 60)

#Качество модели (%), чем выше тем лучше:  57.19
#Абсолютное значение ошибки, чем ниже тем лучше:  55.02
#Параметры:  (-10, 6, 0.29853826189179594, 100, 45)

#Качество модели (%), чем выше тем лучше:  55.27
#Абсолютное значение ошибки, чем ниже тем лучше:  23.37
#Параметры:  (-11, 8, 0.42169650342858223, 100, 60)

#Качество модели (%), чем выше тем лучше:  55.25
#Абсолютное значение ошибки, чем ниже тем лучше:  23.4
#Параметры:  (-11, 8, 0.41686938347033542, 130, 80)

#Качество модели (%), чем выше тем лучше:  67.45
#Абсолютное значение ошибки, чем ниже тем лучше:  15.13
#Параметры:  (-10, 9, 0.34673685045253161, 110, 80)

#Качество модели (%), чем выше тем лучше:  62.29
#Абсолютное значение ошибки, чем ниже тем лучше:  13.36
#Параметры:  (-10, 6, 0.31622776601683794, 110, 100)

#Качество модели (%), чем выше тем лучше:  54.62
#Абсолютное значение ошибки, чем ниже тем лучше:  14.6
#Параметры:  (-12, 8, 0.31622776601683794, 110, 60)

#Качество модели (%), чем выше тем лучше:  57.63
#Абсолютное значение ошибки, чем ниже тем лучше:  11.13
#Параметры:  (-12, 9, 0.31622776601683794, 110, 60)

#Качество модели (%), чем выше тем лучше:  52.92
#Абсолютное значение ошибки, чем ниже тем лучше:  78.53
#Параметры:  (-10, 6, 0.47863009232263831, 110, 100)

#Качество модели (%), чем выше тем лучше:  62.2
#Абсолютное значение ошибки, чем ниже тем лучше:  23.39
#Параметры:  (-12, 9, 0.33113112148259111, 110, 60)

#-11,8,0.31622776601683794,100,80
#==============================================================================
# Блок завершен
#==============================================================================


parameters = product(lower_windows, upper_windows, changepoint_prior_scales,seasonality_prior_scale,holidays_prior_scale)
parameters_list = list(parameters)
        
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

    print('Precess: ',len(parameters_list ),'/',chunksize)
    for i in range(parallel_tasks):
        params=parameters_list [chunksize * i:chunksize * (i + 1)]
        if len(params)==0: continue
        print('start ',(i+1),' process...')
    
        p = multiprocessing.Process(name='chunk '+str(i+1),
                target=mp_worker,
                args=(params,
                        df,pediods,changepoints,
                      out_q,log_queue, worker_configurer))
        procs.append(p)
        p.start()

    print('get queue...')
    # Collect all results into a single result dict. We know how many dicts
    # with results to expect.    
    resultdict = {}
    for _ in procs:
        resultdict.update(out_q.get())

        
    print('store values...')
    #соединяем список параметров
    res_array=[]
    for i in range(parallel_tasks):
        key1='chunk '+str(i+1)
        if key1 not in resultdict: break
        rs=resultdict[key1]
        for j in range(len(parameters_list)):
            key2=key1+'_'+str(j)
            if key2 not in rs: break
            res_array.append(rs[key2])
    #del resultdict
    
    print('wait processes finish...')
    # Wait for all worker processes to finish
    for p in procs:
        p.join()
    
    
        
    #создаем датасет, сортируем
    best_params=pd.DataFrame(data=res_array)
    if best_params.shape[0]>0:
        best_params.sort_values([1], inplace=True)

        print ('Elapsed (min): %s' % (round((time.time() - tstart)/60,0)))
        print ('Лучшие параметры на валидации')    
        print ('Качество модели (%), чем выше тем лучше: ', 100-round(best_params.iloc[0,1],2))
        print ('Абсолютное значение ошибки, чем ниже тем лучше: ', round(best_params.iloc[0,2],2))
        print ('Параметры: ',best_params.iloc[0,0])
        print ('lower_window,upper_window,changepoint_prior_scale,seasonality_prior_scale,holidays_prior_scale')
    else:
        print('empty array')
    
    
    print('wait listener finish...')
    listener.terminate()#Завершаем процессы логгирования
    print('Done')
    