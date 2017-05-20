from fbprophet import Prophet
import numpy as np
import math
import multiprocessing
from itertools import product
import pandas as pd
import datetime as dt
import time


df=pd.read_pickle('MG_Sales.pickle',compression='gzip')

#Формируем выборку
time_series=pd.DataFrame(data=df[(df['ЦветМеталла']=='Красное золото')&(df['Дата']>=dt.date(2015,1,1))].groupby('Дата')['Количество'].sum())
time_series.index.name='ds'
time_series.columns=['y']
time_series['y'] = np.log(time_series['y'])

folds=5
#changepoint_prior_scale=0.003

#для валидации используем хвост временного ряда
last=time_series.iloc[-1].name
startDt=dt.datetime(last.year,last.month,last.day)
lastDay=dt.datetime(last.year,last.month,1)-dt.timedelta(seconds=1)
startmonth=dt.datetime(lastDay.year,lastDay.month,1)

#перебор гиперпараметров
lower_windows=list(range(-11,-9,1))
upper_windows=list(range(2,5,1))
seasonality_prior_scale=[90]#list(range(15,120,5))
holidays_prior_scale=[5]#list(range(1,100,5))
#n_changepoints=list(range(10,110,3))
changepoint_prior_scales=[0.008,0.009]#np.logspace(-2.3, -2.0, num=4, endpoint=True)

parameters = product(lower_windows, upper_windows, changepoint_prior_scales,seasonality_prior_scale,holidays_prior_scale)
parameters_list = list(parameters)


def mp_worker(params_grid, df,pediods,changepoints, out_q):
    name = multiprocessing.current_process().name
    """ The worker function, invoked in a process. 'nums' is a
        list of numbers to factor. The results are placed in
        a dictionary that's pushed to a queue.
    """
    outdict = {}
    count=-1
    for params in params_grid:        
        count+=1
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
        outdict[name+'_'+str(count)]=[params,np.mean(mape),np.mean(mae)]    
    #записываем в очередь потока
    outdict[name]=outdict
    out_q.put(outdict)
    

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
nprocs=22#Одновременное количество задач

lower_windows=list(range(-14,0,1))
upper_windows=list(range(0,7,1))
seasonality_prior_scale=[10,40,90]#list(range(10,110,3))
holidays_prior_scale=[5,55,70]#list(range(5,100,3))
#n_changepoints=list(range(10,110,3))
changepoint_prior_scales=np.logspace(-3, -1., num=20, endpoint=True)
#==============================================================================
# Блок завершен
#==============================================================================


parameters = product(lower_windows, upper_windows, changepoint_prior_scales,seasonality_prior_scale,holidays_prior_scale)
parameters_list = list(parameters)
        
#запускаем процессы
if __name__ == '__main__':
    tstart = time.time()
    
    out_q = multiprocessing.Queue()
    chunksize = int(math.ceil(len(parameters_list ) / float(nprocs)))
    procs = []

    print('Precess: ',len(parameters_list ),'/',chunksize)
    for i in range(nprocs):
        params=parameters_list [chunksize * i:chunksize * (i + 1)]
        if len(params)==0: continue
        print('start ',(i+1),' process...')
    
        p = multiprocessing.Process(name='chunk '+str(i),
                target=mp_worker,
                args=(params,
                        df,pediods,changepoints,
                      out_q))
        procs.append(p)
        p.start()

    print('get queue...')
    # Collect all results into a single result dict. We know how many dicts
    # with results to expect.    
    resultdict = {}
    for _ in procs:
        resultdict.update(out_q.get())

    print('wait processes finish...')
    # Wait for all worker processes to finish
    for p in procs:
        p.join()
    
    #соединяем список параметров
    res_array=[]
    for i in range(nprocs):
        key1='chunk '+str(i)
        if key1 not in resultdict: break
        rs=resultdict[key1]
        for j in range(len(parameters_list)):
            key2=key1+'_'+str(j)
            if key2 not in rs: break
            res_array.append(rs[key2])
    del resultdict
        
    #создаем датасет, сортируем
    best_params=pd.DataFrame(data=res_array)
    best_params.sort_values([1], inplace=True)

    print ('Elapsed (min): %s' % (round((time.time() - tstart)/60,0)))
    print ('Лучшие параметры на валидации')    
    print ('Качество модели (%), чем выше тем лучше: ', 100-round(best_params.iloc[0,1],2))
    print ('Абсолютное значение ошибки, чем ниже тем лучше: ', round(best_params.iloc[0,2],2))
    print ('Параметры: ',best_params.iloc[0,0])
    print ('lower_window,upper_window,changepoint_prior_scale,seasonality_prior_scale,holidays_prior_scale')
        
