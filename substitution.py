import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,6
import datetime as dt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.filters.hp_filter import hpfilter
from statsmodels.tsa.stattools import acf, pacf

#First import to check the data was imported correctly; cleaned in R so shouldn't be a problem
path = "D:/BigDataAnalytics/eChat/Analysis/"
file_ts_data = "contactsbytype.csv"
file_stats_2015 = "contacts&statsbytype2015.csv"

def read_data(path,filename):
    data = pd.read_csv(path+filename)
    return data
    
data1 = read_data(path,file_ts_data)
data2 = read_data(path, file_stats_2015)

data1.head()
data2.head()

#Import the data and overwrite the pandas dataframe currently storing the time series data with date parser 
#object converted to datetime 
dateparse = lambda dates: dt.datetime.strptime(dates, '%d-%b-%y')
data = pd.read_csv(path+file_stats_2015, parse_dates=['DATE_OF_CALL'], index_col = 'DATE_OF_CALL', date_parser=dateparse)
print(data.head())
print(data.columns.values,sep='\n', end='\n')

#Set the timeseries data
#count
ts_outboundcall = data['COUNT_OF_CONTACTS.Outbound Call']
ts_outboundcall = ts_outboundcall.sort_index()
#Effort

ts_offline = data['COUNT_OF_CONTACTS.Offline']
ts_offline = ts_offline.sort_index()

ts_chat = data['COUNT_OF_CONTACTS.Chat']
ts_chat = ts_chat.sort_index()

ts_email = data['COUNT_OF_CONTACTS.Inbound Email']
ts_email = ts_email.sort_index()

ts_voice = data['COUNT_OF_CONTACTS.Inbound Call']
ts_voice = ts_voice.sort_index()

ts_voice_xfer = data['COUNT_OF_CONTACTS.Inbound Call XFER']
ts_voice_xfer = ts_voice_xfer.sort_index()

ts_letter = data['COUNT_OF_CONTACTS.Letter - Inbound']
ts_letter = ts_letter.sort_index()


#Initial plot of the two time series for chat and inbound voice answered
fig1 = plt.figure()
plt.plot(ts_voice_xfer)
plt.plot(ts_voice)
plt.plot(ts_outboundcall)
plt.legend(loc='best')

fig_digi = plt.figure()
plt.plot(ts_chat)
plt.plot(ts_offline)
plt.plot(ts_email)
plt.plot(ts_letter)
plt.legend(loc='best')

def test_stationarity(timeseries, title=""):
    #determing the rolling statistics
    rollmean = timeseries.rolling(center=False, window = 30).mean()
    rolstd = timeseries.rolling(center=False, window = 30).std()

    #Plot rolling stats
    fig = plt.figure()
    #orig = plt.plot(timeseries, color='blue', label = 'Orginal')
    mean = plt.plot(rollmean, color = 'red', label = "Rolling Mean")
    std = plt.plot(rolstd, color = 'black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title(title+' Rolling Mean & Standard Deviation')
    
    #Perform Dickey-Fuller Test:
    print('Results from Dickey Fuller test: '+title)
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic', 
                                'p-value', 
                                '#Lags Used', 
                                'Number of Observations Used'])
    for k,v in dftest[4].items():
        dfoutput['Critical Value (%s)'%k] = v 
    print(dfoutput)

def test_stationarity_AUG(timeseries1, timeseries2, title:list):
    #determing the rolling statistics
    rollmean1 = timeseries1.rolling(center=False, window = 30).mean()
    #rolstd1 = timeseries1.rolling(center=False, window = 30).std()
    rollmean2 = timeseries2.rolling(center=False, window = 30).mean()
    #rollstd2 = timeseries2.rolling(center =False, window =30).std()
    #Plot rolling stats
    fig = plt.figure()
    #orig = plt.plot(timeseries, color='blue', label = 'Orginal')
    mean1 = plt.plot(rollmean1, color = 'darkslateblue', label = "Rolling Mean %s"%title[0])
    mean2 = plt.plot(rollmean2, color = 'darkred', label = "Rolling Mean %s"%title[1])
    #std1 = plt.plot(rolstd1, color = 'black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('%s and %s Rolling Mean'%(title[0],title[1]))
    
    #Perform Dickey-Fuller Test:
    print('Results from Dickey Fuller test: '+title)
    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], 
                         index=['Test Statistic', 
                                'p-value', 
                                '#Lags Used', 
                                'Number of Observations Used'])
    for k,v in dftest[4].items():
        dfoutput['Critical Value (%s)'%k] = v 
    print(dfoutput)
ls = [ts_chat.dropna(), ts_voice.dropna(), ts_voice_xfer.dropna(), ts_outboundcall.dropna(), ts_offline.dropna()]
ls_names = ['Chat', 'Inbound Voice', 'Inbound Voice XFER', 'Oubound Voice', 'Offline']

def process_ts(ls: list, names: list):
    count = 0
    for i in ls:
        test_stationarity(i, names[count])
        count = count+ 1

process_ts(ls, ls_names)

#decomposition
def decomp_ts(timeseries, name:str, freq:int):

    decomposition = seasonal_decompose(timeseries, freq=freq)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    fig = plt.figure()
    plt.subplot(411)
    plt.plot(timeseries, label=name+' Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label=name+' Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal,label=name+' Seasonality')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(residual, label=name+' Residuals')
    plt.legend(loc='best')
    plt.tight_layout()
    return trend, seasonal, residual, decomposition

t,s,r,d = decomp_ts(ts_chat.dropna(), 'Chat', 90)
test_stationarity(residual.dropna())

lag_acf = acf(residual.dropna(), nlags=20)
lag_pacf = pacf(residual.dropna(), nlags=20, method='ols')

plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(residual.dropna())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(residual.dropna())),linestyle='--',color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(residual.dropna())),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(residual.dropna())),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()

#hpfilter
cycle, trend = hpfilter(ts_chat.dropna(), 6.25)
fig = plt.figure()
plt.plot(cycle)
plt.plot(trend)
plt.plot(ts_chat.dropna())

cycle, trend = hpfilter(ts_voice.dropna(), 6.25)
fig = plt.figure()
plt.plot(cycle)
plt.plot(trend)
plt.plot(ts_voice.dropna())

#########################################################################
##                  substitution data breakdown for bar chart
#########################################################################

f_sub = "substitution_data_final.csv"
sub_data = pd.read_csv(path+f_sub, parse_dates=['DATE_OF_CALL'], 
                       index_col = 'DATE_OF_CALL', date_parser=dateparse)

generated = sub_data['generated'] - sub_data['COUNT_OF_CONTACTS']

sub_data['gen_final'] = generated 
percentage = (sub_data['COUNT_OF_CONTACTS.Chat'] - 
              sub_data['generated'])/sub_data['COUNT_OF_CONTACTS.Chat']
sub_data['subs_percent'] = percentage
sub_data = sub_data[sub_data.index.year < 2017]

M_sub_avgs = sub_data.groupby(by =[sub_data.index.month, sub_data.index.year]).mean()

M_sub_sdevs = sub_data.groupby(by = [sub_data.index.month, sub_data.index.year]).std()

mths = np.arange(12)
mths_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']

fig,ax1 = plt.subplots()

p1 = ax1.bar(mths, M_sub_avgs['COUNT_OF_CONTACTS.x'], 
             width = 0.5, yerr=M_sub_sdevs['COUNT_OF_CONTACTS.x'],
             color='darkslateblue',
             label = 'Direct Subsitutions')
p2 = ax1.bar(mths, M_sub_avgs['COUNT_OF_CONTACTS.y'], 
             width = 0.5, bottom = M_sub_avgs['COUNT_OF_CONTACTS.x'],
             yerr=M_sub_sdevs['COUNT_OF_CONTACTS.y'],
             color='blueviolet',
             label = 'New BB')
p3 = ax1.bar(mths, M_sub_avgs['COUNT_OF_CONTACTS'], 
             width = 0.5, yerr=M_sub_sdevs['COUNT_OF_CONTACTS'],
             bottom=[i+j for i,j in zip(M_sub_avgs['COUNT_OF_CONTACTS.x'],M_sub_avgs['COUNT_OF_CONTACTS.y'])],
             color = 'mediumorchid',
             label = 'Long time no faults')
p4 = ax1.bar(mths, M_sub_avgs['gen_final'], 
             width = 0.5, yerr=M_sub_sdevs['gen_final'],
             bottom=[i+j+k for i,j,k in zip(M_sub_avgs['COUNT_OF_CONTACTS.x'],
                                        M_sub_avgs['COUNT_OF_CONTACTS.y'],
                                        M_sub_avgs['COUNT_OF_CONTACTS'])],label = 'Generated',color = 'indianred')

plt.ylabel('Chat Contacts (count)')
plt.title('Chat Substitution: Breakdown of subsitution types by month')
plt.xticks(mths,mths_names)

ax2 = ax1.twinx()
p5 = ax2.plot(mths, M_sub_avgs['subs_percent'], '--k',label = 'percentage_substituted')
ax2.set_ylim([0,100])
ax1.legend(loc='best')
ax2.legend(loc='best')
fig.tight_layout()
fig.savefig("chat_volume_breakdown.png")



