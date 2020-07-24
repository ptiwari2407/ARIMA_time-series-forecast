# Import files and necessary libraries for computation
filename = "SEM_Tech.xlsx"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


df = pd.read_excel(filename, sheet_name= "question 2 raw data")
col_name = ['week', 'acquisitions']
df.columns = col_name

def augmented_dick_fuller_test(ts):
    dftest = adfuller(ts)
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)



# First we check for stationarity of the given time series and analyze visually with a rou
plt.figure(figsize=(12,12), dpi = 80)
plt.plot(df)
plt.xlabel("Weeks")
plt.ylabel("Acquisitions")
plt.title("Stationarity-check")
fig_name = "img/stationarity_check.png"
plt.savefig(fig_name)


# we observe that it has increasing trend and hence it is not stationary.
rolling_mean = df['acquisitions'].rolling(window=4).mean()
rolling_std =  df['acquisitions'].rolling(window=4).std()
orig = plt.plot(df['acquisitions'], color = 'blue', label = 'Original')
mean = plt.plot(rolling_mean, color='red', label = 'Rolling Mean')
std = plt.plot(rolling_std, color='black', label = 'Rolling standard deviation')
plt.title('Rolling Mean and standard Deviation')
plt.legend(loc = 'best')
fig_name = 'img/Rolling_mean_and_std.png'
plt.savefig(fig_name)

# Though stationarity assumption is taken in many TS models, almost none of practical time series are stationary.
# There are 2 major reasons behind non-stationarity of a TS:
# 1. Trend – varying mean over time.
# 2. Seasonality

ts = df['acquisitions']

# Uncomment the line below to check augmented-dick-fuller-test, we get p-value at 0.93 , which implies our time series # is not stationary.

# augmented_dick_fuller_test(ts)

ts_log = np.log(ts)
plt.figure(figsize=(12,12), dpi = 80)
plt.plot(ts_log)
plt.title('Log-OverView of Acquisitions')
plt.xlabel("Weeks")
plt.ylabel("Acquisitions_in_log")
fig_name = 'img/log-acquisitions.png'
plt.savefig(fig_name)

# we observe here an upward trend in the data.

# So, we try to eliminate trend and seasonality. This, we shall do by using the technique called differencing.
ts_log = pd.Series(ts_log)
plt.figure(figsize=(12,12), dpi = 80)
ts_log_diff = ts_log - ts_log.shift()
plt.plot(ts_log_diff)
plt.xlabel("Weeks")
plt.title("Time Series Log Difference through shift ")
plt.show()

ts_log_diff = ts_log_diff[1:]
augmented_dick_fuller_test(ts_log_diff)


# Now i shall apply ARIMA model for making the forecast. For this model we need
# p: lag order
# q: size of moving average window
# d: Degree of differencing, ts_log_diff is already differenced once, so we set it to zero if we use ts_log_diff, if we use ts_log, we set it to 1


# The values for p and q would be calculated through plotting acf( auto-corelation function)
# and PACF (partial auto-corelation Function)


log_acf = acf(ts_log_diff, nlags=20)
log_pacf = pacf(ts_log_diff, nlags= 20, method='ols')


plt.figure(figsize=(12,12), dpi = 80)
plt.plot(log_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function for log difference ')
fig_name = "img/Partial Autocorrelation.png"
plt.savefig(fig_name)


plt.figure(figsize=(12,12), dpi = 80)
plt.plot(log_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function for log difference')
fig_name = "img/Autocorrelation.png"
plt.savefig(fig_name)

# From observing this value, the values for p and q is almost equal to 1 for both.
# so p = q = 1 and d= 1(ts_log), we have parameters for our ARIMA model.

# There is no train-test distribution made to train the model, because of huge differences in the values,
# I usually do that, but for this data I shall be performing with-in the training data as accuracy measure

# fit the ARIMA model
# after tuning the parameters I found p=1, but q = 2 defines our model better
# this could be deduced from graph

model = ARIMA(ts_log, order=(1, 1, 2))
results_ARIMA = model.fit(disp=-1)
plt.figure(figsize=(12,12), dpi = 80)
plt.plot(ts_log_diff, label = 'Time Series log Difference')
plt.plot(results_ARIMA.fittedvalues, color='red', label = 'ARIMA predictions on log of Time-Series')
plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log_diff)**2))
fig_name = "img/ARIMA model on Logarithm of data"
plt.savefig(fig_name)


# looking over the model in the graph above, we can state that it is not overfitting and to a good extent captures the model

# Now, we have to take the model to an original scale(the current model is on logarithm scale)

predict_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)

# The way to convert the differencing to log scale is to add these differences consecutively to the base number


predictions_ARIMA_log = pd.Series(ts_log[1:])
predictions_ARIMA_log = predictions_ARIMA_log + predict_ARIMA_diff
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.figure(figsize=(12, 12), dpi = 80)
plt.plot(ts, label = 'Original data')
plt.plot(predictions_ARIMA, label = 'Predicted data')
plt.title("original data(orange) vs predicted data from ARIMA model(blue)")
fig_name = "img/ARIMA prediction model.png"
plt.savefig(fig_name)


# calculating Error through MSE
print("mean_squared_error is: %.4f" % mean_squared_error(ts[1:], predictions_ARIMA))

# making the predictions for over next 12 weeks
forecast = results_ARIMA.forecast(steps=12)[0]
predictions_12_weeks = np.round(np.exp(forecast))
print(predictions_12_weeks)






