# Time Series Analysis

## Models for Time Series

#### [Moving Average (MA) Models](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc444.htm#MA)

The Moving Average (MA) model treats the time series as a linear regression of the current value against the noise terms of the previous values of the series. It takes the form

<center><a href="https://www.codecogs.com/eqnedit.php?latex=X_{t}=\mu&plus;A_{t}-\theta_{1}A_{t-1}-\theta_{2}A_{t-2}-...-\theta_{q}A_{t-q}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?X_{t}=\mu&plus;A_{t}-\theta_{1}A_{t-1}-\theta_{2}A_{t-2}-...-\theta_{q}A_{t-q}" title="X_{t}=\mu+A_{t}-\theta_{1}A_{t-1}-\theta_{2}A_{t-2}-...-\theta_{q}A_{t-q}" /></a></center>

where X<sub>t</sub> is the time series, &mu; is the mean of the series, A<sub>t-i</sub> are the noise terms, and &theta;<sub>1</sub>,...&theta;<sub>q</sub> are parameters of the model.

#### [Autoregressive (AR) Models](https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc444.htm#AR)

An Autoregressive (AR) model looks like the following:

<center><a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;X_{t}=\delta&plus;\phi_{1}X_{t-1}&plus;\phi_{2}X_{t-2}&plus;...&plus;\phi_{p}X_{t-p}&plus;\epsilon" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\inline&space;X_{t}=\delta&plus;\phi_{1}X_{t-1}&plus;\phi_{2}X_{t-2}&plus;...&plus;\phi_{p}X_{t-p}&plus;\epsilon" title="X_{t}=\delta+\phi_{1}X_{t-1}+\phi_{2}X_{t-2}+...+\phi_{p}X_{t-p}+\epsilon" /></a></center>

where X<sub>t</sub> is the time series in question, &straightepsilon; is a noise term, and

<center><a href="https://www.codecogs.com/eqnedit.php?latex=\delta=\mu\big(1-\sum_{i=1}^{p}\phi_{i})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\delta=\mu\big(1-\sum_{i=1}^{p}\phi_{i})" title="\delta=\mu\big(1-\sum_{i=1}^{p}\phi_{i})" /></a></center>


#### [Autoregressive Integrated Moving Average (ARIMA) Model](https://machinelearningmastery.com/gentle-introduction-box-jenkins-method-time-series-forecasting/)

## Time Series in Java

Jacob Rachiele has written [an amazing set of tools for time series in pure Java](https://github.com/signaflo/java-timeseries) which I'm going to be using to bring all of this into the computational realm.

The [Javadoc can be found here](https://javadoc.io/doc/com.github.signaflo/timeseries/0.4)



[Codecogs Emded Latex](https://www.codecogs.com/latex/eqneditor.php)
