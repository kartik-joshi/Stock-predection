from pandas_datareader import data
import pandas as pd


# Define which online source one should use
data_source = 'yahoo'

# We would like all available data from 01/01/2000 until 12/31/2016.
start_date = '2010-01-01'
end_date = '2016-12-31'

symbols = ['BABA']
# symbols = ['AAPL','AMZN','BABA','MSFT','GOOG','IBM','ORCL','INTC','HPQ','LNVGY']


for i in symbols:
    data.DataReader(i,'yahoo',start_date,end_date).to_csv(i+'.csv')