import pandas as pd

file = "AAPL.CSV"

xl = pd.read_csv(file)
date = xl['Date']
Close = xl['Close']
AD_Position = Close.copy()
Change = [0]* len(date)
Momentum = [0]* len(date)

dates = [i.split(' ', 1)[0] for i in date]

reference = dates[0]


for i in range (1,len(date)):
    if Close[i] > Close[i-1] :
        Momentum[i] = "1"
        Change[i] = (Close[i]-Close[i-1])/Close[i-1]
    else :
        Momentum[i] = "0"
        Change[i] = (Close[i-1] - Close[i]) / Close[i - 1]

xl = pd.DataFrame({'Date':date, 'Close':Close,'Change':Change,'Momentum':Momentum}) # a represents closing date b represents closing value c represents close change and d represents momentum

xl.to_csv("Apple_Modified.csv",index=False,header=True)