import pandas as pd

file1 = "Apple_Modified.CSV"
file2 = "Amazon_Modified.CSV"
# file3 = "Alibaba_Modified.CSV"
file3 = "Google_Modified.CSV"
file4 = "HP_Modified.CSV"
file5 = "IBM_Modified.CSV"
file6 = "Oracle_Modified.CSV"
file7 = "Intel_Modified.CSV"
file8 = "Lenovo_Modified.CSV"
file9 = "Microsoft_Modified.CSV"
file10 = "Nasdaq_Modified.CSV"

xl1 = pd.read_csv(file1)
date_1 = xl1['Date']
Close_1 = xl1['Close']
Change_1 = xl1['Change']
Momentum_1 = xl1['Momentum']

xl2 = pd.read_csv(file2)
date_2 = xl2['Date']
Close_2 = xl2['Close']
Change_2 = xl2['Change']
Momentum_2 = xl2['Momentum']

xl3 = pd.read_csv(file3)
date_3 = xl3['Date']
Close_3 = xl3['Close']
Change_3 = xl3['Change']
Momentum_3 = xl3['Momentum']

xl4 = pd.read_csv(file4)
date_4 = xl4['Date']
Close_4 = xl4['Close']
Change_4 = xl4['Change']
Momentum_4 = xl4['Momentum']


xl5 = pd.read_csv(file5)
date_5 = xl5['Date']
Close_5 = xl5['Close']
Change_5 = xl5['Change']
Momentum_5 = xl5['Momentum']



xl6 = pd.read_csv(file6)
date_6 = xl6['Date']
Close_6 = xl6['Close']
Change_6 = xl6['Change']
Momentum_6 = xl6['Momentum']

xl7 = pd.read_csv(file7)
date_7 = xl7['Date']
Close_7 = xl7['Close']
Change_7 = xl7['Change']
Momentum_7 = xl7['Momentum']

xl8 = pd.read_csv(file8)
date_8 = xl8['Date']
Close_8 = xl8['Close']
Change_8 = xl8['Change']
Momentum_8 = xl8['Momentum']


xl9 = pd.read_csv(file9)
date_9 = xl9['Date']
Close_9 = xl9['Close']
Change_9 = xl9['Change']
Momentum_9 = xl9['Momentum']


xl10 = pd.read_csv(file10)
date_10 = xl10['Date']
Close_10 = xl10['Close']
Change_10 = xl10['Change']
Momentum_10 = xl10['Momentum']

n = 5
Close,Change,Stock_Price_Volatility,Stock_Momentum, Index_Volatility,Index_Momentum, Sector_Momentum = [],[],[],[],[],[],[]
for i in range(5,1762):
    #file1 stock process
    file1_close = (Change_1[i]+Change_1[i-1]+Change_1[i-2]+Change_1[i-3]+Change_1[i-4])/5
    file1_Stock_Momentum = (Momentum_1[i] + Momentum_1[i - 1] + Momentum_1[i - 2] + Momentum_1[i - 3] + Momentum_1[i - 4]) / 5
    #file1 stock process
    file2_close = (Change_2[i]+Change_2[i-1]+Change_2[i-2]+Change_2[i-3]+Change_2[i-4])/5
    file2_Stock_Momentum = (Momentum_2[i] + Momentum_2[i - 1] + Momentum_2[i - 2] + Momentum_2[i - 3] + Momentum_2[i - 4]) / 5
    #file1 stock process
    file3_close = (Change_3[i]+Change_3[i-1]+Change_3[i-2]+Change_3[i-3]+Change_3[i-4])/5
    file3_Stock_Momentum = (Momentum_3[i] + Momentum_3[i - 1] + Momentum_3[i - 2] + Momentum_3[i - 3] + Momentum_3[i - 4]) / 5
    #file1 stock process
    file4_close = (Change_4[i]+Change_4[i-1]+Change_4[i-2]+Change_4[i-3]+Change_4[i-4])/5
    file4_Stock_Momentum = (Momentum_4[i] + Momentum_4[i - 1] + Momentum_4[i - 2] + Momentum_4[i - 3] + Momentum_4[i - 4]) / 5
    #file1 stock process
    file5_close = (Change_5[i]+Change_5[i-1]+Change_5[i-2]+Change_5[i-3]+Change_5[i-4])/5
    file5_Stock_Momentum = (Momentum_5[i] + Momentum_5[i - 1] + Momentum_5[i - 2] + Momentum_5[i - 3] + Momentum_5[i - 4]) / 5
    #file1 stock process
    file6_close = (Change_6[i]+Change_6[i-1]+Change_6[i-2]+Change_6[i-3]+Change_6[i-4])/5
    file6_Stock_Momentum = (Momentum_6[i] + Momentum_6[i - 1] + Momentum_6[i - 2] + Momentum_6[i - 3] + Momentum_6[i - 4]) / 5
    #file1 stock process
    file7_close = (Change_7[i]+Change_7[i-1]+Change_7[i-2]+Change_7[i-3]+Change_7[i-4])/5
    file7_Stock_Momentum = (Momentum_7[i] + Momentum_7[i - 1] + Momentum_7[i - 2] + Momentum_7[i - 3] + Momentum_7[i - 4]) / 5
    #file1 stock process
    file8_close = (Change_8[i]+Change_8[i-1]+Change_8[i-2]+Change_8[i-3]+Change_8[i-4])/5
    file8_Stock_Momentum = (Momentum_8[i] + Momentum_8[i - 1] + Momentum_8[i - 2] + Momentum_8[i - 3] + Momentum_8[i - 4]) / 5
    #file1 stock process
    file9_close = (Change_9[i]+Change_9[i-1]+Change_9[i-2]+Change_9[i-3]+Change_9[i-4])/5
    file9_Stock_Momentum = (Momentum_9[i] + Momentum_9[i - 1] + Momentum_9[i - 2] + Momentum_9[i - 3] + Momentum_9[i - 4]) / 5
    #file1 stock process
    file10_close = (Change_10[i]+Change_10[i-1]+Change_10[i-2]+Change_10[i-3]+Change_10[i-4])/5
    file10_Stock_Momentum = (Momentum_10[i] + Momentum_10[i - 1] + Momentum_10[i - 2] + Momentum_10[i - 3] + Momentum_10[i - 4]) / 5
    #stock momentum for given day
    Sector_momentum = (file1_Stock_Momentum  + file2_Stock_Momentum + file3_Stock_Momentum + file4_Stock_Momentum + file5_Stock_Momentum + file6_Stock_Momentum + file7_Stock_Momentum + file8_Stock_Momentum + file9_Stock_Momentum)/9

    Close.append(Close_1[i])
    Change.append(Momentum_1[i])
    Stock_Price_Volatility.append(file1_close)
    Stock_Momentum.append(file1_Stock_Momentum)
    Index_Volatility.append(file10_close)
    Index_Momentum.append(file10_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

    Close.append(Close_2[i])
    Change.append(Momentum_2[i])
    Stock_Price_Volatility.append(file2_close)
    Stock_Momentum.append(file2_Stock_Momentum)
    Index_Volatility.append(file10_close)
    Index_Momentum.append(file10_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

    Close.append(Close_3[i])
    Change.append(Momentum_3[i])
    Stock_Price_Volatility.append(file3_close)
    Stock_Momentum.append(file3_Stock_Momentum)
    Index_Volatility.append(file10_close)
    Index_Momentum.append(file10_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

    Close.append(Close_4[i])
    Change.append(Momentum_4[i])
    Stock_Price_Volatility.append(file4_close)
    Stock_Momentum.append(file4_Stock_Momentum)
    Index_Volatility.append(file10_close)
    Index_Momentum.append(file10_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

    Close.append(Close_5[i])
    Change.append(Momentum_5[i])
    Stock_Price_Volatility.append(file5_close)
    Stock_Momentum.append(file5_Stock_Momentum)
    Index_Volatility.append(file10_close)
    Index_Momentum.append(file10_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

    Close.append(Close_6[i])
    Change.append(Momentum_6[i])
    Stock_Price_Volatility.append(file6_close)
    Stock_Momentum.append(file6_Stock_Momentum)
    Index_Volatility.append(file10_close)
    Index_Momentum.append(file10_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

    Close.append(Close_7[i])
    Change.append(Momentum_7[i])
    Stock_Price_Volatility.append(file7_close)
    Stock_Momentum.append(file7_Stock_Momentum)
    Index_Volatility.append(file10_close)
    Index_Momentum.append(file10_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

    Close.append(Close_8[i])
    Change.append(Momentum_8[i])
    Stock_Price_Volatility.append(file8_close)
    Stock_Momentum.append(file8_Stock_Momentum)
    Index_Volatility.append(file10_close)
    Index_Momentum.append(file10_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)

    Close.append(Close_9[i])
    Change.append(Momentum_9[i])
    Stock_Price_Volatility.append(file9_close)
    Stock_Momentum.append(file9_Stock_Momentum)
    Index_Volatility.append(file10_close)
    Index_Momentum.append(file10_Stock_Momentum)
    Sector_Momentum.append(Sector_momentum)


xl = pd.DataFrame({'Close':Close, 'Change':Change,'Stock_Price_Volatility':Stock_Price_Volatility,'Stock_Momentum':Stock_Momentum,'Index_Volatility':Index_Volatility,'Index_Momentum':Index_Momentum,'Sector_Momentum':Sector_Momentum}) # a represents closing date b represents closing value c represents close change and d represents momentum
#
xl.to_csv("Input_Dataset.csv",index=False,header=False)

