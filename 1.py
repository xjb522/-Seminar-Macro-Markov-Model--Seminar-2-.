import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import tidyfinance as td

print("succ")

Stock = ["^GSPC","AAPL"]

data = td.download_data(
    domain="Stock Prices",
    symbols=Stock,
    start_date="2012-01-01",
    end_date="2026-01-01"
)

d_colums =data.columns

data = pd.DataFrame(data,columns=d_colums)

data_pivot = pd.pivot(data, columns="symbol",index="date",values="adjusted_close").astype(float)

for i in Stock:
    data_pivot[f"",i]

data_pivot["log"]=np.log(data_pivot["^GSPC"])

def OLS_test(y, x, constant=1):
    """    
    If constant 1 we have a Y = b + a*X. If 2 we have Y = a*X.
    
    """
    if constant==1:
       one=np.ones(x.size)

       Combine = np.column_stack((one,x))

       part_1 = np.linalg.inv(Combine.T @ Combine)

       part_2 = Combine.T @ y

       OlS = part_1 @ part_2
       return OlS
    elif constant==2:
        pass


print(OLS_test(data_pivot["log"], data_pivot["log"]))



    
    