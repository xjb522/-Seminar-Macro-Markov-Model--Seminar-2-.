from patsy.build import test__examine_factor_types
from patsy.mgcv_cubic_splines import test_te_1smooth
from sqlalchemy.dialects.mssql.information_schema import columns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy 
import tidyfinance as td


variables = [
    "PAYEMS",
    "INDPRO",
    "CMRMTSPL",
    "W875RX1",
    "USREC",
    "RECPROUSM156N"
]

def Data_download(vari=None, domain="Stock_Prices", start="1980-01-01", till="2026-01-01"):
    """
    This is a function to download data with a combination of tidyfinance and pandas package.
    Vari is the variabels you want to look up and domain is the database api you want to draw data from (Default is Yahoo prices "Stock_Prices", Fred is "FRED".)
    If you want other databass domain you need to go to tidyfinance or look at this:
    domain : str
        The domain of the dataset to download, given as one of the
        canonical names returned by 'list_supported_datasets()':
        'Fama-French', 'Global Q', 'Goyal-Welch', 'WRDS', 'Pseudo Data',
        'Index Constituents', 'FRED', 'Stock Prices',
        'Open Source Asset Pricing', 'Global Factor Data',
        'Pastor-Stambaugh', 'Stambaugh-Yuan', 'Damodaran',
        'Tidy Finance'. The
        previous short names (e.g. 'famafrench', 'wrds', 'pseudo') are
        still accepted but deprecated and will be removed in a future
        release.
    """
    try:
        Data = td.download_data(
        domain=domain,
        series=vari,
        start_date=start,
        end_date=till)
    except ValueError as e:
        print("You don gerted it up!!! #LOOK AT THE CENTER VALUES!!! This ",e," is wrong")
        return None
    
    try:
        cc = Data.columns

        Data = pd.DataFrame(Data, columns=cc)

        Data = Data.set_index("date")

        Data = pd.pivot(Data, columns="series", values="value")

        print("Gerty done got the data out")
    except ValueError as e:
        print("Big dog... You done gerted it again. Look at the code. This value",e,"is wrong")
        return None
        

    return Data


#Right input
Data = Data_download(variables,"FRED","1980-01-01","2026-01-01")

Data_Macro = Data.drop(["USREC","RECPROUSM156N"], axis=1)

print(Data.shape)

mean = []
std = []

for i in Data_Macro.columns:
    mean.append(np.mean(Data[i]))
    std.append(np.std(Data[i]))

standard = {}

for i, name in enumerate(Data_Macro.columns):
    standard[name]= (Data[name]-mean[i])/std[i]

standard = pd.DataFrame(standard)

standard = standard.rename(columns={"CMRMTSPL":"Real Manufacturing and Trade Industries Sales", "INDPRO":"Industrial Production: Total Index", "PAYEMS":"All Employees, Total Nonfarm","W875RX1":"Real personal income excluding current transfer receipts"}
)

# print(standard.columns)
#["Real Manufacturing and Trade Industries Sales", "Industrial Production: Total Index", "All Employees, Total Nonfarm","Real personal income excluding current transfer receipts"]

plt.clf()
plt.close()

plt.figure(figsize=(12,6))

plt.plot(standard)
plt.legend(standard.columns)
plt.grid()
plt.show()

print(standard.isna().sum())
print(standard.size)

eta = np.random.normal(0,1,2212)

plt.clf()
plt.close()

plt.hist(eta,bins=100)
plt.grid()
plt.show()

print(np.delete(standard,0,0).shape, np.delete(standard,-1,0).shape)


Standard_shift = np.delete(standard,0,0).astype(float)

Standard_array = np.delete(standard,-1,0).astype(float)

print(Standard_array.shape)
print(Standard_shift.shape)

def OLS(Y,X):
    try:
        x_led_1 = np.column_stack((np.ones(X.size), X))
        x_led_2 = np.linalg.inv(x_led_1.T @ x_led_1) 
        y_led = np.linalg.vecdot(x_led_1.T, Y)
        ols = np.linalg.vecdot(x_led_2, y_led)
    except ValueError as e:
        print(f"What is {e} value??? A gert number???")
        return X.shape, len(X), X, x_led_2
    return ols[0], ols[1]

Gamma = []
theta = []

for i in range(0,4):
    Gamma.append(OLS(Standard_array[:,i], Standard_shift[:,i])[0])
    theta.append(OLS(Standard_array[:,i], Standard_shift[:,i])[1])





# %% Test run af ols
print(OLS(Standard_array[:,0], Standard_shift[:,0]))

print(Standard_array[:,0])

test_x = np.array([1,2,4,5])
test_y = np.array([2,3,5,5])

x_led_1 = np.column_stack((np.ones(test_x.size), test_x))

print(x_led_1.T)



y_led = np.linalg.vecdot(x_led_1.T, Y)
ols = np.linalg.vecdot(x_led_2, y_led)