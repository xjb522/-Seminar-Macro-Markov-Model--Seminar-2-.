from patsy.build import test__examine_factor_types
from patsy.mgcv_cubic_splines import test_te_1smooth
from sqlalchemy.dialects.mssql.information_schema import columns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tidyfinance as td

import statsmodels.api as sm


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

eta_t = Standard_array - Gamma - theta * Standard_shift


def AUG_DICKY_Fuller(v):

    v = pd.DataFrame(v)

    Diff_var = np.array(v.diff(1).dropna())

    pi = []

    # Lagged values: y_(t-1)
    Standard_shift = np.array(v.iloc[:-1]).astype(float)

    # Align observations
    for i in range(Diff_var.shape[1]):
        pi.append(OLS(Diff_var[:,i], Standard_shift[:,i])[1])

    li = []

    for i, pi_value in enumerate(pi):

        # Regression for the standard error
        Y = Diff_var[:,i]
        X = Standard_shift[:,i]

        X_matrix = np.column_stack((np.ones(len(X)), X))

        beta = np.linalg.lstsq(X_matrix, Y, rcond=None)[0]

        residuals = Y - X_matrix @ beta

        sigma2 = np.sum(residuals**2) / (len(Y) - 2)

        var_beta = sigma2 * np.linalg.inv(X_matrix.T @ X_matrix)

        se_pi = np.sqrt(var_beta[1,1])

        test = pi_value / se_pi

        if test < -2.86:
            print(f"The model {v.columns[i]} is stationary. T-value {test:.4f}")
            li.append(f"The model is stationary. T-value {test:.4f}")
        else:
            print(f"The model {v.columns[i]} is not stationary. T-value {test:.4f}")
            li.append(f"The model is not stationary. T-value {test:.4f}")

    return li


dddd = AUG_DICKY_Fuller(standard)


Diff_var = np.array(standard.diff(1).dropna())

pi = []

for i in range(0,4):
    pi.append(OLS(Diff_var[:,i], Standard_shift[:,i])[1])

def T_test_zero(pi_value):

    li = []

    for i in list(pi_value):
        pi_value = i
        test = pi_value / pi_value.std()
        if test.any() > 1.96:
            print(f"The model {i} is Stationary")
            li.append(f"The model is stationary. T-value {i}")
        else:
            print(f"The model {i} is not stationay")
            li.append(f"The model is not stationary. T-value {i}")
    return li

ADF = T_test_zero(pi)


## Macro

standard = np.asarray(standard, dtype=np.float64)

cov_matrix = np.cov(standard, rowvar=False)

Eigenvalues, eigenvector = np.linalg.eigh(cov_matrix)

w = eigenvector[:, -1]

F = standard @ w

plt.close()
plt.clf()

plt.plot(F)

plt.show()
plt.grid()

def normal_pdf(x, mu, sigma):

    a = 1 / (sigma * np.sqrt(2 * np.pi))

    b = -0.5 * ((x - mu) / sigma)**2

    return a * np.exp(b)

adf_f = AUG_DICKY_Fuller(F)


#  model

Data_Macro = Data_Macro.astype(float)

for i in Data_Macro.columns:
    Data_Macro[i] = np.log(Data_Macro[i])

Data_Macro = Data_Macro.diff(1).dropna()

mean = []
std = []

for i in Data_Macro.columns:
    mean.append(np.mean(Data_Macro[i]))
    std.append(np.std(Data_Macro[i]))

standard = {}

for i, name in enumerate(Data_Macro.columns):
    standard[name] = (Data_Macro[name] - mean[i]) / std[i]

standard = pd.DataFrame(standard)

standard = np.asarray(standard, dtype=np.float64)

cov_matrix = np.cov(standard, rowvar=False)

Eigenvalues, eigenvector = np.linalg.eigh(cov_matrix)

w = eigenvector[:, -1]

# Orient the factor towards positive economic growth
if np.sum(w) < 0:
    w = -w

F = standard @ w

plt.figure(figsize=(12, 5))
plt.plot(Data_Macro.index, F)
plt.axhline(0, color="black", linestyle="--")
plt.grid()
plt.show()

def Hamilton_filter(F, parameters):

    mu_0, mu_1, sigma_0, sigma_1, p00, p11 = parameters

    P = np.array([
        [p00, 1-p00],
        [1-p11, p11]
    ])

    # Stationary probabilities
    prob = np.array([
        (1-p11)/(2-p00-p11),
        (1-p00)/(2-p00-p11)
    ])

    filtered = []
    predicted_prob = []

    log_likelihood = 0

    for t in range(len(F)):

        # Step 1: Prediction
        predicted = P.T @ prob

        # Step 2: Conditional densities
        density_0 = normal_pdf(F[t], mu_0, sigma_0)
        density_1 = normal_pdf(F[t], mu_1, sigma_1)

        density = np.array([density_0, density_1])

        # Step 3: Joint density
        joint = density * predicted

        likelihood = np.sum(joint)

        if likelihood <= 0 or not np.isfinite(likelihood):
            return -np.inf, None, None

        # Step 4: Bayesian updating
        prob = joint / likelihood

        # Store probabilities
        filtered.append(prob)
        predicted_prob.append(predicted)

        # Log-likelihood
        log_likelihood += np.log(likelihood)

    return (
        log_likelihood,
        np.array(filtered),
        np.array(predicted_prob)
    )

def negative_log_likelihood(parameters, F):

    LL, filtered, predicted = Hamilton_filter(F, parameters)

    return -LL




initial_parameters = [
    -1.5,   # mu_0
    0.5,    # mu_1
    1.2,    # sigma_0
    0.6,    # sigma_1
    0.90,   # p00
    0.95    # p11
]

bounds = [
    (None, None),  # mu_0
    (None, None),  # mu_1
    (0.001, None), # sigma_0
    (0.001, None), # sigma_1
    (0.001, 0.999), # p00
    (0.001, 0.999)  # p11
]

result = minimize(
    negative_log_likelihood,
    initial_parameters,
    args=(F,),
    method="L-BFGS-B",
    bounds=bounds
)

print(result.success)
print(result.message)
print(result.x)

if not result.success:
    raise RuntimeError(result.message)

mu_0, mu_1, sigma_0, sigma_1, p00, p11 = result.x

if mu_0 > mu_1:

    mu_0, mu_1 = mu_1, mu_0

    sigma_0, sigma_1 = sigma_1, sigma_0

    p00, p11 = p11, p00

estimated_parameters = [
    mu_0, mu_1, sigma_0, sigma_1, p00, p11
]

P = np.array([
    [p00, 1-p00],
    [1-p11, p11]
])

print("Mean recession:", mu_0)
print("Mean expansion:", mu_1)

print("Variance recession:", sigma_0**2)
print("Variance expansion:", sigma_1**2)

print("Transition matrix:")
print(P)


LL, filtered, predicted = Hamilton_filter(
    F, estimated_parameters
)


plt.figure(figsize=(14, 5))

plt.plot(
    Data_Macro.index,
    filtered[:, 0],
    label="Estimated recession probability"
)

plt.axhline(0.5, color="red", linestyle="--")

plt.title("US Macroeconomic Regimes")
plt.ylabel("Probability")
plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()

NBER = Data["USREC"].reindex(Data_Macro.index)

plt.figure(figsize=(14, 5))

plt.plot(
    Data_Macro.index,
    filtered[:, 0],
    label="Estimated recession probability"
)

plt.fill_between(
    Data_Macro.index,
    0,
    1,
    where=NBER.eq(1).to_numpy(),
    color="grey",
    alpha=0.25,
    label="NBER recession"
)

plt.ylim(0, 1)
plt.legend()
plt.grid()
plt.show()



url = (
    "https://mba.tuck.dartmouth.edu/pages/"
    "faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)

Market = pd.read_csv(
    url,
    compression="zip",
    skiprows=3,
    index_col=0
)

# Keep observations with valid dates
Market.index = pd.to_datetime(
    Market.index.astype(str).str.strip(),
    format="%Y%m%d",
    errors="coerce"
)

Market = Market.loc[Market.index.notna()].copy()

Market = Market.apply(pd.to_numeric, errors="coerce")

Market = Market.sort_index()

Market = Market.loc["1980-01-01":"2025-12-31"]

# Convert percentages into decimals
Market = Market / 100

print(Market.head())

Market_return = Market["Mkt-RF"]


Market["Rm"] = Market["Mkt-RF"] + Market["RF"]

Monthly_return = (
    (1 + Market["Rm"]).groupby(
        Market.index.to_period("M")
    ).prod()
    -
    (1 + Market["RF"]).groupby(
        Market.index.to_period("M")
    ).prod()
)

def Realized_variance(x):

    x = np.array(x)

    mean = np.mean(x)

    residuals = x - mean

    variance = np.sum(residuals**2)

    return variance

Monthly_variance = Market["Mkt-RF"].groupby(
    Market.index.to_period("M")
).apply(Realized_variance)


Finance = pd.DataFrame({
    "Return": Monthly_return,
    "RV": Monthly_variance
})

print(Finance.head())

Finance.plot(
    subplots=True,
    figsize=(14, 8),
    grid=True,
    title=["Monthly Excess Return", "Realized Variance"]
)

plt.show()

LL, filtered, predicted = Hamilton_filter(
    F, estimated_parameters
)

Regimes = pd.DataFrame({
    "Factor": F,
    "P_bad": filtered[:, 0],
    "P_good": filtered[:, 1]
}, index=pd.to_datetime(Data_Macro.index).to_period("M"))

Empirical = Finance.join(
    Regimes,
    how="inner"
).dropna()

print(Empirical.head())

Empirical["Regime"] = np.where(
    Empirical["P_bad"] > 0.5,
    "Bad",
    "Good"
)


Results = Empirical.groupby("Regime")[
    ["Return", "RV"]
].mean()

Results["Gamma"] = (
    Results["Return"] / Results["RV"]
)

Results["Observations"] = (
    Empirical.groupby("Regime").size()
)

print(Results)

Empirical["Sigma2"] = Empirical["RV"]


RV = np.array(Empirical["RV"])

Y = RV[1:]

X = RV[:-1]

a, b = OLS(Y, X)

print("Constant:", a)
print("AR coefficient:", b)

Empirical["Sigma2_AR"] = a + b * Empirical["RV"]

# Next month's realized excess return
Empirical["Return_next"] = (
    Empirical["Return"].shift(-1)
)

# Conditional market variance
Empirical["Sigma2"] = Empirical["RV"]

# Interaction between variance and macro regime
Empirical["Interaction"] = (
    Empirical["Sigma2"] * Empirical["P_bad"]
)

Regression = Empirical.dropna().copy()


def OLS_Matrix(Y, X, constant=False):

    Y = np.asarray(Y, dtype=float)

    X = np.asarray(X, dtype=float)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if constant:
        X = np.column_stack(
            (np.ones(len(X)), X)
        )

    x_led_1 = X

    x_led_2 = np.linalg.inv(
        x_led_1.T @ x_led_1
    )

    y_led = x_led_1.T @ Y

    ols = x_led_2 @ y_led

    return ols

Y = np.array(Regression["Return_next"])

X = np.column_stack((
    Regression["Sigma2"],
    Regression["Interaction"]
))

alpha, beta = OLS_Matrix(Y, X)

print("Alpha:", alpha)
print("Beta:", beta)

Empirical["Gamma"] = (
    alpha + beta * Empirical["P_bad"]
)

Empirical["Expected_return"] = (
    Empirical["Gamma"] * Empirical["Sigma2"]
)



model = sm.OLS(Y, X)

result_gamma = model.fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 6}
)

print(result_gamma.summary(
    xname=["Alpha", "Beta"]
))

NBER = Data["USREC"].copy()

NBER.index = pd.to_datetime(
    NBER.index
).to_period("M")

NBER = NBER.reindex(Empirical.index)

dates = Empirical.index.to_timestamp()

plt.figure(figsize=(14, 6))

plt.plot(
    dates,
    Empirical["Gamma"],
    label="Conditional Price of Market Risk"
)

plt.axhline(
    alpha,
    color="green",
    linestyle="--",
    label="Gamma Good"
)

plt.axhline(
    alpha + beta,
    color="red",
    linestyle="--",
    label="Gamma Bad"
)

plt.fill_between(
    dates,
    0,
    1,
    where=NBER.eq(1).to_numpy(),
    transform=plt.gca().get_xaxis_transform(),
    color="grey",
    alpha=0.25,
    label="NBER Recession"
)

plt.title("Conditional Price of Market Risk")
plt.ylabel("Gamma")
plt.legend()
plt.grid()
plt.show()