from patsy.build import test__examine_factor_types
from patsy.mgcv_cubic_splines import test_te_1smooth
from sqlalchemy.dialects.mssql.information_schema import columns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import tidyfinance as td

import statsmodels.api as sm


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