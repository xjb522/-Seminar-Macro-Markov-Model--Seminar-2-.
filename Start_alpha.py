import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.regime_switching.markov_regression \
    import MarkovRegression


# %% Variabler
variables = [
    "PAYEMS",
    "INDPRO",
    "CMRMTSPL",
    "W875RX1",
    "USREC",
    "RECPROUSM156N"
]

url = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=" + ",".join(variables)
    + "&cosd=1967-01-01"
    + "&coed=2025-12-01"
)

data = pd.read_csv(
    url,
    index_col=0,
    parse_dates=True
)

# %% tjek

data = data.sort_index().asfreq("MS")

data = data.loc["1967-01-01":"2025-12-01"].copy()

print(data.head())
print(data.tail())
print(data.isna().sum())
print(data.shape)

#%%

data = data.rename(columns={
    "PAYEMS": "Employment",
    "INDPRO": "Production",
    "CMRMTSPL": "Sales",
    "W875RX1": "Income",
    "USREC": "NBER",
    "RECPROUSM156N": "CP_Probability"
})

data["CP_Probability"] /= 100

#%% 

macro_variables = [
    "Employment",
    "Production",
    "Sales",
    "Income"
]

macro = data[macro_variables].copy()

macro_growth = 100 * np.log(macro).diff()

macro_growth = macro_growth.dropna()

print(macro_growth.describe())


#%% plot

macro_growth.plot(
    subplots=True,
    figsize=(12, 9),
    title="Monthly Macroeconomic Growth Rates",
    grid=True
)

plt.tight_layout()
plt.show()

#%% Dickyfuller test

results = []

for variable in macro_variables:

    level = adfuller(
        np.log(macro[variable].dropna()),
        regression="ct",
        autolag="AIC",
        result_object=False

    )

    growth = adfuller(
        macro_growth[variable],
        regression="c",
        autolag="AIC"
    )

    results.append({
        "Variable": variable,
        "ADF Level": level[0],
        "P-value Level": level[1],
        "ADF Growth": growth[0],
        "P-value Growth": growth[1]
    })

adf_table = pd.DataFrame(results)

print(adf_table.round(4))

# %% Samlet funktion
scaler = StandardScaler()

X = scaler.fit_transform(macro_growth)

pca = PCA(n_components=1)

factor = pca.fit_transform(X)

macro_growth["Factor"] = factor[:, 0]

weights = pd.Series(
    pca.components_[0],
    index=macro_variables
)


# %% Tjek

print("Explained variance:")
print(pca.explained_variance_ratio_)

print(weights)

# %% markov model


np.random.seed(42)

model = MarkovRegression(
    endog=macro_growth["Factor"],
    k_regimes=2,
    trend="c",
    switching_variance=False
)

result = model.fit(
    search_reps=30,
    maxiter=1000,
    disp=False
)

print(result.summary())
print(result.mle_retvals)

regime_means = result.params[
    ["const[0]", "const[1]"]
]

bad_regime = int(np.argmin(regime_means.values))
good_regime = 1 - bad_regime

print("Bad regime:", bad_regime)
print("Good regime:", good_regime)
print(regime_means)

macro_growth["p_filtered"] = (
    result.filtered_marginal_probabilities[bad_regime]
)

macro_growth["p_smoothed"] = (
    result.smoothed_marginal_probabilities[bad_regime]
)


transition = result.regime_transition[:, :, 0].T

order = [good_regime, bad_regime]

transition = transition[np.ix_(order, order)]

transition_table = pd.DataFrame(
    transition,
    index=["Good", "Bad"],
    columns=["Good", "Bad"]
)

print(transition_table)


duration = pd.Series(
    1 / (1 - np.diag(transition)),
    index=["Good", "Bad"]
)

print(duration)

# %% plot

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    macro_growth.index,
    macro_growth["p_smoothed"],
    label="Our smoothed probability"
)

ax.plot(
    macro_growth.index,
    macro_growth["p_filtered"],
    label="Our filtered probability",
    alpha=0.6
)

ax.plot(
    data.index,
    data["CP_Probability"],
    label="Chauvet-Piger probability",
    linestyle="--"
)

ax.fill_between(
    data.index,
    0,
    1,
    where=data["NBER"].eq(1),
    color="grey",
    alpha=0.2,
    label="NBER recession"
)

ax.set_title("US Macroeconomic Regime Probabilities")
ax.set_ylabel("Recession Probability")
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
plt.show()

# %% Sammenligning
fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(
    macro_growth.index,
    macro_growth["p_smoothed"],
    label="Model 1: Constant variance",
    alpha=0.7
)

ax.plot(
    macro_growth.index,
    macro_growth["p_smoothed_M2"],
    label="Model 2: Switching variance"
)

ax.plot(
    data.index,
    data["CP_Probability"],
    label="Chauvet-Piger",
    linestyle="--",
    color="green"
)

ax.fill_between(
    data.index,
    0,
    1,
    where=data["NBER"].eq(1),
    color="grey",
    alpha=0.2,
    label="NBER recession"
)

ax.set_title("Comparison of Macroeconomic Regime Models")
ax.set_ylabel("Recession Probability")
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
plt.show()



# %% Sammenligning
comparison = macro_growth[
    [
        "p_filtered",
        "p_smoothed",
        "p_filtered_M2",
        "p_smoothed_M2"
    ]
].join(
    data[["CP_Probability", "NBER"]]
)

print(comparison.corr().round(3))

comparison.to_csv("macroeconomic_regimes.csv")

# %% Trouble shoot 

print(
    macro_growth.nsmallest(10, "Factor")[
        ["Factor", "p_smoothed"]
    ]
)

recession_months = macro_growth[
    macro_growth["p_smoothed"] > 0.5
]

print(recession_months)
print(len(recession_months))

# %% Nye markov
# Model 2: Switching mean and variance

np.random.seed(42)

model2 = MarkovRegression(
    endog=macro_growth["Factor"],
    k_regimes=2,
    trend="c",
    switching_variance=True
)

result2 = model2.fit(
    search_reps=50,
    maxiter=1000,
    disp=False
)

print(result2.summary())
print(result2.mle_retvals)

# %%
regime_means2 = result2.params[
    ["const[0]", "const[1]"]
]

bad_regime2 = int(np.argmin(regime_means2.values))
good_regime2 = 1 - bad_regime2

macro_growth["p_filtered_M2"] = (
    result2.filtered_marginal_probabilities[bad_regime2]
)

macro_growth["p_smoothed_M2"] = (
    result2.smoothed_marginal_probabilities[bad_regime2]
)

# %% 

print(
    macro_growth[
        macro_growth["p_smoothed_M2"] > 0.5
    ]
)

print(
    (macro_growth["p_smoothed_M2"] > 0.5).sum()
)

# %%
fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    macro_growth.index,
    macro_growth["p_smoothed"],
    label="Model 1: Constant variance",
    alpha=0.7
)

ax.plot(
    macro_growth.index,
    macro_growth["p_smoothed_M2"],
    label="Model 2: Switching variance"
)

ax.fill_between(
    data.index,
    0,
    1,
    where=data["NBER"].eq(1),
    color="grey",
    alpha=0.2,
    label="NBER recession"
)

ax.set_title("Comparison of Markov-Switching Models")
ax.set_ylabel("Recession Probability")
ax.set_ylim(0, 1)
ax.legend()

plt.tight_layout()
plt.show()

# %% 4.1
growth_full = macro_growth[macro_variables].copy()

growth_train = growth_full.loc[:"2019-12-01"].copy()

print(growth_train.shape)
print(growth_full.shape)

scaler3 = StandardScaler()

# %% 4.2

X_train = scaler3.fit_transform(growth_train)

pca3 = PCA(n_components=1)

pca3.fit(X_train)

sign3 = 1 if pca3.components_[0].sum() >= 0 else -1

X_full = scaler3.transform(growth_full)

factor_full = sign3 * pca3.transform(X_full)[:, 0]

macro_growth["Factor_M3"] = factor_full

factor_train = macro_growth.loc[
    :"2019-12-01",
    "Factor_M3"
]

print(pca3.explained_variance_ratio_)

print(
    pd.Series(
        sign3 * pca3.components_[0],
        index=macro_variables
    )
)

# %% np.random.seed(42)

model3 = MarkovRegression(
    endog=factor_train,
    k_regimes=2,
    trend="c",
    switching_variance=False
)

result3 = model3.fit(
    search_reps=50,
    maxiter=1000,
    disp=False
)

print(result3.summary())

# %% 4.4
model3_full = MarkovRegression(
    endog=macro_growth["Factor_M3"],
    k_regimes=2,
    trend="c",
    switching_variance=False
)

result3_full = model3_full.smooth(
    result3.params
)

regime_means3 = result3.params[
    ["const[0]", "const[1]"]
]

bad_regime3 = int(
    np.argmin(regime_means3.values)
)

macro_growth["p_filtered_M3"] = (
    result3_full.filtered_marginal_probabilities[
        bad_regime3
    ]
)

macro_growth["p_smoothed_M3"] = (
    result3_full.smoothed_marginal_probabilities[
        bad_regime3
    ]
)


# %%
fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(
    macro_growth.index,
    macro_growth["p_smoothed_M3"],
    label="Model 3: Pre-COVID estimation"
)

ax.plot(
    data.index,
    data["CP_Probability"],
    label="Chauvet-Piger",
    linestyle="--",
    color="green"
)

ax.fill_between(
    data.index,
    0,
    1,
    where=data["NBER"].eq(1),
    color="grey",
    alpha=0.2
)

ax.axvline(
    pd.Timestamp("2020-01-01"),
    linestyle=":",
    color="black",
    label="End of estimation sample"
)

ax.set_ylim(0, 1)
ax.set_title("Pre-COVID Markov-Switching Model")
ax.set_ylabel("Recession Probability")
ax.legend()

plt.tight_layout()
plt.show()
