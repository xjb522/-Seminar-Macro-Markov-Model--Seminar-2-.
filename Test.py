
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


# %%: Model 1.
# Model 1: FREDs eksisterende recessionssandsynligheder
url1 = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=RECPROUSM156N"

fred = pd.read_csv(url1, index_col=0, parse_dates=True)
fred = fred.apply(pd.to_numeric, errors="coerce")

# Model 2: Industriproduktion, som vi selv modellerer
url2 = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=INDPRO"

data = pd.read_csv(url2, index_col=0, parse_dates=True)
data = data.apply(pd.to_numeric, errors="coerce")

# Månedlig logaritmisk vækst i procent
data["growth"] = 100 * np.log(
    data["INDPRO"] / data["INDPRO"].shift(1)
)

data = data.loc["1980":].dropna()

# %%. Model 2
# Markov-switching med 2 regimer
model = sm.tsa.MarkovRegression(
    data["growth"],
    k_regimes=2,
    trend="c",
    switching_variance=True
)

# Maximum likelihood
result = model.fit(
    search_reps=20,
    disp=False
)

print(result.summary())

# Find regimet med lavest gennemsnitlig vækst
mu0 = result.params["const[0]"]
mu1 = result.params["const[1]"]

bad_regime = 0 if mu0 < mu1 else 1

# Filtreret sandsynlighed for svag økonomi
data["P_bad"] = (
    result.filtered_marginal_probabilities[bad_regime]
)

# %% Figures

fig, ax = plt.subplots(figsize=(12, 6))

# Model 1: FRED (omregnet fra 0-100 til 0-1)
ax.plot(
    fred.index,
    fred["RECPROUSM156N"] / 100,
    label="FRED: Smoothed probability"
)

# Model 2: Vores egen Markov-switching-model
ax.plot(
    data.index,
    data["P_bad"],
    label="Own model: Filtered probability"
)

ax.set_xlim(data.index.min(), data.index.max())
ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_title("U.S. Macroeconomic Regime Probabilities")
ax.legend()
plt.show()

# %% Resultater:

print(result.summary())

print("\nRegime means:")
print("Regime 0:", result.params["const[0]"])
print("Regime 1:", result.params["const[1]"])

print("\nRegime variances:")
print("Regime 0:", result.params["sigma2[0]"])
print("Regime 1:", result.params["sigma2[1]"])

# %% Sammenlign:

# Model A: Switching mean, constant variance
model_A = sm.tsa.MarkovRegression(
    data["growth"],
    k_regimes=2,
    trend="c",
    switching_variance=False
)

result_A = model_A.fit(search_reps=20, disp=False)

# Model B: Switching mean and switching variance
model_B = sm.tsa.MarkovRegression(
    data["growth"],
    k_regimes=2,
    trend="c",
    switching_variance=True
)

result_B = model_B.fit(search_reps=20, disp=False)

# Compare information criteria
comparison = pd.DataFrame({
    "Model": ["A: Constant variance",
              "B: Switching variance"],
    "Log Likelihood": [result_A.llf, result_B.llf],
    "AIC": [result_A.aic, result_B.aic],
    "BIC": [result_A.bic, result_B.bic]
})

print(comparison)
print(result_A.summary())