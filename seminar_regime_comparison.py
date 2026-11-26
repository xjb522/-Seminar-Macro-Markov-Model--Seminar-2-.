import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import statsmodels.api as sm

variables = ['PAYEMS', 'INDPRO', 'CMRMTSPL', 'W875RX1', 'USREC', 'RECPROUSM156N']


def Data_download(vari=None, domain="Stock_Prices",
                  start="1980-01-01", till="2026-01-01"):

    import tidyfinance as td

    Data = td.download_data(
        domain=domain,
        series=vari,
        start_date=start,
        end_date=till
    )

    # Convert Polars DataFrame to Pandas DataFrame
    if not isinstance(Data, pd.DataFrame):
        Data = Data.to_pandas()

    Data = Data.set_index("date")

    Data = Data.pivot(
        columns="series",
        values="value"
    )

    Data.index = pd.to_datetime(Data.index)

    return Data.sort_index()


def OLS_Matrix(Y, X, constant=False):
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if constant:
        X = np.column_stack((np.ones(len(X)), X))
    return np.linalg.lstsq(X, Y, rcond=None)[0]

def OLS(Y, X):
    return OLS_Matrix(Y, X, constant=True)

def normal_pdf(x, mu, sigma):
    a = 1 / (sigma * np.sqrt(2 * np.pi))
    b = -0.5 * ((x - mu) / sigma)**2
    return a * np.exp(b)

def Hamilton_filter(F, parameters):
    mu_0, mu_1, sigma_0, sigma_1, p00, p11 = parameters
    P = np.array([[p00, 1-p00], [1-p11, p11]])
    prob = np.array([(1-p11)/(2-p00-p11), (1-p00)/(2-p00-p11)])
    filtered, predicted_prob = [], []
    log_likelihood = 0.0
    for t in range(len(F)):
        predicted = P.T @ prob
        density = np.array([normal_pdf(F[t], mu_0, sigma_0), normal_pdf(F[t], mu_1, sigma_1)])
        joint = density * predicted
        likelihood = np.sum(joint)
        if likelihood <= 0 or not np.isfinite(likelihood):
            return -np.inf, None, None
        prob = joint / likelihood
        filtered.append(prob)
        predicted_prob.append(predicted)
        log_likelihood += np.log(likelihood)
    return log_likelihood, np.array(filtered), np.array(predicted_prob)

def negative_log_likelihood(parameters, F):
    LL, _, _ = Hamilton_filter(F, parameters)
    return -LL if np.isfinite(LL) else 1e100

def Realized_variance(x):
    x = np.asarray(x, dtype=float)
    return np.sum((x - np.mean(x))**2)

def download_market():
    url = ('https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/'
           'F-F_Research_Data_Factors_daily_CSV.zip')
    Market = pd.read_csv(url, compression='zip', skiprows=3, index_col=0)
    Market.index = pd.to_datetime(Market.index.astype(str).str.strip(), format='%Y%m%d', errors='coerce')
    Market = Market.loc[Market.index.notna()].copy()
    Market = Market.apply(pd.to_numeric, errors='coerce').replace([-99.99, -999], np.nan)
    return Market.dropna(subset=['Mkt-RF', 'RF']).sort_index() / 100

def Estimate_period(Data, Market, start, end, variance_model='RV'):
    # End date is exclusive. All estimators are re-fitted in this sample.
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    Data = Data.loc[(Data.index >= start) & (Data.index < end)].copy()
    Market = Market.loc[(Market.index >= start) & (Market.index < end)].copy()
    Data_Macro = Data[variables[:4]].astype(float)
    Data_Macro.index = Data_Macro.index.to_period('M')
    Data_Macro = Data_Macro.reindex(pd.period_range(start.to_period('M'), (end-pd.Timedelta(days=1)).to_period('M'), freq='M'))
    if (Data_Macro <= 0).any().any():
        raise ValueError('Macro levels must be positive for log differences.')
    Data_Macro = np.log(Data_Macro).diff().dropna()
    if not Data_Macro.index.equals(pd.period_range(Data_Macro.index.min(), Data_Macro.index.max(), freq='M')):
        raise ValueError('Missing interior macro months: do not treat gaps as one-month transitions.')
    mean, std = [], []
    for i in Data_Macro.columns:
        mean.append(np.mean(Data_Macro[i]))
        std.append(np.std(Data_Macro[i]))
    standard = {}
    for i, name in enumerate(Data_Macro.columns):
        standard[name] = (Data_Macro[name]-mean[i])/std[i]
    standard = np.asarray(pd.DataFrame(standard), dtype=float)
    Eigenvalues, eigenvector = np.linalg.eigh(np.cov(standard, rowvar=False))
    w = eigenvector[:, -1]
    if np.sum(w) < 0:
        w = -w
    F = standard @ w
    bounds = [(None,None), (None,None), (0.001,None), (0.001,None), (0.001,0.999), (0.001,0.999)]
    starts = [[-1.5,.5,1.2,.6,.90,.95], [-.5,.2,3,.7,.7,.98], [-2,.3,1,1,.85,.95], [-.8,.4,.7,1.5,.85,.95]]
    fits = [minimize(negative_log_likelihood, s, args=(F,), method='L-BFGS-B', bounds=bounds) for s in starts]
    valid = [r for r in fits if r.success and np.isfinite(r.fun) and r.fun < 1e90]
    if not valid:
        raise RuntimeError('No Markov optimization converged.')
    result = min(valid, key=lambda r:r.fun)
    mu_0, mu_1, sigma_0, sigma_1, p00, p11 = result.x
    if mu_0 > mu_1:
        mu_0, mu_1 = mu_1, mu_0
        sigma_0, sigma_1 = sigma_1, sigma_0
        p00, p11 = p11, p00
    parameters = [mu_0, mu_1, sigma_0, sigma_1, p00, p11]
    P = np.array([[p00,1-p00],[1-p11,p11]])
    LL, filtered, predicted = Hamilton_filter(F, parameters)
    Regimes = pd.DataFrame({'Factor':F, 'P_bad':filtered[:,0], 'P_good':filtered[:,1]}, index=Data_Macro.index)
    Market['Rm'] = Market['Mkt-RF'] + Market['RF']
    month = Market.index.to_period('M')
    Monthly_return = (1+Market['Rm']).groupby(month).prod() - (1+Market['RF']).groupby(month).prod()
    Monthly_variance = Market['Mkt-RF'].groupby(month).apply(Realized_variance)
    Finance = pd.DataFrame({'Return':Monthly_return,'RV':Monthly_variance})
    Empirical = Finance.join(Regimes, how='inner').dropna().sort_index()
    if not Empirical.index.equals(pd.period_range(Empirical.index.min(), Empirical.index.max(), freq='M')):
        raise ValueError('Missing interior months in merged sample.')
    Empirical['Regime'] = np.where(Empirical['P_bad'] > .5, 'Bad','Good')
    Results = Empirical.groupby('Regime')[['Return','RV']].mean()
    Results['Gamma'] = Results['Return']/Results['RV']
    Results['Observations'] = Empirical.groupby('Regime').size()
    RV = Empirical['RV'].to_numpy()
    a,b = OLS(RV[1:],RV[:-1])
    Empirical['Sigma2_AR'] = a+b*Empirical['RV']
    Empirical['Return_next'] = Empirical['Return'].shift(-1)
    Empirical['Sigma2'] = Empirical[variance_model]
    if (Empirical['Sigma2'] <= 0).any():
        raise ValueError('Nonpositive variance forecast.')
    Empirical['Interaction'] = Empirical['Sigma2']*Empirical['P_bad']
    Regression = Empirical.dropna(subset=['Return_next','Sigma2','Interaction']).copy()
    Y = Regression['Return_next']
    X = Regression[['Sigma2','Interaction']].rename(columns={'Sigma2':'Alpha','Interaction':'Beta'})
    alpha,beta = OLS_Matrix(Y,X)
    result_gamma = sm.OLS(Y,X).fit(cov_type='HAC',cov_kwds={'maxlags':6})
    Empirical['Gamma'] = alpha+beta*Empirical['P_bad']
    Empirical['Expected_return'] = Empirical['Gamma']*Empirical['Sigma2']
    NBER = Data['USREC'].copy()
    NBER.index = NBER.index.to_period('M')
    NBER = NBER.reindex(Empirical.index)
    return {'Empirical':Empirical,'Regression':Regression,'Markov':pd.Series({'mu_bad':mu_0,'mu_good':mu_1,'var_bad':sigma_0**2,'var_good':sigma_1**2,'p00':p00,'p11':p11,'duration_bad':1/(1-p00),'duration_good':1/(1-p11),'LL':LL}), 'P':P,'Results':Results,'alpha':alpha,'beta':beta,'result_gamma':result_gamma,'NBER':NBER,'a':a,'b':b,'w':w}

def compare(full, restricted):
    rows = {}
    for name,r in [('Full',full),('Restricted',restricted)]:
        fit = r['result_gamma']
        rows[name] = {'Alpha':r['alpha'],'Beta':r['beta'],'Gamma_good':r['alpha'],'Gamma_bad':r['alpha']+r['beta'],'SE_alpha':fit.bse['Alpha'],'SE_beta':fit.bse['Beta'],'p_alpha':fit.pvalues['Alpha'],'p_beta':fit.pvalues['Beta'],'N':int(fit.nobs)}
    Table = pd.DataFrame(rows)
    Table['Full - Restricted'] = Table['Full']-Table['Restricted']
    Table.loc[['SE_alpha','SE_beta','p_alpha','p_beta','N'],'Full - Restricted'] = np.nan
    print('\nGamma regression comparison\n',Table.round(5))
    print('\nMarkov comparison\n',pd.concat({'Full':full['Markov'],'Restricted':restricted['Markov']},axis=1).round(5))
    print('\nDescriptive comparison\n',pd.concat({'Full':full['Results'],'Restricted':restricted['Results']},axis=1).round(5))
    common = full['Regression'].index.intersection(restricted['Regression'].index)
    A = full['Empirical'].loc[common]; B = restricted['Empirical'].loc[common]
    comparison = pd.DataFrame({'P_bad_full':A['P_bad'],'P_bad_restricted':B['P_bad'],'Gamma_full':A['Gamma'],'Gamma_restricted':B['Gamma']})
    comparison['Parameter_effect'] = (full['alpha']-restricted['alpha'])+(full['beta']-restricted['beta'])*B['P_bad']
    comparison['Probability_effect'] = full['beta']*(A['P_bad']-B['P_bad'])
    comparison['Gamma_difference'] = A['Gamma']-B['Gamma']
    print('\nCommon-period probability correlation:', comparison['P_bad_full'].corr(comparison['P_bad_restricted']))
    print('\nMean gamma difference decomposition\n',comparison[['Parameter_effect','Probability_effect','Gamma_difference']].mean())
    for columns,title in [(['P_bad_full','P_bad_restricted'],'Filtered bad-regime probabilities: common sample'),(['Gamma_full','Gamma_restricted'],'Conditional price of market risk: common sample'),(['Parameter_effect','Probability_effect','Gamma_difference'],'Decomposition of gamma difference')]:
        ax = comparison[columns].set_axis(common.to_timestamp()).plot(figsize=(13,5),title=title,grid=True)
        ax.axhline(0,color='black',linewidth=.6)
        plt.tight_layout(); plt.show()
    q = np.linspace(0,1,101)
    plt.figure(figsize=(9,5))
    for name,r in [('Full',full),('Restricted',restricted)]:
        plt.plot(q,r['alpha']+r['beta']*q,label=name)
    plt.xlabel('Bad-regime probability'); plt.ylabel('Gamma'); plt.legend(); plt.grid(); plt.tight_layout(); plt.show()
    return Table, comparison

if __name__ == '__main__':
    Data = Data_download(variables,'FRED','1980-01-01','2026-01-01')
    Market = download_market()
    Full = Estimate_period(Data,Market,'1980-01-01','2026-01-01')
    Restricted = Estimate_period(Data,Market,'1980-01-01','2019-01-01')
    Table, Comparison = compare(Full,Restricted)
    print('\nFull regression\n',Full['result_gamma'].summary())
    print('\nRestricted regression\n',Restricted['result_gamma'].summary())


Data = Data_download(
    variables, "FRED", "1980-01-01", "2026-01-01"
)

Market = download_market()

Full = Estimate_period(
    Data, Market, "1980-01-01", "2026-01-01"
)

Restricted = Estimate_period(
    Data, Market, "1980-01-01", "2019-01-01"
)

Table, Comparison = compare(Full, Restricted)



Empirical_B = Full["Empirical"].copy()

Empirical_B = Empirical_B.loc[
    Empirical_B.index < pd.Period("2019-01", freq="M")
].copy()

# Ensure the dependent variable is also within the sample
Empirical_B["Return_next"] = (
    Empirical_B["Return"].shift(-1)
)

Empirical_B = Empirical_B.dropna(
    subset=["Return_next", "RV", "P_bad"]
)

Y = np.array(Empirical_B["Return_next"])

X = np.column_stack((
    Empirical_B["RV"],
    Empirical_B["RV"] * Empirical_B["P_bad"]
))

result_B = sm.OLS(Y, X).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 6}
)

print(result_B.summary(
    xname=["Alpha", "Beta"]
))


def Gamma_robustness(Empirical, name):

    E = Empirical.copy().sort_index()

    # Future return and realized variance
    E["Return_next"] = E["Return"].shift(-1)

    RV = np.array(E["RV"])

    # Estimate AR(1)
    Y = RV[1:]
    X = RV[:-1]

    a, b = OLS(Y, X)

    E["Sigma2_AR"] = a + b * E["RV"]

    Results = []

    for variance in ["RV", "Sigma2_AR"]:

        Regression = E[
            ["Return_next", "P_bad", variance]
        ].dropna().copy()

        Regression = Regression.loc[
            Regression[variance] > 0
        ]

        Y = np.array(Regression["Return_next"])

        X = np.column_stack((
            Regression[variance],
            Regression[variance] * Regression["P_bad"]
        ))

        for estimator in ["OLS", "WLS"]:

            if estimator == "OLS":

                model = sm.OLS(Y, X)

            else:

                weights = 1 / Regression[variance]

                model = sm.WLS(
                    Y, X, weights=weights
                )

            result = model.fit(
                cov_type="HAC",
                cov_kwds={"maxlags": 6}
            )

            alpha, beta = result.params

            Results.append({
                "Model": name,
                "Variance": variance,
                "Estimator": estimator,
                "Alpha": alpha,
                "Beta": beta,
                "Gamma_good": alpha,
                "Gamma_bad": alpha + beta,
                "SE_beta": result.bse[1],
                "P_beta": result.pvalues[1],
                "N": result.nobs
            })

    return pd.DataFrame(Results)


# Model A
Empirical_A = Full["Empirical"].copy()

# Model B
Empirical_B = Full["Empirical"].copy()

Empirical_B = Empirical_B.loc[
    Empirical_B.index < pd.Period("2019-01", freq="M")
].copy()

# Model C
Empirical_C = Restricted["Empirical"].copy()

# Estimation
Results_A = Gamma_robustness(Empirical_A, "A")

Results_B = Gamma_robustness(Empirical_B, "B")

Results_C = Gamma_robustness(Empirical_C, "C")

# Combine all results
Robustness = pd.concat(
    [Results_A, Results_B, Results_C],
    ignore_index=True
)

print(Robustness.round(4).to_string(index=False))



def PCA_analysis(Data, start, end):

    Data_Macro = Data.loc[
        (Data.index >= pd.Timestamp(start)) &
        (Data.index < pd.Timestamp(end)),
        ["PAYEMS", "INDPRO", "CMRMTSPL", "W875RX1"]
    ].copy()

    Data_Macro = np.log(Data_Macro).diff().dropna()

    standard = (
        Data_Macro - Data_Macro.mean()
    ) / Data_Macro.std(ddof=0)

    cov_matrix = np.cov(standard, rowvar=False)

    Eigenvalues, Eigenvectors = np.linalg.eigh(
        cov_matrix
    )

    w = Eigenvectors[:, -1]

    if np.sum(w) < 0:
        w = -w

    explained_variance = (
        Eigenvalues[-1] / np.sum(Eigenvalues)
    )

    Loadings = pd.Series(
        w,
        index=Data_Macro.columns,
        name="PC1_weights"
    )

    return Loadings, explained_variance


Loadings_Full, EV_Full = PCA_analysis(
    Data, "1980-01-01", "2026-01-01"
)

Loadings_Restricted, EV_Restricted = PCA_analysis(
    Data, "1980-01-01", "2019-01-01"
)

PCA_Table = pd.concat(
    [Loadings_Full, Loadings_Restricted],
    axis=1
)

PCA_Table.columns = ["Full", "Restricted"]

print(PCA_Table.round(4))

print("Explained variance Full:", EV_Full)
print("Explained variance Restricted:", EV_Restricted)

PCA_Table.plot(kind="bar", figsize=(10, 5))
plt.title("First Principal Component Weights")
plt.ylabel("Weight")
plt.xticks(rotation=0)
plt.grid(axis="y")
plt.show()



def Validate_regimes(Empirical, Data, threshold=0.5):

    E = Empirical[["P_bad"]].copy()

    Benchmark = Data[
        ["USREC", "RECPROUSM156N"]
    ].copy()

    Benchmark.index = pd.to_datetime(
        Benchmark.index
    ).to_period("M")

    Benchmark["RECPROUSM156N"] = (
        Benchmark["RECPROUSM156N"] / 100
    )

    E = E.join(Benchmark, how="inner")

    E = E.dropna(subset=["P_bad", "USREC"])

    E["Predicted_regime"] = (
        E["P_bad"] > threshold
    ).astype(int)

    actual = E["USREC"]
    predicted = E["Predicted_regime"]

    TP = np.sum((actual == 1) & (predicted == 1))
    TN = np.sum((actual == 0) & (predicted == 0))
    FP = np.sum((actual == 0) & (predicted == 1))
    FN = np.sum((actual == 1) & (predicted == 0))

    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    Results = {
        "Correlation_NBER":
            E["P_bad"].corr(E["USREC"]),

        "Correlation_CP":
            E["P_bad"].corr(E["RECPROUSM156N"]),

        "Accuracy": (TP + TN) / len(E),

        "Sensitivity": sensitivity,

        "Specificity": specificity,

        "Balanced_accuracy":
            (sensitivity + specificity) / 2,

        "Precision":
            TP / (TP + FP) if TP + FP > 0 else np.nan,

        "True_positive": TP,
        "True_negative": TN,
        "False_positive": FP,
        "False_negative": FN,

        "Observations": len(E)
    }

    return pd.Series(Results), E



Common_index = Full["Empirical"].index.intersection(
    Restricted["Empirical"].index
)

Validation_Full, Benchmark_Full = Validate_regimes(
    Full["Empirical"].loc[Common_index],
    Data
)

Validation_Restricted, Benchmark_Restricted = Validate_regimes(
    Restricted["Empirical"].loc[Common_index],
    Data
)

Validation_Table = pd.concat(
    [Validation_Full, Validation_Restricted],
    axis=1
)

Validation_Table.columns = ["Full", "Restricted"]

print(Validation_Table.round(4))



dates = Common_index.to_timestamp()

plt.figure(figsize=(15, 6))

plt.plot(
    dates,
    Benchmark_Full["P_bad"].reindex(Common_index),
    label="Full sample"
)

plt.plot(
    dates,
    Benchmark_Restricted["P_bad"].reindex(Common_index),
    label="Restricted sample"
)

plt.plot(
    dates,
    Benchmark_Full["RECPROUSM156N"].reindex(Common_index),
    label="Chauvet-Piger",
    linestyle="--"
)

plt.fill_between(
    dates,
    0,
    1,
    where=Benchmark_Full["USREC"].reindex(
        Common_index
    ).eq(1).to_numpy(),
    color="grey",
    alpha=0.25,
    label="NBER recession"
)

plt.ylim(0, 1)
plt.ylabel("Recession probability")
plt.title("Validation of Macroeconomic Regimes")
plt.legend()
plt.grid()
plt.show()



def Block_indices(T, block_size, rng):

    indices = []

    while len(indices) < T:

        start = rng.integers(
            0, T - block_size + 1
        )

        block = np.arange(
            start, start + block_size
        )

        indices.extend(block)

    return np.array(indices[:T])


def Descriptive_gamma(E):

    Bad = E.loc[E["P_bad"] > 0.5]

    Good = E.loc[E["P_bad"] <= 0.5]

    if len(Bad) == 0 or len(Good) == 0:
        return np.nan, np.nan, np.nan

    gamma_bad = (
        Bad["Return"].mean() / Bad["RV"].mean()
    )

    gamma_good = (
        Good["Return"].mean() / Good["RV"].mean()
    )

    difference = gamma_bad - gamma_good

    return gamma_bad, gamma_good, difference


def Bootstrap_gamma(Empirical, B=5000,
                    block_size=12, seed=42):

    E = Empirical[
        ["Return", "RV", "P_bad"]
    ].dropna().sort_index()

    rng = np.random.default_rng(seed)

    Bootstrap_results = []

    T = len(E)

    for b in range(B):

        indices = Block_indices(
            T, block_size, rng
        )

        Sample = E.iloc[indices]

        estimates = Descriptive_gamma(Sample)

        Bootstrap_results.append(estimates)

    Bootstrap_results = pd.DataFrame(
        Bootstrap_results,
        columns=[
            "Gamma_bad",
            "Gamma_good",
            "Difference"
        ]
    ).dropna()

    estimates = Descriptive_gamma(E)

    Results = pd.DataFrame({

        "Estimate": estimates,

        "Bootstrap_SE":
            Bootstrap_results.std().to_numpy(),

        "CI_lower":
            Bootstrap_results.quantile(0.025).to_numpy(),

        "CI_upper":
            Bootstrap_results.quantile(0.975).to_numpy()

    }, index=[
        "Gamma_bad",
        "Gamma_good",
        "Difference"
    ])

    return Results, Bootstrap_results


CI_Full, Boot_Full = Bootstrap_gamma(
    Full["Empirical"]
)

CI_Restricted, Boot_Restricted = Bootstrap_gamma(
    Restricted["Empirical"]
)

print("\nFull sample bootstrap")
print(CI_Full.round(4))

print("\nRestricted sample bootstrap")
print(CI_Restricted.round(4))


plt.figure(figsize=(12, 5))

plt.hist(
    Boot_Full["Difference"],
    bins=60,
    density=True,
    alpha=0.5,
    label="Full sample"
)

plt.hist(
    Boot_Restricted["Difference"],
    bins=60,
    density=True,
    alpha=0.5,
    label="Restricted sample"
)

plt.axvline(
    0,
    color="black",
    linestyle="--",
    label="Null hypothesis"
)

plt.xlabel("Gamma Bad - Gamma Good")
plt.title("Bootstrap Distribution of Gamma Difference")
plt.legend()
plt.grid()
plt.show()




E = Full["Empirical"].copy().sort_index()

E["Return_next"] = E["Return"].shift(-1)

E["Post2019"] = (
    E.index >= pd.Period("2019-01", freq="M")
).astype(int)

E["Sigma2"] = E["RV"]

E["Interaction"] = E["Sigma2"] * E["P_bad"]

E["Break_alpha"] = E["Sigma2"] * E["Post2019"]

E["Break_beta"] = E["Interaction"] * E["Post2019"]

# Remove the observation crossing the sample boundary
E = E.loc[
    E.index != pd.Period("2018-12", freq="M")
]

Regression = E.dropna(
    subset=[
        "Return_next",
        "Sigma2",
        "Interaction",
        "Break_alpha",
        "Break_beta"
    ]
)

Y = Regression["Return_next"]

X = Regression[[
    "Sigma2",
    "Interaction",
    "Break_alpha",
    "Break_beta"
]]

Break_model = sm.OLS(Y, X).fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 6}
)

print(Break_model.summary())

print("\nTest of change in Beta:")

print(
    Break_model.wald_test(
        "Break_beta = 0",
        scalar=True
    )
)

print("\nJoint parameter stability test:")

print(
    Break_model.wald_test(
        "Break_alpha = 0, Break_beta = 0",
        scalar=True
    )
)

alpha_before = Break_model.params["Sigma2"]

beta_before = Break_model.params["Interaction"]

alpha_after = (
    alpha_before + Break_model.params["Break_alpha"]
)

beta_after = (
    beta_before + Break_model.params["Break_beta"]
)

print("Alpha before 2019:", alpha_before)
print("Beta before 2019:", beta_before)

print("Alpha from 2019:", alpha_after)
print("Beta from 2019:", beta_after)



def Estimate_beta(E, probability):

    Y = E["Return_next"].to_numpy()

    X = np.column_stack((
        E["RV"],
        E["RV"] * E[probability]
    ))

    return np.linalg.lstsq(
        X, Y, rcond=None
    )[0][1]


def Bootstrap_beta_difference(
        Full, Restricted, B=5000,
        block_size=12, seed=42):

    A = Full["Empirical"].copy().sort_index()

    A["Return_next"] = A["Return"].shift(-1)

    A = A.rename(columns={"P_bad": "P_full"})

    C = Restricted["Empirical"]["P_bad"].rename(
        "P_restricted"
    )

    A = A.join(C, how="left")

    A = A.dropna(
        subset=["Return_next", "RV", "P_full"]
    )

    cutoff = pd.Period("2019-01", freq="M")

    Pre = A.loc[
        A.index < cutoff - 1
    ].dropna(subset=["P_restricted"])

    Post = A.loc[A.index >= cutoff - 1]

    rng = np.random.default_rng(seed)

    Results = []

    for b in range(B):

        Sample_pre = Pre.iloc[
            Block_indices(len(Pre), block_size, rng)
        ]

        Sample_post = Post.iloc[
            Block_indices(len(Post), block_size, rng)
        ]

        Sample_full = pd.concat([
            Sample_pre, Sample_post
        ])

        beta_A = Estimate_beta(
            Sample_full, "P_full"
        )

        beta_B = Estimate_beta(
            Sample_pre, "P_full"
        )

        beta_C = Estimate_beta(
            Sample_pre, "P_restricted"
        )

        Results.append([
            beta_A - beta_B,
            beta_B - beta_C,
            beta_A - beta_C
        ])

    Results = pd.DataFrame(
        Results,
        columns=["A-B", "B-C", "A-C"]
    )

    actual = [
        Estimate_beta(A, "P_full")
        - Estimate_beta(Pre, "P_full"),

        Estimate_beta(Pre, "P_full")
        - Estimate_beta(Pre, "P_restricted"),

        Estimate_beta(A, "P_full")
        - Estimate_beta(Pre, "P_restricted")
    ]

    Table = pd.DataFrame({

        "Estimate": actual,

        "Bootstrap_SE":
            Results.std().to_numpy(),

        "CI_lower":
            Results.quantile(0.025).to_numpy(),

        "CI_upper":
            Results.quantile(0.975).to_numpy()

    }, index=Results.columns)

    return Table, Results


Beta_CI, Beta_Bootstrap = Bootstrap_beta_difference(
    Full, Restricted
)

print(Beta_CI.round(4))




def Specification_test(Empirical, name):

    E = Empirical.copy().sort_index()

    E["Return_next"] = E["Return"].shift(-1)

    E["Sigma2"] = E["RV"]

    E["Interaction"] = (
        E["Sigma2"] * E["P_bad"]
    )

    E = E.dropna(
        subset=[
            "Return_next",
            "Sigma2",
            "P_bad",
            "Interaction"
        ]
    )

    Y = E["Return_next"]

    X1 = E[["Sigma2", "Interaction"]]

    X2 = sm.add_constant(X1)

    X3 = sm.add_constant(
        E[["Sigma2", "P_bad", "Interaction"]]
    )

    Models = {
        "M1": X1,
        "M2": X2,
        "M3": X3
    }

    Results = []

    for spec, X in Models.items():

        result = sm.OLS(Y, X).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 6}
        )

        Results.append({

            "Sample": name,

            "Specification": spec,

            "Constant":
                result.params.get("const", np.nan),

            "Alpha":
                result.params["Sigma2"],

            "Beta":
                result.params["Interaction"],

            "Delta":
                result.params.get("P_bad", np.nan),

            "P_beta":
                result.pvalues["Interaction"],

            "P_constant":
                result.pvalues.get("const", np.nan),

            "P_delta":
                result.pvalues.get("P_bad", np.nan),

            "AIC": result.aic,

            "BIC": result.bic,

            "N": result.nobs
        })

        if spec == "M2":

            print(name, "Test of constant")

            print(result.wald_test(
                "const = 0", scalar=True
            ))

        if spec == "M3":

            print(name, "Joint specification test")

            print(result.wald_test(
                "const = 0, P_bad = 0",
                scalar=True
            ))

    return pd.DataFrame(Results)


Specification_A = Specification_test(
    Empirical_A, "A"
)

Specification_B = Specification_test(
    Empirical_B, "B"
)

Specification_C = Specification_test(
    Empirical_C, "C"
)

Specifications = pd.concat(
    [
        Specification_A,
        Specification_B,
        Specification_C
    ],
    ignore_index=True
)

print(
    Specifications.round(4).to_string(index=False)
)



print(PCA_Table)
print(Validation_Table)
print(CI_Full)
print(CI_Restricted)
print(Beta_CI)
print(Specifications)