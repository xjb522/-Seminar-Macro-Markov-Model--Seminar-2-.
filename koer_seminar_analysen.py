"""SEMINAR – køres fra top til bund (ingen egne funktioner).

Kør: python koer_seminar_analysen.py
Eller i Positron/Jupyter: %run koer_seminar_analysen.py

Første kørsel kræver internet for faktiske FRED- og Fama–French-data.
Alternativt sættes USE_LOCAL_CSV=True og LOCAL_CSV til en månedlig fil med
kolonnerne: date, ip_growth, unemployment_change, financial_stress, market_excess.

RESULTAT: Mappen seminar_output med SEND_MIG_DENNE_FIL.txt og tre PNG-figurer.

BEMÆRK: Retrospektive in-sample estimater; ikke realtidsestimation.
Første-trins MS-AR-usikkerhed er ikke medtaget i HAC-standardfejlene.
"""

# ============================================================
# 0. INDSTILLINGER OG IMPORTS
# ============================================================
from pathlib import Path
from urllib.request import Request, urlopen
import io
import re
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

START = "1990-01"                 # Første måned i analysen
USE_LOCAL_CSV = False             # Skift til True for din egen CSV
LOCAL_CSV = "macro_market_monthly.csv"
VARIANCE_WINDOW = 12              # Rullende varians over 12 måneder
HAC_LAGS = 6
SEED = 123

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT = ROOT / "seminar_output"
OUTPUT.mkdir(exist_ok=True)

print("\n" + "=" * 66)
print("SEMINAR: MACROECONOMIC REGIMES AND CONDITIONAL PRICE OF RISK")
print("=" * 66, flush=True)

# ============================================================
# 1. HENT MAANEDLIGE MAKRODATA + MARKEDSMERAFKAST
# ============================================================
print("\n[1/6] Indlæser data ...", flush=True)

if USE_LOCAL_CSV:
    path = ROOT / LOCAL_CSV
    raw = pd.read_csv(path)
    raw.index = pd.to_datetime(raw.pop("date")).dt.to_period("M")
    required = ["ip_growth", "unemployment_change", "financial_stress", "market_excess"]
    missing = [col for col in required if col not in raw]
    if missing:
        raise ValueError(f"Din CSV mangler kolonnerne: {missing}")
    df = raw[required + (["USREC"] if "USREC" in raw else [])].copy()
    source = f"Lokal CSV ({path.name}) – KONTROLLÉR DATAGRUNDLAGET"

else:
    # FRED CSV: industriproduktion, arbejdsløshed, kreditspænd og recessioner
    fred = {}
    for code in ["INDPRO", "UNRATE", "BAA10Y", "USREC"]:
        print(f"   Henter FRED: {code}", flush=True)
        req = Request(
            f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urlopen(req, timeout=40) as response:
            blob = response.read()
        table = pd.read_csv(io.BytesIO(blob), na_values=["."])
        time_col = "observation_date" if "observation_date" in table else "DATE"
        dates = pd.to_datetime(table[time_col]).dt.to_period("M")
        vals = pd.to_numeric(table[code], errors="coerce")
        series = pd.Series(vals.to_numpy(dtype=float), index=pd.PeriodIndex(dates), name=code)
        fred[code] = series.groupby(level=0).mean()

    # Kenneth French: FF3 ZIP-filen indeholder månedlige Mkt-RF i procent
    print("   Henter Fama–French: Mkt-RF", flush=True)
    ff_url = (
        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
        "F-F_Research_Data_Factors_CSV.zip"
    )
    req = Request(ff_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=50) as response:
        ff_blob = response.read()
    with zipfile.ZipFile(io.BytesIO(ff_blob)) as archive:
        fname = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        ff_lines = archive.read(fname).decode("utf-8-sig", errors="replace").splitlines()

    ff_months = {}
    for line in ff_lines:
        if not re.match(r"^\s*\d{6}\s*,", line):
            continue
        cells = [s.strip() for s in line.split(",")]
        if len(cells) < 5:
            continue
        try:
            period = pd.Period(cells[0], freq="M")
            percent = float(cells[1])
            if percent not in (-99.99, -999.0, -999.99):
                ff_months[period] = percent / 100.0
        except ValueError:
            continue
    if len(ff_months) < 100:
        raise ValueError("Kunne ikke læse månedlige Mkt-RF-afkast fra Fama–French.")

    returns = pd.Series(ff_months, name="market_excess", dtype=float).sort_index()
    df = pd.DataFrame(fred).join(returns, how="inner").sort_index()
    df["ip_growth"] = 100 * np.log(df["INDPRO"] / df["INDPRO"].shift(12))
    df["unemployment_change"] = df["UNRATE"].diff(3)
    df["financial_stress"] = df["BAA10Y"]
    source = "FRED (INDPRO, UNRATE, BAA10Y, USREC) + Kenneth French (Mkt-RF)"

# Månedligt afkast i decimaler – fx 0.01 er 1 pct.
cols = ["ip_growth", "unemployment_change", "financial_stress", "market_excess"]
df = df.loc[df.index >= pd.Period(START, freq="M")].copy()
for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna(subset=cols)

if df.index.has_duplicates:
    raise ValueError("Datasættet indeholder flere observationer for samme måned.")
if not df.index.equals(pd.period_range(df.index.min(), df.index.max(), freq="M")):
    raise ValueError("Der er manglende måneder i datamaterialet – tjek data før estimation.")
if len(df) < 100:
    raise ValueError("For få månedlige observationer. Der skal være mindst 100.")

print(f"   Kilde: {source}")
print(f"   Periode: {df.index.min()} til {df.index.max()}, N={len(df)}")

# ============================================================
# 2. KONSTRUÉR MAKROØKONOMISK INDEKS
# ============================================================
print("\n[2/6] Konstruerer makroindeks ...", flush=True)

x = pd.DataFrame(index=df.index)
x["activity"] = df["ip_growth"]
x["labor"] = -df["unemployment_change"]
x["credit"] = -df["financial_stress"]
z_scores = (x - x.mean()) / x.std(ddof=0)
df["macro_index"] = z_scores.mean(axis=1)  # Høj værdi = bedre økonomiske forhold

print("   Indeks = (Z(IP-vækst) - Z(ændring arbejdsløshed) - Z(kreditspænd))/3")
print(f"   Gennemsnit: {df['macro_index'].mean():.4f}")
print(f"   Standardafvigelse: {df['macro_index'].std():.4f}")

# ============================================================
# 3. ESTIMÉR MARKOV SWITCHING AR(1) VIA MAXIMUM LIKELIHOOD
# ============================================================
print("\n[3/6] Estimerer Markov Switching-modeller (kan tage lidt tid) ...", flush=True)

np.random.seed(SEED)
ms_ar = MarkovAutoregression(
    endog=df["macro_index"],
    k_regimes=2,
    order=1,
    trend="c",
    switching_trend=True,
    switching_ar=False,
    switching_variance=True,
)
fit_ar = ms_ar.fit(em_iter=10, search_reps=20, search_iter=10, maxiter=500, disp=False)

# Sammenlign med den gamle model uden AR-led (samme antal observationer)
np.random.seed(SEED)
ms_no_ar = MarkovRegression(
    endog=df["macro_index"].iloc[1:],
    k_regimes=2,
    trend="c",
    switching_variance=True,
)
fit_no_ar = ms_no_ar.fit(em_iter=10, search_reps=20, search_iter=10, maxiter=500, disp=False)

mu0 = float(fit_ar.params["const[0]"])
mu1 = float(fit_ar.params["const[1]"])
bad = 0 if mu0 < mu1 else 1
p00 = float(fit_ar.params["p[0->0]"])
p10 = float(fit_ar.params["p[1->0]"])
P = np.array([[p00, 1 - p00], [p10, 1 - p10]])

if not fit_ar.mle_retvals.get("converged", True):
    warnings.warn("MS-AR-estimationen konvergerede ikke. Fortolk ikke estimater ukritisk.")

print(f"   MS-AR(1): LogL={fit_ar.llf:.3f} | AIC={fit_ar.aic:.2f} | BIC={fit_ar.bic:.2f}")
print(f"   Uden AR:   LogL={fit_no_ar.llf:.3f} | AIC={fit_no_ar.aic:.2f} | BIC={fit_no_ar.bic:.2f}")
print(f"   'Dårligt' regime = {bad} (lavest makroindeksmiddelværdi)")

# ============================================================
# 4. HAMILTON: FILTREREDE SANDSYNLIGHEDER OG q_t
# ============================================================
print("\n[4/6] Beregner regimesandsynligheder og forventet markedsvarians ...")

# AR(1)-modellen mister første observation, så indeks starter i måned 2.
regime_dates = df.index[1:]
filtered = pd.DataFrame(
    np.asarray(fit_ar.filtered_marginal_probabilities),
    index=regime_dates,
    columns=["state0", "state1"],
)
smoothed = pd.DataFrame(
    np.asarray(fit_ar.smoothed_marginal_probabilities),
    index=regime_dates,
    columns=["state0", "state1"],
)

if not np.allclose(filtered.sum(axis=1), 1.0, atol=1e-6):
    raise AssertionError("Filtrerede sandsynligheder summerer ikke til 1.")

df["p_bad_filtered"] = filtered[f"state{bad}"]
df["p_bad_smoothed"] = smoothed[f"state{bad}"]

# Konservativ timing: ved udgangen af måned t anvendes makroinformation til t-1.
# Vi vil forudsige dårlig tilstand ved t+1: to regimeovergange fra t-1.
q = filtered.shift(1).to_numpy() @ np.linalg.matrix_power(P, 2)
df["q_bad_next_month"] = pd.Series(q[:, bad], index=regime_dates)

# Tidspunkt-t-afkast og varians bruger KUN afkast frem til tidspunkt t.
df["variance_t"] = df["market_excess"].rolling(VARIANCE_WINDOW).var(ddof=1)
df["return_next_month"] = df["market_excess"].shift(-1)

# ============================================================
# 5. ESTIMÉR PRISEN PÅ MARKEDSRISIKO
# ============================================================
print("\n[5/6] Estimerer r_(t+1)^e = alpha*v_t + beta*q_t*v_t + u_(t+1) ...")

sample = df[["return_next_month", "variance_t", "q_bad_next_month"]].dropna().copy()
sample["v_x_q"] = sample["variance_t"] * sample["q_bad_next_month"]

X = sample[["variance_t", "v_x_q"]]
y = sample["return_next_month"]

if np.linalg.matrix_rank(X.to_numpy()) < 2:
    raise ValueError("Regressorerne har ikke fuld rang. Kan ikke identificere alpha/beta.")

pricing = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
benchmark = sm.OLS(y, X[["variance_t"]]).fit(
    cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS}
)

alpha = float(pricing.params["variance_t"])
beta = float(pricing.params["v_x_q"])
p_beta = float(pricing.pvalues["v_x_q"])
ci_low, ci_high = pricing.conf_int().loc["v_x_q"].tolist()
sample["gamma_hat"] = alpha + beta * sample["q_bad_next_month"]
sample["expected_excess_return"] = pricing.predict(X)

pricing_table = pd.DataFrame({
    "koefficient": pricing.params,
    "HAC_standardfejl": pricing.bse,
    "t_stat": pricing.tvalues,
    "p_vaerdi": pricing.pvalues,
    "CI95_nedre": pricing.conf_int()[0],
    "CI95_oevre": pricing.conf_int()[1],
})
macro_table = pd.DataFrame({
    "koefficient": fit_ar.params,
    "standardfejl": fit_ar.bse,
    "p_vaerdi": fit_ar.pvalues,
})

rmse_full = np.sqrt(np.mean((y - pricing.predict(X)) ** 2))
rmse_base = np.sqrt(np.mean((y - benchmark.predict(X[["variance_t"]])) ** 2))

if sample["q_bad_next_month"].std() < 0.03:
    warnings.warn("Den forventede regimeprobabilitet varierer meget lidt; beta kan være svær at identificere.")

# ============================================================
# 6. PRINT ALT I KONSOLLEN + GEM OUTPUT TIL CHATGPT
# ============================================================
print("\n[6/6] Udskriver resultater og gemmer fil ...", flush=True)

modelvalg = pd.DataFrame({
    "model": ["Markov uden AR", "MS-AR(1)"],
    "N": [int(fit_no_ar.nobs), int(fit_ar.nobs)],
    "LogL": [fit_no_ar.llf, fit_ar.llf],
    "AIC": [fit_no_ar.aic, fit_ar.aic],
    "BIC": [fit_no_ar.bic, fit_ar.bic],
})

report = "\n".join([
    "=" * 76,
    "SEND DENNE TEKST TIL CHATGPT FOR FORTOLKNING",
    "=" * 76,
    f"Kilde: {source}",
    f"Stikprøve: {df.index.min()} - {df.index.max()}",
    f"N (makro-model): {int(fit_ar.nobs)} | N (risk-pricing): {int(pricing.nobs)}",
    "",
    "1) MODELVALG – AIC OG BIC",
    modelvalg.to_string(index=False, float_format=lambda a: f"{a:.4f}"),
    "",
    "2) MS-AR(1) – MLE-PARAMETRE",
    macro_table.to_string(float_format=lambda a: f"{a:.5f}"),
    f"Dårligt regime er state {bad} (lavest gennemsnit for makroindekset).",
    "Overgangsmatrix (rækker=i, kolonner=j):",
    np.array2string(P, precision=4),
    f"Forventet varighed i state 0: {1/(1-P[0,0]):.2f} måneder",
    f"Forventet varighed i state 1: {1/(1-P[1,1]):.2f} måneder",
    f"MS-AR konvergerede: {fit_ar.mle_retvals.get('converged', 'ukendt')}",
    "",
    "3) RISK PRICING – OLS MED NEWEY-WEST/HAC STANDARDERROR",
    "Regression: r^e_(t+1) = alpha*variance_t + beta*q_t*variance_t + u_(t+1)",
    pricing_table.to_string(float_format=lambda a: f"{a:.6f}"),
    f"H0: beta = 0 | beta_hat={beta:.5f}, p(HAC)={p_beta:.5f}",
    f"95% CI for beta: [{ci_low:.5f}, {ci_high:.5f}]",
    f"Gamma ved q=0: {alpha:.5f}",
    f"Gamma ved q=1: {alpha + beta:.5f}",
    f"Gns. forventet sandsynlighed for dårligt regime: {sample['q_bad_next_month'].mean():.4f}",
    f"Standardafvigelse i q: {sample['q_bad_next_month'].std():.4f}",
    f"Middelværdi markedsvarians: {sample['variance_t'].mean():.6f}",
    f"Beta-effekt ved gns. varians (q:0->1): {beta*sample['variance_t'].mean()*10000:.2f} basispunkter pr. måned",
    f"In-sample RMSE fuld model: {rmse_full:.6f}",
    f"In-sample RMSE uden regimeled: {rmse_base:.6f}",
    "",
    "4) DATA- OG INFERENSFORBEHOLD",
    "- In-sample model og fuld-stikprøve-standardisering (IKKE realtidsprognoser).",
    "- Makroinformation antages tilgængelig med minimum 1 måneds lag.",
    "- FRED-serier er reviderede; der anvendes ikke realtidsvintages.",
    "- HAC-standardfejl behandler q_t som observeret, dvs. ignorerer first-stage usikkerhed.",
    "- Rullende afkastvarians er en proxy; beta er en association, ikke kausal effekt.",
    "- Ingen konstant i risk-pricing regressionen: direkte test af mu_t=gamma_t*v_t.",
    "=" * 76,
])

print("\n" + report, flush=True)
(OUTPUT / "SEND_MIG_DENNE_FIL.txt").write_text(report, encoding="utf-8")
modelvalg.to_csv(OUTPUT / "modelvalg.csv", index=False)
macro_table.to_csv(OUTPUT / "msar_parametre.csv")
pricing_table.to_csv(OUTPUT / "pricing_resultater.csv")
df.to_csv(OUTPUT / "fuldt_datasaet.csv", index_label="date")
sample.to_csv(OUTPUT / "pricing_dataset.csv", index_label="date")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df.index.to_timestamp(), df["p_bad_filtered"], label="Filtered P(bad)")
ax.plot(df.index.to_timestamp(), df["p_bad_smoothed"], label="Smoothed P(bad)", alpha=0.7)
if "USREC" in df:
    ax.fill_between(df.index.to_timestamp(), 0, 1,
                    where=(df["USREC"].fillna(0).to_numpy() == 1),
                    alpha=0.15, label="NBER-recession")
ax.set(title="Makroøkonomiske regimer", ylabel="Sandsynlighed", ylim=(0, 1))
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "figur_1_makroregimer.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sample.index.to_timestamp(), sample["gamma_hat"])
ax.set(title="Estimeret pris på markedsrisiko: gamma_t = alpha + beta*q_t",
       ylabel="Pris pr. enhed markedsvarians")
fig.tight_layout()
fig.savefig(OUTPUT / "figur_2_risikopris.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(sample.index.to_timestamp(), 100*sample["return_next_month"], alpha=0.5,
        label="Realiseret merafkast (%)")
ax.plot(sample.index.to_timestamp(), 100*sample["expected_excess_return"],
        label="Modelbaseret forventet merafkast (%)")
ax.set(title="Forventet og realiseret markedsmerafkast", ylabel="Procent pr. måned")
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT / "figur_3_afkast.png", dpi=160)
plt.close(fig)

print(f"\nFÆRDIG – send denne fil til ChatGPT: {OUTPUT / 'SEND_MIG_DENNE_FIL.txt'}")
print("Figurer: figur_1_makroregimer.png, figur_2_risikopris.png, figur_3_afkast.png")
