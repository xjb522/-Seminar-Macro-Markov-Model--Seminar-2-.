"""Seminar: Macro regimes and the conditional price of market risk.

Build monthly US macro data (FRED) and market excess returns (Fama-French),
estimate a two-state Markov switching AR(1) for a transparent macro index,
and run an illustrative conditional risk-return regression.

Install: pip install numpy pandas scipy statsmodels pandas-datareader matplotlib
Run:     python seminar_msar_fred.py

NOTE: This is a retrospective, IN-SAMPLE benchmark. Full-sample standardization
and full-sample parameter estimation are not point-in-time realistic. The pricing
regression conservatively assumes monthly macro data for period t-1 become
available by the end of calendar month t. Actual releases and revisions require
ALFRED vintages and a release-date-aware database for a real-time study.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas_datareader import data as web
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

START = '1990-01-01'
FRED_IDS = ['INDPRO', 'UNRATE', 'BAA10Y', 'PAYEMS', 'CPIAUCSL',
            'T10Y2Y', 'USREC']


def by_month(series: pd.Series, method='last') -> pd.Series:
    """Monthly level series: last observed point; daily spreads: monthly mean."""
    s = series.dropna().astype(float)
    grouped = s.groupby(s.index.to_period('M'))
    return grouped.mean() if method == 'mean' else grouped.last()


def download_monthly_data() -> pd.DataFrame:
    raw = web.DataReader(FRED_IDS, 'fred', START)
    macro = pd.DataFrame({
        'INDPRO': by_month(raw['INDPRO']),
        'UNRATE': by_month(raw['UNRATE']),
        'BAA10Y': by_month(raw['BAA10Y'], 'mean'),
        'PAYEMS': by_month(raw['PAYEMS']),
        'CPIAUCSL': by_month(raw['CPIAUCSL']),
        'T10Y2Y': by_month(raw['T10Y2Y'], 'mean'),
        'USREC': by_month(raw['USREC']),
    })

    # Monthly market factor is already market excess return, in percent units.
    ff = web.DataReader('F-F_Research_Data_Factors', 'famafrench', start=START)[0]
    ff.index = pd.PeriodIndex(ff.index, freq='M')
    macro['market_excess'] = (ff['Mkt-RF'] / 100).reindex(macro.index)

    # Real macro measures, annual production growth and 3-month unemployment change.
    macro['ip_growth'] = 100 * np.log(macro['INDPRO'] / macro['INDPRO'].shift(12))
    macro['unemployment_change'] = macro['UNRATE'].diff(3)
    macro['financial_stress'] = macro['BAA10Y']
    return macro


def create_macro_index(df: pd.DataFrame) -> pd.Series:
    """Higher means healthier macro-financial conditions; equal weights."""
    x = pd.DataFrame({
        'activity': df['ip_growth'],
        'labor': -df['unemployment_change'],
        'credit': -df['financial_stress'],
    }, index=df.index)
    means = x.mean()
    sds = x.std(ddof=0)
    if (sds <= 0).any():
        raise ValueError('A macro input has zero variation.')
    return ((x - means) / sds).mean(axis=1)


def estimate_model(df: pd.DataFrame):
    model = MarkovAutoregression(
        endog=df['macro_index'], k_regimes=2, order=1,
        trend='c', switching_ar=False,
        switching_trend=True, switching_variance=True,
    )
    result = model.fit(em_iter=10, search_reps=20, search_iter=10,
                       maxiter=500, disp=False)
    params = result.params
    bad_state = int(params['const[1]'] < params['const[0]'])

    p00 = float(params['p[0->0]'])
    p10 = float(params['p[1->0]'])
    P = np.array([[p00, 1 - p00], [p10, 1 - p10]])

    # The AR(1) likelihood starts with the second observation.
    date_index = df.index[1:]
    filtered = pd.DataFrame(np.asarray(result.filtered_marginal_probabilities),
                            index=date_index, columns=[0, 1])
    smoothed = pd.DataFrame(np.asarray(result.smoothed_marginal_probabilities),
                            index=date_index, columns=[0, 1])
    return result, P, bad_state, filtered, smoothed


def pricing_regression(df: pd.DataFrame, P: np.ndarray, bad_state: int,
                       filtered: pd.DataFrame):
    out = df.copy()
    out['p_bad_filtered'] = filtered[bad_state]
    out['p_bad_smoothed'] = np.nan   # set separately for ex-post plots only

    # Conservative calendar alignment for monthly macro releases:
    # at the end of month t, macro state for t-1 is available.
    # Therefore, forecast state t+1 with TWO transition steps from state t-1.
    pi_asof_t = filtered.shift(1)
    p_next = pi_asof_t.to_numpy() @ np.linalg.matrix_power(P, 2)
    out.loc[filtered.index, 'p_bad_next'] = p_next[:, bad_state]

    # Market variance estimate and the NEXT month's realized excess return.
    out['variance_forecast'] = out['market_excess'].rolling(12).var()
    out['return_next'] = out['market_excess'].shift(-1)
    sample = out[['return_next', 'variance_forecast', 'p_bad_next']].dropna()
    X = pd.DataFrame({
        'variance': sample['variance_forecast'],
        'variance_x_pbad': (sample['variance_forecast'] * sample['p_bad_next']),
    }, index=sample.index)
    # No separate intercept: E_t[r_excess(t+1)] = (alpha + beta*q_t)*v_t.
    regression = sm.OLS(sample['return_next'], X).fit(
        cov_type='HAC', cov_kwds={'maxlags': 6})
    return out, regression


def main():
    df = download_monthly_data()
    needed = ['ip_growth', 'unemployment_change', 'financial_stress',
              'market_excess']
    df = df.dropna(subset=needed).copy()
    df['macro_index'] = create_macro_index(df)
    df.to_csv('macro_market_monthly.csv', index_label='date')

    result, P, bad_state, filtered, smoothed = estimate_model(df)
    out, pricing = pricing_regression(df, P, bad_state, filtered)
    out.loc[smoothed.index, 'p_bad_smoothed'] = smoothed[bad_state]
    out.to_csv('macro_msar_outputs.csv', index_label='date')

    print('\n=== MACRO MS-AR(1) ===')
    print(result.summary())
    print('Bad state:', bad_state, '(lower estimated macro mean)')
    print('Transition matrix (row = previous state):\n', P)
    print('\n=== CONDITIONAL RISK PRICING (IN SAMPLE) ===')
    print(pricing.summary())
    print('Beta significance is conditional on estimated first-stage states.')
    print('Saved macro_market_monthly.csv and macro_msar_outputs.csv')


if __name__ == '__main__':
    main()
