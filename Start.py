# %% 0. Imports and settings
from pathlib import Path
import inspect
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

START = '1967-01-01'
END = '2025-12-01'
TRAIN_END = '2019-12-01'
SEED = 42
OUT = Path(__file__).resolve().parent / 'results'
OUT.mkdir(exist_ok=True)

# %% 1. Download and validate monthly data
variables = ['PAYEMS', 'INDPRO', 'CMRMTSPL', 'W875RX1', 'USREC', 'RECPROUSM156N']
macro_variables = ['Employment', 'Production', 'Sales', 'Income']
url = ('https://fred.stlouisfed.org/graph/fredgraph.csv?id=' + ','.join(variables)
       + '&cosd=' + START + '&coed=' + END)

data = pd.read_csv(url, index_col=0, parse_dates=True, na_values=['.'])
data = data.sort_index().reindex(pd.date_range(START, END, freq='MS', name='observation_date'))
data = data.rename(columns=dict(zip(variables, macro_variables + ['NBER', 'CP_Probability'])))
data = data.apply(pd.to_numeric, errors='raise')
data['CP_Probability'] /= 100
print('\nData coverage:', data.index.min(), data.index.max(), data.shape)
print(data.isna().sum())
macro = data[macro_variables].copy()
if macro.isna().any().any():
    raise ValueError(f'Missing macro observations:\n{macro[macro.isna().any(axis=1)]}')
if (macro <= 0).any().any():
    raise ValueError('Macro levels must be positive before taking logs.')
if not data['CP_Probability'].dropna().between(0, 1).all():
    raise ValueError('Check Chauvet-Piger probability units.')
data.to_csv(OUT / 'macro_levels.csv')

# %% 2. Log growth and descriptive statistics
macro_growth = (100 * np.log(macro).diff()).iloc[1:].copy()
if not np.isfinite(macro_growth.to_numpy()).all():
    raise ValueError('Non-finite growth observations.')
print('\nDescriptive statistics:\n', macro_growth.describe().round(4))
print('\nMinimum dates:\n', macro_growth.idxmin())
macro_growth.describe().to_csv(OUT / 'descriptive_statistics.csv')
axes = macro_growth.plot(subplots=True, figsize=(12, 9), grid=True,
                         title='Monthly macroeconomic log growth')
fig = axes[0].get_figure()
fig.tight_layout()
fig.savefig(OUT / '01_growth.png', dpi=180)
plt.show()

# %% 3. ADF tests (compatible with older/newer statsmodels)
adf_kwargs = {'result_object': False} if 'result_object' in inspect.signature(adfuller).parameters else {}
adf_rows = []
for variable in macro_variables:
    level = adfuller(np.log(macro[variable]), regression='ct', autolag='AIC', **adf_kwargs)
    growth = adfuller(macro_growth[variable], regression='c', autolag='AIC', **adf_kwargs)
    adf_rows.append({'Variable': variable, 'ADF Level': level[0], 'P-value Level': level[1],
                     'ADF Growth': growth[0], 'P-value Growth': growth[1]})
adf_table = pd.DataFrame(adf_rows).set_index('Variable')
print('\nADF tests:\n', adf_table.round(4))
adf_table.to_csv(OUT / 'adf_tests.csv')

# %% 4. Full-sample PCA (M1 and M2)
growth_full = macro_growth[macro_variables].copy()
scaler = StandardScaler()
X = scaler.fit_transform(growth_full)
pca = PCA(n_components=1)
pca.fit(X)
sign = 1 if pca.components_[0].sum() >= 0 else -1
macro_growth['Factor'] = sign * pca.transform(X)[:, 0]
weights = pd.Series(sign * pca.components_[0], index=macro_variables, name='Full sample')
print('\nFull-sample PCA explained variance:', pca.explained_variance_ratio_[0])
print(weights)

# %% 5. Helper functions for model estimation and diagnostics
def fit_model(factor, switching_variance, search_reps=50):
    np.random.seed(SEED)
    model = MarkovRegression(factor, k_regimes=2, trend='c',
                             switching_variance=switching_variance)
    result = model.fit(search_reps=search_reps, maxiter=1000, disp=False)
    if not result.mle_retvals.get('converged', False):
        warnings.warn('Model did not converge; interpret results with caution.')
    return model, result


def store_probabilities(target, fitted, evaluated, suffix):
    means = fitted.params[['const[0]', 'const[1]']].to_numpy()
    bad = int(np.argmin(means))
    good = 1 - bad
    target['p_filtered' + suffix] = evaluated.filtered_marginal_probabilities[bad]
    target['p_smoothed' + suffix] = evaluated.smoothed_marginal_probabilities[bad]
    return good, bad


def diagnostics(fitted, good, bad, label):
    order = [good, bad]
    transition = fitted.regime_transition[:, :, 0].T[np.ix_(order, order)]
    transition_table = pd.DataFrame(transition, index=['Good', 'Bad'], columns=['Good', 'Bad'])
    durations = pd.Series(1 / (1 - np.diag(transition)), index=['Good', 'Bad'])
    params = fitted.params
    print(f'\n{label}\n', fitted.summary())
    print('Transition matrix (rows = previous state):\n', transition_table)
    print('Expected duration in months:\n', durations)
    transition_table.to_csv(OUT / f'{label}_transition.csv')
    return {'Model': label, 'N estimation': fitted.nobs, 'LogLik': fitted.llf,
            'AIC': fitted.aic, 'BIC': fitted.bic,
            'Mean Good': params[f'const[{good}]'], 'Mean Bad': params[f'const[{bad}]'],
            'Variance Good': params.get(f'sigma2[{good}]', params.get('sigma2', np.nan)),
            'Variance Bad': params.get(f'sigma2[{bad}]', params.get('sigma2', np.nan)),
            'P Good-Good': transition[0, 0], 'P Bad-Bad': transition[1, 1],
            'Duration Good': durations.iloc[0], 'Duration Bad': durations.iloc[1],
            'Converged': fitted.mle_retvals.get('converged', False)}

# %% 6. M1: switching mean, constant variance
model, result = fit_model(macro_growth['Factor'], False, search_reps=30)
good_regime, bad_regime = store_probabilities(macro_growth, result, result, '')
row1 = diagnostics(result, good_regime, bad_regime, 'M1')
print('\nM1: lowest factor observations:\n', macro_growth.nsmallest(10, 'Factor')[['Factor', 'p_smoothed']])

# %% 7. M2: switching mean and variance
model2, result2 = fit_model(macro_growth['Factor'], True)
good_regime2, bad_regime2 = store_probabilities(macro_growth, result2, result2, '_M2')
row2 = diagnostics(result2, good_regime2, bad_regime2, 'M2')

# %% 8. M3: PCA and switching-mean estimation before COVID
# Fit BOTH the scaler and PCA using only the training sample.
growth_train = growth_full.loc[:TRAIN_END].copy()
scaler3 = StandardScaler()
X_train = scaler3.fit_transform(growth_train)
pca3 = PCA(n_components=1).fit(X_train)
sign3 = 1 if pca3.components_[0].sum() >= 0 else -1
macro_growth['Factor_M3'] = sign3 * pca3.transform(scaler3.transform(growth_full))[:, 0]
weights3 = pd.Series(sign3 * pca3.components_[0], index=macro_variables, name='Pre-COVID')
print('\nPre-COVID PCA explained variance:', pca3.explained_variance_ratio_[0])
print(weights3)
pd.concat([weights, weights3], axis=1).to_csv(OUT / 'pca_weights.csv')
pd.Series({'Full sample': pca.explained_variance_ratio_[0],
           'Pre-COVID': pca3.explained_variance_ratio_[0]}).to_csv(OUT / 'pca_explained_variance.csv')
factor_train = macro_growth.loc[:TRAIN_END, 'Factor_M3']
model3, result3 = fit_model(factor_train, False)
# Apply fixed pre-COVID parameters to the complete series; do NOT refit.
model3_full = MarkovRegression(macro_growth['Factor_M3'], k_regimes=2,
                              trend='c', switching_variance=False)
result3_full = model3_full.smooth(result3.params)
good_regime3, bad_regime3 = store_probabilities(macro_growth, result3, result3_full, '_M3')
row3 = diagnostics(result3, good_regime3, bad_regime3, 'M3')

# %% 9. Compare and validate (only AFTER all models are fitted)
prob_columns = ['p_filtered', 'p_smoothed', 'p_filtered_M2', 'p_smoothed_M2',
                'p_filtered_M3', 'p_smoothed_M3']
comparison = macro_growth[prob_columns].join(data[['CP_Probability', 'NBER']])
model_table = pd.DataFrame([row1, row2, row3]).set_index('Model')
print('\nModel parameters (M3 uses a different sample and factor scale):\n', model_table.round(4).T)
print('\nCorrelations:\n', comparison.corr().round(3))
validation = []
for col in prob_columns:
    for sample, frame in [('Full', comparison), ('Pre-COVID', comparison.loc[:TRAIN_END]),
                          ('2020 onward', comparison.loc['2020-01-01':])]:
        valid = frame[[col, 'NBER', 'CP_Probability']]
        nber = valid[[col, 'NBER']].dropna()
        pred = nber[col] > .5
        actual = nber['NBER'].eq(1)
        tp = int((pred & actual).sum())
        fp = int((pred & ~actual).sum())
        fn = int((~pred & actual).sum())
        validation.append({'Probability': col, 'Sample': sample, 'N': len(nber),
                           'Months above 0.5': int(pred.sum()), 'TP': tp, 'FP': fp, 'FN': fn,
                           'Recall': tp / (tp + fn) if tp + fn else np.nan,
                           'Precision': tp / (tp + fp) if tp + fp else np.nan,
                           'Corr NBER': valid[col].corr(valid['NBER']),
                           'Corr CP': valid[col].corr(valid['CP_Probability'])})
validation_table = pd.DataFrame(validation)
print('\nValidation, full sample:\n', validation_table.query("Sample == 'Full'").round(3).to_string(index=False))
for label, col in [('M1', 'p_smoothed'), ('M2', 'p_smoothed_M2'), ('M3', 'p_smoothed_M3')]:
    print(f'\n{label}: months above 0.5:', int((comparison[col] > .5).sum()))

# %% 10. Plots: same benchmark in each panel
fig, axes = plt.subplots(3, 1, figsize=(15, 10), sharex=True, sharey=True)
for ax, suffix, title in zip(axes, ['', '_M2', '_M3'],
        ['M1: Switching mean', 'M2: Switching mean and variance', 'M3: Pre-COVID estimation']):
    ax.fill_between(comparison.index, 0, 1, where=comparison['NBER'].eq(1).to_numpy(),
                    color='grey', alpha=.2, label='NBER recession')
    ax.plot(comparison.index, comparison['CP_Probability'], '--', color='green', label='Chauvet-Piger (smoothed)')
    ax.plot(comparison.index, comparison['p_smoothed' + suffix], label='Our smoothed probability')
    ax.plot(comparison.index, comparison['p_filtered' + suffix], alpha=.55, label='Our filtered probability')
    if suffix == '_M3':
        ax.axvline(pd.Timestamp('2020-01-01'), color='black', linestyle=':', label='End of training')
    ax.set(title=title, ylim=(0, 1.03), ylabel='Probability')
    ax.legend(loc='upper right', fontsize=8)
axes[-1].set_xlabel('Observation month')
fig.tight_layout()
fig.savefig(OUT / '02_model_comparison.png', dpi=200)
plt.show()

# %% 11. Export all results
# These probabilities are based on revised data. Filtered != real-time vintage.
# M1/M2 use full-sample parameters; M3 only has fixed-parameter evaluation after 2019.
# Smoothed probabilities use future observations, even for M3.
comparison.to_csv(OUT / 'macroeconomic_regimes.csv')
macro_growth.to_csv(OUT / 'macro_growth_and_factors.csv')
model_table.to_csv(OUT / 'model_parameters.csv')
validation_table.to_csv(OUT / 'model_validation.csv', index=False)
print('\nSaved results to:', OUT)
