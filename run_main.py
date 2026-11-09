import atomica as at
import numpy as np
import os
import pandas as pd
import utils_Canada as ut
import utils_plotting as ut_plt

dirname = os.path.dirname(os.path.abspath(__file__)) + '/'  # directory name

if not os.path.exists('results'): os.makedirs('results')
np.seterr(all='raise')

#%% Atomica project definition
application = 'Canada'
F = 'pnsp_oat_framework_{}.xlsx'.format(application)
D = 'pnsp_oat_databook_{}.xlsx'.format(application)

P = at.Project(framework=F, databook=D, do_run=False)

P.settings.sim_dt    = 0.25 # simulation timestep
P.settings.sim_start = 2015 # simulation start year
P.settings.sim_end   = 2040 # simulation end year
P.settings.multinomial = True
ut.run_initialization(P) # initialise model

#%% Settings
start_year = 2026   # first year of the outcomes window
end_year = 2035     # last year of the outcomes window
ref_year = 2025     # the counterfactual is frozen at the values reached at the end of 2024
n_samples = 2000    # simulations per scenario (the sensitivity analyses use the same number)

input_pars_path = dirname + 'input_pars.xlsx'

#%% Calibration and policy change scenario
calibration_name = 'Status-quo\nmodel prediction'
res_baseline, P = ut.auto_calibration(P, calibration_name, n_samples=n_samples)
ut.export_calibration_table(P)

res_baseline.name = 'Policy change'

#%% Counterfactual: OAMT parameters frozen at their end-of-2024 values
def get_ts_up_to(P, par_name, ref_year, pop='Prisoners', freeze_val=None, end_year=None):
    """Get time series up to ref_year, optionally extending flat to end_year"""
    t = P.data.tdve[par_name].ts[pop].t
    v = P.data.tdve[par_name].ts[pop].vals
    t_out = [ti for ti in t if ti <= ref_year]
    v_out = [vi for ti, vi in zip(t, v) if ti <= ref_year]
    if end_year is not None:
        t_out = t_out + [end_year]
        v_out = v_out + [freeze_val if freeze_val is not None else v_out[-1]]
    return t_out, v_out

params_to_freeze = [
    'prop_oat_meth', 'prop_oat_subx', 'prop_oat_subl',
    'prop_pwid_oat',
    'uptake_cst_meth', 'uptake_cst_subx', 'uptake_cst_subl',
    'subx_to_subl', 'entry_subx_to_subl'
]

scenario_args = []
for par in params_to_freeze:
    t_out, v_out = get_ts_up_to(P, par, ref_year, end_year=end_year)
    scenario_args += [par, t_out, v_out]

res_counter, P_counter = ut.run_scenario(P, 'Counterfactual', *scenario_args, n_samples=n_samples)

#%% Calibration plots
ut_plt.multi_plot(P, [res_baseline], ylimit=True)
ut_plt.plot_oat_stocks(P, res_baseline)

#%% Main analysis
results = ut.calculate_all_outcomes(P, res_counter, res_baseline, input_pars_path, start_year, end_year,
                                    ci_method='bootstrap', community_oat_cost='medication')
ut.write_economic_outcomes(results, 'results/outcomes_economic.xlsx')

ut_plt.plot_cca_summary(results, start_year=start_year, end_year=end_year, include_total=False)
ut_plt.plot_trajectories_difference(P, res_baseline, res_counter, start_year=start_year, end_year=end_year)

#%% Sensitivity analyses (medians only)
sens = {}
sens['Main analysis (frozen at end of 2024)'] = results

def outcomes(P, res_c, res_p, econ=input_pars_path, end=end_year):
    return ut.calculate_all_outcomes(P, res_c, res_p, econ, start_year, end,
                                     ci_method='bootstrap', community_oat_cost='medication')

def _override_args(par, fn, from_year=2026):
    # Keep the databook series up to 2025 and replace it from from_year with fn(value), held flat to end_year
    ts = P.data.tdve[par].ts['Prisoners']
    t_out = [ti for ti in ts.t if ti < from_year]
    v_out = [vi for ti, vi in zip(ts.t, ts.vals) if ti < from_year]
    base = ts.get(from_year - 1)
    if base is None:           # parameter entered as a constant in the databook
        base = ts.assumption
    if base is None and len(ts.vals):
        base = ts.vals[-1]
    if not t_out:              # no earlier time points: pin the original value from the start
        t_out, v_out = [P.settings.sim_start], [base]
    return [par, t_out + [from_year, end_year], v_out + [fn(base), fn(base)]]

def _run_both(label, extra_args):
    # Both arms are rerun with the same changed assumption so the difference stays paired
    res_p = ut.run_scenario(P, 'Policy change - ' + label, *extra_args, n_samples=n_samples)[0]
    res_c = ut.run_scenario(P, 'Counterfactual - ' + label, *scenario_args, *extra_args, n_samples=n_samples)[0]
    return outcomes(P, res_c, res_p)

# Counterfactual frozen at July 2025 instead of the end of 2024
label = 'Counterfactual frozen at July 2025'
args = []
for par in params_to_freeze:
    t_out, v_out = get_ts_up_to(P, par, 2025.5, end_year=end_year)
    args += [par, t_out, v_out]
res_alt = ut.run_scenario(P, label, *args, n_samples=n_samples)[0]
sens[label] = outcomes(P, res_alt, res_baseline)

# 1 instead of 2 correctional officers per administration (cost only)
df_2co = pd.read_excel(input_pars_path, index_col=0)
df_1co = ut.sensitivity_coverage_co_staffing(df_2co, co_number=1.0)
sens['1 officer per administration'] = outcomes(P, res_counter, res_baseline, econ=df_1co)

# 50% of entrants with OUD instead of 30%. Entry on OAMT is calibrated to CSC counts, so the OUD share
# only matters for the post-release tree (no model rerun)
df_oud = pd.read_excel(input_pars_path, index_col=0)
df_oud.loc['prop_oud_entry', ['point_estimate', 'lower_bound', 'upper_bound']] = 0.5
sens['50% of entrants with OUD'] = outcomes(P, res_counter, res_baseline, econ=df_oud)

# HCV prevalence among entrants with a history of injecting halves 2026-2030
t_p, v_p = get_ts_up_to(P, 'prev_pwid', 2025)
v_last = v_p[-1]
hcv_args = ['prev_pwid', t_p + [2026, 2030, end_year], v_p + [v_last, 0.5*v_last, 0.5*v_last]]
sens['Lower HCV burden (entrant prevalence halves by 2030)'] = _run_both('lower HCV burden', hcv_args)

# OAMT effect on injecting and IRIs halved from 2026
eff_args = []
for par in ('meth_efficacy', 'subx_efficacy', 'subl_efficacy'):
    eff_args += _override_args(par, lambda x: 0.5*x)
for par in ('OR_iri_alone', 'OR_iri_combined'):
    # the model reads these as the size of the reduction, so halving them halves the effect
    eff_args += _override_args(par, lambda x: 0.5*x)
sens['OAMT effect on injecting and IRIs halved'] = _run_both('halved OAMT effect', eff_args)

# LAI-BPN retention equal to SL-BPN from 2026 (both arms). In prison, initiation is proportional to dropout
# (1 - retention), so the LAI-BPN uptake constant is scaled by (1 - old retention)/(1 - new retention)
# to keep initiation constant and only change how long people stay on treatment
def _val(par):
    ts_ = P.data.tdve[par].ts['Prisoners']
    return float(ts_.vals[-1]) if len(ts_.vals) else float(ts_.assumption)
ret_old, ret_new = _val('subl_retention'), _val('subx_retention')
comp = (1 - ret_old) / (1 - ret_new)
from_year = 2026

def _scaled_from(t_list, v_list, factor):
    # keep the original path up to a quarter-step before from_year, then scale
    t_arr, v_arr = np.array(t_list, dtype=float), np.array(v_list, dtype=float)
    t_new = [float(t) for t in t_arr if t < from_year - 0.25]
    v_new = [float(v) for t, v in zip(t_arr, v_arr) if t < from_year - 0.25]
    t_new += [from_year - 0.25, float(from_year)]
    v_new += [float(np.interp(from_year - 0.25, t_arr, v_arr)), float(np.interp(from_year, t_arr, v_arr)) * factor]
    for t, v in zip(t_arr, v_arr):
        if t > from_year:
            t_new.append(float(t)); v_new.append(float(v) * factor)
    return t_new, v_new

retention_args = ['subl_retention', [2015, from_year - 0.25, from_year], [ret_old, ret_old, ret_new]]
# policy arm: databook uptake series; counterfactual arm: the frozen series already in scenario_args
t_pol, v_pol = ut._read_current_series(P, 'uptake_cst_subl')
pol_args = retention_args + ['uptake_cst_subl'] + list(_scaled_from(t_pol, v_pol, comp))
cf_base = {scenario_args[i]: (scenario_args[i + 1], scenario_args[i + 2]) for i in range(0, len(scenario_args), 3)}
cf_other = [x for i in range(0, len(scenario_args), 3) if scenario_args[i] != 'uptake_cst_subl' for x in scenario_args[i:i + 3]]
cf_args = cf_other + retention_args + ['uptake_cst_subl'] + list(_scaled_from(*cf_base['uptake_cst_subl'], comp))
res_p_ret = ut.run_scenario(P, 'Policy change - LAI-BPN retention = SL-BPN', *pol_args, n_samples=n_samples)[0]
res_c_ret = ut.run_scenario(P, 'Counterfactual - LAI-BPN retention = SL-BPN', *cf_args, n_samples=n_samples)[0]
df_cont = ut.sensitivity_lai_bup_continuity(input_pars_path, match_to='subx')   # post-release retention = SL-BPN
sens['LAI-BPN retention = SL-BPN from 2026, in prison'] = outcomes(P, res_c_ret, res_p_ret)
sens['LAI-BPN retention = SL-BPN from 2026, in the community'] = outcomes(P, res_counter, res_baseline, econ=df_cont)
sens['LAI-BPN retention = SL-BPN from 2026, in prison and community'] = outcomes(P, res_c_ret, res_p_ret, econ=df_cont)

# Switch success of 20% and 80% instead of 55%. The counterfactual has no switching, so it is reused
for p_succ in [0.2, 0.8]:
    lab = 'Switch success {:.0f}%'.format(100 * p_succ)
    res_sw = ut.run_scenario(P, 'Policy change - ' + lab, 'prop_subx_subl_success', [2015, end_year],
                             [p_succ, p_succ], n_samples=n_samples)[0]
    sens[lab] = outcomes(P, res_counter, res_sw)

# Hypothetical equal-coverage scenario: LAI-BPN stays first-line but initiation is scaled by one multiplier so
# the number on OAMT matches the counterfactual. Switch success is set to 100% so fewer SL-BPN patients are
# lost in the transition
eq = ut.sensitivity_equal_coverage(P, P_counter, res_baseline, res_counter, start_year, end_year, n_samples=n_samples,
                                   extra_overrides=['prop_subx_subl_success', [2015, end_year], [1.0, 1.0]],
                                   safe_uptake_ceiling=0.9)
res_eq, P_eq = eq['ensemble'], eq['P']
sens['Coverage maintained, 100% transition success (hypothetical)'] = outcomes(P_eq, res_counter, res_eq)

# Discount rate (0%, 3% instead of 1.5%) and time horizon (to 2030, to 2040 instead of 2035).
# These reuse the main arms and only change the discounting or the years that are counted
for rate in (0.0, 0.03):
    df_disc = pd.read_excel(input_pars_path, index_col=0)
    df_disc.loc['discount', ['point_estimate', 'lower_bound', 'upper_bound']] = rate
    sens['Discount rate {:.0f}%'.format(100 * rate)] = outcomes(P, res_counter, res_baseline, econ=df_disc)
for end_h in (2030, 2040):
    sens['Time horizon 2026–{}'.format(end_h)] = outcomes(P, res_counter, res_baseline, end=end_h)

ut.write_lean_sensitivity(sens, 'results/sensitivity_lean.xlsx', n_sens=n_samples)
