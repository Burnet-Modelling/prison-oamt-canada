import atomica as at
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import math
import zlib
import re
import bisect
import collections


#%% Scenario calculations   
def update_pars(P, pars):
    """
    Updates the dataset with values specified in pars
    :param P: atomica project
    :param pars: (array) contains parameter names and values to be updated
    #the structure is ['par_name', 'population', year_list, value_list]
    :return:
    """

    pop_groups = list(P.data.pops.keys())

    for i in range(len(pars)):
        if pars[i][1] == 'Transfer':
            P.data.transfers[0].ts[pars[i][0]].assumption = pars[i][3]
        else:
            if not isinstance(pars[i][0], list):
                pars[i][0] = [pars[i][0]]
            for k in range(len(pars[i][0])):
                if pars[i][2] is None:
                    # Apply assumption parameterss
                    if pars[i][1] == 'All':
                        for j in pop_groups:
                            if len(P.data.tdve[pars[i][0][k]].ts[j].vals):
                                P.data.tdve[pars[i][0][k]].ts[j].vals = [pars[i][3]] * len(P.data.tdve[pars[i][0][k]].ts[j].vals)
                            else:
                                P.data.tdve[pars[i][0][k]].ts[j].assumption = pars[i][3]
                    else:
                        if len(P.data.tdve[pars[i][0][k]].ts[pars[i][1]]):
                            P.data.tdve[pars[i][0][k]].ts[pars[i][1]].vals = [pars[i][3]] * len(P.data.tdve[pars[i][0][k]].ts[pars[i][1]])
                        else:
                            P.data.tdve[pars[i][0][k]].ts[pars[i][1]].assumption = pars[i][3]
                else:
                    # Apply time varying parameters
                    if pars[i][1] == 'All':
                        for j in pop_groups:
                            P.data.tdve[pars[i][0][k]].ts[j].t = pars[i][2]
                            P.data.tdve[pars[i][0][k]].ts[j].vals = pars[i][3]
                    else:
                        P.data.tdve[pars[i][0][k]].ts[pars[i][1]].t = pars[i][2]
                        P.data.tdve[pars[i][0][k]].ts[pars[i][1]].vals = pars[i][3]


def run_initialization(P):
    '''
    Initializes the OAT compartments (databook page 'init') by drug type, IDU history and infection status.
    Splits the total prisoners on OAT (pop_oat) by current OAT type (prop_oat_meth/subx/subl), then by IDU
    history (prop_oat_idu) and HCV RNA prevalence (prev_pwid for IDU, prev_gp for non-IDU), mirroring how the
    inflow_* parameters (Parameters sheet) split new entrants going onto OAT.

    :param P: project
    :return:
    '''
    pop_groups = list(P.data.pops.keys())
    drugs = ['meth', 'subx', 'subl']

    for i in pop_groups:
        prop_drug = {d: P.data.tdve['prop_oat_' + d].ts[i].vals[0] for d in drugs}
        if abs(1 - sum(prop_drug.values())) > 1e-6:
            print('ERROR: OAT type proportions do not add up to 1')

        pop_oat = P.data.tdve['pop_oat'].ts[i].vals[0]
        prop_idu = P.data.tdve['prop_oat_idu'].ts[i].vals[0]
        prev_pwid = P.data.tdve['prev_pwid'].ts[i].vals[0]
        prev_gp = P.data.tdve['prev_gp'].ts[i].vals[0]

        for d in drugs:
            pop_drug = pop_oat * prop_drug[d]

            P.data.tdve['idu_' + d + '_sus'].ts[i].vals[0] = pop_drug * prop_idu * (1 - prev_pwid)
            P.data.tdve['idu_' + d + '_inf'].ts[i].vals[0] = pop_drug * prop_idu * prev_pwid
            P.data.tdve['nidu_' + d + '_sus'].ts[i].vals[0] = pop_drug * (1 - prop_idu) * (1 - prev_gp)
            P.data.tdve['nidu_' + d + '_inf'].ts[i].vals[0] = pop_drug * (1 - prop_idu) * prev_gp


def run_scenario(P, scenario_name, *args, n_samples=500, rand_seed=20240430):
    '''
    Run scenarios by changing parameter values (stochastic)
    :param P: atomica project
    :param scenario_name: name of scenario
    :param n_samples: number of samples (optional, default = 100)
    :param *args: optional: provide a set of variables to change: the parameter code name,
    the implementation year and the new parameter value, in that order.
    Multiple {par_code, year, value} sets can be specified
    
    :return: simulation result
    
    :example: changing the value of parameter 'nsp_cov' to 0.5 starting 2023, and 'nsp_cst' to 0.9 starting 2023.
    result_nsp_50 = run_scenario(P, 'NSP 50%-90%', 'nsp_cov', 2023, 0.5, 'nsp_cst', 2023, 0.9)
    '''
    
    P_scen = P.copy()
    
    pop = P_scen.data.pops.keys() # define population    
    outputs = ['pop_prison',
               'iri_m',
               'inci_m',
               'iri_per100py',
               'inci_per100py',
               'inci_per1000py',
               'iri_idu_per100py',
               'inci_idu_per100py',
               'prev',
               'prop_idu',
               'pop_inf',
               'pop_sus',
               'nsp_cov_prisons',
               'nsp_cov_prisoners',
               'nidu_inf:release',
               'idu_inf:release',
               'inflow_idu_inf:flow',
               'inflow_nidu_inf:flow',
               'treat_m',
               'incarceration_r',
               'prop_pwid',
               'prob_iri',
               'prob_treat',
               'foi',
               'pop_oat','prop_oat_meth_m','prop_oat_subl_m','prop_oat_subx_m',
               'pop_oat_meth','pop_oat_subl','pop_oat_subx',
               'prop_oat_meth','prop_oat_subx','prop_oat_subl',
               'prop_entry_oat_meth','prop_entry_oat_subx','prop_entry_oat_subl',
               'uptake_cst_meth','uptake_cst_subl','uptake_cst_subx',
               'subx_to_subl',
               'prop_pwid_oat',
               'prop_subx_subl_success',
               'idu_sus::sentence_len',
               'idu_inf::sentence_len',
               'idu_meth_sus::sentence_len',
               'idu_meth_inf::sentence_len',
               'idu_subx_sus::sentence_len',
               'idu_subx_inf::sentence_len',
               'idu_subl_sus::sentence_len',
               'idu_subl_inf::sentence_len',
               'nidu_sus::sentence_len',
               'nidu_inf::sentence_len',
               'nidu_meth_sus::sentence_len',
               'nidu_meth_inf::sentence_len',
               'nidu_subx_sus::sentence_len',
               'nidu_subx_inf::sentence_len',
               'nidu_subl_sus::sentence_len',
               'nidu_subl_inf::sentence_len',
               # 'release_oud:prop_pwid_oat',
               # 'release_oud:prop_pwid_oat',
               'max_treat_rate',
               'oat_intiate',
               'sentence_len',
               'meth_uptake:flow','subx_uptake:flow','subl_uptake:flow',
               'meth_dropout:flow','subx_dropout:flow','subl_dropout:flow'
                ]
    
    mapping_function = lambda x: at.PlotData(x,outputs=outputs,pops='Prisoners')
    
    if (len(args) % 3 != 0): 
        raise Exception('Length of *args must be a multiple of 3 (par_code, year, value)')
    if (len(args) > 0):
        pars = []
        i = 0
        while i < len(args)-1:
            year_val = args[i+1]
            val_val = args[i+2]
            if not isinstance(year_val, list):
                year_val = [year_val]
            if not isinstance(val_val, list):
                val_val = [val_val]
            pars.append([[args[i]], pop[0], year_val, val_val])
            i += 3
        update_pars(P_scen, pars)
    # run_initialization(P_scen)
    parset_scen = P_scen.make_parset(name=scenario_name) 
    result_scen = P_scen.run_sampled_sims(parset_scen,n_samples=n_samples,rand_seed=rand_seed)
    
    ensemble = at.Ensemble(mapping_function=mapping_function)
    ensemble.update(result_scen)
    ensemble.name = scenario_name
    
    return ensemble, P_scen


def extract_data(ensemble, output, alpha=0.05, ub=None):
    """
    Extract data from ensemble results
    :param ensemble: atomica ensemble results object
    :param output: parameter to extract
    :return:
    :param t: time vector
    :param median_vals: median of the results (time-series)
    :param lb_vals: lower bound of the results (time-series)
    :param ub_vals: upper bound of the results (time-series)
    :param vals: values of all the results (time-series)
    """
    pop = ensemble.pops[0]
    t = ensemble.tvec
    ens_series = ensemble._get_series()[ensemble.results[0], pop, output]
    vals = np.vstack([x.vals.ravel() for x in ens_series])
    if ub is not None:
        vals = vals[:ub]
    median_vals = np.median(vals, axis=0)
    lb_vals = np.quantile(vals, alpha/2, axis=0)
    ub_vals = np.quantile(vals, 1-alpha/2, axis=0)
    
    return t, median_vals, lb_vals, ub_vals, vals


#%% OAT functions
# How lb/ub are computed for every reported difference (see bootstrap_median):
#   'bootstrap'  - bootstrap CI of the median (Monte Carlo precision only; very narrow)
#   'percentile' - 2.5th-97.5th percentile of the per-draw (paired) values, i.e. the
#                  spread across stochastic runs and sampled parameters
# Set ut.CI_METHOD = 'percentile' (e.g. in run_main.py) to switch globally.
CI_METHOD = 'bootstrap'


def bootstrap_median(diffs, n_iter=1000, method=None):
    # Median of an array of per-draw values (differences or otherwise), with lb/ub chosen by
    # `method` (default CI_METHOD). Used everywhere a reported figure needs an interval, so
    # every row in the outputs uses the same convention.
    method = method or CI_METHOD
    diffs = np.asarray(diffs)
    if method == 'percentile':
        return {
            'pe':  np.median(diffs),
            'lb':  np.percentile(diffs, 2.5),
            'ub':  np.percentile(diffs, 97.5),
            'vals': diffs
        }
    if method != 'bootstrap':
        raise ValueError("method must be 'bootstrap' or 'percentile'")
    rng = np.random.RandomState(20240430)
    boot_medians = [np.median(rng.choice(diffs, size=len(diffs), replace=True))
                    for _ in range(n_iter)]
    return {
        'pe':  np.median(diffs),
        'lb':  np.percentile(boot_medians, 2.5),
        'ub':  np.percentile(boot_medians, 97.5),
        'vals': diffs
    }


def calc_difference(results_baseline, results_scenario, n_iter=1000):
    # Calculate difference between outcomes & bootstrap
    diffs = np.array(results_scenario) - np.array(results_baseline)
    return bootstrap_median(diffs, n_iter=n_iter)


def adjust_time_range(par_values, year_start, year_end):
    # Trim vectors to desired time range
    time_values = par_values['t']
    t_range = (time_values >= year_start) & (time_values < year_end + 1)  # year_end = end of that year
    par_values_t = dict()
    
    for key in par_values.keys():
        if key == 't':
            par_values_t['t'] = par_values['t'][t_range]
            continue
        par_values_t[key] = []
        for i,rel in enumerate(par_values[key]):
            par_values_t[key].append(rel[t_range])
    return par_values_t


def sample_par(i, par, df_economic_pars):
    # Sample parameters (triangular or linear distribution). The generator is seeded from
    # (simulation index, parameter name): the same (i, par) always returns the same draw, so
    # baseline and scenario arms stay paired, but different parameters within one simulation
    # are drawn independently (seeding with i alone gave every parameter the same quantile).
    rng = np.random.RandomState((i * 1000003 + zlib.crc32(par.encode())) % (2**32))
    low = df_economic_pars.loc[par]['lower_bound']
    point = df_economic_pars.loc[par]['point_estimate']
    high = df_economic_pars.loc[par]['upper_bound']
    if (low < point) and (high > point):
        return rng.triangular(low, point, high)
    else:
        return rng.uniform(low, high) # or choose to sample within 10%?


def calculate_post_release(P, ensemble_baseline, ensemble_scenario, df_economic_pars, year_start, year_end,
                           community_oat_cost=None):
    # Calculate post-release outcomes (reincarcerations and deaths, cost of reincarceration and QALYs)
    # time-stratified decision tree
    def extract_releases(ens, prop_oud_override=None):
        t, _, _, _, idu_sus = extract_data(ens, 'idu_sus::sentence_len') # not oat but at risk
        _, _, _, _, idu_inf = extract_data(ens, 'idu_inf::sentence_len') # not oat but at risk
        _, _, _, _, nidu_sus = extract_data(ens, 'nidu_sus::sentence_len') # not oat but oud
        _, _, _, _, nidu_inf = extract_data(ens, 'nidu_inf::sentence_len') # not oat but oud
        _, _, _, _, idu_meth_sus = extract_data(ens, 'idu_meth_sus::sentence_len') # oat_meth
        _, _, _, _, idu_meth_inf = extract_data(ens, 'idu_meth_inf::sentence_len') # oat_meth
        _, _, _, _, idu_subx_sus = extract_data(ens, 'idu_subx_sus::sentence_len') # oat_subx
        _, _, _, _, idu_subx_inf = extract_data(ens, 'idu_subx_inf::sentence_len') # oat_subx
        _, _, _, _, idu_subl_sus = extract_data(ens, 'idu_subl_sus::sentence_len') # oat_subl
        _, _, _, _, idu_subl_inf = extract_data(ens, 'idu_subl_inf::sentence_len') # oat_subl
        _, _, _, _, nidu_meth_sus = extract_data(ens, 'nidu_meth_sus::sentence_len') # oat_meth
        _, _, _, _, nidu_meth_inf = extract_data(ens, 'nidu_meth_inf::sentence_len') # oat_meth
        _, _, _, _, nidu_subx_sus = extract_data(ens, 'nidu_subx_sus::sentence_len') # oat_subx
        _, _, _, _, nidu_subx_inf = extract_data(ens, 'nidu_subx_inf::sentence_len') # oat_subx
        _, _, _, _, nidu_subl_sus = extract_data(ens, 'nidu_subl_sus::sentence_len') # oat_subl
        _, _, _, _, nidu_subl_inf = extract_data(ens, 'nidu_subl_inf::sentence_len') # oat_subl
        # Non-injecting, off-OAT people with OUD, by conservation. Policy changes who is on
        # treatment, not who has OUD, so total OUD among releases is the same in both arms:
        # prop_oud_entry (share of CSC entrants with OUD, includes people who inject) x all
        # releases. People on OAT come straight from the model's pools, as do people injecting
        # off OAT, of whom prop_idu_with_oud have OUD (the rest inject without OUD, e.g.
        # stimulants, but are still at risk in the tree below). The remainder is the
        # non-injecting, off-OAT OUD group - OAT drop-outs in prison land there
        # automatically. Capped at the non-injecting pool.
        n_draws = idu_sus.shape[0]
        prop_oud = np.array([sample_par(i, 'prop_oud_entry', df_economic_pars) for i in range(n_draws)])[:, None]
        if prop_oud_override is not None:
            prop_oud = prop_oud_override[:n_draws]
        prop_idu_oud = np.array([sample_par(i, 'prop_idu_with_oud', df_economic_pars) for i in range(n_draws)])[:, None]
        oat_all = (idu_meth_sus + idu_meth_inf + nidu_meth_sus + nidu_meth_inf
                   + idu_subx_sus + idu_subx_inf + nidu_subx_sus + nidu_subx_inf
                   + idu_subl_sus + idu_subl_inf + nidu_subl_sus + nidu_subl_inf)
        nidu_off = nidu_sus + nidu_inf
        all_releases = oat_all + idu_sus + idu_inf + nidu_off
        oud_total = prop_oud*all_releases
        # smallest OUD share consistent with people already counted (on OAT + injecting off OAT
        # with OUD); below it the derived non-injecting group would be clipped at zero
        with np.errstate(divide='ignore', invalid='ignore'):
            oud_needed = np.where(all_releases > 0, (oat_all + prop_idu_oud*(idu_sus + idu_inf))/all_releases, 0.0)
        no_oat_oud = np.clip(oud_total - oat_all - prop_idu_oud*(idu_sus + idu_inf), 0, nidu_off)
        with np.errstate(divide='ignore', invalid='ignore'):
            frac_nidu_oud = np.where(nidu_off > 0, no_oat_oud/nidu_off, 0.0)


        return {
            't': t,
            'oud_needed': oud_needed,
            'no_oat_oud': no_oat_oud,
            'no_oat_idu': idu_sus + idu_inf,
            'oat_meth': idu_meth_sus + idu_meth_inf + nidu_meth_sus + nidu_meth_inf,
            'oat_subx': idu_subx_sus + idu_subx_inf + nidu_subx_sus + nidu_subx_inf,
            'oat_subl': idu_subl_sus + idu_subl_inf + nidu_subl_sus + nidu_subl_inf,
            # HCV-infected component of each group above, used to split post-release
            # QALYs by HCV status at release (mortality/dropout/reincarceration don't
            # depend on HCV status, so this fraction carries through the decision tree)
            'no_oat_oud_inf': nidu_inf*frac_nidu_oud,
            'no_oat_idu_inf': idu_inf,
            'oat_meth_inf': idu_meth_inf + nidu_meth_inf,
            'oat_subx_inf': idu_subx_inf + nidu_subx_inf,
            'oat_subl_inf': idu_subl_inf + nidu_subl_inf,
            # Injecting (IDU-compartment) component of each group, and its HCV-infected part.
            # HCV acquisition and IRIs only happen among people who inject, so the tree
            # applies them to this component only. The non-injecting OUD group has none;
            # OAT pools mix idu_* (injecting) and nidu_* (not) releases.
            'no_oat_oud_inj': np.zeros_like(no_oat_oud),
            'no_oat_oud_inf_inj': np.zeros_like(no_oat_oud),
            'no_oat_idu_inj': idu_sus + idu_inf,
            'no_oat_idu_inf_inj': idu_inf,
            'oat_meth_inj': idu_meth_sus + idu_meth_inf,
            'oat_meth_inf_inj': idu_meth_inf,
            'oat_subx_inj': idu_subx_sus + idu_subx_inf,
            'oat_subx_inf_inj': idu_subx_inf,
            'oat_subl_inj': idu_subl_sus + idu_subl_inf,
            'oat_subl_inf_inj': idu_subl_inf
        }
    
    # Pass 1: find, per draw, the smallest OUD share that is feasible in BOTH arms. If the sampled
    # share is below it, the non-injecting OUD group is clipped to zero in the arm with more
    # people on OAT but not in the other, shrinking the tree in one arm only and producing
    # spurious negative reincarceration/death differences. Use the same effective share in both.
    pass1_b = adjust_time_range(extract_releases(ensemble_baseline), year_start, year_end)
    pass1_s = adjust_time_range(extract_releases(ensemble_scenario), year_start, year_end)
    n_eff = min(len(pass1_b['oud_needed']), len(pass1_s['oud_needed']))
    needed = np.array([max(np.max(pass1_b['oud_needed'][i]), np.max(pass1_s['oud_needed'][i])) for i in range(n_eff)])
    prop_oud_sampled = np.array([sample_par(i, 'prop_oud_entry', df_economic_pars) for i in range(n_eff)])
    prop_oud_effective = np.maximum(prop_oud_sampled, needed)[:, None]

    releases_baseline = extract_releases(ensemble_baseline, prop_oud_effective)
    releases_scenario = extract_releases(ensemble_scenario, prop_oud_effective)
    
    releases_baseline = adjust_time_range(releases_baseline, year_start, year_end)
    releases_scenario = adjust_time_range(releases_scenario, year_start, year_end)
    # min, not baseline's length alone: baseline and scenario ensembles can have different
    # sample counts (e.g. the in-prison retention sweep compares a fresh, deliberately
    # smaller per-gridpoint trial ensemble against the full-size main-analysis ensemble)
    n_sample = min(len(releases_baseline['no_oat_idu']), len(releases_scenario['no_oat_idu']))
    years_dt = releases_baseline['t']
    years_annual = np.arange(year_start, year_end+1)
    dt = releases_baseline['t'][1]-releases_baseline['t'][0]

    # Loop over each simulation and calculate outcomes.
    deaths_baseline = []
    deaths_scenario = []
    deaths_baseline_sus = []
    deaths_scenario_sus = []
    deaths_baseline_inf = []
    deaths_scenario_inf = []
    reincarcerated_baseline_oat = []
    reincarcerated_scenario_oat = []
    reincarcerated_baseline_off = []
    reincarcerated_scenario_off = []
    reincarceration_baseline_total = []
    reincarceration_scenario_total = []
    persontime_baseline_sus = []
    persontime_baseline_inf = []
    persontime_scenario_sus = []
    persontime_scenario_inf = []
    iri_events_baseline = []
    iri_events_scenario = []
    sero_events_baseline = []
    sero_events_scenario = []
    treat_events_baseline = []
    treat_events_scenario = []
    uptake_events_baseline = []
    uptake_events_scenario = []
    oat_py_baseline = []   # undiscounted post-release person-years on OAT (all modalities)
    oat_py_scenario = []
    cost_reinc_vals = []
    # Discounted (to year_start, by release year) versions of the event / person-time totals.
    # Used for costs and QALYs only; reported event counts stay undiscounted.
    discount_rate = df_economic_pars.loc['discount']['point_estimate']
    DISC_NAMES = ('death_sus_b', 'death_inf_b', 'death_sus_s', 'death_inf_s',
                  'pt_sus_b', 'pt_inf_b', 'pt_sus_s', 'pt_inf_s',
                  'iri_b', 'iri_s', 'treat_b', 'treat_s',
                  'reinc_oat_b', 'reinc_oat_s', 'reinc_off_b', 'reinc_off_s',
                  'ptoat_meth_b', 'ptoat_subx_b', 'ptoat_subl_b',
                  'ptoat_meth_s', 'ptoat_subx_s', 'ptoat_subl_s')
    disc_vals = {k: [] for k in DISC_NAMES}
    oat_keys = ['oat_meth','oat_subx','oat_subl']
    off_keys = ['no_oat_oud','no_oat_idu']
    HCV_STATES = ('sus', 'inf')
    POOL_STATES = ('sus', 'inf', 'sus_inj', 'inf_inj')  # '_inj' = injecting subset of each

    time_to_inc_oat_in_years = df_economic_pars.loc['time_to_reinc_on']['point_estimate']/365
    time_to_inc_off_in_years = df_economic_pars.loc['time_to_reinc_off']['point_estimate']/365
    reinc_oat = df_economic_pars.loc['p_reincar_on_oat']['point_estimate']
    reinc_off = df_economic_pars.loc['p_reincar_off_oat']['point_estimate']

    # Time windows (in years). 1 month used for both retention checkpoint and
    # elevated-mortality window (aligning 4-week Santo et al. period with Ivasiy 1-month
    # retention checkpoint — 2-day difference is clinically negligible).
    W1_END   = 1/12          # 1 month
    W2_END   = 6/12          # 6 months
    W3_END   = 12/12         # 12 months
    W1_MID   = W1_END / 2                    # 0.5 mo — dropout applied here
    W2_MID   = (W1_END + W2_END) / 2        # 3.5 mo — dropout applied here
    W3_MID   = (W2_END + W3_END) / 2        # 9 mo   — dropout applied here
    W1_HALF  = W1_MID                        # half-width of W1
    W2_HALF  = W2_MID - W1_END              # half-width of W2 (2.5 mo)

    # Modality key mapping for retention parameters
    oat_to_mod = {'oat_meth': 'meth', 'oat_subx': 'subx', 'oat_subl': 'subl'}

    def new_pool(releases, par, i, t_range_year):
        total = releases[par][i][t_range_year].sum() * dt
        inf = releases[f'{par}_inf'][i][t_range_year].sum() * dt
        inj = releases[f'{par}_inj'][i][t_range_year].sum() * dt
        inf_inj = releases[f'{par}_inf_inj'][i][t_range_year].sum() * dt
        # 'sus'/'inf' = whole pool by HCV status; 'sus_inj'/'inf_inj' = the injecting part of
        # each (a subset), carried through every move so injecting status travels with people
        return {'sus': max(total - inf, 0.0), 'inf': min(inf, total),
                'sus_inj': max(inj - inf_inj, 0.0), 'inf_inj': min(inf_inj, inj)}

    def advance_phase(pools, keys, mort_rate, duration, hcv_rate_per_year, iri_rate_per_year, treat_rate_per_year,
                       death_acc, persontime_acc, iri_acc, sero_acc, treat_acc):
        """Advance every pool in `keys` through one phase of the decision tree: seroconvert,
        treat (cure a fraction of the infected pool - not disaggregated by OAT status, since
        the source paper stratified treatment initiation by OAT status and found no
        significant effect, i.e. the CI crossed 1), accumulate person-time, IRI events and
        seroconversion events at the resulting composition, then apply mortality. All
        processes use the same per-phase step, a discretisation consistent with how deaths
        already use survivors-at-phase-start * rate * duration.

        HCV seroconversion/treatment, IRI incidence, and OAT uptake (see apply_uptake below)
        are all modelled here because they drive realistic clinical outcomes (mortality
        risk, HCV-status utility) or are epidemiologically implausible to zero out post-
        release. Their COSTS (HCV treatment, IRI management, OAT delivery) are deliberately
        NOT tracked in the community, though: this manuscript is scoped to CSC's own costs
        (in-prison HCV/IRI/OAT delivery + reincarceration), and community treatment costs
        are a provincial health-system cost, not CSC's - modelling them properly would need
        its own costing study with its own data and limitations, out of scope here."""
        p_convert = 1 - np.exp(-hcv_rate_per_year * duration)
        p_treat = 1 - np.exp(-treat_rate_per_year * duration)
        for par in keys:
            pool = pools[par]
            # HCV acquisition only among people who inject
            newly_inf = pool['sus_inj'] * p_convert
            pool['sus'] -= newly_inf
            pool['sus_inj'] -= newly_inf
            pool['inf'] += newly_inf
            pool['inf_inj'] += newly_inf
            sero_acc['events'] += newly_inf

            # Treatment cures anyone infected; the injecting part is cured pro rata
            newly_treated = pool['inf'] * p_treat
            treated_inj = newly_treated * (pool['inf_inj'] / pool['inf']) if pool['inf'] > 0 else 0.0
            pool['inf'] -= newly_treated
            pool['inf_inj'] -= treated_inj
            pool['sus'] += newly_treated
            pool['sus_inj'] += treated_inj
            treat_acc['events'] += newly_treated

            persontime_acc['sus'] += pool['sus'] * duration
            persontime_acc['inf'] += pool['inf'] * duration
            if par in persontime_acc:  # per-modality OAT person-time (OAT pools only)
                persontime_acc[par] += (pool['sus'] + pool['inf']) * duration
            # IRIs only among people who inject
            iri_acc['events'] += (pool['sus_inj'] + pool['inf_inj']) * iri_rate_per_year * duration

            for s in HCV_STATES:
                d = pool[s] * mort_rate * duration
                pool[s] -= d
                pool[s + '_inj'] *= (1 - mort_rate * duration)
                death_acc[s] += d

    def apply_dropout(pools_from, keys, p_drop, pool_to):
        for par in keys:
            pool = pools_from[par]
            for s in POOL_STATES:
                moved = pool[s] * p_drop[par]
                pool[s] -= moved
                pool_to[s] += moved

    def apply_reincarceration(pools, keys, reinc_rate, reinc_acc):
        for par in keys:
            pool = pools[par]
            for s in POOL_STATES:
                r = pool[s] * reinc_rate
                pool[s] -= r
                if s in HCV_STATES:  # '_inj' keys are subsets - don't double count events
                    reinc_acc['events'] += r

    def apply_uptake(pools_off, off_group_keys, pools_oat, uptake_rate_per_year, duration,
                     meth_share, subx_share, subl_share, uptake_acc):
        """Move a fraction of the off-OAT pools into OAT (split oat_meth/oat_subx/oat_subl
        by modality shares from Ontario dispensing data) at each phase."""
        p_uptake = 1 - np.exp(-uptake_rate_per_year * duration)
        for par in off_group_keys:
            pool = pools_off[par]
            for s in POOL_STATES:
                moved = pool[s] * p_uptake
                pool[s] -= moved
                pools_oat['oat_meth'][s] += moved * meth_share
                pools_oat['oat_subx'][s] += moved * subx_share
                pools_oat['oat_subl'][s] += moved * subl_share
                if s in HCV_STATES:
                    uptake_acc['events'] += moved

    for i in range(n_sample):
        mort_4wk_oat_sampled = sample_par(i, 'mortality_oat_4wk', df_economic_pars)
        mort_4wk_off_sampled = sample_par(i, 'mortality_off_oat_4wk', df_economic_pars)
        mort_rem_oat_sampled = sample_par(i, 'mortality_oat_rem', df_economic_pars)
        mort_rem_off_sampled = sample_par(i, 'mortality_off_oat_rem', df_economic_pars)

        # Cumulative retention at each checkpoint (modality-specific)
        r1  = {par: sample_par(i, f'retain_{oat_to_mod[par]}_1mo',  df_economic_pars) for par in oat_keys}
        r6  = {par: sample_par(i, f'retain_{oat_to_mod[par]}_6mo',  df_economic_pars) for par in oat_keys}
        r12 = {par: sample_par(i, f'retain_{oat_to_mod[par]}_12mo', df_economic_pars) for par in oat_keys}

        # Conditional P(discontinue | retained at window start)
        p_drop_w1 = {par: 1 - r1[par]            for par in oat_keys}
        p_drop_w2 = {par: 1 - r6[par]  / r1[par] for par in oat_keys}
        p_drop_w3 = {par: 1 - r12[par] / r6[par] for par in oat_keys}

        # Post-release HCV incidence (per year), by OAT status (Drug Policy 2026, Table 2)
        hcv_rate_oat = sample_par(i, 'hcv_inci_oat', df_economic_pars) / 100.0
        hcv_rate_off = sample_par(i, 'hcv_inci_off', df_economic_pars) / 100.0
        # Post-release IRI incidence (per year), by OAT status (reused in-prison databook parameter)
        iri_rate_off_year = sample_par(i, 'iri_inci_postrelease', df_economic_pars) / 100.0
        iri_rate_oat_year = iri_rate_off_year * sample_par(i, 'iri_efficacy_postrelease', df_economic_pars)
        # Post-release HCV treatment initiation rate (per year) - one rate, not split by OAT
        # status (see hcv_treat_rate_postrelease's own comment for the derivation/caveats)
        hcv_treat_rate = sample_par(i, 'hcv_treat_rate_postrelease', df_economic_pars) / 100.0
        # Post-release OAT uptake rate (per year) and modality split on initiation
        # (Ontario Opioid Indicator Tool / Garg et al. 2026, new-user dispensing data)
        oat_uptake_rate = sample_par(i, 'oat_uptake_rate_postrelease', df_economic_pars) / 100.0
        oat_uptake_meth_share = sample_par(i, 'oat_uptake_meth_share', df_economic_pars)
        oat_uptake_subx_share = sample_par(i, 'oat_uptake_subx_share', df_economic_pars)
        oat_uptake_subl_share = sample_par(i, 'oat_uptake_subl_share', df_economic_pars)

        death_acc_baseline = {'sus': 0.0, 'inf': 0.0}
        death_acc_scenario = {'sus': 0.0, 'inf': 0.0}
        persontime_acc_baseline = {'sus': 0.0, 'inf': 0.0, **{k: 0.0 for k in oat_keys}}
        persontime_acc_scenario = {'sus': 0.0, 'inf': 0.0, **{k: 0.0 for k in oat_keys}}
        iri_acc_baseline = {'events': 0.0}
        iri_acc_scenario = {'events': 0.0}
        sero_acc_baseline = {'events': 0.0}
        sero_acc_scenario = {'events': 0.0}
        treat_acc_baseline = {'events': 0.0}
        treat_acc_scenario = {'events': 0.0}
        uptake_acc_baseline = {'events': 0.0}
        uptake_acc_scenario = {'events': 0.0}
        reinc_acc_baseline_oat = {'events': 0.0}
        reinc_acc_scenario_oat = {'events': 0.0}
        reinc_acc_baseline_off = {'events': 0.0}
        reinc_acc_scenario_off = {'events': 0.0}

        tracked = [('death_sus_b', death_acc_baseline, 'sus'), ('death_inf_b', death_acc_baseline, 'inf'),
                   ('death_sus_s', death_acc_scenario, 'sus'), ('death_inf_s', death_acc_scenario, 'inf'),
                   ('pt_sus_b', persontime_acc_baseline, 'sus'), ('pt_inf_b', persontime_acc_baseline, 'inf'),
                   ('pt_sus_s', persontime_acc_scenario, 'sus'), ('pt_inf_s', persontime_acc_scenario, 'inf'),
                   ('iri_b', iri_acc_baseline, 'events'), ('iri_s', iri_acc_scenario, 'events'),
                   ('treat_b', treat_acc_baseline, 'events'), ('treat_s', treat_acc_scenario, 'events'),
                   ('reinc_oat_b', reinc_acc_baseline_oat, 'events'), ('reinc_oat_s', reinc_acc_scenario_oat, 'events'),
                   ('reinc_off_b', reinc_acc_baseline_off, 'events'), ('reinc_off_s', reinc_acc_scenario_off, 'events')]
        for _k in ('meth', 'subx', 'subl'):
            tracked.append((f'ptoat_{_k}_b', persontime_acc_baseline, f'oat_{_k}'))
            tracked.append((f'ptoat_{_k}_s', persontime_acc_scenario, f'oat_{_k}'))
        disc_acc = {name: 0.0 for name in DISC_NAMES}
        prev_acc = {name: 0.0 for name in DISC_NAMES}

        for year in years_annual:
            t_range_year = (years_dt >= year) & (years_dt < year+1)

            pools_oat_baseline = {par: new_pool(releases_baseline, par, i, t_range_year) for par in oat_keys}
            pools_off_baseline = {par: new_pool(releases_baseline, par, i, t_range_year) for par in off_keys}
            pools_oat_scenario = {par: new_pool(releases_scenario, par, i, t_range_year) for par in oat_keys}
            pools_off_scenario = {par: new_pool(releases_scenario, par, i, t_range_year) for par in off_keys}

            def do_phase(duration):
                advance_phase(pools_oat_baseline, oat_keys, mort_4wk_oat_sampled, duration,
                               hcv_rate_oat, iri_rate_oat_year, hcv_treat_rate, death_acc_baseline, persontime_acc_baseline, iri_acc_baseline, sero_acc_baseline, treat_acc_baseline)
                advance_phase(pools_off_baseline, off_keys, mort_4wk_off_sampled, duration,
                               hcv_rate_off, iri_rate_off_year, hcv_treat_rate, death_acc_baseline, persontime_acc_baseline, iri_acc_baseline, sero_acc_baseline, treat_acc_baseline)
                apply_uptake(pools_off_baseline, off_keys, pools_oat_baseline, oat_uptake_rate, duration, oat_uptake_meth_share, oat_uptake_subx_share, oat_uptake_subl_share, uptake_acc_baseline)
                advance_phase(pools_oat_scenario, oat_keys, mort_4wk_oat_sampled, duration,
                               hcv_rate_oat, iri_rate_oat_year, hcv_treat_rate, death_acc_scenario, persontime_acc_scenario, iri_acc_scenario, sero_acc_scenario, treat_acc_scenario)
                advance_phase(pools_off_scenario, off_keys, mort_4wk_off_sampled, duration,
                               hcv_rate_off, iri_rate_off_year, hcv_treat_rate, death_acc_scenario, persontime_acc_scenario, iri_acc_scenario, sero_acc_scenario, treat_acc_scenario)
                apply_uptake(pools_off_scenario, off_keys, pools_oat_scenario, oat_uptake_rate, duration, oat_uptake_meth_share, oat_uptake_subx_share, oat_uptake_subl_share, uptake_acc_scenario)

            def do_phase_remission(duration, oat_only=False):
                advance_phase(pools_oat_baseline, oat_keys, mort_rem_oat_sampled, duration,
                               hcv_rate_oat, iri_rate_oat_year, hcv_treat_rate, death_acc_baseline, persontime_acc_baseline, iri_acc_baseline, sero_acc_baseline, treat_acc_baseline)
                advance_phase(pools_oat_scenario, oat_keys, mort_rem_oat_sampled, duration,
                               hcv_rate_oat, iri_rate_oat_year, hcv_treat_rate, death_acc_scenario, persontime_acc_scenario, iri_acc_scenario, sero_acc_scenario, treat_acc_scenario)
                if not oat_only:
                    advance_phase(pools_off_baseline, off_keys, mort_rem_off_sampled, duration,
                                   hcv_rate_off, iri_rate_off_year, hcv_treat_rate, death_acc_baseline, persontime_acc_baseline, iri_acc_baseline, sero_acc_baseline, treat_acc_baseline)
                    apply_uptake(pools_off_baseline, off_keys, pools_oat_baseline, oat_uptake_rate, duration, oat_uptake_meth_share, oat_uptake_subx_share, oat_uptake_subl_share, uptake_acc_baseline)
                    advance_phase(pools_off_scenario, off_keys, mort_rem_off_sampled, duration,
                                   hcv_rate_off, iri_rate_off_year, hcv_treat_rate, death_acc_scenario, persontime_acc_scenario, iri_acc_scenario, sero_acc_scenario, treat_acc_scenario)
                    apply_uptake(pools_off_scenario, off_keys, pools_oat_scenario, oat_uptake_rate, duration, oat_uptake_meth_share, oat_uptake_subx_share, oat_uptake_subl_share, uptake_acc_scenario)

            # ── Window 1 Phase 1a: 0 → W1 midpoint (0.5 mo) ─────────────────────
            do_phase(W1_HALF)

            # ── W1 Dropout at 0.5 mo midpoint ────────────────────────────────────
            apply_dropout(pools_oat_baseline, oat_keys, p_drop_w1, pools_off_baseline['no_oat_idu'])
            apply_dropout(pools_oat_scenario, oat_keys, p_drop_w1, pools_off_scenario['no_oat_idu'])

            # ── Window 1 Phase 1b: W1 midpoint → 1 mo ────────────────────────────
            do_phase(W1_HALF)

            # ── Window 2 Phase 2a: 1 mo → W2 midpoint (3.5 mo) ──────────────────
            do_phase_remission(W2_HALF)

            # ── W2 Dropout at 3.5 mo midpoint (conditional on W1 retention) ──────
            apply_dropout(pools_oat_baseline, oat_keys, p_drop_w2, pools_off_baseline['no_oat_idu'])
            apply_dropout(pools_oat_scenario, oat_keys, p_drop_w2, pools_off_scenario['no_oat_idu'])

            # ── Window 2 Phase 2b: W2 midpoint → 6 mo ────────────────────────────
            do_phase_remission(W2_HALF)

            # ── Window 3 Phase 3a: 6 mo → OAT reincarceration time (~6.4 mo) ─────
            w3_to_oat_reinc = time_to_inc_oat_in_years - W2_END
            do_phase_remission(w3_to_oat_reinc)

            # ── OAT Reincarceration ───────────────────────────────────────────────
            apply_reincarceration(pools_oat_baseline, oat_keys, reinc_oat, reinc_acc_baseline_oat)
            apply_reincarceration(pools_oat_scenario, oat_keys, reinc_oat, reinc_acc_scenario_oat)

            # ── Window 3 Phase 3b: OAT reinc time → off-OAT reinc time (~7.7 mo) ─
            # (non-reincarc'd OAT survivors only — off_keys has no phase here, matching
            # the original decision tree, which leaves off_keys' mortality gap between
            # w3_to_oat_reinc and their own reincarceration checkpoint unmodelled)
            w3_oat_to_off_reinc = time_to_inc_off_in_years - time_to_inc_oat_in_years
            do_phase_remission(w3_oat_to_off_reinc, oat_only=True)

            # ── off-OAT Reincarceration ───────────────────────────────────────────
            apply_reincarceration(pools_off_baseline, off_keys, reinc_off, reinc_acc_baseline_off)
            apply_reincarceration(pools_off_scenario, off_keys, reinc_off, reinc_acc_scenario_off)

            # ── Window 3 Phase 3c: off-OAT reinc time → W3 midpoint (9 mo) ───────
            w3_to_mid = W3_MID - time_to_inc_off_in_years
            do_phase_remission(w3_to_mid)

            # ── W3 Dropout at 9 mo midpoint (non-reincarc'd OAT survivors only) ──
            apply_dropout(pools_oat_baseline, oat_keys, p_drop_w3, pools_off_baseline['no_oat_idu'])
            apply_dropout(pools_oat_scenario, oat_keys, p_drop_w3, pools_off_scenario['no_oat_idu'])

            # ── Window 3 Phase 3d: W3 midpoint → 12 mo ───────────────────────────
            W3_FINAL = W3_END - W3_MID
            do_phase_remission(W3_FINAL)

            # Discount this release-year's increments back to year_start
            year_factor = 1 / (1 + discount_rate) ** (year - year_start)
            for name, acc, key in tracked:
                disc_acc[name] += (acc[key] - prev_acc[name]) * year_factor
                prev_acc[name] = acc[key]

        for name in DISC_NAMES:
            disc_vals[name].append(disc_acc[name])
        deaths_baseline.append(death_acc_baseline['sus'] + death_acc_baseline['inf'])
        deaths_scenario.append(death_acc_scenario['sus'] + death_acc_scenario['inf'])
        deaths_baseline_sus.append(death_acc_baseline['sus'])
        deaths_baseline_inf.append(death_acc_baseline['inf'])
        deaths_scenario_sus.append(death_acc_scenario['sus'])
        deaths_scenario_inf.append(death_acc_scenario['inf'])
        persontime_baseline_sus.append(persontime_acc_baseline['sus'])
        persontime_baseline_inf.append(persontime_acc_baseline['inf'])
        persontime_scenario_sus.append(persontime_acc_scenario['sus'])
        persontime_scenario_inf.append(persontime_acc_scenario['inf'])
        oat_py_baseline.append(sum(persontime_acc_baseline[k] for k in oat_keys))
        oat_py_scenario.append(sum(persontime_acc_scenario[k] for k in oat_keys))
        iri_events_baseline.append(iri_acc_baseline['events'])
        iri_events_scenario.append(iri_acc_scenario['events'])
        sero_events_baseline.append(sero_acc_baseline['events'])
        sero_events_scenario.append(sero_acc_scenario['events'])
        treat_events_baseline.append(treat_acc_baseline['events'])
        treat_events_scenario.append(treat_acc_scenario['events'])
        uptake_events_baseline.append(uptake_acc_baseline['events'])
        uptake_events_scenario.append(uptake_acc_scenario['events'])
        reincarcerated_baseline_oat.append(reinc_acc_baseline_oat['events'])
        reincarcerated_scenario_oat.append(reinc_acc_scenario_oat['events'])
        reincarcerated_baseline_off.append(reinc_acc_baseline_off['events'])
        reincarcerated_scenario_off.append(reinc_acc_scenario_off['events'])
        reincarceration_baseline_total.append(reinc_acc_baseline_oat['events']+reinc_acc_baseline_off['events'])
        reincarceration_scenario_total.append(reinc_acc_scenario_oat['events']+reinc_acc_scenario_off['events'])

    # Deaths averted (exact HCV-status split - each death is drawn from whichever pool,
    # sus or inf, it actually occurred in, tracked through the decision tree above)
    diff_deaths = calc_difference(deaths_baseline,deaths_scenario)
    # QALY inputs: deaths and person-time discounted back to year_start by release year
    diff_deaths_sus = calc_difference(disc_vals['death_sus_b'], disc_vals['death_sus_s'])
    diff_deaths_inf = calc_difference(disc_vals['death_inf_b'], disc_vals['death_inf_s'])

    # Reincarcerations averted
    diff_reinc = calc_difference(reincarceration_baseline_total,reincarceration_scenario_total)

    # Post-release survivor person-time (12-month window only), IRI events and HCV
    # seroconversion events, by HCV status - the latter two combined with their in-prison
    # counterparts by the caller (calculate_all_outcomes), same as the IRI cost below.
    diff_persontime_inf = calc_difference(disc_vals['pt_inf_b'], disc_vals['pt_inf_s'])
    diff_iri_events_postrelease = calc_difference(iri_events_baseline, iri_events_scenario)
    diff_sero_events_postrelease = calc_difference(sero_events_baseline, sero_events_scenario)

    def summarise(vals):
        return {'pe': np.median(vals), 'lb': np.percentile(vals, 2.5), 'ub': np.percentile(vals, 97.5), 'vals': vals}
    postrelease_epi = {
        'iri_baseline': summarise(iri_events_baseline), 'iri_scenario': summarise(iri_events_scenario),
        'sero_baseline': summarise(sero_events_baseline), 'sero_scenario': summarise(sero_events_scenario),
    }

    ## QALYs
    discount = df_economic_pars.loc['discount']['point_estimate']
    life_expectancy = df_economic_pars.loc['life_expectancy_oud']['point_estimate']
    discounted_life_expectancy = sum(1 / (1 + discount)**t for t in range(int(life_expectancy)))

    # In-prison HCV morbidity QALYs: atomica already gives exact pop_sus/pop_inf stocks at
    # every timestep, so this is a straightforward discounted person-time x utility sum -
    # no approximation needed here, unlike the post-release side above.
    def inprison_hcv_qalys(ens):
        t, _, _, _, pop_sus = extract_data(ens, 'pop_sus')
        _, _, _, _, pop_inf = extract_data(ens, 'pop_inf')
        t_range = (t >= year_start) & (t < year_end + 1)
        t_trim = t[t_range]
        dt_inprison = t_trim[1] - t_trim[0]
        discount_factors = 1 / (1 + discount) ** (t_trim - year_start)
        qalys = np.zeros(n_sample)
        for i in range(n_sample):
            u_neg = sample_par(i, 'utility_hcv_neg', df_economic_pars)
            u_pos = sample_par(i, 'utility_hcv_pos', df_economic_pars)
            # HCV-attributable loss only (negative): infected person-time x (u_pos - u_neg). Counting all person-time x
            # utility would make the result depend on the size of the prison population, not on HCV.
            qalys[i] = np.sum(pop_inf[i][t_range]*(u_pos - u_neg) * discount_factors) * dt_inprison
        return qalys

    inprison_qalys_baseline = inprison_hcv_qalys(ensemble_baseline)
    inprison_qalys_scenario = inprison_hcv_qalys(ensemble_scenario)
    diff_inprison_qalys = inprison_qalys_scenario - inprison_qalys_baseline

    # QALYs = (1) lost to premature deaths (NOTE this is based on deaths, not survivors!
    # Hence the negative sign), split by HCV status at release, each death assumed to buy
    # back a full discounted life expectancy at that status's utility (Saeed et al. 2025);
    # (2) post-release survivor morbidity, bounded to the 12-month decision-tree window
    # (not extrapolated - see note above persontime_baseline_sus); (3) in-prison morbidity,
    # over the full modelled horizon.
    qalys_diff_vals = []
    death_qalys_vals = []
    morbidity_postrelease_qalys_vals = []
    # Per-arm (absolute) components, for the CCA tables. Each arm's death QALYs are the QALYs
    # lost to its deaths (negative); the scenario-minus-baseline of each component equals the
    # corresponding difference above by construction (everything is linear in the counts).
    abs_qalys = {k: {'baseline': [], 'scenario': []} for k in ('death', 'morb_post', 'total')}
    for i in range(n_sample):
        u_neg = sample_par(i, 'utility_hcv_neg', df_economic_pars)
        u_pos = sample_par(i, 'utility_hcv_pos', df_economic_pars)
        for arm, dsus, dinf, psus, pinf, inprison in (
                ('baseline', disc_vals['death_sus_b'], disc_vals['death_inf_b'], disc_vals['pt_sus_b'], disc_vals['pt_inf_b'], inprison_qalys_baseline),
                ('scenario', disc_vals['death_sus_s'], disc_vals['death_inf_s'], disc_vals['pt_sus_s'], disc_vals['pt_inf_s'], inprison_qalys_scenario)):
            d_q = -(dsus[i]*u_neg + dinf[i]*u_pos) * discounted_life_expectancy
            m_q = pinf[i]*(u_pos - u_neg)   # HCV-attributable loss only (negative)
            abs_qalys['death'][arm].append(d_q)
            abs_qalys['morb_post'][arm].append(m_q)
            abs_qalys['total'][arm].append(d_q + m_q + inprison[i])
        death_qalys = -(diff_deaths_sus['vals'][i]*u_neg + diff_deaths_inf['vals'][i]*u_pos) * discounted_life_expectancy
        morbidity_qalys_postrelease = diff_persontime_inf['vals'][i]*(u_pos - u_neg)
        death_qalys_vals.append(death_qalys)
        morbidity_postrelease_qalys_vals.append(morbidity_qalys_postrelease)
        qalys_diff_vals.append(death_qalys + morbidity_qalys_postrelease + diff_inprison_qalys[i])

    diff_qalys = bootstrap_median(qalys_diff_vals)
    # QALY breakdown by component, for diagnosing which mechanism drives the total
    # (e.g. whether post-release morbidity offsets the deaths-averted benefit).
    diff_qalys_death = bootstrap_median(death_qalys_vals)
    diff_qalys_morbidity_postrelease = bootstrap_median(morbidity_postrelease_qalys_vals)
    diff_qalys_morbidity_inprison = bootstrap_median(diff_inprison_qalys)

    # Post-release IRI and HCV treatment cost (12-month window). cost_iri_management is a
    # flat per-occurrence cost applied equally in prison and the community; HCV treatment
    # uses the community-specific cost_hcv_management_com (in-prison uses cost_hcv_management);
    # both are drug + test + RN time, built from the cells in input_pars.xlsx.
    # Always computed and returned (cheap), but NOT included in this manuscript's headline
    # CSC-scoped totals by default (see calculate_all_outcomes's include_community_costs
    # switch) - community treatment costs are a provincial health-system cost, not CSC's,
    # and modelling them properly would need their own costing study with its own data/
    # limitations. Post-release OAT delivery cost is only computed when community_oat_cost
    # is set (medication-only or in-prison formulas; no community dispensing cost data);
    # the events/person-time driving mortality/QALY outcomes (IRI, HCV
    # seroconversion/treatment, OAT uptake) are still tracked regardless of this switch.
    cost_iri_postrelease_baseline = [disc_vals['iri_b'][i]*sample_par(i, 'cost_iri_management', df_economic_pars) for i in range(n_sample)]
    cost_iri_postrelease_scenario = [disc_vals['iri_s'][i]*sample_par(i, 'cost_iri_management', df_economic_pars) for i in range(n_sample)]
    diff_costs_iri_postrelease = calc_difference(cost_iri_postrelease_baseline, cost_iri_postrelease_scenario)

    cost_hcv_postrelease_baseline = [disc_vals['treat_b'][i]*sample_par(i, 'cost_hcv_management_com', df_economic_pars) for i in range(n_sample)]
    cost_hcv_postrelease_scenario = [disc_vals['treat_s'][i]*sample_par(i, 'cost_hcv_management_com', df_economic_pars) for i in range(n_sample)]
    diff_costs_hcv_postrelease = calc_difference(cost_hcv_postrelease_baseline, cost_hcv_postrelease_scenario)

    # Optional post-release (community) OAT delivery cost: discounted post-release person-years
    # on each modality x annual cost per person-year. 'medication' = drug only; 'same_as_prison'
    # = drug + RN + CO time with the in-prison formulas (CO observation time is a custodial
    # cost, so this likely overstates the community cost).
    postrelease_oat_cost = {}
    if community_oat_cost:
        if community_oat_cost not in ('medication', 'same_as_prison'):
            raise ValueError("community_oat_cost must be None, 'medication' or 'same_as_prison'")
        def annual_cost(i, key, rn_wage, co_wage):
            unit = lambda par: df_economic_pars.loc[par]['unit']
            drug = df_economic_pars.loc[f'cost_{key}']['point_estimate'] * (365 if 'per day' in unit(f'cost_{key}') else 12)
            if community_oat_cost == 'medication':
                return drug
            rn_time = sample_par(i, f'rn_time_{key}', df_economic_pars)
            co_time = sample_par(i, f'co_time_{key}', df_economic_pars)
            co_number = sample_par(i, 'co_number', df_economic_pars)
            rn = rn_wage/60*rn_time*(365 if 'per day' in unit(f'rn_time_{key}') else 12)
            co = co_wage/60*co_time*co_number*(365 if 'per day' in unit(f'co_time_{key}') else 12)
            return drug + rn + co
        for arm, suffix in (('baseline', 'b'), ('scenario', 's')):
            vals = []
            for i in range(n_sample):
                rn_wage = sample_par(i, 'rn_wage', df_economic_pars)
                co_wage = sample_par(i, 'co_wage', df_economic_pars)
                vals.append(sum(disc_vals[f'ptoat_{k}_{suffix}'][i] * annual_cost(i, k, rn_wage, co_wage)
                                for k in ('meth', 'subx', 'subl')))
            postrelease_oat_cost[arm] = vals
        postrelease_oat_cost['diff'] = calc_difference(postrelease_oat_cost['baseline'], postrelease_oat_cost['scenario'])

    postrelease_epi['uptake_baseline'] = summarise(uptake_events_baseline)
    postrelease_epi['uptake_scenario'] = summarise(uptake_events_scenario)
    diff_uptake_events_postrelease = calc_difference(uptake_events_baseline, uptake_events_scenario)

    ## Cost of reincarceration
    inc_cost_fed = df_economic_pars.loc['inc_cost_fed']['point_estimate']
    inc_cost_prov = df_economic_pars.loc['inc_cost_prov']['point_estimate']
    new_orders_federal = df_economic_pars.loc['new_orders_federal']['point_estimate']
    provincial_sen_len = df_economic_pars.loc['provincial_sen_len']['point_estimate']
    discount = df_economic_pars.loc['discount']['point_estimate']
    reinc_new_offence_oat = df_economic_pars.loc['reinc_new_offence_oat']['point_estimate']
    reinc_new_offence_off = df_economic_pars.loc['reinc_new_offence_off']['point_estimate']
    # fed_sentence_len_vals: shape [n_samples, n_time_points], in years
    _, _, _, _, fed_sentence_len_vals = extract_data(ensemble_baseline, 'sentence_len')

    def pv_sentence(daily_cost, T_days, r):
        # Present value of daily_cost paid over T_days at annual discount rate r
        T_years = T_days / 365
        if r == 0:
            return daily_cost * T_days
        return daily_cost * 365 * (1 - (1 + r) ** (-T_years)) / r

    def reinc_cost(n_oat, n_off, fed_len_days):
        # Cost of n_oat / n_off reincarceration events (linear in the counts, so it works
        # equally for per-arm totals and for scenario-minus-baseline differences)
        reinc_fed_oat = n_oat*(1-reinc_new_offence_oat)
        reinc_rem_oat = n_oat - reinc_fed_oat

        reinc_fed_off = n_off*(1-reinc_new_offence_off)
        reinc_rem_off = n_off - reinc_fed_off

        reinc_fed_oat += reinc_rem_oat*new_orders_federal
        reinc_prov_oat = reinc_rem_oat - reinc_rem_oat*new_orders_federal

        reinc_fed_off += reinc_rem_off*new_orders_federal
        reinc_prov_off = reinc_rem_off - reinc_rem_off*new_orders_federal

        cost_fed_oat  = pv_sentence(inc_cost_fed, fed_len_days, discount) * reinc_fed_oat
        cost_fed_off  = pv_sentence(inc_cost_fed, fed_len_days, discount) * reinc_fed_off
        cost_prov_oat = pv_sentence(inc_cost_prov, provincial_sen_len, discount) * reinc_prov_oat
        cost_prov_off = pv_sentence(inc_cost_prov, provincial_sen_len, discount) * reinc_prov_off
        return cost_fed_oat + cost_fed_off + cost_prov_oat + cost_prov_off

    cost_reinc_baseline = []
    cost_reinc_scenario = []
    for i in range(n_sample):
        # Federal sentence length: per-draw value from ensemble (years → days)
        fed_len_days = np.mean(fed_sentence_len_vals[i]) * 365
        # discounted event counts (by release year) - cost is linear in the counts
        cost_reinc_baseline.append(reinc_cost(disc_vals['reinc_oat_b'][i], disc_vals['reinc_off_b'][i], fed_len_days))
        cost_reinc_scenario.append(reinc_cost(disc_vals['reinc_oat_s'][i], disc_vals['reinc_off_s'][i], fed_len_days))
        cost_reinc_vals.append(reinc_cost(disc_vals['reinc_oat_s'][i] - disc_vals['reinc_oat_b'][i],
                                          disc_vals['reinc_off_s'][i] - disc_vals['reinc_off_b'][i], fed_len_days))

    diff_reinc_cost = bootstrap_median(cost_reinc_vals)

    # Absolute per-arm post-release outcomes (12-month decision tree), for the CCA tables
    postrelease_epi.update({
        'deaths_baseline': summarise(deaths_baseline), 'deaths_scenario': summarise(deaths_scenario),
        'reinc_baseline': summarise(reincarceration_baseline_total), 'reinc_scenario': summarise(reincarceration_scenario_total),
        'cost_reinc_baseline': summarise(cost_reinc_baseline), 'cost_reinc_scenario': summarise(cost_reinc_scenario),
        'cost_iri_baseline': summarise(cost_iri_postrelease_baseline), 'cost_iri_scenario': summarise(cost_iri_postrelease_scenario),
        'cost_hcv_baseline': summarise(cost_hcv_postrelease_baseline), 'cost_hcv_scenario': summarise(cost_hcv_postrelease_scenario),
        'qalys_death_baseline': summarise(abs_qalys['death']['baseline']), 'qalys_death_scenario': summarise(abs_qalys['death']['scenario']),
        'qalys_morb_post_baseline': summarise(abs_qalys['morb_post']['baseline']), 'qalys_morb_post_scenario': summarise(abs_qalys['morb_post']['scenario']),
        'qalys_morb_inprison_baseline': summarise(list(inprison_qalys_baseline)), 'qalys_morb_inprison_scenario': summarise(list(inprison_qalys_scenario)),
        'qalys_baseline': summarise(abs_qalys['total']['baseline']), 'qalys_scenario': summarise(abs_qalys['total']['scenario']),
    })
    postrelease_epi.update({
        'oat_py_baseline': summarise(oat_py_baseline), 'oat_py_scenario': summarise(oat_py_scenario),
        'diff_oat_py': calc_difference(oat_py_baseline, oat_py_scenario),
    })
    if postrelease_oat_cost:
        postrelease_epi.update({
            'cost_oat_com_baseline': summarise(postrelease_oat_cost['baseline']),
            'cost_oat_com_scenario': summarise(postrelease_oat_cost['scenario']),
            'diff_cost_oat_com': postrelease_oat_cost['diff'],
        })

    return (diff_reinc, diff_deaths, diff_qalys, diff_reinc_cost,
            diff_iri_events_postrelease, diff_sero_events_postrelease, diff_uptake_events_postrelease,
            diff_qalys_death, diff_qalys_morbidity_postrelease, diff_qalys_morbidity_inprison,
            diff_costs_iri_postrelease, diff_costs_hcv_postrelease,
            postrelease_epi)


def calculate_oat_costs(P, ensemble_baseline, ensemble_scenario, df_economic_pars, year_start, year_end):
    def extract_stock(ens):
        t, _, _, _, pop_oat_meth = extract_data(ens, 'pop_oat_meth') 
        _, _, _, _, pop_oat_subx = extract_data(ens, 'pop_oat_subx')
        _, _, _, _, pop_oat_subl = extract_data(ens, 'pop_oat_subl')
        
        return {
            't': t,
            'meth': pop_oat_meth,
            'subx': pop_oat_subx,
            'subl': pop_oat_subl
        }

    def extract_starts(ens):
        # PlotData annualises :flow outputs (people/year), so starts per year = sum * dt
        t = extract_data(ens, 'meth_uptake:flow')[0]
        return {'t': t, **{k: extract_data(ens, f'{k}_uptake:flow')[4] for k in ['meth', 'subx', 'subl']}}

    stock_baseline = extract_stock(ensemble_baseline)
    stock_scenario = extract_stock(ensemble_scenario)
    starts_baseline = adjust_time_range(extract_starts(ensemble_baseline), year_start, year_end)
    starts_scenario = adjust_time_range(extract_starts(ensemble_scenario), year_start, year_end)

    stock_baseline = adjust_time_range(stock_baseline, year_start, year_end)
    stock_scenario = adjust_time_range(stock_scenario, year_start, year_end)
    # min, not baseline's length alone - see calculate_post_release's matching comment
    n_sample = min(len(stock_baseline['meth']), len(stock_scenario['meth']))
    years_dt = stock_baseline['t']
    years_annual = np.arange(year_start, year_end+1)
    dt = stock_baseline['t'][1]-stock_baseline['t'][0]

    oat_keys = ['meth','subx','subl']

    cost_oat = dict()
    for key in oat_keys:
        cost_oat[key] = df_economic_pars.loc[f'cost_{key}']['point_estimate']
    rn_time_init = df_economic_pars.loc['rn_time_init']['point_estimate']

    cost_drug_vals = {k1: {k2: [] for k2 in oat_keys} for k1 in ['Baseline','Scenario']}
    cost_rn_vals = {k1: {k2: [] for k2 in oat_keys} for k1 in ['Baseline','Scenario']}
    cost_co_vals = {k1: {k2: [] for k2 in oat_keys} for k1 in ['Baseline','Scenario']}
    cost_total_vals = {k1: {k2: [] for k2 in oat_keys} for k1 in ['Baseline','Scenario']}
    cost_init_vals = {k1: {k2: [] for k2 in oat_keys} for k1 in ['Baseline','Scenario']}
    starts_vals = {k1: {k2: [] for k2 in oat_keys} for k1 in ['Baseline','Scenario']}
    py_vals = {k1: {k2: [] for k2 in oat_keys} for k1 in ['Baseline','Scenario']}   # person-years on each modality
    per_patient_cost = {k1: [] for k1 in ['Baseline','Scenario']}
    unit_vals = {k: [] for k in oat_keys}   # annual delivery cost per person-year, same in both arms
    discount = df_economic_pars.loc['discount']['point_estimate']

    for i in range(n_sample):
        rn_wage = sample_par(i, 'rn_wage', df_economic_pars)
        co_wage = sample_par(i, 'co_wage', df_economic_pars)
        totals = {name: {key: 0 for key in oat_keys} for name in ['Baseline','Scenario']}
        drug_totals = {name: {key: 0 for key in oat_keys} for name in ['Baseline','Scenario']}
        rn_totals = {name: {key: 0 for key in oat_keys} for name in ['Baseline','Scenario']}
        co_totals = {name: {key: 0 for key in oat_keys} for name in ['Baseline','Scenario']}
        init_totals = {name: {key: 0 for key in oat_keys} for name in ['Baseline','Scenario']}
        starts_totals = {name: {key: 0 for key in oat_keys} for name in ['Baseline','Scenario']}
        py_totals = {name: {key: 0 for key in oat_keys} for name in ['Baseline','Scenario']}
        # Per-simulation cost per patient (discounted OAT delivery cost / OAT person-years),
        # accumulated over all years so each simulation contributes one value per arm
        pp_cost = {name: 0.0 for name in ['Baseline','Scenario']}
        pp_stock = {name: 0.0 for name in ['Baseline','Scenario']}
        for year in years_annual:
            t_range_year = (years_dt >= year) & (years_dt < year+1)    
            discount_factor = 1 / (1 + discount) ** (np.floor(year) - year_start)
            for result, starts, name in zip([stock_baseline, stock_scenario], [starts_baseline, starts_scenario], ['Baseline','Scenario']):
                stock_init=0
                cost_init=0
                for key in oat_keys:
                    rn_time = sample_par(i, f'rn_time_{key}', df_economic_pars)
                    co_time = sample_par(i, f'co_time_{key}', df_economic_pars)
                    co_number = sample_par(i, 'co_number', df_economic_pars)
                    stock = result[key][i][t_range_year].sum()*dt
                    py_totals[name][key] += stock
                    drug_cost = cost_oat[key]*365 if "per day" in df_economic_pars.loc[f'cost_{key}']['unit'] else cost_oat[key]*12
                    rn_cost = rn_wage/60*rn_time*365 if "per day" in df_economic_pars.loc[f'rn_time_{key}']['unit'] else rn_wage/60*rn_time*12
                    co_cost = co_wage/60*co_time*365*co_number if "per day" in df_economic_pars.loc[f'co_time_{key}']['unit'] else co_wage/60*co_time*12*co_number
                    n_starts = starts[key][i][t_range_year].sum()*dt
                    init_total = n_starts*(rn_time_init/60)*rn_wage  # undiscounted; discount_factor is applied once below
                    drug_totals[name][key] += stock*drug_cost*discount_factor
                    rn_totals[name][key] += stock*rn_cost*discount_factor
                    co_totals[name][key] += stock*co_cost*discount_factor
                    init_totals[name][key] += init_total*discount_factor
                    starts_totals[name][key] += n_starts
                    totals[name][key] += (stock*(drug_cost + rn_cost + co_cost) + init_total)*discount_factor
                    stock_init += stock
                    cost_init += (stock*(drug_cost + rn_cost + co_cost) + init_total)*discount_factor
                pp_cost[name] += cost_init
                pp_stock[name] += stock_init
                
        for key in oat_keys:
            # Ongoing annual delivery cost per person-year on this modality (medication + nurse + officer time);
            # no initiation cost and no discounting, so it does not depend on the scenario
            u = df_economic_pars.loc[f'cost_{key}']['unit']
            r_u = df_economic_pars.loc[f'rn_time_{key}']['unit']
            c_u = df_economic_pars.loc[f'co_time_{key}']['unit']
            per = lambda unit: 365 if 'per day' in unit else 12
            unit_vals[key].append(
                cost_oat[key]*per(u)
                + rn_wage/60*sample_par(i, f'rn_time_{key}', df_economic_pars)*per(r_u)
                + co_wage/60*sample_par(i, f'co_time_{key}', df_economic_pars)*per(c_u)*sample_par(i, 'co_number', df_economic_pars))
        for name in ['Baseline','Scenario']:
            per_patient_cost[name].append(pp_cost[name]/pp_stock[name])
            for key in oat_keys:               
                cost_drug_vals[name][key].append(drug_totals[name][key])
                cost_rn_vals[name][key].append(rn_totals[name][key])
                cost_co_vals[name][key].append(co_totals[name][key])
                cost_init_vals[name][key].append(init_totals[name][key])
                starts_vals[name][key].append(starts_totals[name][key])
                cost_total_vals[name][key].append(totals[name][key])
                py_vals[name][key].append(py_totals[name][key])

    # Flatten to per-simulation totals across modalities
    def flatten(vals_dict, name):
        return [sum(vals_dict[name][k][i] for k in oat_keys) for i in range(n_sample)]

    def summarise(flat):
        return {'pe': np.median(flat), 'lb': np.percentile(flat, 2.5), 'ub': np.percentile(flat, 97.5), 'vals': flat}

    total_baseline_flat = flatten(cost_total_vals, 'Baseline')
    total_scenario_flat = flatten(cost_total_vals, 'Scenario')
    diff_costs_total = calc_difference(total_baseline_flat, total_scenario_flat)
    
    def by_modality(name):
        """Per modality: person-years (undiscounted), share of all OAMT person-years (%), discounted delivery
        cost, and discounted cost per person-year - each summarised across simulations."""
        py = {k: np.asarray(py_vals[name][k], dtype=float) for k in oat_keys}
        cost = {k: np.asarray(cost_total_vals[name][k], dtype=float) for k in oat_keys}
        all_py = sum(py.values())

        def nan_summary(v):
            v = np.asarray(v, dtype=float)
            if np.all(np.isnan(v)):
                return {'pe': np.nan, 'lb': np.nan, 'ub': np.nan, 'vals': v}
            return {'pe': np.nanmedian(v), 'lb': np.nanpercentile(v, 2.5), 'ub': np.nanpercentile(v, 97.5), 'vals': v}

        out = {}
        for k in oat_keys:
            with np.errstate(divide='ignore', invalid='ignore'):
                per_py = np.where(py[k] > 0, cost[k] / py[k], np.nan)
                share = 100 * py[k] / all_py
            out[k] = {'person_years': nan_summary(py[k]), 'share': nan_summary(share),
                      'cost': nan_summary(cost[k]), 'cost_per_py': nan_summary(per_py),
                      'unit_cost': nan_summary(unit_vals[k])}
        return out

    costs = {}
    for name in ['Baseline', 'Scenario']:
        costs[name] = {
            'by modality': by_modality(name),
            'total': summarise(flatten(cost_total_vals, name)),
            'drug': summarise(flatten(cost_drug_vals, name)),
            'rn': summarise(flatten(cost_rn_vals, name)),
            'co': summarise(flatten(cost_co_vals, name)),
            'init': summarise(flatten(cost_init_vals, name)),
            'starts': summarise(flatten(starts_vals, name)),
            'per patient': summarise(per_patient_cost[name])
        }

    return diff_costs_total, costs


def calculate_epi_outcomes(P, ensemble_baseline, ensemble_scenario, df_economic_pars, year_start, year_end):    
    def extract_inci(ens):
        t, _, _, _, hcv_inci = extract_data(ens, 'inci_m') 
        _, _, _, _, iri_inci = extract_data(ens, 'iri_m')
        _, _, _, _, hcv_treat = extract_data(ens, 'treat_m')
        
        return {
            't': t,
            'hcv_inci': hcv_inci,
            'iri_inci': iri_inci,
            'hcv_treat': hcv_treat,
        }
    
    inci_baseline = extract_inci(ensemble_baseline)
    inci_scenario = extract_inci(ensemble_scenario)
    inci_baseline = adjust_time_range(inci_baseline, year_start, year_end)
    inci_scenario = adjust_time_range(inci_scenario, year_start, year_end)
    # min, not baseline's length alone - see calculate_post_release's matching comment
    n_sample = min(len(inci_baseline['hcv_inci']), len(inci_scenario['hcv_inci']))
    years_dt = inci_baseline['t']
    years_annual = np.arange(year_start, year_end+1)
    dt = inci_baseline['t'][1]-inci_baseline['t'][0]

    discount = df_economic_pars.loc['discount']['point_estimate']
    hcv_baseline = []
    hcv_scenario = []
    iri_baseline = []
    iri_scenario = []
    cost_hcv_baseline = []
    cost_hcv_scenario = []
    cost_iri_baseline = []
    cost_iri_scenario = []
    for i in range(n_sample):
        cost_hcv = sample_par(i, 'cost_hcv_management', df_economic_pars)
        cost_iri = sample_par(i, 'cost_iri_management', df_economic_pars)
        hcv_b, hcv_s, iri_b, iri_s = 0, 0, 0, 0
        # discounted counts (same convention as calculate_oat_costs), used for costs only
        # HCV cost follows treatment initiations (treat_m = prob_treat:flow), not new infections
        hcv_b_d, hcv_s_d, iri_b_d, iri_s_d = 0, 0, 0, 0
        for year in years_annual:
            t_range_year = (years_dt >= year) & (years_dt < year+1)
            discount_factor = 1 / (1 + discount) ** (year - year_start)
            hcv_y_b = inci_baseline['hcv_inci'][i][t_range_year].sum()*dt
            hcv_y_s = inci_scenario['hcv_inci'][i][t_range_year].sum()*dt
            iri_y_b = inci_baseline['iri_inci'][i][t_range_year].sum()*dt
            iri_y_s = inci_scenario['iri_inci'][i][t_range_year].sum()*dt
            trt_y_b = inci_baseline['hcv_treat'][i][t_range_year].sum()*dt
            trt_y_s = inci_scenario['hcv_treat'][i][t_range_year].sum()*dt
            hcv_b += hcv_y_b; hcv_s += hcv_y_s; iri_b += iri_y_b; iri_s += iri_y_s
            hcv_b_d += trt_y_b*discount_factor; hcv_s_d += trt_y_s*discount_factor
            iri_b_d += iri_y_b*discount_factor; iri_s_d += iri_y_s*discount_factor
        hcv_baseline.append(hcv_b)
        hcv_scenario.append(hcv_s)
        iri_baseline.append(iri_b)
        iri_scenario.append(iri_s)
        cost_hcv_baseline.append(hcv_b_d * cost_hcv)
        cost_hcv_scenario.append(hcv_s_d * cost_hcv)
        cost_iri_baseline.append(iri_b_d * cost_iri)
        cost_iri_scenario.append(iri_s_d * cost_iri)
    
    diff_hcv = calc_difference(hcv_baseline, hcv_scenario)
    diff_iri = calc_difference(iri_baseline, iri_scenario)   
    diff_costs_hcv = calc_difference(cost_hcv_baseline, cost_hcv_scenario)
    diff_costs_iri = calc_difference(cost_iri_baseline, cost_iri_scenario)
    def summarise(par_vals):
        return {'pe': np.median(par_vals), 'lb': np.percentile(par_vals, 2.5), 'ub': np.percentile(par_vals, 97.5), 'vals': par_vals}
    
    return (diff_hcv, diff_iri, diff_costs_hcv, diff_costs_iri,
            summarise(hcv_baseline), summarise(hcv_scenario),
            summarise(iri_baseline), summarise(iri_scenario),
            summarise(cost_hcv_baseline), summarise(cost_hcv_scenario),
            summarise(cost_iri_baseline), summarise(cost_iri_scenario))


TRAJECTORY_OUTPUTS = [
    ('iri_idu_per100py', 'IRI incidence among people who inject drugs (per 100 person-years)', 1, 1),
    ('inci_per100py', 'HCV incidence, prison-wide (per 100 person-years)', 1, 3),
    ('prev', 'HCV prevalence, prison-wide (%)', 100, 2),
]


def calculate_trajectory_summary(ensemble_baseline, ensemble_scenario, year_start, year_end):
    """
    Model outputs behind the trajectory figure, as numbers: IRI incidence, HCV incidence and HCV
    prevalence averaged over the analysis period (year_start to the end of year_end) and over the
    final year, for each scenario (median and 2.5th-97.5th percentiles across runs) and as the
    paired difference scenario - baseline (bootstrap_median, same interval as the other tables).
    ensemble_baseline is the counterfactual and ensemble_scenario the policy change, as in
    calculate_all_outcomes.
    """
    def summ(v):
        return {'pe': float(np.median(v)), 'lb': float(np.percentile(v, 2.5)), 'ub': float(np.percentile(v, 97.5))}

    out = {}
    for code, label, scale, dp in TRAJECTORY_OUTPUTS:
        t, _, _, _, vb = extract_data(ensemble_baseline, code)
        _, _, _, _, vs = extract_data(ensemble_scenario, code)
        n = min(len(vb), len(vs))
        windows = {
            'period': (t >= year_start) & (t < year_end + 1),
            'final': (t >= year_end) & (t < year_end + 1),
        }
        entry = {'label': label, 'dp': dp}
        for key, mask in windows.items():
            mb = scale * np.asarray(vb)[:n, mask].mean(axis=1)
            ms = scale * np.asarray(vs)[:n, mask].mean(axis=1)
            d = bootstrap_median(ms - mb)
            entry[key] = {'baseline': summ(mb), 'scenario': summ(ms),
                          'diff': {'pe': float(d['pe']), 'lb': float(d['lb']), 'ub': float(d['ub']), 'vals': ms - mb}}
        out[code] = entry
    out['year_start'], out['year_end'] = year_start, year_end
    return out


def calculate_cost_offset(results):
    """
    How much of the policy change's savings in OAMT delivery (in prison plus community medication after
    release) is offset by higher HCV, IRI and reincarceration costs. Computed per simulation from the
    paired cost differences, then summarised with bootstrap_median:
        savings = -(diff OAMT delivery + diff community OAMT)
        extra   = diff IRI + diff HCV + diff reincarceration   (prison and community)
        offset  = 100 * extra / savings   (% of the savings that is offset)
        net     = savings - extra         (= minus the total cost difference)
    """
    keys = ('diff_costs_oat_total', 'diff_costs_oat_com', 'diff_costs_iri_total',
            'diff_costs_hcv_total', 'diff_reinc_cost')
    arrays = {k: np.asarray(results[k]['vals'], dtype=float) for k in keys if results.get(k)}
    n = min(len(v) for v in arrays.values())
    g = lambda k: arrays[k][:n] if k in arrays else np.zeros(n)
    savings = -(g('diff_costs_oat_total') + g('diff_costs_oat_com'))
    extra = g('diff_costs_iri_total') + g('diff_costs_hcv_total') + g('diff_reinc_cost')
    with np.errstate(divide='ignore', invalid='ignore'):
        offset = np.where(savings > 0, 100 * extra / savings, np.nan)
    return {
        'savings': bootstrap_median(savings),
        'extra': bootstrap_median(extra),
        'net': bootstrap_median(savings - extra),
        'offset_pct': bootstrap_median(offset[~np.isnan(offset)]),
    }


def calculate_all_outcomes(P, ensemble_baseline, ensemble_scenario, economic_pars_path, year_start, year_end,
                            include_community_costs=True, ci_method=None, community_oat_cost=None):
    """
    Public entry point; see _calculate_all_outcomes for what is computed.

    :param community_oat_cost: None (default) leaves post-release OAT delivery cost out.
        'medication' adds the drug cost of the post-release (12-month decision tree) OAT
        person-years; 'same_as_prison' adds drug + RN + CO time with the in-prison formulas.

    :param ci_method: how lb/ub are computed for every reported figure in the returned dict
        (and in write_economic_outcomes, which reads it back from the results):
        'bootstrap'  - bootstrap CI of the median (Monte Carlo precision only; narrow)
        'percentile' - 2.5th-97.5th percentile of the per-draw values/paired differences
                       (includes parameter uncertainty; wider)
        None (default) - module-level CI_METHOD.
    """
    global CI_METHOD
    previous = CI_METHOD
    if ci_method is not None:
        CI_METHOD = ci_method
    try:
        results = _calculate_all_outcomes(P, ensemble_baseline, ensemble_scenario, economic_pars_path,
                                          year_start, year_end, include_community_costs=include_community_costs,
                                          community_oat_cost=community_oat_cost)
        results['ci_method'] = CI_METHOD
        try:
            results['cost_offset'] = calculate_cost_offset(results)
        except Exception as e:
            print('Cost offset skipped:', e)
        try:
            results['trajectory_summary'] = calculate_trajectory_summary(
                ensemble_baseline, ensemble_scenario, year_start, year_end)
        except Exception as e:  # never let this extra summary break the main outcomes
            print('Trajectory summary skipped:', e)
        return results
    finally:
        CI_METHOD = previous


def _calculate_all_outcomes(P, ensemble_baseline, ensemble_scenario, economic_pars_path, year_start, year_end,
                            include_community_costs=True, community_oat_cost=None):
    """
    Wrapper that calls all outcome functions and aggregates total costs for the
    cost-consequence analysis (CCA): costs and consequences are reported side by side, not
    combined into an ICER / net monetary benefit.

    Costs default to CSC only (in-prison HCV/IRI/OAT delivery + reincarceration) - this
    manuscript's primary scope, since community treatment costs are a provincial health-
    system cost, not CSC's, and modelling them properly would need their own costing
    study. Set include_community_costs=True to fold post-release HCV treatment and IRI
    management cost into the headline total cost instead (post-release OAT delivery
    cost still isn't tracked at all either way - see calculate_post_release's docstring).

    HCV infections, IRI events, and (regardless of the switch) HCV/IRI *cost* are all
    returned three ways - '_inprison', '_postrelease', '_total' (their sum) - so
    write_economic_outcomes can show the split alongside whichever total is "official"
    for headline reporting under the current switch setting.

    diff_pop_oat_inprison is also split by modality (diff_pop_oat_meth/_subx/_subl) - these
    can carry a different sign than the total (e.g. a shift onto/off LAI-BUP nets out in the
    total but not within each modality).
    Returns a dict of summarised outcomes ready for write_economic_outcomes.
    """
    def summarise(vals):
        return {'pe': np.median(vals), 'lb': np.percentile(vals, 2.5), 'ub': np.percentile(vals, 97.5), 'vals': vals}

    def combine(a, b):
        return bootstrap_median([a['vals'][i] + b['vals'][i] for i in range(len(a['vals']))])

    # Average difference in in-prison OAMT population, start_year-end_year (mean per-sample
    # difference over the modelled horizon, same style as write_sensitivity_summary_table's
    # pop_oat_diff). Truncated to the smaller ensemble's sample count in case baseline and
    # scenario were run with different n_samples (e.g. a retention-threshold sweep).
    t, _, _, _, vals_a = extract_data(ensemble_scenario, 'pop_oat')
    _, _, _, _, vals_b = extract_data(ensemble_baseline, 'pop_oat')
    n_pop = min(vals_a.shape[0], vals_b.shape[0])
    mask = (t >= year_start) & (t < year_end + 1)
    pop_oat_annual = (vals_a[:n_pop, mask] - vals_b[:n_pop, mask]).mean(axis=1)
    diff_pop_oat_inprison = bootstrap_median(pop_oat_annual)
    pop_oat_baseline_abs = summarise(vals_b[:n_pop, mask].mean(axis=1).tolist())
    pop_oat_scenario_abs = summarise(vals_a[:n_pop, mask].mean(axis=1).tolist())
    if isinstance(economic_pars_path,str):
        df_economic_pars = pd.read_excel(economic_pars_path, index_col=0)
    else:
        df_economic_pars = economic_pars_path

    # Same average yearly difference, split by OAT modality - the total above can net out a
    # decline in one modality against a rise in another (e.g. a shift onto/off LAI-BUP), so
    # the by-modality split can carry a different sign than the total.
    diff_pop_oat_by_modality = {}
    for modality, output in [('meth', 'pop_oat_meth'), ('subx', 'pop_oat_subx'), ('subl', 'pop_oat_subl')]:
        _, _, _, _, mod_vals_a = extract_data(ensemble_scenario, output)
        _, _, _, _, mod_vals_b = extract_data(ensemble_baseline, output)
        n_mod = min(mod_vals_a.shape[0], mod_vals_b.shape[0])
        mod_annual = (mod_vals_a[:n_mod, mask] - mod_vals_b[:n_mod, mask]).mean(axis=1)
        diff_pop_oat_by_modality[modality] = bootstrap_median(mod_annual)

    # --- run all three functions ---
    diff_hcv_inprison, diff_iri_inprison, diff_costs_hcv_inprison, diff_costs_iri_inprison, \
        hcv_baseline, hcv_scenario, iri_baseline, iri_scenario, \
        cost_hcv_baseline, cost_hcv_scenario, cost_iri_baseline, cost_iri_scenario = \
        calculate_epi_outcomes(P, ensemble_baseline, ensemble_scenario, df_economic_pars, year_start, year_end)

    diff_costs_oat, costs_oat = \
        calculate_oat_costs(P, ensemble_baseline, ensemble_scenario, df_economic_pars, year_start, year_end)

    diff_reinc, diff_deaths, diff_qalys, diff_reinc_cost, \
        diff_iri_events_postrelease, diff_sero_events_postrelease, diff_uptake_events_postrelease, \
        diff_qalys_death, diff_qalys_morbidity_postrelease, diff_qalys_morbidity_inprison, \
        diff_costs_iri_postrelease, diff_costs_hcv_postrelease, \
        postrelease_epi = \
        calculate_post_release(P, ensemble_baseline, ensemble_scenario, df_economic_pars, year_start, year_end,
                               community_oat_cost=community_oat_cost)

    n_sample = len(diff_deaths['vals'])

    # Combined (in-prison + post-release) totals for HCV infections and IRI events, and
    # for HCV/IRI cost regardless of the switch (so the report can show the split even
    # when the "official" total below is in-prison only).
    diff_iri_total = combine(diff_iri_inprison, diff_iri_events_postrelease)
    diff_hcv_total = combine(diff_hcv_inprison, diff_sero_events_postrelease)
    diff_costs_hcv_total = combine(diff_costs_hcv_inprison, diff_costs_hcv_postrelease)
    diff_costs_iri_total = combine(diff_costs_iri_inprison, diff_costs_iri_postrelease)

    # "Official" HCV/IRI cost used for headline total cost - CSC (in-prison) only
    # unless include_community_costs asks for post-release HCV treatment/IRI management
    # cost to be folded in too.
    diff_costs_hcv = diff_costs_hcv_total if include_community_costs else diff_costs_hcv_inprison
    diff_costs_iri = diff_costs_iri_total if include_community_costs else diff_costs_iri_inprison

    # OAT cost by category (in-prison only)
    diff_costs_oat_drug = calc_difference(costs_oat['Baseline']['drug']['vals'], costs_oat['Scenario']['drug']['vals'])
    diff_costs_oat_rn = calc_difference(costs_oat['Baseline']['rn']['vals'], costs_oat['Scenario']['rn']['vals'])
    diff_costs_oat_co = calc_difference(costs_oat['Baseline']['co']['vals'], costs_oat['Scenario']['co']['vals'])
    diff_costs_oat_init = calc_difference(costs_oat['Baseline']['init']['vals'], costs_oat['Scenario']['init']['vals'])
    diff_oat_starts_inprison = calc_difference(costs_oat['Baseline']['starts']['vals'], costs_oat['Scenario']['starts']['vals'])
    diff_costs_oat_perpatient = calc_difference(costs_oat['Baseline']['per patient']['vals'], costs_oat['Scenario']['per patient']['vals'])

    # --- total costs (positive = policy costs more) ---
    diff_costs_oat_com = postrelease_epi.get('diff_cost_oat_com')
    total_cost_vals = [
        diff_costs_oat['vals'][i] + diff_costs_hcv['vals'][i] + diff_costs_iri['vals'][i] + diff_reinc_cost['vals'][i]
        + (diff_costs_oat_com['vals'][i] if diff_costs_oat_com else 0.0)
        for i in range(n_sample)
    ]
    diff_total_costs = bootstrap_median(total_cost_vals)
    # CCA framing: 'cost' = cost of the intervention (OAMT delivery, in prison + community medication); 'financial
    # consequences' = HCV treatment, IRI management and reincarceration costs that follow from the health consequences
    cost_interv_vals = [diff_costs_oat['vals'][i] + (diff_costs_oat_com['vals'][i] if diff_costs_oat_com else 0.0)
                        for i in range(n_sample)]
    cost_fin_vals = [diff_costs_hcv['vals'][i] + diff_costs_iri['vals'][i] + diff_reinc_cost['vals'][i]
                     for i in range(n_sample)]
    diff_cost_intervention = bootstrap_median(cost_interv_vals)
    diff_cost_fin = bootstrap_median(cost_fin_vals)

    def add_abs(a, b):
        n = min(len(a['vals']), len(b['vals']))
        return summarise([a['vals'][i] + b['vals'][i] for i in range(n)])

    # Absolute in-prison + post-release totals per arm (combined CCA view)
    pr = postrelease_epi
    total_abs = {}
    for arm, hcv_in, iri_in, chcv_in, ciri_in in (
            ('baseline', hcv_baseline, iri_baseline, cost_hcv_baseline, cost_iri_baseline),
            ('scenario', hcv_scenario, iri_scenario, cost_hcv_scenario, cost_iri_scenario)):
        total_abs[f'hcv_total_{arm}'] = add_abs(hcv_in, pr[f'sero_{arm}'])
        total_abs[f'iri_total_{arm}'] = add_abs(iri_in, pr[f'iri_{arm}'])
        total_abs[f'cost_hcv_total_{arm}'] = add_abs(chcv_in, pr[f'cost_hcv_{arm}'])
        total_abs[f'cost_iri_total_{arm}'] = add_abs(ciri_in, pr[f'cost_iri_{arm}'])

    # Total cost per arm, on the same basis as diff_total_costs (CSC only unless
    # include_community_costs): OAT delivery + HCV + IRI + re-incarceration
    cost_oat_arm = {'baseline': costs_oat['Baseline']['total'], 'scenario': costs_oat['Scenario']['total']}
    for arm, hcv_in, iri_in in (('baseline', cost_hcv_baseline, cost_iri_baseline),
                                ('scenario', cost_hcv_scenario, cost_iri_scenario)):
        hcv_c = total_abs[f'cost_hcv_total_{arm}'] if include_community_costs else hcv_in
        iri_c = total_abs[f'cost_iri_total_{arm}'] if include_community_costs else iri_in
        total_abs[f'cost_total_{arm}'] = add_abs(add_abs(cost_oat_arm[arm], hcv_c),
                                                 add_abs(iri_c, pr[f'cost_reinc_{arm}']))
        if community_oat_cost:
            total_abs[f'cost_total_{arm}'] = add_abs(total_abs[f'cost_total_{arm}'], pr[f'cost_oat_com_{arm}'])
        interv = cost_oat_arm[arm]
        if community_oat_cost:
            interv = add_abs(interv, pr[f'cost_oat_com_{arm}'])
        total_abs[f'cost_intervention_{arm}'] = interv
        total_abs[f'cost_fin_{arm}'] = add_abs(add_abs(hcv_c, iri_c), pr[f'cost_reinc_{arm}'])

    return {
        'include_community_costs': include_community_costs,
        'community_oat_cost': community_oat_cost,
        'diff_costs_oat_com': diff_costs_oat_com,
        # Post-release (12mo decision tree) absolute per-arm values and in-prison + post-release totals
        'postrelease_abs': postrelease_epi,
        'total_abs': total_abs,
        'diff_pop_oat_inprison': diff_pop_oat_inprison,
        'diff_pop_oat_meth': diff_pop_oat_by_modality['meth'],
        'diff_pop_oat_subx': diff_pop_oat_by_modality['subx'],
        'diff_pop_oat_subl': diff_pop_oat_by_modality['subl'],
        # Epi outcomes: in-prison / post-release (12mo) / combined
        'diff_hcv_inprison': diff_hcv_inprison, 'diff_hcv_postrelease': diff_sero_events_postrelease, 'diff_hcv_total': diff_hcv_total,
        'diff_iri_inprison': diff_iri_inprison, 'diff_iri_postrelease': diff_iri_events_postrelease, 'diff_iri_total': diff_iri_total,
        'diff_oat_uptake_postrelease': diff_uptake_events_postrelease,
        # OAT delivery cost, by category (in-prison only; post-release OAT is a provincial cost)
        'diff_costs_oat_drug': diff_costs_oat_drug,
        'diff_costs_oat_rn':   diff_costs_oat_rn,
        'diff_costs_oat_co':   diff_costs_oat_co,
        'diff_costs_oat_init': diff_costs_oat_init,
        'diff_oat_starts_inprison': diff_oat_starts_inprison,
        'diff_costs_oat_total': diff_costs_oat,
        'diff_cost_per_patient': diff_costs_oat_perpatient,
        # Downstream costs: in-prison / post-release (12mo) / combined always available;
        # 'diff_costs_hcv'/'diff_costs_iri' (no suffix) are whichever is "official" under
        # include_community_costs, and are what feeds diff_total_costs below.
        'diff_costs_hcv_inprison': diff_costs_hcv_inprison, 'diff_costs_hcv_postrelease': diff_costs_hcv_postrelease, 'diff_costs_hcv_total': diff_costs_hcv_total,
        'diff_costs_iri_inprison': diff_costs_iri_inprison, 'diff_costs_iri_postrelease': diff_costs_iri_postrelease, 'diff_costs_iri_total': diff_costs_iri_total,
        'diff_costs_hcv':       diff_costs_hcv,
        'diff_costs_iri':       diff_costs_iri,
        'diff_reinc_cost':      diff_reinc_cost,
        # Mortality & QALYs (differences only)
        'diff_deaths':          diff_deaths,
        'diff_qalys':           diff_qalys,
        'diff_qalys_death':                 diff_qalys_death,
        'diff_qalys_morbidity_postrelease': diff_qalys_morbidity_postrelease,
        'diff_qalys_morbidity_inprison':    diff_qalys_morbidity_inprison,
        # Reincarceration count
        'diff_reinc':           diff_reinc,
        # Summary
        'diff_total_costs':     diff_total_costs,
        'diff_cost_intervention': diff_cost_intervention,
        'diff_cost_fin':         diff_cost_fin,
        # ── Absolute in-prison values for CCA display ──────────────────────
        'pop_oat_inprison_baseline': pop_oat_baseline_abs,
        'pop_oat_inprison_scenario': pop_oat_scenario_abs,
        'hcv_inprison_baseline':      hcv_baseline,
        'hcv_inprison_scenario':      hcv_scenario,
        'iri_inprison_baseline':      iri_baseline,
        'iri_inprison_scenario':      iri_scenario,
        'cost_hcv_inprison_baseline': cost_hcv_baseline,
        'cost_hcv_inprison_scenario': cost_hcv_scenario,
        'cost_iri_inprison_baseline': cost_iri_baseline,
        'cost_iri_inprison_scenario': cost_iri_scenario,
        'cost_oat_total_baseline':    costs_oat['Baseline']['total'],
        'cost_oat_total_scenario':    costs_oat['Scenario']['total'],
        'cost_oat_drug_baseline':     costs_oat['Baseline']['drug'],
        'cost_oat_drug_scenario':     costs_oat['Scenario']['drug'],
        'cost_oat_rn_baseline':       costs_oat['Baseline']['rn'],
        'cost_oat_rn_scenario':       costs_oat['Scenario']['rn'],
        'cost_oat_co_baseline':       costs_oat['Baseline']['co'],
        'cost_oat_co_scenario':       costs_oat['Scenario']['co'],
        'oat_starts_inprison_baseline': costs_oat['Baseline']['starts'],
        'oat_starts_inprison_scenario': costs_oat['Scenario']['starts'],
        'cost_oat_init_baseline':     costs_oat['Baseline']['init'],
        'cost_oat_init_scenario':     costs_oat['Scenario']['init'],
        'oat_by_modality': {'Baseline': costs_oat['Baseline']['by modality'], 'Scenario': costs_oat['Scenario']['by modality']},
        'n_years': year_end - year_start + 1,
        'cost_per_patient_baseline':  costs_oat['Baseline']['per patient'],
        'cost_per_patient_scenario':  costs_oat['Scenario']['per patient'],
    }


def _mdot(s):
    """Replace decimal points with middle dot in numeric contexts."""
    return re.sub(r'(?<=\d)\.(?=\d)', '·', s)


def _fmt_sigfig(x, n=3):
    """Millions+: n sig figs with M/B suffix. Below 1M: plain $#,##0."""
    if x == 0:
        return '$0'
    sign = '-' if x < 0 else ''
    ax = abs(x)
    def dp(val):
        if val == 0:
            return n
        digits_before = max(1, int(math.floor(math.log10(val))) + 1)
        return max(0, n - digits_before)
    if ax >= 1e9:
        scaled = ax / 1e9
        s = f'{sign}${scaled:.{dp(scaled)}f} billion'
    elif ax >= 1e6:
        scaled = ax / 1e6
        s = f'{sign}${scaled:.{dp(scaled)}f} million'
    else:
        s = f'{sign}${ax:,.0f}'
    return _mdot(s)


def _t2_formatter(kind):
    if kind == 'cost':
        # 3 significant figures with an M/B suffix on every value ($1.48B, $147M), true minus sign
        return lambda x: _fmt_sigfig(x).replace(' million', 'M').replace(' billion', 'B').replace('-', '\u2212')
    if kind.startswith('rate'):
        dp = int(kind[4:])
        return lambda x: f'{x:.{dp}f}'.replace('.', '\u00b7').replace('-', '\u2212')
    return lambda x: f'{int(round(x)):,}'.replace('-', '\u2212')


def _t2_median_ci_ui(d, kind):
    """
    For a difference dict that carries its per-run values ('vals'): the median, the bootstrap 95% CI of
    the median and the 95% UI (2.5th-97.5th percentiles of the paired differences), each formatted.
    Falls back to the dict's own lb/ub for both intervals if 'vals' is missing.
    """
    fmt = _t2_formatter(kind)
    vals = d.get('vals') if isinstance(d, dict) else None
    if vals is None:
        ci = (d['lb'], d['ub'])
        ui = ci
    else:
        v = np.asarray(vals, dtype=float)
        b = bootstrap_median(v, method='bootstrap')
        ci = (b['lb'], b['ub'])
        ui = (np.percentile(v, 2.5), np.percentile(v, 97.5))
    return fmt(d['pe']), f'{fmt(ci[0])} to {fmt(ci[1])}', f'{fmt(ui[0])} to {fmt(ui[1])}'


def _t2_labelled(d, kind, intervals):
    """
    Cell like '110 (95% CI: 108 to 112; 95% UI: 29 to 188)': the median, then every interval in the
    parentheses starting with its label. intervals: 'ui' (the dict's own lb/ub, used for scenario values)
    or 'ci_ui' (bootstrap CI of the median and the UI of the paired differences, for differences).
    """
    fmt = _t2_formatter(kind)
    if intervals == 'ui':
        return f"{fmt(d['pe'])} (95% UI: {fmt(d['lb'])} to {fmt(d['ub'])})"
    med, ci, ui = _t2_median_ci_ui(d, kind)
    return f"{med} (95% CI: {ci}; 95% UI: {ui})"


def _t2_cell(d, kind, with_interval=True):
    """
    Table 2 cell from a result dict {'pe', 'lb', 'ub'}. kind: 'count' (integers) or 'cost'
    (dollars, 3 significant figures with an M or B suffix).
    """
    fmt = _t2_formatter(kind)
    if not with_interval:
        return fmt(d['pe'])
    return f"{fmt(d['pe'])} ({fmt(d['lb'])} to {fmt(d['ub'])})"


def write_economic_outcomes(results, file_name='results/outcomes_economic.xlsx', baseline='no-policy-change',
                             label=None, baseline_name=None, scenario_name=None):
    """
    Write the cost-consequence analysis to Excel (Table 2, OAMT by modality, QALYs and cost offset),
    with Baseline | Scenario | Difference columns over the full study period. Costs and consequences
    are reported side by side, with no ICER or net monetary benefit.

    :param label: override for the Difference column header.
    :param baseline_name, scenario_name: used to build the default label.
    """
    if baseline == 'policy-change':
        default_baseline_name, default_scenario_name = "Policy change", "Counterfactual"
    else:
        default_baseline_name, default_scenario_name = "Counterfactual", "Policy change"
    if baseline_name is None:
        baseline_name = default_baseline_name
    if scenario_name is None:
        scenario_name = default_scenario_name
    if label is None:
        label = f'({scenario_name} − {baseline_name})'

    def add_bootstrap(a, b):
        """Sum two bootstrap result dicts element-wise across resamples."""
        av = np.asarray(a['vals']); bv = np.asarray(b['vals'])
        n = min(len(av), len(bv))
        return bootstrap_median(av[:n] + bv[:n], method=results.get('ci_method'))

    pr, ta = results['postrelease_abs'], results['total_abs']

    # Total cost per arm, on the same basis as results['diff_total_costs'] (CSC only unless
    # include_community_costs): OAT delivery + HCV + IRI + re-incarceration
    community = results.get('include_community_costs')
    total_cost_b = ta['cost_total_baseline']
    total_cost_s = ta['cost_total_scenario']

    # Table 2: manuscript-format table (median with 95% UI per scenario and for the difference)
    T2_COLS = ['', baseline_name, scenario_name, f'Difference {label} (95% CI)']
    t2_rows = []   # (cells, style) with style in {'header', 'row', 'sub', 'total'}

    def t2_hdr(title):
        t2_rows.append(([title, '', '', ''], 'header'))

    def t2(label1, b, s_, d, kind, style='row'):
        t2_rows.append(([label1, _t2_cell(b, kind), _t2_cell(s_, kind), _t2_cell(d, kind)], style))

    t2_hdr('Costs: OAMT intervention (2025 CAD, discounted)')
    t2('OAMT delivery in prison', results['cost_oat_total_baseline'], results['cost_oat_total_scenario'],
       results['diff_costs_oat_total'], 'cost')
    t2('   of which medication', results['cost_oat_drug_baseline'], results['cost_oat_drug_scenario'],
       results['diff_costs_oat_drug'], 'cost', 'sub')
    t2('   of which registered nurse (includes initiation)',
       add_bootstrap(results['cost_oat_rn_baseline'], results['cost_oat_init_baseline']),
       add_bootstrap(results['cost_oat_rn_scenario'], results['cost_oat_init_scenario']),
       add_bootstrap(results['diff_costs_oat_rn'], results['diff_costs_oat_init']), 'cost', 'sub')
    t2('   of which correctional officer', results['cost_oat_co_baseline'], results['cost_oat_co_scenario'],
       results['diff_costs_oat_co'], 'cost', 'sub')
    t2('Delivery cost per person-year on OAMT in prison (2025 CAD)', results['cost_per_patient_baseline'],
       results['cost_per_patient_scenario'], results['diff_cost_per_patient'], 'cost')
    if results.get('community_oat_cost'):
        t2('OAMT ' + ('medication' if results['community_oat_cost'] == 'medication' else 'delivery') + ' after release (community, 12 months)',
           pr['cost_oat_com_baseline'], pr['cost_oat_com_scenario'], results['diff_costs_oat_com'], 'cost')
    t2('Total cost of OAMT', ta['cost_intervention_baseline'], ta['cost_intervention_scenario'],
       results['diff_cost_intervention'], 'cost', 'total')

    t2_hdr('Consequences: OAMT coverage, health and justice outcomes')
    t2('Average people on OAMT in prison', results['pop_oat_inprison_baseline'], results['pop_oat_inprison_scenario'],
       results['diff_pop_oat_inprison'], 'count')
    t2('OAMT person-years after release (undiscounted)', pr['oat_py_baseline'], pr['oat_py_scenario'], pr['diff_oat_py'], 'count')
    t2('HCV infections in prison', results['hcv_inprison_baseline'], results['hcv_inprison_scenario'],
       results['diff_hcv_inprison'], 'count')
    t2('IRIs in prison', results['iri_inprison_baseline'], results['iri_inprison_scenario'],
       results['diff_iri_inprison'], 'count')
    traj_summary = results.get('trajectory_summary')
    if traj_summary:
        # final-year (2035) values, not the period average
        for code in ('inci_per100py', 'iri_idu_per100py', 'prev'):
            w = traj_summary[code]['final']
            t2('{} in {}'.format(traj_summary[code]['label'], traj_summary['year_end']),
               w['baseline'], w['scenario'], w['diff'], 'rate{}'.format(traj_summary[code]['dp']))
    t2('HCV infections after release (12 months)', pr['sero_baseline'], pr['sero_scenario'], results['diff_hcv_postrelease'], 'count')
    t2('IRIs after release (12 months)', pr['iri_baseline'], pr['iri_scenario'], results['diff_iri_postrelease'], 'count')
    t2('HCV infections, prison and community', ta['hcv_total_baseline'], ta['hcv_total_scenario'], results['diff_hcv_total'], 'count', 'total')
    t2('IRIs, prison and community', ta['iri_total_baseline'], ta['iri_total_scenario'], results['diff_iri_total'], 'count', 'total')
    t2('Drug-related deaths after release', pr['deaths_baseline'], pr['deaths_scenario'], results['diff_deaths'], 'count')
    t2('Reincarceration events after release', pr['reinc_baseline'], pr['reinc_scenario'], results['diff_reinc'], 'count')

    t2_hdr('Consequences: financial (2025 CAD, discounted)')
    t2('HCV treatment (prison and community)', ta['cost_hcv_total_baseline'], ta['cost_hcv_total_scenario'],
       results['diff_costs_hcv_total'], 'cost')
    t2('IRI management (prison and community)', ta['cost_iri_total_baseline'], ta['cost_iri_total_scenario'],
       results['diff_costs_iri_total'], 'cost')
    t2('Reincarceration', pr['cost_reinc_baseline'], pr['cost_reinc_scenario'],
       results['diff_reinc_cost'], 'cost')
    t2('Total financial consequences', ta['cost_fin_baseline'], ta['cost_fin_scenario'],
       results['diff_cost_fin'], 'cost', 'total')

    t2_hdr('Net cost')
    t2('Net cost (cost of OAMT plus financial consequences)', total_cost_b, total_cost_s, results['diff_total_costs'], 'cost', 'total')

    ci_desc = ('percentile range of the paired differences' if results.get('ci_method') == 'percentile'
               else 'bootstrap 95% confidence interval of the median difference')
    t2_notes = [
        'Incidence and prevalence are values in the final year of the analysis (2035): HCV incidence and prevalence are prison-wide, IRI incidence is among people who inject drugs, and incidence is per 100 person-years.',
        'Values for each scenario are medians (95% UI: 2.5th to 97.5th percentiles across simulations). '
        f'Differences are {scenario_name.lower()} minus {baseline_name.lower()}; the interval is the {ci_desc}.',
        'Cost is the cost of the intervention, OAMT delivery. Consequences are the health and justice outcomes of the change in OAMT and their financial consequences: '
        'the costs of HCV treatment, IRI management and reincarceration. Net cost is the cost of OAMT plus the financial consequences.',
        'Costs are in 2025 Canadian dollars, discounted at 1.5% to 2026. Delivery cost per person-year is the '
        'discounted OAMT delivery cost in prison divided by person-years on OAMT. '
        + ('Financial consequences include community HCV treatment and IRI management after release.' if community
           else 'Financial consequences are CSC costs only.')
        + (' Community OAMT cost is the ' + ('medication cost' if results['community_oat_cost'] == 'medication'
                                             else 'drug, nursing and officer cost')
           + ' of person-years on OAMT in the 12 months after release.' if results.get('community_oat_cost') else ''),
    ]

    from openpyxl.styles import Font, Alignment, Border, Side
    with pd.ExcelWriter(file_name, engine='openpyxl') as writer:
        pd.DataFrame([r for r, _ in t2_rows], columns=T2_COLS).to_excel(writer, sheet_name='Table 2', index=False)
        ws2 = writer.sheets['Table 2']
        for col_idx, width in enumerate([52, 30, 30, 34], start=1):
            ws2.column_dimensions[chr(64 + col_idx)].width = width
        thin = Side(style='thin', color='000000')
        for c in ws2[1]:
            c.font = Font(bold=True)
            c.alignment = Alignment(horizontal='left' if c.column == 1 else 'center', wrap_text=True)
            c.border = Border(top=thin, bottom=thin)
        for i, (_, style) in enumerate(t2_rows, start=2):
            for c in ws2[i]:
                c.alignment = Alignment(horizontal='left' if c.column == 1 else 'center')
                if style in ('header', 'total'):
                    c.font = Font(bold=True)
                if style == 'total':
                    c.border = Border(top=thin, bottom=thin)
        last = len(t2_rows) + 2
        for k, note in enumerate(t2_notes):
            ws2.cell(row=last + 1 + k, column=1, value=note).font = Font(italic=True, size=9)

        if 'oat_by_modality' in results:
            names = {'meth': 'Methadone', 'subx': 'SL-BPN', 'subl': 'LAI-BPN'}
            f_int = lambda x: 'n/a' if np.isnan(x) else f'{int(round(x)):,}'
            f_pct = lambda x: 'n/a' if np.isnan(x) else f'{x:.0f}%'
            f_usd = lambda x: 'n/a' if np.isnan(x) else '${:,.0f}'.format(x)

            def cell_(d, f):
                return f"{f(d['pe'])} (95% UI: {f(d['lb'])} to {f(d['ub'])})"

            n_years_ = results.get('n_years', 10)

            def avg_share(d_py, d_share):
                # average number of people on the modality (person-years / years) with its share of all OAMT person-years
                return f"{f_int(d_py['pe'] / n_years_)} ({f_pct(d_share['pe'])})"

            mrows = []
            for key, nm in names.items():
                b = results['oat_by_modality']['Baseline'][key]
                s_ = results['oat_by_modality']['Scenario'][key]
                mrows.append({
                    'Modality': nm,
                    f'Average people on OAMT (share), {baseline_name}': avg_share(b['person_years'], b['share']),
                    f'Average people on OAMT (share), {scenario_name}': avg_share(s_['person_years'], s_['share']),
                    'Delivery cost per person-year (annual)': cell_(b['unit_cost'], f_usd),
                })
            pd.DataFrame(mrows).to_excel(writer, sheet_name='OAMT by modality', index=False)
            ws5 = writer.sheets['OAMT by modality']
            for col_idx, width in enumerate([14, 36, 36, 34], start=1):
                ws5.column_dimensions[chr(64 + col_idx)].width = width
            ws5.cell(row=len(mrows) + 3, column=1, value=(
                'Average people on OAMT = person-years over the analysis period divided by the number of years; the share is the percentage of all OAMT person-years. Delivery cost per person-year is the annual cost of medication, nurse time and '
                'correctional officer time for one person on the modality; it is the same in both scenarios (it excludes one-off '
                'initiation nursing time and discounting). Values are medians with 95% UI across simulations.')).font = Font(italic=True, size=9)

        if 'postrelease_abs' in results:
            # QALYs lost (positive numbers) per scenario, and the additional QALYs lost under the policy change
            pr_q = results['postrelease_abs']
            comps = [('Deaths', 'death'), ('HCV morbidity after release (12 months)', 'morb_post'),
                     ('HCV morbidity in prison', 'morb_inprison')]
            def _loss(key, arm):
                return -np.asarray(pr_q[f'qalys_{key}_{arm}']['vals'], dtype=float)
            arrs = {}
            for label_, key in comps:
                arrs[label_] = (_loss(key, 'baseline'), _loss(key, 'scenario'))
            arrs['Total QALYs lost'] = tuple(sum(arrs[l][j] for l, _ in comps) for j in (0, 1))
            fq = _t2_formatter('count')
            q_rows = []
            for label_, (b_, s_) in arrs.items():
                n_ = min(len(b_), len(s_))
                b_, s_ = b_[:n_], s_[:n_]
                d_ = bootstrap_median(s_ - b_)
                q_rows.append({
                    'QALYs lost': label_,
                    baseline_name: f"{fq(np.median(b_))} (95% UI: {fq(np.percentile(b_, 2.5))} to {fq(np.percentile(b_, 97.5))})",
                    scenario_name: f"{fq(np.median(s_))} (95% UI: {fq(np.percentile(s_, 2.5))} to {fq(np.percentile(s_, 97.5))})",
                    f'Additional QALYs lost under {scenario_name.lower()} (95% CI)': f"{fq(d_['pe'])} (95% CI: {fq(d_['lb'])} to {fq(d_['ub'])})",
                })
            pd.DataFrame(q_rows).to_excel(writer, sheet_name='QALYs', index=False)
            ws7 = writer.sheets['QALYs']
            for col_idx, width in enumerate([42, 34, 34, 50], start=1):
                ws7.column_dimensions[chr(64 + col_idx)].width = width
            ws7.cell(row=len(q_rows) + 3, column=1, value=(
                'QALYs lost are positive numbers. Deaths: utility x discounted remaining life expectancy of each death. HCV morbidity: '
                'person-years with HCV x the utility decrement between HCV-negative and HCV-positive states (after release, '
                'within 12 months; in prison, over the full period). Values are medians with 95% UI across simulations; the difference is the '
                'median of the paired differences with the bootstrap 95% CI of the median. Discounted at 1.5%.')).font = Font(italic=True, size=9)

        if 'cost_offset' in results:
            co = results['cost_offset']
            off_rows = [
                ['Savings in OAMT delivery and community OAMT medication', _t2_labelled(co['savings'], 'cost', 'ci_ui')],
                ['Increase in HCV, IRI and reincarceration costs', _t2_labelled(co['extra'], 'cost', 'ci_ui')],
                ['Net saving (minus the total cost difference)', _t2_labelled(co['net'], 'cost', 'ci_ui')],
            ]
            f1 = lambda x: f'{x:.1f}'.replace('.', '\u00b7')
            v = np.asarray(co['offset_pct']['vals'], dtype=float)
            ui = (np.percentile(v, 2.5), np.percentile(v, 97.5))
            off_rows.append(['Share of the savings offset by the increases (%)',
                             f"{f1(co['offset_pct']['pe'])}% (95% CI: {f1(co['offset_pct']['lb'])}% to {f1(co['offset_pct']['ub'])}%; "
                             f"95% UI: {f1(ui[0])}% to {f1(ui[1])}%)"])
            pd.DataFrame(off_rows, columns=['', f'{scenario_name} minus {baseline_name}']).to_excel(
                writer, sheet_name='Cost offset', index=False)
            ws6 = writer.sheets['Cost offset']
            ws6.column_dimensions['A'].width = 62
            ws6.column_dimensions['B'].width = 70
            ws6.cell(row=len(off_rows) + 3, column=1, value=(
                'Computed for each simulation from the paired cost differences, then summarised (median, bootstrap 95% CI of the median, '
                '95% UI of the paired values). Savings = reduction in in-prison OAMT delivery and community OAMT medication; '
                'increases = IRI, HCV (prison and community) and reincarceration costs.')).font = Font(italic=True, size=9)

    print(f"Saved to {file_name}")


def write_lean_sensitivity(named_results, file_path='results/sensitivity_lean.xlsx', n_sens=None, n_main=None):
    """
    One table for the lean sensitivity analyses: one column per analysis, medians only (the analyses are run
    with a small number of simulations, so intervals are not reported). Each cell is the median difference
    (policy change - counterfactual) from calculate_all_outcomes for that analysis.

    :param named_results: dict {column name: results dict from calculate_all_outcomes}
    """
    # Each cell: the median difference (policy change - counterfactual) for that analysis and, in brackets, its relative
    # difference from the same quantity in the first analysis (the reference, shown without brackets), e.g. '80 (−18%)'.
    ref_res = next(iter(named_results.values()))

    def _cell(res, key, kind):
        if not res.get(key):
            return ''
        d = res[key]['pe']
        text = _t2_formatter(kind)(d)
        if res is ref_res or not ref_res.get(key):
            return text
        r = ref_res[key]['pe']
        if abs(r) < 1e-9:
            return text
        pct = 100 * (d - r) / abs(r)
        pct_text = '{:+.0f}'.format(pct) if abs(pct) >= 10 else '{:+.1f}'.format(pct)
        if pct_text in ('+0.0', '-0.0'):
            pct_text = '0'
        return '{} ({}%)'.format(text, pct_text.replace('-', '−').replace('.', '·'))

    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # One row per analysis, one column per outcome (difference, policy change - counterfactual)
        tbl_specs = [
            ('Average people on OAMT in prison', 'diff_pop_oat_inprison', 'count'),
            ('HCV infections (prison and community)', 'diff_hcv_total', 'count'),
            ('IRIs (prison and community)', 'diff_iri_total', 'count'),
            ('Reincarceration events after release', 'diff_reinc', 'count'),
            ('Drug-related deaths after release', 'diff_deaths', 'count'),
            ('Cost of OAMT', 'diff_cost_intervention', 'cost'),
            ('Financial consequences', 'diff_cost_fin', 'cost'),
            ('Net cost', 'diff_total_costs', 'cost'),
        ]
        tbl_rows = []
        for name, res in named_results.items():
            row = {'Analysis': name}
            for label, key, kind in tbl_specs:
                row[label] = _cell(res, key, kind)
            tbl_rows.append(row)
        pd.DataFrame(tbl_rows).to_excel(writer, sheet_name='Sensitivity analyses', index=False)
        ws2 = writer.sheets['Sensitivity analyses']
        ws2.column_dimensions['A'].width = 52
        for col_idx in range(2, len(tbl_specs) + 2):
            ws2.column_dimensions[chr(64 + col_idx)].width = 26
        ws2.cell(row=len(tbl_rows) + 3, column=1, value=(
            'Difference (policy change minus counterfactual), median of the paired differences over {} simulations; in brackets, '
            'the relative difference from the main analysis (first row); intervals are not reported. Costs are in 2025 '
            'Canadian dollars, discounted at 1.5% to 2026.'
            + (' The main analysis is the full run of {:,} simulations.'.format(n_main) if n_main else '')).format(n_sens if n_sens else 'a small number of'))
    print('Saved lean sensitivity table: {}'.format(file_path))


def run_scen(P, scenario_name, *args):
    '''
    Run a scenario by changing parameter values (non-stochastic only, for optimisation purposes)
    :param P: atomica project
    :param scenario_name: name of scenario
    :param *args: a set of three variables to change: the parameter code name,
    the implementation year and the new parameter value, in that order.
    Multiple {par_code, year, value} sets can be specified
    
    :return: simulation result
    
    :example: changing the value of parameter 'nsp_cov' to 0.5 starting 2023, and 'nsp_cst' to 0.9 starting 2023.
    result_nsp_50 = run_scen(P, 'NSP 50%-90%', 'nsp_cov', 2023, 0.5, 'nsp_cst', 2023, 0.9)
    '''
    if (len(args) % 3 != 0): 
        raise Exception('Length of *args must be a multiple of 3 (par_code, year, value)')
        
    P.settings.multinomial = False
    pop = P.data.pops.keys() # define population
    i = 0
    scen = at.ParameterScenario(name=scenario_name)
    
    if (len(args) > 0):
        while i < len(args)-1:
            scen.add(args[i],pop[0],args[i+1],args[i+2])
            i += 3
    parset = P.make_parset()
    result = scen.run(P,parset)
        
    return result


SENTENCE_WEIGHT_YEAR = 2021  # years at/after this (the post-switch tail) matter most - upweight them


SENTENCE_WEIGHT = 5


def objective_sentence(params, P, scen_name, knot_years, groups, *args):
    """
    Objective function to calculate the sum of the squared differences
    between the model outcomes and the data points for the HepC incidence
    Works on non-stochastic model
    :param: all input parameters to the run_scenario function
    :return: sum of the squared differences
    """
    sen_t, sen_v = _step_series(knot_years, [params[g] for g in groups])
    result = run_scen(P, scen_name, 'sentence_len', sen_t, sen_v, *args)
    pop = result.pop_names[0]

    # Sentence length
    par = 'pop_prison'
    years = np.array(P.data.tdve['pop_prison'].ts['Prisoners'].t)
    data = np.array(P.data.tdve['pop_prison'].ts['Prisoners'].vals)
    t = result.get_variable(par, pop)[0].t
    model_outcomes = []
    for year in years:
        index = bisect.bisect_left(t, year)
        model_outcomes.append(result.get_variable(par, pop)[0].vals[index-1])

    mask = years > 2010
    model_outcomes = np.array(model_outcomes)
    # Unweighted, this objective let the optimiser trade away the post-2021 fit (the region
    # that actually matters - it's what's visible/reported) for a marginally better fit in
    # 2011-2020, since a plain sum of squares treats every year equally. Upweight the tail.
    weights = np.where(years[mask] >= SENTENCE_WEIGHT_YEAR, SENTENCE_WEIGHT, 1.0)
    error = np.sum(weights * (model_outcomes[mask] - data[mask]) ** 2)

    return error


def objective_iri(params, P, scen_name, *args):
    """
    Objective function to calculate the sum of the squared differences 
    between the model outcomes and the data points for the HepC incidence
    Works on non-stochastic model
    :param: all input parameters to the run_scenario function
    :return: sum of the squared differences
    """
    iri = params
    result = run_scen(P, scen_name, 'iri', 2010, iri, *args)
    step_size = result.dt 
    pop = result.pop_names[0]
    
    # IRI incidence data
    par = 'iri_idu_per100py'
    years = P.data.tdve['iri_inci'].ts['Prisoners'].t
    data = np.array(P.data.tdve['iri_inci'].ts['Prisoners'].vals)
    model_outcomes = []
    t = result.get_variable(par, pop)[0].t
    
    for year in years:
        t_range = (t >= year) & (t < year+1)
        model_outcomes.append(sum(result.get_variable(par, pop)[0].vals[t_range])*step_size)
        
    model_outcomes = np.array(model_outcomes)
    error = np.sum((model_outcomes-data)**2)
    
    return error


def objective_infx(params, P, scen_name, *args):
    """
    Objective function to calculate the sum of the squared differences 
    between the model outcomes and the data points for the HepC incidence
    Works on non-stochastic model
    :param: all input parameters to the run_scenario function
    :return: sum of the squared differences
    """
    infx = params
    result = run_scen(P, scen_name, 'infx', 2010, infx, *args)
    step_size = result.dt
    pop = result.pop_names[0]

    # HCV incidence data
    par = 'inci_per100py'
    years = P.data.tdve['hcv_inci'].ts['Prisoners'].t
    data = np.array(P.data.tdve['hcv_inci'].ts['Prisoners'].vals)
    model_outcomes = []
    t = result.get_variable(par, pop)[0].t

    for year in years:
        t_range = (t >= year) & (t < year+1)
        model_outcomes.append(sum(result.get_variable(par, pop)[0].vals[t_range])*step_size)

    model_outcomes = np.array(model_outcomes)
    error = np.sum((model_outcomes-data)**2)

    # HCV prevalence data
    par = 'prev'
    years = P.data.tdve['prev'].ts['Prisoners'].t
    data = np.array(P.data.tdve['prev'].ts['Prisoners'].vals)
    model_outcomes = []
    t = result.get_variable(par, pop)[0].t

    for year in years:
        t_range = (t >= year) & (t < year+1)
        model_outcomes.append(sum(result.get_variable(par, pop)[0].vals[t_range])*step_size)

    model_outcomes = np.array(model_outcomes)
    error += np.sum((model_outcomes-data)**2)

    return error


def _match(P, result, par, pop):
    """
    Build (years, data, model) triples for every databook time point of `par`, at full
    resolution (including the quarterly points) - see _match_weights for how those get
    fairly weighted without collapsing/averaging them away.

    The model value follows the resolution of the data: a calendar year with a single point
    is an annual mean, so it's compared with the model's mean over that year; a year with
    several points (the quarterly data) is compared point by point with the model value at
    that time.
    """
    years = list(P.data.tdve[par].ts['Prisoners'].t)
    data = list(P.data.tdve[par].ts['Prisoners'].vals)
    series = result.get_variable(par, pop)[0]
    t = np.asarray(series.t)
    vals = np.asarray(series.vals)
    points_per_year = collections.Counter(int(y) for y in years)
    model_outcomes = []
    for year in years:
        if points_per_year[int(year)] == 1:
            model_outcomes.append(vals[(t >= year) & (t < year + 1)].mean())
        else:
            model_outcomes.append(vals[np.argmin(np.abs(t - year))])

    return years, data, model_outcomes


PEAK_WEIGHT_YEARS = (2023, 2026.5)  # extra-weighted range: the LAI-BPN rollout and its lead-up


PEAK_WEIGHT = 5  # how much more each year in that range counts, relative to 1.0 elsewhere


PLATEAU_WEIGHT_YEARS = (2025.0, 2026.75)  # Dec 2024 - Jul 2025 point-in-time counts: the stall in OAMT growth


PLATEAU_WEIGHT = 5  # weight of each of those points (NOT shared across the calendar year), so the fit


def _match_weights(years):
    # Divide by how many points share a calendar year, so 2026's 3 quarterly points sum to the
    # same total weight as any other single year, without discarding the shape between them.
    counts = collections.Counter(int(y) for y in years)
    w = np.array([
        (PEAK_WEIGHT if PEAK_WEIGHT_YEARS[0] <= y <= PEAK_WEIGHT_YEARS[1] else 1.0) / counts[int(y)]
        for y in years
    ])
    years_arr = np.asarray(years, dtype=float)
    in_plateau = (years_arr >= PLATEAU_WEIGHT_YEARS[0]) & (years_arr <= PLATEAU_WEIGHT_YEARS[1])
    w[in_plateau] = PLATEAU_WEIGHT
    return w


SENTENCE_POLISH_MARGIN = 0.15  # +/- fraction around whatever is currently in the databook


def auto_calibration_sentence(P, scen_name, *args, margin=SENTENCE_POLISH_MARGIN, n_samples=500, rand_seed=20240430):
    """
    Small local polish around the databook-tuned sentence_len values, analogous to
    auto_calibration_uptake_polish - NOT a free wide-bounds search. A free search (bounds
    (2,6) from an arbitrary starting point) pulled the post-2021 knot away from the
    manually-tuned value (e.g. 3.542 -> 3.018) even though that value already gave a good
    fit, because objective_sentence's plain sum of squares was happy to trade away fit
    quality in the region that matters (post-2021) for a marginally better fit in
    2011-2020. Bounding each knot to +/-margin of its current databook value keeps the
    search in a small, plausible neighbourhood instead.
    :param: all input parameters to the run_scenario function
    :return: optimised infectiousness constant, calibrated ensemble results
    """
    knot_years, current_vals = _read_current_series(P, 'sentence_len')
    groups = _tie_successive_equal(current_vals)  # identical successive databook values calibrate as one
    x0 = [current_vals[groups.index(g)] for g in range(max(groups) + 1)]
    bounds = [(v * (1 - margin), v * (1 + margin)) for v in x0]

    res_optim = minimize(objective_sentence, x0, args=(P, scen_name, knot_years, groups, *args), bounds=bounds)
    sen_optim = [float(res_optim.x[g]) for g in groups]

    print('Scenario: {}\nOptimised sentence length at {}: {}'.format(scen_name, knot_years, [round(v, 3) for v in sen_optim]))
    P.settings.multinomial = True
    sen_t, sen_v = _step_series(knot_years, sen_optim)
    ensemble_calibration, P_calibration = run_scenario(P, scen_name, 'sentence_len', sen_t, sen_v, *args, n_samples=n_samples, rand_seed=rand_seed)

    return ensemble_calibration, P_calibration


def _step_series(t, v, eps=0.25):
    t_full, v_full = [t[0]], [v[0]]
    for i in range(1, len(t)):
        t_full.append(t[i] - eps)
        v_full.append(v[i - 1])
        t_full.append(t[i])
        v_full.append(v[i])
    return t_full, v_full


UPTAKE_POLISH_PARS = ['uptake_cst_meth', 'uptake_cst_subx', 'uptake_cst_subl']#, 'prop_subx_subl_success']


UPTAKE_POLISH_MARGIN = 0.1  # default +/- margin around whatever is currently in the databook


UPTAKE_POLISH_MARGINS = {
    'uptake_cst_meth': 0.1,  # methadone was already well fitted by hand, so it gets less room to move
}


def _read_current_series(P, par, pop='Prisoners'):
    ts = P.data.tdve[par].ts[pop]
    return list(ts.t), list(ts.vals)


def objective_uptake_polish(params, P, scen_name, knot_structure, *args):
    """
    Small local polish around the manually tuned databook values for uptake_cst_meth/subx/subl.
    Wide-bounds searches on these knots landed on implausible values (e.g. a switch success of 0),
    so each knot is bounded to +/-UPTAKE_POLISH_MARGIN of its current databook value instead.
    The knots are read from the databook, so the number of time points can differ per parameter.
    """
    idx = 0
    overrides = []
    for par_name, t_list, groups in knot_structure:
        n_groups = max(groups) + 1
        vals = [params[idx + g] for g in groups]
        overrides += [par_name, t_list, vals]
        idx += n_groups

    result = run_scen(P, scen_name, *overrides, *args)
    pop = result.pop_names[0]

    total_error = 0.0
    for par in ('pop_oat_meth', 'pop_oat_subx', 'pop_oat_subl', 'pop_oat'):
        years, d, m = _match(P, result, par, pop)
        d = np.array(d)
        m = np.array(m)
        scale = d.max() if d.max() > 0 else 1.0
        w = _match_weights(years)
        total_error += np.sum(w * ((m - d) / scale) ** 2)

    return total_error


def _tie_successive_equal(v_list, tol=1e-9):
    # Group index per knot: successive knots with identical databook values share one group, so
    # they're polished as a single value and stay equal.
    groups = [0]
    for prev, cur in zip(v_list[:-1], v_list[1:]):
        groups.append(groups[-1] if abs(cur - prev) < tol else groups[-1] + 1)
    return groups


def auto_calibration_uptake_polish(P, scen_name, *args, margin=UPTAKE_POLISH_MARGIN, margins=UPTAKE_POLISH_MARGINS, n_samples=500, rand_seed=20240430):
    knot_structure = []
    x0 = []
    bounds = []
    for par in UPTAKE_POLISH_PARS:
        par_margin = margins.get(par, margin)
        t_list, v_list = _read_current_series(P, par)
        groups = _tie_successive_equal(v_list)
        knot_structure.append((par, t_list, groups))
        for g in range(max(groups) + 1):
            v = v_list[groups.index(g)]
            x0.append(v)
            if v == 0:
                bounds.append((0, 0))  # structural zero (e.g. modality phased out) - not polished
            else:
                lo, hi = v * (1 - par_margin), v * (1 + par_margin)
                bounds.append((min(lo, hi), max(lo, hi)))

    res_optim = minimize(objective_uptake_polish, x0, args=(P, scen_name, knot_structure, *args), bounds=bounds)

    print('Scenario: {}'.format(scen_name))
    idx = 0
    final_overrides = []
    for par, t_list, groups in knot_structure:
        n_groups = max(groups) + 1
        before = [x0[idx + g] for g in groups]
        after = [float(res_optim.x[idx + g]) for g in groups]
        idx += n_groups
        print('  {} at {}: {} (was {})'.format(par, t_list, [round(v, 4) for v in after], [round(v, 4) for v in before]))
        final_overrides += [par, t_list, after]
    print('Final error: {:.4f}'.format(res_optim.fun))

    P.settings.multinomial = True
    ensemble_calibration, P_calibration = run_scenario(P, scen_name, *final_overrides, *args, n_samples=n_samples, rand_seed=rand_seed)
    return ensemble_calibration, P_calibration, res_optim.fun


def auto_calibration_infx(P, scen_name, *args, n_samples=500, rand_seed=20240430):
    """
    Auto-calibration by changing the infectiousness and IRI parameters
    :param: all input parameters to the run_scenario function
    :return: optimised infectiousness constant, calibrated ensemble results
    """
    infx_baseline=0.2
    res_optim = minimize(objective_infx, (infx_baseline), args = (P, scen_name, *args), bounds=((0, 1),))
    infx_optim = res_optim.x[0]
    print('Scenario: {}\nOptimised infectiousness constant: {:.2f}'.format(scen_name,infx_optim))
    P.settings.multinomial = True
    ensemble_calibration, P_calibration = run_scenario(P, scen_name, 'infx', 2015, infx_optim, *args, n_samples=n_samples, rand_seed=rand_seed)

    return ensemble_calibration, P_calibration


def auto_calibration_iri(P, scen_name, *args, n_samples=500, rand_seed=20240430):
    """
    Auto-calibration by changing the infectiousness and IRI parameters
    :param: all input parameters to the run_scenario function
    :return: optimised infectiousness constant, calibrated ensemble results
    """
    iri_baseline=3.1/100
    res_optim = minimize(objective_iri, (iri_baseline), args = (P, scen_name, *args), bounds=((0, 5),))
    iri_optim = res_optim.x[0]
    print('Scenario: {}\nOptimised IRI: {:.2f}'.format(scen_name,iri_optim))
    P.settings.multinomial = True
    ensemble_calibration, P_calibration = run_scenario(P, scen_name, 'iri', 2015, iri_optim, *args, n_samples=n_samples, rand_seed=rand_seed)     
    
    return ensemble_calibration, P_calibration


def auto_calibration(P, scen_name, *args, n_samples=500, rand_seed=20240430):
    """
    Calibration in four steps: sentence length, IRI and the infectiousness constant / max treatment rate
    are fitted with minimize(). The methadone, SL-BPN and LAI-BPN uptake constants are tuned by hand in
    the databook and only polished within a small margin (auto_calibration_uptake_polish).
    """
    # Only the project is carried forward from the first three steps, so they run a single sample;
    # the last step uses n_samples
    _, P_cal = auto_calibration_sentence(P, scen_name, *args, n_samples=1, rand_seed=rand_seed)
    _, P_cal = auto_calibration_iri(P_cal, scen_name, *args, n_samples=1, rand_seed=rand_seed)
    _, P_cal = auto_calibration_infx(P_cal, scen_name, *args, n_samples=1, rand_seed=rand_seed)
    ensemble_calibration, P_calibration, _ = auto_calibration_uptake_polish(
        P_cal, scen_name, *args, n_samples=n_samples, rand_seed=rand_seed)

    return ensemble_calibration, P_calibration


def sensitivity_equal_coverage(P, P_counter, res_baseline, res_counter,
                                start_year, end_year, n_samples=1000, rand_seed=20240430,
                                coverage_par='pop_oat', k_bounds=None, safe_uptake_ceiling=0.6, extra_overrides=None):
    """
    Equal OAMT coverage scenario. Separates the effect of the LAI-BPN formulation change from the
    effect of the lower OAMT coverage that comes with it: LAI-BPN stays first-line but in-prison
    uptake is scaled up (same multiplier on the methadone, SL-BPN and LAI-BPN uptake constants, so the
    mix between modalities is kept) until the number on OAMT matches the counterfactual.

    :param k_bounds: bounds for the uptake multiplier. If None, the upper bound is the largest k that
        keeps the combined uptake probability (meth + subx + subl) below safe_uptake_ceiling, since
        the three are probabilities in a multinomial split and their sum cannot exceed 1.
    """
    pop = 'Prisoners'
    extra_overrides = list(extra_overrides or [])   # e.g. ['prop_subx_subl_success', [2015, 2035], [1.0, 1.0]]

    # Deterministic counterfactual trajectory to match against
    result_target = run_scen(P_counter, 'Counterfactual (target, deterministic)')
    t_target = np.array(result_target.get_variable(coverage_par, pop)[0].t)
    v_target = np.array(result_target.get_variable(coverage_par, pop)[0].vals)
    mask = (t_target >= start_year) & (t_target <= end_year)

    uptake_pars = ['uptake_cst_meth', 'uptake_cst_subx', 'uptake_cst_subl']
    knot_structure = [(par,) + tuple(_read_current_series(P, par)) for par in uptake_pars]

    def scaled_series(t_list, v_list, k):
        # Scale the uptake constant by k from start_year onward only, so the calibrated history is unchanged
        t_arr, v_arr = np.array(t_list, dtype=float), np.array(v_list, dtype=float)
        t_new = [float(ti) for ti in t_arr if ti < start_year - 0.25]
        v_new = [float(vi) for ti, vi in zip(t_arr, v_arr) if ti < start_year - 0.25]
        t_new.append(start_year - 0.25)
        v_new.append(float(np.interp(start_year - 0.25, t_arr, v_arr)))
        t_new.append(float(start_year))
        v_new.append(float(np.interp(start_year, t_arr, v_arr)) * k)
        for ti, vi in zip(t_arr, v_arr):
            if ti > start_year:
                t_new.append(float(ti))
                v_new.append(float(vi) * k)
        return t_new, v_new

    if k_bounds is None:
        result_k1 = run_scen(P, 'Equal coverage (k=1 safety check)')
        combined_uptake = (np.array(result_k1.get_variable('meth_uptake', pop)[0].vals)
                            + np.array(result_k1.get_variable('subx_uptake', pop)[0].vals)
                            + np.array(result_k1.get_variable('subl_uptake', pop)[0].vals))
        combined_max = float(combined_uptake.max())
        safe_k_max = safe_uptake_ceiling / combined_max if combined_max > 0 else 6.0
        k_bounds = (0.5, min(6.0, safe_k_max))
        print('Equal coverage scenario: dynamic k upper bound = {:.3f} '
              '(keeps combined uptake probability <= {} at k=1 baseline)'.format(k_bounds[1], safe_uptake_ceiling))

    def objective(k):
        k = k[0]
        overrides = []
        for par, t_list, v_list in knot_structure:
            overrides += [par] + list(scaled_series(t_list, v_list, k))
        overrides += extra_overrides
        result = run_scen(P, 'Equal coverage (trial)', *overrides)
        v_cand = np.array(result.get_variable(coverage_par, pop)[0].vals)
        return float(np.sum((v_cand[mask] - v_target[mask]) ** 2))

    res_optim = minimize(objective, [1.0], bounds=[k_bounds])
    k_optim = res_optim.x[0]
    print('Equal coverage scenario: uptake multiplier k={:.3f} (fit error={:.4f})'.format(k_optim, res_optim.fun))

    # A stochastic draw can push the combined uptake probability above 1 even when the deterministic fit is valid.
    # If the ensemble fails, lower k by 10% and retry
    P.settings.multinomial = True   # run_scen leaves it False
    res_equal = P_equal = None
    for attempt in range(8):
        final_overrides = []
        for par, t_list, v_list in knot_structure:
            final_overrides += [par] + list(scaled_series(t_list, v_list, k_optim))
        final_overrides += extra_overrides
        try:
            res_equal, P_equal = run_scenario(P, 'Equal coverage (LAI-BUP first-line, backfilled uptake)',
                                               *final_overrides, n_samples=n_samples, rand_seed=rand_seed)
            break
        except Exception as e:
            print('Equal coverage: ensemble failed at k={:.3f} ({}); retrying with k={:.3f}'.format(k_optim, e, k_optim * 0.9))
            k_optim *= 0.9
    if res_equal is None:
        raise RuntimeError('Equal coverage scenario: ensemble failed for every k tried (last k={:.3f})'.format(k_optim))
    print('Equal coverage scenario: final uptake multiplier k={:.3f}'.format(k_optim))

    return {'ensemble': res_equal, 'P': P_equal, 'k': k_optim, 'fit_error': res_optim.fun}


def sensitivity_lai_bup_continuity(economic_pars_path, match_to='subx'):
    """
    Optimised LAI-BUP + community continuity scenario: builds the modified economic-pars
    table (LAI-BUP's post-release retention matched to `match_to`'s) only.
    calculate_all_outcomes for it is called in run_main.py, not here.
    """
    df_economic_pars = pd.read_excel(economic_pars_path, index_col=0)
    df_pars_continuity = df_economic_pars.copy()
    for cp in ['1mo', '6mo', '12mo']:
        df_pars_continuity.loc[f'retain_subl_{cp}', ['point_estimate', 'lower_bound', 'upper_bound']] = \
            df_economic_pars.loc[f'retain_{match_to}_{cp}', ['point_estimate', 'lower_bound', 'upper_bound']].values

    return df_pars_continuity


def sensitivity_coverage_co_staffing(economic_pars_path, co_number=1.0):
    """
    Build a modified economic-pars table with `co_number` (COs needed per OAT
    administration, currently point estimate 2) overridden to a fixed value -
    e.g. 1 instead of 2 - leaving co_time_meth/subx/subl untouched.
    """
    if isinstance(economic_pars_path, str):
        df = pd.read_excel(economic_pars_path, index_col=0)
    else:
        df = economic_pars_path
    df = df.copy()
    df.loc['co_number', ['point_estimate', 'lower_bound', 'upper_bound']] = co_number
    return df


def _collapse_periods(t, v, tol=1e-6):
    """
    Collapse a (possibly step-encoded) time series into distinct (start_year, value)
    periods, dropping consecutive points that share the same value. This recovers the
    original calibration knots even when the series has been step-encoded with tiny
    epsilon-offset points (see `_step_series`) to force staircase rather than linear
    interpolation - e.g. [(2010,v0),(2014.99,v0),(2015,v1)] collapses to just
    [(2010,v0),(2015,v1)].
    """
    periods = []
    i = 0
    n = len(t)
    while i < n:
        j = i
        while j + 1 < n and abs(v[j + 1] - v[i]) < tol:
            j += 1
        periods.append((t[i], v[i]))
        i = j + 1
    return periods


def export_calibration_table(P, filename='results/calibrated_parameters.xlsx'):
    """
    Write an Excel table of every parameter set by auto_calibration() (plus the
    policy-driven, non-calibrated new-entrant modality split), with real parameter
    names/formats, for direct use in a manuscript methods/supplementary table.

    Values are read directly from the final calibrated P.data, not from hardcoded
    numbers, so this always reflects whatever auto_calibration() actually produced -
    however the knot years or fitted values change in future runs.

    :param P: The calibrated atomica Project returned by auto_calibration()
    :param filename: Output .xlsx path
    """
    F = P.framework
    pop = list(P.data.pops.keys())[0]

    def display(code):
        return F.pars.at[code, 'display name'] if code in F.pars.index else code

    def fmt(code):
        return F.pars.at[code, 'format'] if code in F.pars.index else ''

    def interp_at(code, year):
        return float(P.data.tdve[code].ts[pop].interpolate([year])[0])

    rows = []

    def _series_str(years, vals):
        return '; '.join('{:g}: {:.4g}'.format(y, v) for y, v in zip(years, vals))

    def add_param(category, code, note=''):
        ts = P.data.tdve[code].ts[pop]
        periods = _collapse_periods(list(ts.t), list(ts.vals))
        years = [y for y, _ in periods]
        vals = [v for _, v in periods]
        rows.append({
            'Category': category,
            'Parameter': display(code),
            'Code name': code,
            'Value': _series_str(years, vals),
            'Format': fmt(code),
            'Notes': note,
        })

    def add_constant(category, code, note=''):
        # Single constant (no year) - falls back to the full series if it isn't actually constant.
        ts = P.data.tdve[code].ts[pop]
        periods = _collapse_periods(list(ts.t), list(ts.vals))
        value = '{:.4g}'.format(periods[0][1]) if len(periods) == 1 else _series_str(*zip(*periods))
        rows.append({
            'Category': category,
            'Parameter': display(code),
            'Code name': code,
            'Value': value,
            'Format': fmt(code),
            'Notes': note,
        })

    def add_value_at(category, code, year, note=''):
        rows.append({
            'Category': category,
            'Parameter': display(code),
            'Code name': code,
            'Value': _series_str([year], [interp_at(code, year)]),
            'Format': fmt(code),
            'Notes': note,
        })

    # Successive knots with identical values were calibrated as one value (see
    # _tie_successive_equal), so report them as one range, e.g. '2010-2018: 0.3'.
    # stepped=True for series written via _step_series (sentence_len), whose last point of each
    # period sits at the next knot minus 0.25 (e.g. 2020.75) - report that as the period's last year.
    def add_ranges(category, code, note='', stepped=False):
        ts = P.data.tdve[code].ts[pop]
        years, vals = list(ts.t), list(ts.vals)
        groups = _tie_successive_equal(vals)
        parts = []
        for g in range(max(groups) + 1):
            idx = [i for i, gi in enumerate(groups) if gi == g]
            first, last = years[idx[0]], years[idx[-1]]
            if stepped:
                last = math.floor(last)
            label = '{:g}'.format(first) if first == last else '{:g}-{:g}'.format(first, last)
            parts.append('{}: {:.4g}'.format(label, vals[idx[0]]))
        rows.append({
            'Category': category,
            'Parameter': display(code),
            'Code name': code,
            'Value': '; '.join(parts),
            'Format': fmt(code),
            'Notes': note,
        })

    uptake_cat = 'OAT uptake (in-prison initiation, not yet on any OAT)'
    uptake_note = ('Manually tuned in the databook, then auto-polished within a small margin (see '
                   'auto_calibration_uptake_polish); identical successive values calibrated as one. '
                   'Initiation probability = prop_pwid_oat * prop_oat_x * dropout * this constant. The constant is a '
                   'calibration scalar applied on top of the modality share and dropout rate, so it is not comparable '
                   'between modalities; see the effective initiation rate column.')

    add_ranges('Sentence length', 'sentence_len',
               'Auto-calibrated against incarcerated population size (panel a); identical successive databook values calibrated as one',
               stepped=True)
    add_ranges(uptake_cat, 'uptake_cst_meth', uptake_note + ' Reflects CSC first-line policy eras.')
    add_ranges(uptake_cat, 'uptake_cst_subx', uptake_note)
    add_ranges(uptake_cat, 'uptake_cst_subl', uptake_note)
    add_param('Suboxone-to-sublocade switch attrition', 'prop_subx_subl_success',
              'Proportion of switched patients who successfully initiate sublocade rather than dropping off OAT. '
              'Calibrated manually (not part of the automated routine) to the observed fall in the OAMT population.')
    add_constant('IRI incidence', 'iri',
                 'Auto-calibrated against IRI incidence data (panel d)')
    add_constant('HCV transmission coefficient', 'infx',
                 'Auto-calibrated against in-prison HCV incidence and prevalence (panels e-f). Annual probability that a '
                 'susceptible person who injects (not on OAMT) is infected = coefficient x HCV prevalence among people who inject '
                 'in prison x (1 - NSP coverage x NSP effect); people on OAMT have it reduced by their OAMT effect on injecting.')
    # prop_entry_oat_meth/subx/subl (Oct-2025 new-entrant modality split) deliberately excluded -
    # policy-driven databook assumptions, not calibrated/tuned values.

    # Effective initiation rate behind each uptake constant, per 100 person-years not on OAMT:
    # prop_pwid_oat * modality share of OAMT * dropout (1 - retention) * constant (* uptake_mult for SL-BPN),
    # before (2024) and after (2027) the policy. This is the quantity that is comparable between modalities.
    EFF_COL = 'Effective initiation rate (per 100 person-years off OAMT)'

    def _constant(code):
        ts_c = P.data.tdve[code].ts[pop]
        return float(ts_c.vals[-1]) if len(ts_c.vals) else float(ts_c.assumption)

    def effective_initiation(modality, year):
        mult = interp_at('uptake_mult', year) if modality == 'subx' else 1.0
        return (100 * interp_at('prop_pwid_oat', year) * interp_at('prop_oat_' + modality, year)
                * (1 - _constant(modality + '_retention')) * interp_at('uptake_cst_' + modality, year) * mult)

    for row in rows:
        if row['Code name'].startswith('uptake_cst_'):
            modality = row['Code name'].split('_')[-1]
            row[EFF_COL] = '2024: {:.1f}; 2027: {:.1f}'.format(
                effective_initiation(modality, 2024), effective_initiation(modality, 2027))

    # Framework leaves 'format' blank for these two; set it here (not in the framework, which would invalidate the calibration cache)
    for row in rows:
        if row['Code name'] == 'iri':
            row['Format'] = 'probability (per year, person who injects, not on OAMT)'
        elif row['Code name'] == 'infx':
            row['Format'] = 'coefficient (annual probability when prevalence = 100%, no NSP)'

    # Transmission coefficient: name it as one and give an example of what it means at a typical prevalence
    for row in rows:
        if row['Code name'] == 'infx':
            row['Parameter'] = 'Transmission coefficient'
            try:
                v_infx = float(row['Value'])
                row['Notes'] += ' Example: at 20% prevalence among people who inject and no NSP, the annual infection probability is {:.1f}%.'.format(100 * v_infx * 0.2)
            except ValueError:
                pass

    # Uptake: report the effective initiation rate (comparable between modalities) in the main table instead of the calibration
    # constant; the constants themselves go to a separate sheet.
    uptake_const_rows = []
    modality_names = {'meth': 'Methadone', 'subx': 'SL-BPN (suboxone)', 'subl': 'LAI-BPN (sublocade)'}
    for row in rows:
        if row['Code name'].startswith('uptake_cst_'):
            modality = row['Code name'].split('_')[-1]
            uptake_const_rows.append({'Parameter': display(row['Code name']), 'Code name': row['Code name'],
                                      'Value': row['Value'], 'Notes': row['Notes']})
            row['Parameter'] = modality_names[modality] + ' initiation rate in prison'
            row['Value'] = row[EFF_COL]
            row['Code name'] = modality + '_uptake'
            row['Format'] = 'per 100 person-years off OAMT'
            row[EFF_COL] = ''
            row['Notes'] = ('Initiation of OAMT in prison among people not on OAMT = proportion of people with OUD entering on OAMT x '
                            'modality share of OAMT x dropout rate (1 - retention) x calibration constant. The constant was calibrated; '
                            'the rate shown is the resulting value before (2024) and after (2027) the policy change. Constants are in the '
                            '"Uptake constants" sheet.')
    df_uptake_const = pd.DataFrame(uptake_const_rows)

    df = pd.DataFrame(rows)
    df = df[['Category', 'Parameter', 'Code name', 'Value', 'Format', 'Notes']].fillna('')

    # Fixed (not calibrated) assumptions - values read from the project, not hardcoded.
    fixed_rows = []

    def add_fixed(parameter, code, value, fmt_, note):
        fixed_rows.append({'Parameter': parameter, 'Code name': code, 'Value': value, 'Format': fmt_, 'Notes': note})

    def fixed_value(code):
        ts = P.data.tdve[code].ts[pop]
        return float(ts.vals[-1]) if len(ts.vals) else float(ts.assumption)

    for modality, label, note in (
            ('meth', 'Methadone', 'Databook assumption.'),
            ('subx', 'SL-BPN (suboxone)', 'Databook assumption.'),
            ('subl', 'LAI-BPN (sublocade)', 'Ivasiy et al. 2025 (community outpatient retention, so likely understates in-prison retention).')):
        code = '{}_retention'.format(modality)
        add_fixed('{} in-prison retention'.format(label), code, '{:.4g}'.format(fixed_value(code)), fmt(code), note)

    treat_fn = str(P.framework.pars.at['prob_treat', 'function']) if 'prob_treat' in P.framework.pars.index else ''
    m = re.search(r'\(t>(\d+)\)\*([0-9.eE+-]+)', treat_fn)
    if m:
        add_fixed('HCV treatment rate after {}'.format(m.group(1)), 'prob_treat', '{:.4g}'.format(float(m.group(2))), fmt('prob_treat'),
                  'Held at the {0} rate: {0} treatments / observed infected people in {0} (prevalence x prison population).'.format(m.group(1)))

    oud_fn = str(P.framework.pars.at['prop_entry_oat', 'function']) if 'prop_entry_oat' in P.framework.pars.index else ''
    m = re.match(r'([0-9.]+)\*prop_pwid_oat', oud_fn)
    if m:
        add_fixed('Proportion of prison entrants with OUD', 'prop_oud_entry', m.group(1), 'proportion',
                  'Assumed (plausible range 0.2-0.4); no direct CSC source for OUD at intake.')

    ts = P.data.tdve['prev_pwid'].ts[pop]
    add_fixed(display('prev_pwid'), 'prev_pwid', _series_str([ts.t[-1]], [ts.vals[-1]]), fmt('prev_pwid'),
              'Last databook data point, held constant afterwards.')
    df_fixed = pd.DataFrame(fixed_rows)

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        workbook = writer.book
        header_fmt = workbook.add_format({'bold': True, 'bg_color': '#1F3A5F', 'font_color': 'white', 'border': 1, 'text_wrap': True})
        wrap_fmt = workbook.add_format({'text_wrap': True, 'valign': 'top'})
        widths = {'Category': 32, 'Parameter': 34, 'Code name': 20, 'Value': 40, 'Format': 12, 'Notes': 60, EFF_COL: 34}
        for sheet_name, frame in (('Calibrated parameters', df), ('Fixed assumptions', df_fixed), ('Uptake constants', df_uptake_const)):
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for col_idx, col in enumerate(frame.columns):
                worksheet.write(0, col_idx, col, header_fmt)
                worksheet.set_column(col_idx, col_idx, widths.get(col, 20), wrap_fmt)
            worksheet.freeze_panes(1, 0)

    print('Saved calibration parameter table: {}'.format(filename))
    return df
