import atomica.plotting as aplt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import utils_Canada as ut


def _save_fig(fig, fig_name, dpi=300):
    fig.savefig(fig_name, dpi=dpi, bbox_inches='tight')


def multi_plot(P, ensemble, ylimit: bool = False, fig_name='results/multi_plot.png'):
    '''
    Panel plot: 3 cols x 2 rows.
    Row 1: prison pop size | OAT pop size | prop on each OAT modality
    Row 2: IRI incidence | HCV incidence | HCV prevalence
    '''
    LW   = 2.5
    MS   = 120
    FM = 1.5 # font multiplier
    FS   = 20*FM    # base font pt
    FS_L = 22*FM    # axis label pt
    FS_T = 22*FM    # title pt
    FS_K = 20*FM    # tick / legend pt
    ACCENT     = '#1F3A68'
    DATA_COLOR = 'black'

    plt.rcParams.update({
        'font.family':       'Times New Roman',
        'font.size':          FS,
        'axes.labelsize':     FS_L,
        'axes.titlesize':     FS_T,
        'xtick.labelsize':    FS_K,
        'ytick.labelsize':    FS_K,
        'legend.fontsize':    FS_K,
        'axes.linewidth':     1.0,
        'xtick.major.width':  1.0,
        'ytick.major.width':  1.0,
        'lines.linewidth':    LW,
    })

    if isinstance(ensemble, list):
        ensemble = ensemble[0]

    fig, axes = plt.subplots(2, 3, figsize=(22, 13), constrained_layout=True)
    fig.get_layout_engine().set(hspace=0.15)

    def style_ax(ax, xlim=(2015, 2030), step=5):
        ax.set_xlim(xlim)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(step))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
        ax.set_xlabel('Year', fontsize=FS_L)
        ax.tick_params(labelsize=FS_K)

    def plot_ens(ax, output, color=ACCENT, ls='-'):
        t, med, lb, ub, _ = ut.extract_data(ensemble, output)
        ax.plot(t, med, color=color, linestyle=ls, linewidth=LW)
        ax.fill_between(t, lb, ub, alpha=0.2, color=color)
        return t, med

    def scatter_data(ax, tdve_key, x_offset=0, marker='o'):
        t = P.data.tdve[tdve_key].ts['Prisoners'].t
        v = P.data.tdve[tdve_key].ts['Prisoners'].vals
        ax.scatter([x + x_offset for x in t], v,
                   facecolors='none', edgecolors=DATA_COLOR, linewidths=1.5, s=MS, marker=marker)

    # (a) Prison population
    ax = axes[0, 0]
    plot_ens(ax, 'pop_prison')
    scatter_data(ax, 'pop_prison')
    ax.set_title('(a) Incarcerated population', fontsize=FS_T)
    ax.set_ylabel('Number', fontsize=FS_L)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    if ylimit: ax.set_ylim([0, 20000])
    style_ax(ax)

    # (b) OAT population
    ax = axes[0, 1]
    plot_ens(ax, 'pop_oat')
    scatter_data(ax, 'pop_oat')
    ax.set_title('(b) Incarcerated OAMT population', fontsize=FS_T)
    ax.set_ylabel('Number', fontsize=FS_L)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    if ylimit: ax.set_ylim([0, 4500])
    style_ax(ax)

    # (c) Modality proportions â€” inline labels
    ax = axes[0, 2]
    mod_outputs = [('prop_oat_meth_m', '-',  'o'), ('prop_oat_subx_m', '--', 's'), ('prop_oat_subl_m', ':', '^')]
    mod_tdve    = ['prop_oat_meth', 'prop_oat_subx', 'prop_oat_subl']
    inline_labels = [('Methadone', 2019, .5), ('SL-BPN', 2021.5, 0.03), ('LAI-BPN', 2027.2, 0.03)]
    for (out, ls, mk), tdve_key, (lbl, x_lbl, dy) in zip(mod_outputs, mod_tdve, inline_labels):
        t, med, lb, ub, _ = ut.extract_data(ensemble, out)
        ax.plot(t, med, color=ACCENT, linestyle=ls, linewidth=LW)
        ax.fill_between(t, lb, ub, alpha=0.1, color=ACCENT)
        t_d = P.data.tdve[tdve_key].ts['Prisoners'].t[:-1]
        v_d = P.data.tdve[tdve_key].ts['Prisoners'].vals[:-1]
        ax.scatter(t_d, v_d, facecolors='none', edgecolors=DATA_COLOR,
                   linewidths=1.5, s=MS, marker=mk)
        idx = np.argmin(np.abs(t - x_lbl))
        ax.text(x_lbl, med[idx] + dy, lbl, color=ACCENT, fontsize=FS_K,
                ha='center', va='bottom', fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))
    ax.set_title('(c) OAMT modality mix', fontsize=FS_T)
    ax.set_ylabel('Percentage', fontsize=FS_L)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    if ylimit: ax.set_ylim([0, 1])
    ax.set_ylim([0, 1.1])
    style_ax(ax)

    # (d) IRI incidence
    ax = axes[1, 0]
    plot_ens(ax, 'iri_idu_per100py')
    scatter_data(ax, 'iri_inci')
    ax.set_title('(d) IRIs (people who inject drugs)', fontsize=FS_T)
    ax.set_ylabel('Incidence (per 100 PY)', fontsize=FS_L)
    if ylimit: ax.set_ylim([0, 45])
    style_ax(ax)

    # (e) HCV incidence
    ax = axes[1, 1]
    plot_ens(ax, 'inci_per100py')
    scatter_data(ax, 'hcv_inci', x_offset=0.5)
    ax.set_title('(e) HCV (prison-wide)', fontsize=FS_T)
    ax.set_ylabel('Incidence (per 100 PY)', fontsize=FS_L)
    if ylimit: ax.set_ylim([0, 4])
    style_ax(ax)

    # (f) HCV prevalence
    ax = axes[1, 2]
    plot_ens(ax, 'prev')
    scatter_data(ax, 'prev')
    ax.set_title('(f) HCV (prison-wide)', fontsize=FS_T)
    ax.set_ylabel('Prevalence', fontsize=FS_L)
    if ylimit: ax.set_ylim([0, 0.15])
    style_ax(ax)

    fit_line = Line2D([0], [0], color=ACCENT, linewidth=LW, label='Model fit')
    data_pt  = Line2D([0], [0], marker='o', color='none', markeredgecolor=DATA_COLOR,
                      markerfacecolor='none', markersize=10, label='Data')
    fig.legend(handles=[fit_line, data_pt], loc='lower center',
               bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=True, fontsize=FS_T)

    _save_fig(fig, fig_name)
    print(f'Saved: {fig_name}')


def plot_trajectories_difference(P, ensemble_baseline, ensemble_scenario,
                                 label_baseline='Policy change', label_scenario='Counterfactual',
                                 fig_name='results/plot_trajectories_difference.png',
                                 start_year=2026, end_year=2035, ci_method=None):
    '''
    Trajectories and their divergence, 2x3. Top row: each scenario (median and 95% UI);
    bottom row: policy change minus counterfactual, as the median of the paired (matched-run)
    differences at each time point with the same interval as the reported tables
    (ut.bootstrap_median: bootstrap 95% CI of the median by default, or the 2.5th-97.5th
    percentiles with ci_method='percentile'), so figure and Table 2 agree.
    (a)/(d) incarcerated OAMT population | (b)/(e) HCV incidence (prison-wide)
    (c)/(f) IRI incidence (people who inject drugs)
    Policy change = solid navy, Counterfactual = dashed red, difference = grey; circles = data.

    :param ensemble_baseline: Policy change ensemble
    :param ensemble_scenario: Counterfactual ensemble
    '''
    FM = 1.35
    FS, FS_L, FS_T, FS_K = 20*FM, 22*FM, 22*FM, 20*FM
    LW = 2.5
    MS = 100
    COL_B, COL_S, COL_D = '#1F3A68', '#B85450', '#444444'
    COLS = [COL_B, COL_S]
    STYLES = ['-', '--']
    LABELS = [label_baseline, label_scenario]
    DATA_COLOR = 'black'

    plt.rcParams.update({
        'font.family':       'Times New Roman',
        'font.size':          FS,
        'axes.labelsize':     FS_L,
        'axes.titlesize':     FS_T,
        'xtick.labelsize':    FS_K,
        'ytick.labelsize':    FS_K,
        'legend.fontsize':    FS_K,
        'axes.linewidth':     1.0,
        'xtick.major.width':  1.0,
        'ytick.major.width':  1.0,
    })

    fig, axes = plt.subplots(2, 3, figsize=(24, 13), constrained_layout=True)
    fig.get_layout_engine().set(hspace=0.15, wspace=0.08, rect=(0, 0.07, 1, 1))

    def style_ax(ax):
        ax.set_xlim([2022, end_year])
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x)}'))
        ax.set_xlabel('Year', fontsize=FS_L)
        ax.tick_params(labelsize=FS_K)

    def plot_two(ax, output):
        for i, ens in enumerate([ensemble_baseline, ensemble_scenario]):
            t, med, lb, ub, _ = ut.extract_data(ens, output)
            ax.plot(t, med, color=COLS[i], linestyle=STYLES[i], linewidth=LW)
            ax.fill_between(t, lb, ub, alpha=0.1, color=COLS[i])

    def plot_diff(ax, output):
        # paired differences: run i of the policy change minus run i of the counterfactual
        t, _, _, _, vb = ut.extract_data(ensemble_baseline, output)
        _, _, _, _, vs = ut.extract_data(ensemble_scenario, output)
        n = min(len(vb), len(vs))
        diff = vb[:n] - vs[:n]
        summ = [ut.bootstrap_median(diff[:, j], method=ci_method) for j in range(diff.shape[1])]
        pe = np.array([x['pe'] for x in summ])
        lb = np.array([x['lb'] for x in summ])
        ub = np.array([x['ub'] for x in summ])
        ax.plot(t, pe, color=COL_D, linewidth=LW)
        ax.fill_between(t, lb, ub, alpha=0.25, color=COL_D)
        ax.axhline(0, color='black', linewidth=1.0, linestyle=':')

    def scatter_data(ax, tdve_key):
        t = P.data.tdve[tdve_key].ts['Prisoners'].t
        v = P.data.tdve[tdve_key].ts['Prisoners'].vals
        ax.scatter(t, v, facecolors='none', edgecolors=DATA_COLOR, linewidths=1.5, s=MS, marker='o')

    panels = [
        ('pop_oat',          '(a) Incarcerated OAMT population',    '(d) Difference in OAMT population',
         'Number',                '{x:,.0f}', None,   True),
        ('inci_per100py',    '(b) HCV incidence (prison-wide)',     '(e) Difference in HCV incidence',
         'Incidence (per 100 PY)', None,      [0, 1.2], False),
        ('iri_idu_per100py', '(c) IRIs (people who inject drugs)',  '(f) Difference in IRI incidence',
         'Incidence (per 100 PY)', None,      [0, 45], False),
    ]
    for j, (output, title_top, title_bot, ylab, yfmt, ylim, has_data) in enumerate(panels):
        ax = axes[0, j]
        plot_two(ax, output)
        if has_data:
            scatter_data(ax, output)
        ax.set_title(title_top, fontsize=FS_T)
        ax.set_ylabel(ylab, fontsize=FS_L)
        if yfmt:
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(yfmt))
        if ylim:
            ax.set_ylim(ylim)
        style_ax(ax)

        ax = axes[1, j]
        plot_diff(ax, output)
        ax.set_title(title_bot, fontsize=FS_T)
        ax.set_ylabel('Difference' + (' (per 100 PY)' if 'per100py' in output else ' in number'), fontsize=FS_L)
        if yfmt:
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter(yfmt))
        style_ax(ax)

    data_pt = Line2D([0], [0], marker='o', color='none', markeredgecolor=DATA_COLOR,
                      markerfacecolor='none', markersize=10, label='Data')
    diff_line = Line2D([0], [0], color=COL_D, linewidth=LW,
                       label=f'Difference ({label_baseline} − {label_scenario}); band: 95% CI of the median')
    leg_handles = [Line2D([0], [0], color=COLS[i], linestyle=STYLES[i], linewidth=LW, label=LABELS[i])
                   for i in range(2)] + [data_pt, diff_line]
    fig.legend(handles=leg_handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=4, frameon=True, fontsize=FS_K)

    _save_fig(fig, fig_name)
    print(f'Saved: {fig_name}')


def _money_compact(x, _=None):
    """3 significant figures with M/B suffix (e.g. $1.24B, $342M, $9.2M); '$' escaped for matplotlib."""
    if x == 0:
        return '\\$0'
    sign = '-' if x < 0 else ''
    ax = abs(x)
    if ax >= 1e9:
        s = f'{ax/1e9:.2f}B'
    elif ax >= 1e8:
        s = f'{ax/1e6:.0f}M'
    elif ax >= 1e7:
        s = f'{ax/1e6:.1f}M'
    elif ax >= 1e6:
        s = f'{ax/1e6:.2f}M'
    else:
        s = f'{ax/1e3:.0f}K'
    return f'{sign}\\${s}'


def _money_tick(x, _=None):
    """Axis tick label: no trailing zeros ($50M, $1.2B, $800M); '$' escaped for matplotlib."""
    if x == 0:
        return '\\$0'
    sign = '-' if x < 0 else ''
    ax = abs(x)
    if ax >= 1e9:
        s = f'{ax/1e9:g}B'
    elif ax >= 1e6:
        s = f'{ax/1e6:g}M'
    else:
        s = f'{ax/1e3:g}K'
    return f'{sign}\\${s}'


def _fold_community_oat(results):
    """Copy of `results` with post-release (community) OAT cost folded into the OAMT delivery
    totals (per-arm and difference), so the OAMT bar matches the total cost. In 'medication'
    mode it is also folded into the medication segment; otherwise it stays a separate
    'diff_costs_oat_com' segment. No-op when community_oat_cost wasn't used."""
    com = results.get('diff_costs_oat_com')
    if not com:
        return results

    def add(a, b):
        n = min(len(a['vals']), len(b['vals']))
        return ut.bootstrap_median([a['vals'][i] + b['vals'][i] for i in range(n)])

    r = dict(results)
    r['diff_costs_oat_total'] = add(results['diff_costs_oat_total'], com)
    if results.get('community_oat_cost') == 'medication':
        r['diff_costs_oat_drug'] = add(results['diff_costs_oat_drug'], com)
    for arm in ('baseline', 'scenario'):
        r[f'cost_oat_total_{arm}'] = add(results[f'cost_oat_total_{arm}'], results['postrelease_abs'][f'cost_oat_com_{arm}'])
    return r


def _shade(hex_color, frac=0.3):
    """Blend a colour towards black (frac = share of black)."""
    rgb = np.array(mpl.colors.to_rgb(hex_color))
    return tuple(rgb * (1 - frac))


def _tint(hex_color, frac=0.6):
    """Blend a colour towards white (frac = share of white)."""
    rgb = np.array(mpl.colors.to_rgb(hex_color))
    return tuple(rgb + (1 - rgb) * frac)


def plot_cca_summary(results, label_baseline='Policy change', label_scenario='Counterfactual',
                     start_year=2026, end_year=2035,
                     fig_name='results/plot_cca_summary.png', include_total=True):
    '''
    Cost-consequence summary figure, 4 rows x 2 columns, in the same order as Table 2: cost first, then consequences.
      Row 1 - Cost: OAMT delivery (medication, registered nurse, correctional officer) and its total.
      Row 2 - Consequences: OAMT coverage (average people on OAMT in prison, by modality and in total).
      Row 3 - Consequences: health and justice outcomes (HCV infections, IRIs, reincarceration events, deaths).
      Row 4 - Consequences: financial (HCV treatment, IRI management, reincarceration) and their total.
      Left column  - by scenario over start_year-end_year (medians): hatched pale bar = Counterfactual, solid = Policy change.
      Right column - difference (Policy change - Counterfactual, as calculate_all_outcomes).
    No error bars; intervals are in Table 2. Drawn at print size (7.2 x 9.6 in) so the fonts are real point sizes; insert at
    100% width. QALYs are reported in the appendix, not here.

    :param results: dict returned by calculate_all_outcomes.
    :param include_total: unused (kept for compatibility); the rows always show their totals, and net cost is not shown.
    '''
    FM = 0.62
    FS, FS_T = 14 * FM, 16 * FM
    plt.rcParams.update({
        'font.family':    'Times New Roman',
        'font.size':       FS,
        'axes.titlesize':  FS_T,
        'axes.labelsize':  FS,
        'xtick.labelsize': FS * 1.0,
        'ytick.labelsize': FS * 1.05,
        'axes.linewidth':  0.8,
        'hatch.linewidth': 0.7,
    })
    pr, ta = results['postrelease_abs'], results['total_abs']
    results = _fold_community_oat(results)
    community = results.get('include_community_costs')
    period = f'{start_year}–{end_year}'

    COL_HCV, COL_IRI = '#8E44AD', '#16A085'
    COL_REINC, COL_DEATH = '#B85450', '#1F3A68'
    COL_TOTAL = '#595959'
    COL_METH, COL_SUBX, COL_SUBL = '#5B9BD5', '#2E75B6', '#1F4E79'
    COL_MED, COL_RN, COL_CO = '#2E5597', '#4472C4', '#8FAADC'

    fig = plt.figure(figsize=(7.2, 9.6))
    gs = fig.add_gridspec(4, 2, left=0.115, right=0.995, top=0.935, bottom=0.045, wspace=0.34, hspace=0.62)
    ax_k0, ax_k1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])    # cost: by scenario | difference
    ax_c0, ax_c1 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])    # consequences, OAMT coverage
    ax_h0, ax_h1 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])    # consequences, health and justice
    ax_f0, ax_f1 = fig.add_subplot(gs[3, 0]), fig.add_subplot(gs[3, 1])    # consequences, financial

    def style(ax):
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#DDDDDD', linewidth=0.8)
        for side in ('right', 'top'):
            ax.spines[side].set_visible(False)

    count_fmt = lambda x, _=None: f'{x:,.0f}'
    MINUS = lambda t: t.replace('-', '−')   # true minus sign, as in the tables
    DOT = lambda t: t.replace('.', '·')     # Lancet style: decimal point is a mid-line dot
    money_tick = lambda x, _=None: DOT(MINUS(_money_tick(x, _)))
    MONEY = lambda x: DOT(MINUS(_money_compact(x)))
    cnt_label = lambda x: MINUS(f'{x:,.0f}')

    def add(a, b):
        """Sum two result dicts (each with 'vals') per simulation."""
        n = min(len(a['vals']), len(b['vals']))
        return ut.bootstrap_median(np.asarray(a['vals'][:n], dtype=float) + np.asarray(b['vals'][:n], dtype=float))

    def by_scenario(ax, specs, fmt, ytick_fmt, ylabel, title, top_pad=1.15, label_fs=None):
        """Grouped bars: one slot per item, pale hatched bar (Counterfactual) beside solid bar (Policy change)."""
        ymax = top_pad * max(d['pe'] for _, _, d_cf, d_pol in specs for d in (d_cf, d_pol))
        for k, (_, colour, d_cf, d_pol) in enumerate(specs):
            for dx, d, face, hatch in ((-0.245, d_cf, _tint(colour), '///'), (0.245, d_pol, _shade(colour), None)):
                ax.bar([k + dx], [d['pe']], color=face, edgecolor=_shade(colour, 0.15) if hatch is None else colour,
                       linewidth=0.9, width=0.44, zorder=3, hatch=hatch)
                ax.text(k + dx, d['pe'] + 0.012 * ymax, fmt(d['pe']), ha='center', va='bottom',
                        fontsize=label_fs or FS * 0.9, fontweight='bold')
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.5, len(specs) - 0.5)
        ax.set_xticks(range(len(specs)))
        ax.set_xticklabels([lab for lab, _, _, _ in specs])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(ytick_fmt))
        ax.set_ylabel(ylabel, fontsize=FS)
        ax.set_title(title, fontsize=FS_T)
        style(ax)

    def diff_panel(ax, specs, fmt, ytick_fmt, ylabel, title, label_fs=None, pad_frac=0.18):
        """Difference panel: one bar per (value dict, label, colour) with a signed value label."""
        n = len(specs)
        meds = [d['pe'] for d, _, _ in specs]
        for x, (d, _, color) in enumerate(specs):
            ax.bar([x], [d['pe']], color=color, width=0.62, zorder=3)
        ax.axhline(0, color='black', linewidth=0.9, zorder=5)
        ax.set_xticks(range(n))
        ax.set_xticklabels([lab for _, lab, _ in specs])
        ax.set_xlim(-0.5, n - 0.5)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(ytick_fmt))
        span = (max(meds + [0]) - min(meds + [0])) or 1.0
        for x, m in enumerate(meds):
            sign = '+' if m > 0 else ''
            off = 0.015 * span
            ax.text(x, m + off if m >= 0 else m - off, sign + fmt(m), ha='center', va='bottom' if m >= 0 else 'top',
                    fontsize=label_fs or FS * 0.95, fontweight='bold')
        lo, hi = min(meds + [0]), max(meds + [0])
        ax.set_ylim(lo - (pad_frac * span if lo < 0 else 0), hi + pad_frac * span)
        ax.set_ylabel(ylabel, fontsize=FS)
        ax.set_title(title, fontsize=FS_T)
        style(ax)

    # ── Row 1: Cost - OAMT delivery ────────────────────────────────────────────
    drug_b, drug_s = results['cost_oat_drug_baseline'], results['cost_oat_drug_scenario']
    if results.get('community_oat_cost') == 'medication' and results.get('diff_costs_oat_com'):
        drug_b = add(drug_b, pr['cost_oat_com_baseline'])      # medication includes community medication after release
        drug_s = add(drug_s, pr['cost_oat_com_scenario'])
    rn_b = add(results['cost_oat_rn_baseline'], results['cost_oat_init_baseline'])
    rn_s = add(results['cost_oat_rn_scenario'], results['cost_oat_init_scenario'])
    k_specs = [
        ('Medication',              COL_MED, drug_b, drug_s),
        ('Registered\nnurse',       COL_RN,  rn_b, rn_s),
        ('Correctional\nofficer',   COL_CO,  results['cost_oat_co_baseline'], results['cost_oat_co_scenario']),
    ]
    if True:   # totals are part of the cost and financial-consequence rows (net cost is not shown)
        k_specs.append(('Total cost\nof OAMT', COL_TOTAL, results['cost_oat_total_baseline'], results['cost_oat_total_scenario']))
    by_scenario(ax_k0, k_specs, MONEY, money_tick, '2025 CAD', '(a) Cost: OAMT delivery', top_pad=1.3, label_fs=FS * 0.8)
    kd_specs = [
        (results['diff_costs_oat_drug'], 'Medication', COL_MED),
        (add(results['diff_costs_oat_rn'], results['diff_costs_oat_init']), 'Registered\nnurse', COL_RN),
        (results['diff_costs_oat_co'], 'Correctional\nofficer', COL_CO),
    ]
    if True:
        kd_specs.append((results['diff_costs_oat_total'], 'Total cost\nof OAMT', COL_TOTAL))
    diff_panel(ax_k1, kd_specs, MONEY, money_tick, 'Difference, 2025 CAD', '(b) Cost: OAMT delivery', label_fs=FS * 0.9)

    # ── Row 2: Consequences - OAMT coverage ────────────────────────────────────
    n_years = results.get('n_years', end_year - start_year + 1)
    obm = results.get('oat_by_modality')

    def avg_people(d):
        v = np.asarray(d['person_years']['vals'], dtype=float) / n_years
        return {'pe': float(np.nanmedian(v)), 'vals': v}

    c_specs = []
    cd_specs = []
    if obm:
        for key, lab, col in (('meth', 'Methadone', COL_METH), ('subx', 'SL-BPN', COL_SUBX), ('subl', 'LAI-BPN', COL_SUBL)):
            c_specs.append((lab, col, avg_people(obm['Baseline'][key]), avg_people(obm['Scenario'][key])))
            cd_specs.append((results['diff_pop_oat_' + key], lab, col))
    c_specs.append(('All\nOAMT', COL_TOTAL, results['pop_oat_inprison_baseline'], results['pop_oat_inprison_scenario']))
    cd_specs.append((results['diff_pop_oat_inprison'], 'All\nOAMT', COL_TOTAL))
    by_scenario(ax_c0, c_specs, count_fmt, count_fmt, 'Average people', '(c) Consequences: OAMT coverage')
    diff_panel(ax_c1, cd_specs, cnt_label, lambda x, _=None: MINUS(count_fmt(x)), 'Difference in people',
               '(d) Consequences: OAMT coverage')

    # ── Row 3: Consequences - health and justice ───────────────────────────────
    h_specs = [
        ('HCV\ninfections',        COL_HCV,   ta['hcv_total_baseline'], ta['hcv_total_scenario']),
        ('IRI\nevents',            COL_IRI,   ta['iri_total_baseline'], ta['iri_total_scenario']),
        ('Reincarceration\nevents', COL_REINC, pr['reinc_baseline'],    pr['reinc_scenario']),
        ('Deaths\n(drug-related)', COL_DEATH, pr['deaths_baseline'],    pr['deaths_scenario']),
    ]
    by_scenario(ax_h0, h_specs, count_fmt, count_fmt, 'Events', '(e) Consequences: health and justice')
    hd_specs = [
        (results['diff_hcv_total'], 'HCV\ninfections', COL_HCV),
        (results['diff_iri_total'], 'IRI\nevents', COL_IRI),
        (results['diff_reinc'], 'Reincarceration\nevents', COL_REINC),
        (results['diff_deaths'], 'Deaths\n(drug-related)', COL_DEATH),
    ]
    diff_panel(ax_h1, hd_specs, cnt_label, lambda x, _=None: MINUS(count_fmt(x)), 'Difference in events',
               '(f) Consequences: health and justice')

    # ── Row 4: Consequences - financial ────────────────────────────────────────
    hcv_keys = ('cost_hcv_total_baseline', 'cost_hcv_total_scenario') if community else \
               ('cost_hcv_inprison_baseline', 'cost_hcv_inprison_scenario')
    iri_keys = ('cost_iri_total_baseline', 'cost_iri_total_scenario') if community else \
               ('cost_iri_inprison_baseline', 'cost_iri_inprison_scenario')
    src = (lambda keys: (ta[keys[0]], ta[keys[1]])) if community else \
          (lambda keys: (results[keys[0]], results[keys[1]]))
    f_specs = [
        ('HCV\ntreatment',   COL_HCV,   *src(hcv_keys)),
        ('IRI\nmanagement',  COL_IRI,   *src(iri_keys)),
        ('Reincarceration',  COL_REINC, pr['cost_reinc_baseline'], pr['cost_reinc_scenario']),
    ]
    fd_specs = [
        (results['diff_costs_hcv'], 'HCV\ntreatment', COL_HCV),
        (results['diff_costs_iri'], 'IRI\nmanagement', COL_IRI),
        (results['diff_reinc_cost'], 'Reincarceration', COL_REINC),
    ]
    if 'cost_fin_baseline' in ta and results.get('diff_cost_fin'):
        f_specs.append(('Total', COL_TOTAL, ta['cost_fin_baseline'], ta['cost_fin_scenario']))
        fd_specs.append((results['diff_cost_fin'], 'Total', COL_TOTAL))
    by_scenario(ax_f0, f_specs, MONEY, money_tick, f'2025 CAD{"" if community else "; CSC costs only"}',
                '(g) Consequences: financial', top_pad=1.3, label_fs=FS * 0.8)
    diff_panel(ax_f1, fd_specs, MONEY, money_tick, f'Difference, 2025 CAD{"" if community else "; CSC costs only"}',
               '(h) Consequences: financial', label_fs=FS * 0.9)

    # Column headers (once), instead of repeating them in every panel title
    left_x = (ax_k0.get_position().x0 + ax_k0.get_position().x1) / 2
    right_x = (ax_k1.get_position().x0 + ax_k1.get_position().x1) / 2
    fig.text(left_x, 0.985, f'By scenario, {period}', ha='center', va='top', fontsize=FS_T, fontweight='bold')
    fig.text(right_x, 0.985, f'Difference (policy change − counterfactual), {period}', ha='center', va='top',
             fontsize=FS_T, fontweight='bold')

    arm_handles = [Patch(facecolor=_tint('#555555'), edgecolor='#555555', linewidth=0.9, hatch='///', label=label_scenario),
                   Patch(facecolor=_shade('#555555', 0.3), edgecolor=_shade('#555555', 0.45), linewidth=0.9, label=label_baseline)]
    ax_k0.legend(handles=arm_handles, loc='upper left', frameon=True, framealpha=1.0, edgecolor='#BBBBBB',
                 fontsize=FS * 0.8, handlelength=1.5)

    _save_fig(fig, fig_name)
    plt.show()
    plt.close(fig)
    print(f'Saved: {fig_name}')


def plot_oat_stocks(P, ensemble, ylimit: bool = False):
    '''
    Standalone plot showing modality-specific OAT stocks over time.
    '''
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 14,
        'axes.labelsize': 15,
        'axes.titlesize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'lines.linewidth': 3.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    aplt.settings['legend_mode'] = ''
    
    ACCENT = '#1F3A68'
    DATA_COLOR = 'black'
    LINESTYLES = ['-', '--', ':']
    
    fig = plt.figure(figsize=(10, 6))
    plt.subplots_adjust(left=0.10, right=0.78, top=0.88, bottom=0.13)
    
    # Build the plot â€” one ensemble call per modality stock
    for ens in (ensemble if isinstance(ensemble, list) else [ensemble]):
        for output_name, ls in zip(['pop_oat_meth', 'pop_oat_subx', 'pop_oat_subl'], LINESTYLES):
            t, median_vals, _, _, vals = ut.extract_data(ens, output_name)
            plt.plot(t, median_vals, color=ACCENT, linestyle=ls, linewidth=3.0)
            plt.fill_between(
                t,
                np.quantile(vals, 0.025, axis=0),
                np.quantile(vals, 0.975, axis=0),
                alpha=0.08,
                color=ACCENT,
            )
    
    ax = plt.gca()
    plt.title('Total prisoners on OAT, by modality', pad=12)
    plt.xlabel('Year')
    plt.ylabel('Number')
    plt.xlim([2015, 2030])
    plt.ylim(bottom=0)
    if ylimit:
        plt.ylim([0, 3000])
    
    # Integer years
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{int(x)}'))
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter('{x:,.0f}'))
    
    # Data scatter â€” derive each modality stock from databook
    
    modality_data = {
        'pop_oat_meth': ('o', 'Methadone', '-'),
        'pop_oat_subx': ('s', 'Suboxone', '--'),
        'pop_oat_subl': ('^', 'Sublocade', ':'),
    }
    
    for tdve_name, (marker, label, ls) in modality_data.items():
        t_pop = P.data.tdve[tdve_name].ts['Prisoners'].t
        v_pop = np.array(P.data.tdve[tdve_name].ts['Prisoners'].vals)
        plt.scatter(t_pop, v_pop, facecolors='none', edgecolors=DATA_COLOR,
                    linewidths=2, s=70, marker=marker)
    
    # Build legend with proxy artists
    legend_handles = [
        Line2D([0], [0], color=ACCENT, linestyle='-',  linewidth=3.0,
               marker='o', markerfacecolor='none', markeredgecolor=DATA_COLOR,
               markeredgewidth=2, markersize=9, label='Methadone'),
        Line2D([0], [0], color=ACCENT, linestyle='--', linewidth=3.0,
               marker='s', markerfacecolor='none', markeredgecolor=DATA_COLOR,
               markeredgewidth=2, markersize=9, label='Suboxone'),
        Line2D([0], [0], color=ACCENT, linestyle=':',  linewidth=3.0,
               marker='^', markerfacecolor='none', markeredgecolor=DATA_COLOR,
               markeredgewidth=2, markersize=9, label='Sublocade'),
    ]
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
              frameon=False, handlelength=2.5, handletextpad=0.8)

    return fig
