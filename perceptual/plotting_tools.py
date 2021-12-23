import os
from collections import OrderedDict

import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from joblib import Parallel, delayed
from scipy.stats import (f_oneway, kendalltau, kruskal, pearsonr, spearmanr,
                         wilcoxon, ttest_rel, shapiro)
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

var1_map = {
    'rest': 'restoration',
    'resynth': 'resynthesis',
    'transcr': 'transcription',
    'hr': 'hr',
    'nr': 'nr',
    'si': 'si',
    'o&f': 'o&f'
}


def plot(df,
         obj_eval,
         measure_name,
         var1,
         var2,
         excerpts_mean=True,
         variable=None):
    """
    Arguments
    ---------

    `df` : pd.DataFrame
        the dataframe built from the `saves` dir

    `excerpt_mean` : bool
        if True, plots the average over all the excerpts per each var1_val type

    `variable` : str or None
        the variable to be observed: each value of the variable will be a
        different line and the Wilcoxon rank test will be computed for all the
        combinations of the variable value

    `measure_name` : str
        the name used for the measure in the plots

    `var1` : str
        a column of `df` that is iterated first (e.g. `'task'` or `'method'`)

    `var2` : str
        a column of `df` that is iterated next (e.g. `'task'` or `'method'`)

    Returns
    -------

    `list[dash_core_components.Graph]` :
        the list of graphs to be plotted in Dash
    """

    var1_vals = df[var1].unique()
    excerpts = df['excerpt_num'].unique()

    # selecting data
    print("Plotting")

    def process(var1_val):
        sort_by = [var2] + ([variable] if variable is not None else [])
        if excerpts_mean:
            excerpt = 'mean'
            selected_data = df.loc[df[var1] == var1_val].sort_values(sort_by)
            # plotting means of the excerpts
            _plot_data(var2, selected_data, var1_val, excerpt, variable,
                       obj_eval, measure_name)
        else:
            for excerpt in sorted(excerpts):
                selected_data = df.loc[df[var1] == var1_val].loc[
                    df['excerpt_num'] == excerpt].sort_values(sort_by)
                # plotting each excerpt
                _plot_data(var2, selected_data, var1_val, excerpt, variable,
                           obj_eval, measure_name)

    Parallel(n_jobs=1)(delayed(process)(var1_val)
                       for var1_val in tqdm(var1_vals))


def compute_correlations(groupby, obj_eval, excerpt):
    # correlations = np.zeros((3, 2, 3, 2, 2))
    funcs = ['Pearsons', 'Spearman', 'Kendall']
    measures = ['Precision', 'Recall', 'F-Measure']
    vel = ['With Velocity', 'Without Velocity']
    type_ = ['Mean', 'Median']
    correlations = pd.DataFrame(
        index=pd.MultiIndex.from_product([vel, measures]),
        columns=pd.MultiIndex.from_product(
            [type_, funcs, ['Correlation', 'p-value']]))
    for i in range(2):
        for j in range(3):
            for k, correl_func in enumerate([pearsonr, spearmanr, kendalltau]):
                mean_val = correl_func(obj_eval[excerpt, :, i, j],
                                       groupby.mean())
                median_val = correl_func(obj_eval[excerpt, :, i, j],
                                         groupby.median())
                correlations.loc[vel[i], measures[j]]['Mean',
                                                      funcs[k]] = mean_val
                correlations.loc[vel[i], measures[j]]['Median',
                                                      funcs[k]] = median_val
    return correlations.T


def _plot_data(var2, selected_data, var1_val, excerpt, variable, obj_eval,
               measure_name):

    st.write(f"""
    ## Var1: _{var1_map[var1_val]}_
    ## Excerpt: _{excerpt}_
    ## Controlled variable: _{variable}_
    """)

    # computing the data related to this excerpt
    if excerpt == 'mean':
        obj_eval = np.mean(obj_eval[:], axis=0)[np.newaxis]
        groupby = selected_data.groupby(var2)['rating']
        excerpt_num = 0
    else:
        groupby = selected_data.loc[selected_data['excerpt_num'] ==
                                    excerpt].groupby(var2)['rating']
        excerpt_num = excerpt

    # creating plot
    fig_plot = px.violin(
        selected_data,
        x=var2,
        y='rating',
        box=True,
        # points="all",
        color=variable,
        violinmode='group',
        title=
        f'Var1: {var1_map[var1_val]} - Excerpt: {excerpt} - Controlled variable: {variable}'
    )
    # customizing plot
    fig_plot.update_traces(spanmode='manual', span=[0, 1])
    # changing color of boxplots and adding mean line
    for data in fig_plot.data:
        data.meanline = dict(visible=True, color='white', width=1)
        data.box.line.color = 'white'
        data.box.line.width = 1
    var2_vals = selected_data[var2].unique()
    # adding measure line to the plot
    if len(var2_vals) == 4:
        fig_plot.add_trace(
            go.Scatter(x=var2_vals,
                       y=obj_eval[excerpt_num, :, 1, 2],
                       name=measure_name))
    # saving plot and output to streamlit
    if not os.path.exists('imgs'):
        os.mkdir('imgs')
    fig_plot.write_image(f"imgs/{var1_map[var1_val]}_{excerpt}_{variable}.svg")
    st.write(fig_plot)

    # computing correlations
    st.write("### Correlations and error margins")
    if groupby.sum().shape[0] == 4:
        correlations = compute_correlations(groupby, obj_eval, excerpt_num)
        st.write("Correlations for all the data:")
        st.table(correlations.style.format('{:.2e}'))

    # a function to compute error margins
    def _compute_error_margins(groupby, selected_data_variable, var):

        # error_margin_text.append(f"var {var}: ")
        error_margins = pd.DataFrame(index=var2_vals,
                                     columns=['Sample Size', 'Error Margin'])
        for var2_val in var2_vals:
            # computing std, and error margin
            samples = selected_data.loc[selected_data_variable]
            samples = samples.loc[samples[var2] == var2_val]['rating']
            sample_size = samples.count()
            std = samples.std()
            error_margins.loc[var2_val] = [
                sample_size,
                np.sqrt((1.96**2) * (std**2) / sample_size)
            ]
        st.write(f"Error margins for control group **{var}**")
        st.write(error_margins)

    if type(excerpt) is not int:
        groupby = selected_data
    else:
        groupby = selected_data.loc[selected_data['excerpt_num'] == excerpt]

    if variable:
        # the values available for this variable
        variable_vals = selected_data[variable].unique()

        distributions = {}
        for var in variable_vals:
            # computing the distributions for each value of the variable and
            # each var2_val
            for var2_val in var2_vals:
                distributions[f"{var2_val}, group {var}"] = selected_data[
                    (selected_data[variable] == var).values *
                    (selected_data[var2] == var2_val).values]['rating'].values

            # computing correlations and error margins for each variable value
            groupby_variable = groupby[variable] == var
            print(groupby_variable)
            selected_data_variable = selected_data[variable] == var
            this_groupby = groupby.loc[groupby_variable].groupby(
                var2)['rating']
            if this_groupby.sum().shape[0] == 4:
                correlations = compute_correlations(this_groupby, obj_eval,
                                                    excerpt_num)
                st.write(f"Correlations for control group **{var}**")
                st.table(correlations.style.format('{:.2e}'))

            _compute_error_margins(this_groupby, selected_data_variable, var)

    else:
        # no variable provided, using all the data
        # computing error margins for all the data
        groupby_variable = groupby['rating'] > -1
        groupby = groupby.loc[groupby_variable].groupby(var2)['rating']
        _compute_error_margins(groupby, selected_data['rating'] > -1, 'all')
        distributions = {
            var2_val:
            selected_data[selected_data[var2] == var2_val]['rating'].values
            for var2_val in var2_vals
        }

    st.write("### Statistical significance analysis")

    # normality test
    shapiro_pvals(distributions, var2_vals)

    # computing wilcoxon, and t-test test
    st.write("#### Wilcoxon test")
    _compute_pvals(var2, selected_data, excerpt, variable, wilcoxon)
    st.write("#### t-test (related variables)")
    _compute_pvals(var2, selected_data, excerpt, variable, ttest_rel)

    st.write("---")


def omnibus(distributions):
    st.write("###### Omnibus tests")
    kruskal_h, kruskal_pval = kruskal(*distributions)
    st.write(
        f"Kruskal-Wallis (p-value, h-statistics): {kruskal_pval:.2e}, {kruskal_h:.2f}"
    )
    f_h, f_pval = f_oneway(*distributions)
    st.write(
        f"ANOVA (p-value, h-statistics):          {f_pval:.2e}, {f_h:.2f}")


def shapiro_pvals(distributions, var2_vals):
    st.write("#### Shapiro-Wilk tests:")
    shapiro_pval = OrderedDict(
        {k: shapiro(d)[1]
         for k, d in distributions.items()})
    if len(distributions) > 2:
        st.write("###### Using Bonferroni-Holm correction!")
        _, shapiro_pval_corrected, _, _ = multipletests(list(
            shapiro_pval.values()),
                                                        method='holm')
    st.write({
        k: f"{shapiro_pval_corrected[i]:.2e}"
        for i, k in enumerate(shapiro_pval.keys())
    })


def correct_pvalues(pval):
    pval_indices = np.nonzero(~np.isnan(pval))
    _, corrected_pval, _, _ = multipletests(pval[pval_indices].flatten(),
                                            method='holm')
    pval[pval_indices] = corrected_pval


def _compute_pvals(var2, selected_data, excerpt, variable, statistics_func):
    var2_vals = selected_data[var2].unique()
    if variable:
        variable_vals = selected_data[variable].unique()

        # computing pval for each var2_val between the variable values
        for var2_val in var2_vals:
            samples = selected_data.loc[selected_data[var2] == var2_val]
            pval = np.full((len(variable_vals), len(variable_vals)), np.nan)
            distributions = []
            for i, expi in enumerate(variable_vals):
                for j, expj in enumerate(variable_vals):
                    if i > j:
                        try:
                            datai = samples.loc[samples[variable] == expi]
                            dataj = samples.loc[samples[variable] == expj]
                            maxlen = min(len(datai), len(dataj))
                            x = datai['rating'][:maxlen]
                            y = dataj['rating'][:maxlen]
                            _, pval[i, j] = statistics_func(x, y)
                            distributions += [x, y]
                        except Exception as e:
                            print(
                                f"\nError while computing pvals with {statistics_func} test!:"
                            )
                            print(e)
                            print()
            st.write(
                f"p-values for var2_val **{var2_val}** and variable **{variable}**"
            )
            if pval.shape[0] > 2:
                omnibus(distributions)
                correct_pvalues(pval)
                st.write("###### using Bonferroni-Holm correction!")
            st.table(
                pd.DataFrame(pval, columns=variable_vals,
                             index=variable_vals).style.format('{:.2e}'))

        # computing pval for each variable
        for var in variable_vals:
            samples = selected_data.loc[selected_data[variable] == var]
            _pval_on_var2_vals(var2, var2_vals, samples, var, statistics_func)
    else:
        samples = selected_data
        _pval_on_var2_vals(var2, var2_vals, samples, 'all', statistics_func)


def _pval_on_var2_vals(var2, var2_vals, samples, var, statistics_func):
    pval = np.full((len(var2_vals), len(var2_vals)), np.nan)
    distributions = []
    for i, expi in enumerate(var2_vals):
        for j, expj in enumerate(var2_vals):
            if i > j:
                try:
                    datai = samples.loc[samples[var2] == expi]
                    dataj = samples.loc[samples[var2] == expj]
                    maxlen = min(len(datai), len(dataj))
                    x = datai['rating'][:maxlen]
                    y = dataj['rating'][:maxlen]
                    _, pval[i, j] = statistics_func(x, y)
                    distributions += [x, y]
                except Exception as e:
                    print(
                        f"\nError while computing pvals with {statistics_func} test!:"
                    )

                    print(e)
                    print()
    st.write(f"##### p-values for variable value **{var}**")
    if pval.shape[0] > 2:
        omnibus(distributions)
        correct_pvalues(pval)
        st.write("###### using Bonferroni-Holm correction!")
    st.table(
        pd.DataFrame(pval, columns=var2_vals,
                     index=var2_vals).style.format('{:.2e}'))
