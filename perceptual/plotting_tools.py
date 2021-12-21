import os

import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from joblib import Parallel, delayed
from scipy.stats import (f_oneway, kendalltau, kruskal, pearsonr, spearmanr,
                         wilcoxon)
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

task_map = {
    'rest': 'restoration',
    'resynth': 'resynthesis',
    'transcr': 'transcription'
}


def plot(df, obj_eval, measure_name, excerpts_mean=True, variable=None):
    """
    Arguments
    ---------

    `df` : pd.DataFrame
        the dataframe built from the `saves` dir

    `excerpt_mean` : bool
        if True, plots the average over all the excerpts per each question type

    `variable` : str or None
        the variable to be observed: each value of the variable will be a
        different line and the Wilcoxon rank test will be computed for all the
        combinations of the variable value

    Returns
    -------

    `list[dash_core_components.Graph]` :
        the list of graphs to be plotted in Dash
    """

    # taking the number of questions and excerpts
    questions = df.question.unique()
    excerpts = df.excerpt_num.unique()

    # selecting data
    print("Plotting")

    def process(question):
        sort_by = ['method'] + ([variable] if variable is not None else [])
        if excerpts_mean:
            excerpt = 'mean'
            selected_data = df.loc[df['question'] == question].sort_values(
                sort_by)
            # plotting means of the excerpts
            _plot_data(selected_data, question, excerpt, variable, obj_eval,
                       measure_name)
        else:
            for excerpt in sorted(excerpts):
                selected_data = df.loc[df['question'] == question].loc[
                    df['excerpt_num'] == excerpt].sort_values(sort_by)
                # plotting each excerpt
                _plot_data(selected_data, question, excerpt, variable,
                           obj_eval, measure_name)

    Parallel(n_jobs=1)(delayed(process)(question)
                       for question in tqdm(questions))


def compute_correlations(groupby, obj_eval, excerpt):
    # correlations = np.zeros((3, 2, 3, 2, 2))
    funcs = ['Pearsons', 'Spearman', 'Kendall']
    measures = ['Precision', 'Recall', 'F-Measure']
    vel = ['With Velocity', 'Without Velocity']
    type_ = ['Mean', 'Median']
    correlations = pd.DataFrame(index=pd.MultiIndex.from_product(
        [funcs, measures]),
                                columns=pd.MultiIndex.from_product(
                                    [vel, type_, ['Correlation', 'Pvalue']]))
    for i in range(2):
        for j in range(3):
            for k, correl_func in enumerate([pearsonr, spearmanr, kendalltau]):
                mean_val = correl_func(obj_eval[excerpt, :, i, j],
                                       groupby.mean())
                median_val = correl_func(obj_eval[excerpt, :, i, j],
                                         groupby.median())
                correlations.loc[funcs[k], measures[j]][vel[i],
                                                        'Mean'] = mean_val
                correlations.loc[funcs[k], measures[j]][vel[i],
                                                        'Median'] = median_val
    return correlations


def _plot_data(selected_data, question, excerpt, variable, obj_eval,
               measure_name):

    st.write(f"""
    ### Task: {task_map[question]} - Excerpt: {excerpt} - Controlled variable: {variable}
    """)
    fig_plot = px.violin(
        selected_data,
        x='method',
        y='rating',
        box=True,
        # points="all",
        color=variable,
        violinmode='group',
        title=
        f'Task: {task_map[question]} - Excerpt: {excerpt} - Controlled variable: {variable}'
    )

    fig_plot.update_traces(meanline_visible=True,
                           spanmode='manual',
                           span=[0, 1])

    if excerpt == 'mean':
        obj_eval = np.mean(obj_eval[:], axis=0)[np.newaxis]
        groupby = selected_data.groupby('method')['rating']
        excerpt_num = 0
    else:
        groupby = selected_data.loc[selected_data['excerpt_num'] ==
                                    excerpt].groupby('method')['rating']
        excerpt_num = excerpt
    correlations = compute_correlations(groupby, obj_eval, excerpt_num)
    st.write("Correlations:")
    st.table(correlations)
    methods = selected_data['method'].unique()
    fig_plot.add_trace(
        go.Scatter(x=methods,
                   y=obj_eval[excerpt_num, :, 1, 2],
                   name=measure_name))

    if not os.path.exists('imgs'):
        os.mkdir('imgs')
    fig_plot.write_image(f"imgs/{task_map[question]}_{excerpt}_{variable}.svg")
    st.write(fig_plot)

    if type(excerpt) is not int:
        groupby = selected_data
    else:
        groupby = selected_data.loc[selected_data['excerpt_num'] == excerpt]

    def _compute_errors_correlations(groupby_variable, selected_data_variable,
                                     var):
        this_groupby = groupby.loc[groupby_variable].groupby(
            'method')['rating']
        correlations = compute_correlations(this_groupby, obj_eval,
                                            excerpt_num)

        # error_margin_text.append(f"var {var}: ")
        error_margins = pd.DataFrame(index=methods,
                                     columns=['Sample Size', 'Error Margin'])
        for method in methods:
            # computing std, and error margin
            samples = selected_data.loc[selected_data_variable]
            samples = samples.loc[samples['method'] == method]['rating']
            sample_size = samples.count()
            std = samples.std()
            error_margins.loc[method] = [
                sample_size,
                np.sqrt((1.96**2) * (std**2) / sample_size)
            ]
        st.write("Error margins")
        st.write(error_margins)

        st.write(f"Correlations for {var}")
        st.table(correlations)

    if variable:
        variables = selected_data.unique()

        # computing all the distributions and kruskal-wallis test
        distributions = []
        for var in variables:
            for method in methods:
                distributions.append(
                    selected_data[(selected_data[variable] == var).values *
                                  (selected_data['method']
                                   == method).values]['rating'].values)

        # computing wilcoxon test
        _compute_wilcoxon_pvals(selected_data, question, excerpt, variable)
        for var in variables:
            groupby_variable = groupby[variable] == var
            print(groupby_variable)
            selected_data_variable = selected_data[variable] == var
            _compute_errors_correlations(groupby_variable,
                                         selected_data_variable, var)

    else:
        _compute_errors_correlations(groupby['rating'] > -1,
                                     selected_data['rating'] > -1, 'all')
        distributions = [
            selected_data[selected_data['method'] == method]['rating'].values
            for method in methods
        ]

        _compute_wilcoxon_pvals(selected_data, question, excerpt, variable)

    kruskal_h, kruskal_pval = kruskal(*distributions)
    st.write(
        f"Kruskal-Wallis (p-value, h-statistics): {kruskal_pval:.2e}, {kruskal_h:.2f}"
    )
    f_h, f_pval = f_oneway(*distributions)
    st.write(f"ANOVA (p-value, h-statistics): {f_pval:.2e}, {f_h:.2f}")

    st.write("---")


def correct_pvalues(pval):
    pval_indices = np.nonzero(~np.isnan(pval))
    _, corrected_pval, _, _ = multipletests(pval[pval_indices].flatten(),
                                            method='holm')
    pval[pval_indices] = corrected_pval


def _compute_wilcoxon_pvals(selected_data, question, excerpt, variable):
    methods = selected_data['method'].unique()
    if variable:
        variables = selected_data[variable].unique()
        # computing wilcoxon matries for each method
        for method in methods:
            samples = selected_data.loc[selected_data['method'] == method]
            pval = np.full((len(variables), len(variables)), np.nan)
            for i, expi in enumerate(variables):
                for j, expj in enumerate(variables):
                    if i > j:
                        try:
                            datai = samples.loc[samples[variable] == expi]
                            dataj = samples.loc[samples[variable] == expj]
                            maxlen = min(len(datai), len(dataj))
                            _, pval[i, j] = wilcoxon(datai['rating'][:maxlen],
                                                     dataj['rating'][:maxlen])
                        except Exception as e:
                            print("\nError in Wilcoxon test!:")
                            print(e)
                            print()
            if pval.shape[0] > 2:
                correct_pvalues(pval)
            st.write(pd.DataFrame(pval, columns=variables, index=variables))

        # computing wilcoxon matrices for each variable
        for var in variables:
            samples = selected_data.loc[selected_data[variable] == var]
            _wilcoxon_on_methods(methods, samples, var)
    else:
        samples = selected_data
        _wilcoxon_on_methods(methods, samples, 'all')


def _wilcoxon_on_methods(methods, samples, var):
    pval = np.full((len(methods), len(methods)), np.nan)
    for i, expi in enumerate(methods):
        for j, expj in enumerate(methods):
            if i > j:
                try:
                    datai = samples.loc[samples['method'] == expi]
                    dataj = samples.loc[samples['method'] == expj]
                    maxlen = min(len(datai), len(dataj))
                    _, pval[i, j] = wilcoxon(datai['rating'][:maxlen],
                                             dataj['rating'][:maxlen])
                except Exception as e:
                    print("\nError in Wilcoxon test!:")
                    print(e)
                    print()
    if pval.shape[0] > 2:
        correct_pvalues(pval)
    st.write(f"p-values for {var}")
    st.write(pd.DataFrame(pval, columns=methods, index=methods))
