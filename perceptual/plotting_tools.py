import os

import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from joblib import Parallel, delayed
from scipy.stats import (f_oneway, kendalltau, kruskal, pearsonr, spearmanr,
                         wilcoxon)
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm


def plot(df, obj_eval, excerpts_mean=True, variable=None):
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
        graphs = []
        if excerpts_mean:
            excerpt = 'mean'
            selected_data = df.loc[df['question'] == question].sort_values(
                sort_by)
            # plotting means of the excerpts
            graphs.append(
                _plot_data(selected_data, question, excerpt, variable,
                           obj_eval))
        else:
            for excerpt in sorted(excerpts):
                selected_data = df.loc[df['question'] == question].loc[
                    df['excerpt_num'] == excerpt].sort_values(sort_by)
                # plotting each excerpt
                graphs.append(
                    _plot_data(selected_data, question, excerpt, variable,
                               obj_eval))
        return graphs

    graphs = Parallel(n_jobs=-2)(delayed(process)(question)
                                 for question in tqdm(questions))

    return [g0 for g1 in graphs for g0 in g1]


def compute_correlations(groupby, obj_eval, excerpt):
    correlations = np.zeros((3, 2, 3, 2, 2))
    for i in range(2):
        for j in range(3):
            for k, correl_func in enumerate([pearsonr, spearmanr, kendalltau]):
                correlations[k, i, j,
                             0, :] = correl_func(obj_eval[excerpt, :, i, j],
                                                 groupby.mean())
                correlations[k, i, j,
                             1, :] = correl_func(obj_eval[excerpt, :, i, j],
                                                 groupby.median())
    return correlations


def create_text_for_correlations(correlations):
    correl_text = [
        html.Br(), "correlation coeffs (corr, pval)",
        html.Br(), "[mean, median] for each variable",
        html.Br()
    ]
    for k in range(correlations.shape[0]):
        correl_text.append(["pearson", "spearman", "kendall"][k] +
                           " coefficients")
        correl_text.append(html.Br())
        for i in range(correlations.shape[1]):
            for j in range(correlations.shape[2]):
                if i == 0:
                    correl_text.append("w/o vel, ")
                elif i == 1:
                    correl_text.append("w/ vel, ")

                if j == 0:
                    correl_text.append("p [")
                elif j == 1:
                    correl_text.append("r [")
                elif j == 2:
                    correl_text.append("f [")

                for var in range(correlations.shape[3]):
                    val = correlations[k, i, j, var]
                    correl_text.append(f"({val[0]:.2f}, {val[1]:.2f})")
                    if var % 2 == 1:
                        correl_text.append("] - [")
                    else:
                        correl_text.append(" ")
                correl_text.append(html.Br())
            correl_text += ["----------", html.Br()]

        correl_text += ["----------", html.Br(), "-----------", html.Br()]
    return correl_text


def _plot_data(selected_data, question, excerpt, variable, obj_eval):
    fig_plot = px.violin(
        selected_data,
        x='method',
        y='rating',
        box=True,
        # points="all",
        color=variable,
        violinmode='group',
        title=f'question {question}, excerpt {excerpt}, variable {variable}')

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
    methods = selected_data['method'].unique()
    fig_plot.add_trace(go.Scatter(x=methods, y=obj_eval[excerpt_num, :, 1, 2]))

    if not os.path.exists('imgs'):
        os.mkdir('imgs')
    fig_plot.write_image(f"imgs/{question}_{excerpt}_{variable}.svg")

    error_margin_text = [
        "(size, margin error) with 95% of confidence",
        html.Br(), "methods: "
    ]
    error_margin_text += [str(method) + ", " for method in methods]
    error_margin_text.append(html.Br())
    if type(excerpt) is not int:
        groupby = selected_data
    else:
        groupby = selected_data.loc[selected_data['excerpt_num'] == excerpt]

    def _compute_errors_correlations(groupby_variable, selected_data_variable,
                                     correlations, var):
        this_groupby = groupby.loc[groupby_variable].groupby(
            'method')['rating']
        correlations = np.concatenate([
            correlations,
            compute_correlations(this_groupby, obj_eval, excerpt_num)
        ],
            axis=-2)

        error_margin_text.append(f"var {var}: ")
        for method in methods:
            # computing std, and error margin
            samples = selected_data.loc[selected_data_variable]
            samples = samples.loc[samples['method'] == method]['rating']
            sample_size = samples.count()
            std = samples.std()
            margin_error = np.sqrt((1.96**2) * (std**2) / sample_size)
            error_margin_text.append(f"({sample_size} {margin_error:.2f}) ")
        error_margin_text.append(html.Br())
        return correlations

    if variable:
        variables = selected_data[variable].unique()

        # computing all the distributions and kruskal-wallis test
        distributions = []
        for var in variables:
            for method in methods:
                distributions.append(
                    selected_data[(selected_data[variable] == var).values *
                                  (selected_data['method'] == method).values]
                    ['rating'].values)

        # computing wilcoxon test
        fig_pvals = _compute_wilcoxon_pvals(selected_data, question, excerpt,
                                            variable)

        for var in variables:
            groupby_variable = groupby[variable] == var
            print(groupby_variable)
            selected_data_variable = selected_data[variable] == var
            correlations = _compute_errors_correlations(
                groupby_variable, selected_data_variable, correlations, var)

        correl_text = create_text_for_correlations(correlations)

    else:
        correlations = _compute_errors_correlations(
            groupby['rating'] > -1, selected_data['rating'] > -1, correlations,
            'all')
        correl_text = create_text_for_correlations(correlations)
        distributions = [
            selected_data[selected_data['method'] == method]['rating'].values
            for method in methods
        ]

        fig_pvals = _compute_wilcoxon_pvals(selected_data, question, excerpt,
                                            variable)
    kruskal_h, kruskal_pval = kruskal(*distributions)
    kruskal_text = [
        f"Kruskal-Wallis (p-value, h-statistics): {kruskal_pval}, {kruskal_h}"
    ]
    f_h, f_pval = f_oneway(*distributions)
    f_text = [f"ANOVA (p-value, h-statistics): {f_pval}, {f_h}"]

    return get_html_div(fig_plot, fig_pvals, kruskal_text, error_margin_text,
                        correl_text, f_text)


def correct_pvalues(pval):
    pval_indices = np.tril_indices(pval.shape[0], -1)
    _, corrected_pval, _, _ = multipletests(pval[pval_indices].flatten(),
                                            method='holm')
    pval[pval_indices] = corrected_pval


def _compute_wilcoxon_pvals(selected_data, question, excerpt, variable):
    methods = selected_data['method'].unique()
    fig_pvals = []
    if variable:
        variables = selected_data[variable].unique()
        # computing wilcoxon matries for each method
        for method in methods:
            samples = selected_data.loc[selected_data['method'] == method]
            pval = np.ones((len(variables), len(variables))) * 2
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
            fig_pval = px.imshow(pval,
                                 x=variables,
                                 y=variables,
                                 title=f'method {method}',
                                 range_color=[0, 0.5])

            # fig_pval.write_image(
            #     os.path.join(SAVE_PATH,
            #                  f'pval-{question}_{excerpt}_{variable}.svg'))

            fig_pvals.append(fig_pval)

        # computing wilcoxon matrices for each variable
        for var in variables:
            samples = selected_data.loc[selected_data[variable] == var]
            fig_pval = _wilcoxon_on_methods(methods, samples, var)
            fig_pvals.append(fig_pval)
    else:
        samples = selected_data
        fig_pval = _wilcoxon_on_methods(methods, samples, 'all')
        fig_pvals.append(fig_pval)

    return fig_pvals


def _wilcoxon_on_methods(methods, samples, var):
    pval = np.ones((len(methods), len(methods))) * 2
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
    fig_pval = px.imshow(pval,
                         x=methods,
                         y=methods,
                         title=f'var {var}',
                         range_color=[0, 0.5])
    return fig_pval

    # fig_pval.write_image(
    #     os.path.join(SAVE_PATH,
    #                  f'pval-{question}_{excerpt}_{variable}.svg'))


def get_html_div(fig_plot, fig_pvals, kruskal_text, error_margin_text,
                 correl_text, f_text):
    return html.Div([
        html.Div([
            html.Div([dcc.Graph(figure=fig_plot)], className="col-md-12"),
        ],
            className="row"),
        html.Div([
            html.Div([dcc.Graph(figure=fig)], className="col-md-12")
            for fig in fig_pvals
        ],
            className="row"),
        html.P(kruskal_text),
        html.P(f_text),
        html.P(error_margin_text),
        html.P(correl_text)
    ])
