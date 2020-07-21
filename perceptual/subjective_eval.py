import sqlite3
import os
import re
import datetime
import pandas as pd
import plotly.express as px
import scipy.stats
import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import xml.etree.ElementTree as ET
from tqdm import tqdm
from joblib import Parallel, delayed

PATH = "/home/sapo/Develop/http/listening/saves"
DISCARD_BEFORE_THAN = datetime.datetime(year=2020,
                                        month=6,
                                        day=5,
                                        hour=12,
                                        minute=45,
                                        second=0)
MAP_VALUES = {'0': 0, '1': 0, '2': 1, '3': 1, '4': 1}

METHODS = {
    '0': 'hidden reference',
    '1': 'negative reference',
    '2': 'score-informed',
    '3': 'vienna',
    '4': 'o&f',
    't': 'reference'
}

EXCERPTS = {'n0': 0, 'n1': 1, 'n2': 2, 'n3': 3, 'medoid': 4}

SAVE_PATH = './figures'

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)


def xml2sqlite(path):
    """
    Parse an xml save from WAET and creates a sqlite database in RAM with two
    tables:

        CREATE TABLE "USERS" (
            "id"    INTEGER,
            "expertise" TEXT,
            "habits_classical"  TEXT,
            "habits_general"    TEXT,
            "headphones"    TEXT,
            PRIMARY KEY("id")
        );
        CREATE TABLE "ANSWERS" (
            "id"	INTEGER,
            "user_id"	INTEGER NOT NULL,
            "listen_time"	REAL,
            "cursor_moved"	BOOLEAN,
            "question"	TEXT,
            "excerpt_num"	INTEGER,
            "method"	TEXT,
            "rating"	REAL,
            PRIMARY KEY("id")
        );

    It also maps variable values according to the `MAP_VALUES` dict

    """
    db = sqlite3.connect(':memory:')
    db_cursor = db.cursor()
    db_cursor.execute("""
CREATE TABLE "USERS" (
    "id"    INTEGER,
    "expertise" TEXT,
    "habits_classical"  TEXT,
    "habits_general"    TEXT,
    "headphones"    TEXT,
    PRIMARY KEY("id")
); """)

    db_cursor.execute("""
CREATE TABLE "ANSWERS" (
    "id"	INTEGER,
    "user_id"	INTEGER NOT NULL,
    "listen_time"	REAL,
    "cursor_moved"	BOOLEAN,
    "question"	TEXT,
    "excerpt_num"	INTEGER,
    "method"	TEXT,
    "rating"	REAL,
    PRIMARY KEY("id")
); """)

    print("Loading files")
    for root, dir, files in os.walk(path):
        for file in files:
            if file.endswith('.xml'):
                doc = ET.parse(os.path.join(root, file))

                # skipping if old
                dates = doc.find("./datetime")
                if dates:
                    time = dates.find("./time")
                    date = dates.find("./date")
                    execution_time = datetime.datetime(
                        year=int(date.get('year')),
                        month=int(date.get('month')),
                        day=int(date.get('day')),
                        hour=int(time.get('hour')),
                        minute=int(time.get('minute')),
                        second=int(time.get('secs')))
                    if execution_time < DISCARD_BEFORE_THAN:
                        print(
                            f"Skipping one save because before than {DISCARD_BEFORE_THAN}"
                        )
                        continue

                # checking variables
                variables = doc.findall(
                    "./survey[@location='pre']/surveyresult")
                variables_dict = {
                    'expertise': 'NULL',
                    'habits_general': 'NULL',
                    'habits_classical': 'NULL',
                    'headphones': 'NULL'
                }
                for variable in variables:
                    value = variable.find('./response').get('name')
                    name = variable.get('ref')
                    variables_dict[name] = MAP_VALUES[value]

                # inserting user
                db_cursor.execute(f"""
INSERT INTO "USERS"
("expertise", "habits_classical", "habits_general", "headphones")
VALUES (
{variables_dict['expertise']},
{variables_dict['habits_classical']},
{variables_dict['habits_general']},
{variables_dict['headphones']}); """)
                user_id = db_cursor.lastrowid

                # inserting answers
                audio_answers = doc.findall("./page/audioelement")
                for answer in audio_answers:
                    # metrics
                    metrics = answer.findall('./metric/metricresult')
                    listen_time = cursor_moved = None
                    for m in metrics:
                        if m.get('name') == 'enableElementTimer':
                            listen_time = float(m.text)
                        else:
                            cursor_moved = m.text.lower() == 'true'
                    if listen_time is None or cursor_moved is None:
                        continue

                    # inferring question and method
                    m = re.match(r"(.*)_(.*)-(.)", answer.get('ref'))
                    if m is None:
                        continue
                    question = m.group(1)
                    excerpt_num = EXCERPTS[m.group(2)]
                    method = METHODS[m.group(3)]
                    if method == 'reference':
                        # this is the reference, no rating
                        continue
                    rating = float(answer.find('./value').text)

                    db_cursor.execute(f"""
INSERT INTO "ANSWERS"
("user_id", "listen_time", "cursor_moved", "question",
"excerpt_num", "method", "rating")
VALUES (
{user_id}, {listen_time}, {cursor_moved},
"{question}", {excerpt_num}, "{method}", {rating}); """)

    db_cursor.close()
    db.commit()
    return db


def sqlite2pandas(db,
                  variable=None,
                  ordinal=False,
                  min_listen_time=5,
                  cursor_moved=True):
    """
    Given an sqlite db, this does a query and returns the table in a dataframe
    """

    # building SQL query
    if variable is not None:
        SQL = f'SELECT "{variable}", '
    else:
        SQL = 'SELECT '

    SQL += '"question", "excerpt_num", "method", "rating", "user_id"\
        FROM ANSWERS JOIN USERS ON user_id == USERS.id'

    need_where = True
    if variable:
        SQL += f' WHERE "{variable}" IS NOT NULL'
        need_where = False
    if cursor_moved:
        if need_where:
            SQL += ' WHERE '
            need_where = False
        else:
            SQL += ' AND '
        SQL += f'"cursor_moved" IS {cursor_moved}'
    if min_listen_time is not None:
        if need_where:
            SQL += ' WHERE '
            need_where = False
        else:
            SQL += ' AND '
        SQL += f'"listen_time" >= {min_listen_time}'

    print("Extracting data")
    data = pd.read_sql(SQL, db)
    if ordinal:
        # changing ratings according to the ordinal value
        questions = data['question'].unique()
        excerpts = data['excerpt_num'].unique()
        users = data['user_id'].unique()
        for question in questions:
            d0 = data.loc[data['question'] == question]
            for excerpt in excerpts:
                d1 = d0.loc[d0['excerpt_num'] == excerpt]
                for user in users:
                    d2 = d1.loc[d1['user_id'] == user]
                    arr = d2['rating'].to_numpy()
                    sort_idx = np.argsort(arr)
                    arr[sort_idx] = list(range(len(arr)))
                    data.loc[d2.index, 'rating'] = arr
    return data


def count_users(db):
    db_cursor = db.cursor()

    print("\nCounting users for each variable:")
    print("------------------------")
    for variable in [
            'expertise', 'headphones', 'habits_classical', 'habits_general'
    ]:
        tot = 0
        for value in range(5):
            SQL = f'SELECT COUNT(*) FROM USERS WHERE "{variable}" == {value}'
            db_cursor.execute(SQL)
            answer = db_cursor.fetchall()[0][0]
            tot += int(answer)
            print(f"{variable} {value}: {answer}")
        print(f"Tot: {tot}")
        print("------------------------")
    print("\n")


def plot(path,
         excerpts_mean=True,
         variable=None,
         ordinal=False,
         *args,
         **kwargs):
    """
    Arguments
    ---------

    `path` : str
        the path to the `saves` folder

    `excerpt_mean` : bool
        if True, plots the average over all the excerpts per each question type

    `variable` : str or None
        the variable to be observed: each value of the variable will be a
        different line and the Wilcoxon rank test will be computed for all the
        combinations of the variable value

    `ordinal` : bool
        if ratings should be considered by ordinal or not

    `*args, **kwargs` :
        other arguments passed to `sqlite2pandas` :

        `min_listen_time` : float or int or None
            the minimum time that an audio should have been listened to for
            being taking into account; if a user listened to an audio for less
            than `min_listen_time`, his answer is discarded

        `cursor_moved` : bool
            if True, only answers where users moved the cursor are taken into
            account

    Returns
    -------

    `list[dash_core_components.Graph]` :
        the list of graphs to be plotted in Dash
    """

    # loading data from xml to pandas
    db = xml2sqlite(path)
    count_users(db)
    df = sqlite2pandas(db, variable=variable, ordinal=ordinal, *args, **kwargs)

    # taking the number of questions and excerpts
    questions = df.question.unique()
    excerpts = df.excerpt_num.unique()

    # selecting data
    print("Plotting")

    def process(question):
        graphs = []
        if excerpts_mean:
            excerpt = 'mean'
            selected_data = df.loc[df['question'] == question].sort_values(
                ['method', variable])
            # plotting means of the excerpts
            graphs.append(
                _plot_data(selected_data, question, excerpt, variable))
        else:
            for excerpt in sorted(excerpts):
                selected_data = df.loc[df['question'] == question].loc[
                    df['excerpt_num'] == excerpt].sort_values(['method', variable])
                # plotting each excerpt
                graphs.append(
                    _plot_data(selected_data, question, excerpt, variable))
        return graphs

    graphs = Parallel(n_jobs=-2)(delayed(process)(question)
                                 for question in tqdm(questions))

    return [g0 for g1 in graphs for g0 in g1]


def _plot_data(selected_data, question, excerpt, variable):
    fig_plot = px.violin(
        selected_data,
        x='method',
        y='rating',
        box=True,
        points="all",
        color=variable,
        violinmode='group',
        title=f'question {question}, excerpt {excerpt}, variable {variable}')

    fig_plot.update_traces(meanline_visible=True)

    # fig_plot.write_image(
    #     os.path.join(SAVE_PATH, f'plot-{question}_{excerpt}_{variable}.svg'))

    if variable:
        variables = selected_data[variable].unique()
        methods = selected_data['method'].unique()

        fig_pvals = _compute_wilcoxon_pvals(selected_data, question, excerpt,
                                            variable)

        text = [
            "(size, margin error) with 95% of confidence",
            html.Br(), "methods: "
        ]
        text += [str(method) + ", " for method in methods]
        text.append(html.Br())
        for var in variables:
            text.append(f"var {var}: ")
            for method in methods:
                # computing std, expected sample size and real sample size:
                samples = selected_data.loc[selected_data[variable] == var]
                samples = samples.loc[samples['method'] == method]['rating']
                sample_size = samples.count()
                std = samples.std()
                margin_error = np.sqrt((1.96**2) * (std**2) / sample_size)
                text += [f"({sample_size} {margin_error:.2f}) "]
            text.append(html.Br())
        return html.Div([
            html.Div([
                html.Div([dcc.Graph(figure=fig_plot)], className="col-md-12"),
            ],
                className="row"),
            html.Div([
                html.Div([dcc.Graph(figure=fig)], className="col-md-2")
                for fig in fig_pvals
            ],
                className="row"),
            html.P(text)
        ])

    else:
        return dcc.Graph(figure=fig_plot)


def _compute_wilcoxon_pvals(selected_data, question, excerpt, variable):
    variables = selected_data[variable].unique()
    methods = selected_data['method'].unique()
    # computing wilcoxon matries for each method
    fig_pvals = []
    for method in methods:
        samples = selected_data.loc[selected_data['method'] == method]
        pval = np.ones((len(variables), len(variables)))
        for i, expi in enumerate(variables):
            for j, expj in enumerate(variables):
                if i != j:
                    try:
                        datai = samples.loc[samples[variable] == expi]
                        dataj = samples.loc[samples[variable] == expj]
                        maxlen = min(len(datai), len(dataj))
                        _, pval[i, j] = scipy.stats.wilcoxon(
                            datai['rating'][:maxlen], dataj['rating'][:maxlen])
                    except Exception as e:
                        print("\nError in Wilcoxon test!:")
                        print(e)
                        print()
        fig_pval = px.imshow(pval,
                             x=variables,
                             y=variables,
                             title=f'method {method}',
                             range_color=[0, 0.5])

        # fig_pval.write_image(
        #     os.path.join(SAVE_PATH,
        #                  f'pval-{question}_{excerpt}_{variable}.svg'))

        fig_pvals.append(fig_pval)

    # computing wilcoxon matries for each variable
    for var in variables:
        samples = selected_data.loc[selected_data[variable] == var]
        pval = np.ones((len(methods), len(methods)))
        for i, expi in enumerate(methods):
            for j, expj in enumerate(methods):
                if i != j:
                    try:
                        datai = samples.loc[samples['method'] == expi]
                        dataj = samples.loc[samples['method'] == expj]
                        maxlen = min(len(datai), len(dataj))
                        _, pval[i, j] = scipy.stats.wilcoxon(
                            datai['rating'][:maxlen], dataj['rating'][:maxlen])
                    except Exception as e:
                        print("\nError in Wilcoxon test!:")
                        print(e)
                        print()
        fig_pval = px.imshow(pval,
                             x=methods,
                             y=methods,
                             title=f'var {var}',
                             range_color=[0, 0.5])

        # fig_pval.write_image(
        #     os.path.join(SAVE_PATH,
        #                  f'pval-{question}_{excerpt}_{variable}.svg'))
        fig_pvals.append(fig_pval)
    return fig_pvals


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"""
Syntax:

{sys.argv[0]} [variable] [true|false] [ordinal]

* `variable`: one of 'expertise', 'headphones', 'habits_classical', 'habits_general'. Default: 'expertise'

* `true|false`: if true, the average over the excerpts are computed, if false, the results for each single excerpts are shown. Default: true

* `ordinal` : if used, absolute ratings given by users are substituted with their their ordinal position (higher is more similar)
""")
        variable = 'expertise'
    else:
        variable = sys.argv[1]
    if len(sys.argv) > 2:
        if sys.argv[2] == 'false':
            excerpt_mean = False
        else:
            excerpt_mean = True
    else:
        excerpt_mean = True

    ordinal = False
    if len(sys.argv) > 3:
        if sys.argv[3] == 'ordinal':
            ordinal = True

    graphs = plot(PATH,
                  variable=variable,
                  excerpts_mean=excerpt_mean,
                  min_listen_time=1,
                  cursor_moved=False,
                  ordinal=ordinal)

    app = dash.Dash(external_stylesheets=[
        'https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css'
    ])
    app.layout = html.Div(graphs)
    app.run_server(debug=False, use_reloader=False)
