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

PATH = "/home/sapo/Develop/http/listening/saves"
DISCARD_BEFORE_THAN = datetime.datetime(year=2020,
                                        month=6,
                                        day=5,
                                        hour=12,
                                        minute=45,
                                        second=0)
MAP_VALUES = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4}

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


def sqlite2pandas(db, variable=None, min_listen_time=5, cursor_moved=True):
    """
    Given an sqlite db, this does a query and returns the table in a dataframe
    """

    # building SQL query
    if variable is not None:
        SQL = f'SELECT "{variable}", '
    else:
        SQL = 'SELECT '

    SQL += '"question", "excerpt_num", "method", "rating"\
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
    return pd.read_sql(SQL, db)


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


def plot(path, excerpts_mean=True, variable=None, *args, **kwargs):
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
    def _plot_data(selcted_data):
        groupby = [variable, 'method']
        if not variable:
            groupby = 'method'
        data = selected_data.groupby(
            groupby, as_index=True)['rating'].mean().to_frame()
        data['std'] = selected_data.groupby(groupby,
                                            as_index=True)['rating'].std()
        data['count'] = selected_data.groupby(groupby,
                                              as_index=True)['rating'].count()
        data = data.reset_index()
        fig_plot = px.bar(
            data,
            x='method',
            y='rating',
            color=variable,
            barmode='group',
            text='count',
            title=f'question {question}, excerpt {excerpt}, variable {variable}',
            error_y='std')

        # fig_plot.update_traces(textposition='outside')
        fig_plot.write_image(
            os.path.join(SAVE_PATH,
                         f'plot-{question}_{excerpt}_{variable}.svg'))

        # computing wilcoxon matrix
        if variable:
            variables = data[variable].unique()
            pval = np.ones((len(variables), len(variables)))
            for i, expi in enumerate(variables):
                for j, expj in enumerate(variables):
                    if i != j:
                        try:
                            _, pval[i, j] = scipy.stats.wilcoxon(
                                data.loc[data[variable] == expi]['rating'],
                                data.loc[data[variable] == expj]['rating'])
                        except Exception as e:
                            print("\nError in Wilcoxon test!:")
                            print(e)
                            print()
            fig_pval = px.imshow(
                pval,
                x=variables,
                y=variables,
                title=f'question {question}, excerpt {excerpt}, variable {variable}',
                range_color=[0, 0.5])

            fig_pval.write_image(
                os.path.join(SAVE_PATH,
                             f'pval-{question}_{excerpt}_{variable}.svg'))
            return html.Div([
                html.Div([dcc.Graph(figure=fig_plot)], className="col-md-8"),
                html.Div([dcc.Graph(figure=fig_pval)], className="col-md-4")
            ],
                className="row")

        else:
            return dcc.Graph(figure=fig_plot)

    graphs = []
    # loading data from xml to pandas
    db = xml2sqlite(path)
    count_users(db)
    df = sqlite2pandas(db, variable=variable, *args, **kwargs)

    # taking the number of questions and excerpts
    questions = df.question.unique()
    excerpts = df.excerpt_num.unique()

    # selecting data
    print("Plotting")
    for question in tqdm(questions):
        if excerpts_mean:
            excerpt = 'mean'
            selected_data = df.loc[df['question'] == question]
            # plotting means of the excerpts
            graphs.append(_plot_data(selected_data))
        else:
            for excerpt in sorted(excerpts):
                selected_data = df.loc[df['question'] == question].loc[
                    df['excerpt_num'] == excerpt]
                # plotting each excerpt
                graphs.append(_plot_data(selected_data))

    return graphs


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"""
Syntax:

{sys.argv[0]} [variable] [true|false]

* `variable`: one of 'expertise', 'headphones', 'habits_classical', 'habits_general'. Default: 'expertise'

* `true|false`: if true, the average over the excerpts are computed, if false, the results for each single excerpts are shown. Default: true

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
    graphs = plot(PATH,
                  variable=variable,
                  excerpts_mean=excerpt_mean,
                  min_listen_time=1,
                  cursor_moved=False)
    # graphs = plot(PATH,
    #               variable='expertise',
    #               excerpts_mean=True,
    #               min_listen_time=5,
    #               cursor_moved=True)

    app = dash.Dash(external_stylesheets=[
        'https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css'
    ])
    app.layout = html.Div(graphs)
    app.run_server(debug=False, use_reloader=False)
