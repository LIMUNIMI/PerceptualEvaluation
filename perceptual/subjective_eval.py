import sqlite3
import os
import re
import datetime
import pandas as pd
import dash
import dash_html_components as html
import numpy as np
import xml.etree.ElementTree as ET
import pickle
from .plotting_tools import plot
from .objective_eval import excerpts_test, EXCERPTS
from . import utils, objective_eval, excerpt_search

#################################
# SETTINGS:
PATH = "/home/sapo/Develop/http/listening/saves"
DISCARD_BEFORE_THAN = datetime.datetime(year=2020,
                                        month=6,
                                        day=5,
                                        hour=12,
                                        minute=45,
                                        second=0)
MAP_VALUES = {'0': 0, '1': 0, '2': 1, '3': 1, '4': 1}

#################################

METHODS = {
    '0': 'hr',
    '1': 'nr',
    '2': 'si',
    '3': 'vienna',
    '4': 'o&f',
    't': 'ref'
}

SAVE_PATH = './figures'

if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)


def get_peamt():
    import sys
    root = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(os.path.realpath(os.path.join(root, "PEAMT/")))
    from .PEAMT.peamt import PEAMT
    # there's a subtle incompatibility in PEAMT:
    import pickle
    old_pickle_load = pickle.load

    def new_pickle_load(x, encoding='latin1'):
        return old_pickle_load(x, encoding=encoding)

    pickle.load = new_pickle_load
    return PEAMT(
        parameters=os.path.join(root, 'PEAMT/model_parameters/PEAMT.pkl'))


def symbolic_bpms(mat):

    pr = utils.make_pianoroll(mat,
                              res=0.01,
                              only_onsets=True,
                              velocities=False)
    windows = []
    for win_len in [10, 100, 1000]:
        # [10, 100, 1000]:
        if win_len > pr.shape[1]:
            win_len = pr.shape[1]
        end = win_len * (pr.shape[1] // win_len)
        windows.append(np.sum(pr[:, :end].reshape((128, -1, win_len)),
                              axis=-1))

    bpms = []
    for wins in windows:
        if np.any(wins):
            bpms.append(np.mean(wins))
            bpms.append(np.std(wins))
        else:
            bpms.append(0)
            bpms.append(0)

    return bpms


def evaluate(target,
             pred):

    scaler = pickle.load(open('scaler.pkl', 'rb'))
    model = pickle.load(open('metric.pkl', 'rb'))
    if type(target) is str:
        target = utils.midipath2mat(target)
    if type(pred) is str:
        pred = utils.midipath2mat(pred)
    midi = utils.make_pianoroll(pred, res=0.005)
    features1 = excerpt_search.score_features(midi)
    features1 += symbolic_bpms(pred)
    features1 = scaler.transform(np.array(features1).reshape(1, -1))[0]
    midi = utils.make_pianoroll(target, res=0.005)
    features2 = excerpt_search.score_features(midi)
    features2 += symbolic_bpms(target)
    features2 = scaler.transform(np.array(features2).reshape(1, -1))[0]

    fmeasure = objective_eval.evaluate(target, pred)[1][2]

    features = features1 - features2
    features = np.concatenate([features, [fmeasure]])[model.selected_features]

    return model.predict(features[None])


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
                    if method == 'ref':
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


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(f"""
Syntax:

{sys.argv[0]} variable ['average'] ['ordinal'] ['rescale'] ['our_eval'|'peamt']

* `variable`:  is the controlled variable: one of 'expertise', 'headphones', 'habits_classical', 'habits_general', 'all'

* `average`: if used, all answers from all excerpts are used for each question type; this will results in one plot per question type instead of one plot per (question type, excerpt). Also not that the median and mean values are *averaged* over all the excerpts.

* `ordinal` : if used, absolute ratings given by users are substituted with their ordinal position (higher is more similar); in this way the pearson coefficient should change, but the pearsman and kendall coefficients shouldn't.

* `rescale` : if used, objective_evaluation is rescaled with `f(x) = 1 - 10**(-x)`

* `our_eval` : if used, objective evaluation metric is substituted by our metric (based on the linear regression of the median subjective metric)

* `peamt` : if used, objective evaluation metric is substituted by peamt metric (another subjective metric)
""")
        sys.exit()
    else:
        variable = sys.argv[1]
        if variable == 'all':
            variable = None

    excerpt_mean = 'average' in sys.argv

    ordinal = 'ordinal' in sys.argv

    rescale = 'rescale' in sys.argv

    # loading data from xml to pandas
    db = xml2sqlite(PATH)
    count_users(db)
    df = sqlite2pandas(db,
                       variable=variable,
                       min_listen_time=5,
                       cursor_moved=True,
                       ordinal=ordinal)
    if 'our_eval' in sys.argv:
        obj_eval = excerpts_test(ordinal=ordinal, evaluate=evaluate)
    elif 'peamt' in sys.argv:
        peamt = get_peamt()
        obj_eval = excerpts_test(ordinal=ordinal,
                                 evaluate=peamt.evaluate_from_midi)
    else:
        obj_eval = excerpts_test(ordinal=ordinal)

    if rescale:
        obj_eval = 1 - 10**(-obj_eval)

    graphs = plot(df, obj_eval, variable=variable, excerpts_mean=excerpt_mean)

    app = dash.Dash(external_stylesheets=[
        'https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css'
    ])
    app.layout = html.Div(graphs)
    app.run_server(debug=False, use_reloader=False)
