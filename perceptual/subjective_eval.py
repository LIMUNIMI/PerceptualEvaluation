import sqlite3
import os
import re
import datetime
import pandas as pd
import xml.etree.ElementTree as ET

PATH = "/home/sapo/Develop/http/listening/saves"
DISCARD_BEFORE_THAN = datetime.datetime(year=2020,
                                        month=6,
                                        day=5,
                                        hour=12,
                                        minute=45,
                                        second=0)

METHODS = {
    '0': 'hidden reference',
    '1': 'negative reference',
    '2': 'score-informed',
    '3': 'vienna',
    '4': 'o&f',
    't': 'reference'
}

EXCERPTS = {'n0': 0, 'n1': 1, 'n2': 2, 'n3': 3, 'medoid': 4}


def xml2sqlite(path):
    db = sqlite3.connect(':memory:')
    db_cursor = db.cursor()
    db_cursor.execute("""
CREATE TABLE "USERS" (
    "id"    INTEGER,
    "expertise" INTEGER,
    "habits_classical"  INTEGER,
    "habits_general"    INTEGER,
    "headphones"    INTEGER,
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
                            f"Skipping one save because before than f{DISCARD_BEFORE_THAN}"
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
                    value = int(variable.find('./response').get('name'))
                    name = variable.get('ref')
                    variables_dict[name] = value

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


def sqlite2pandas(path,
                  db,
                  variable=None,
                  min_listen_time=5,
                  cursor_moved=True):

    # beuilding SQL query
    if variable is not None:
        SQL = f'SELECT "{variable}", '
    else:
        SQL = 'SELECT '

    SQL += '"question", "excerpt_num", "method", "rating"\
        FROM ANSWERS JOIN USERS ON user_id == USERS.id'

    need_and = False
    need_where = True
    if variable is not None:
        SQL += f' WHERE "{variable}" IS NOT NULL'
        need_and = True
        need_where = False
    if cursor_moved is not None:
        if need_and:
            SQL += ' AND '
        if need_where:
            SQL += ' WHERE '
        SQL += f'"cursor_moved" IS {cursor_moved}'
    if min_listen_time is not None:
        if need_and:
            SQL += ' AND '
        if need_where:
            SQL += ' WHERE '
        SQL += f'"listen_time" >= {min_listen_time}'

    return pd.read_sql(SQL, db)


def plot(path, *args, **kwargs):
    db = xml2sqlite(path)
    df = sqlite2pandas(path, db, *args, **kwargs)

    return df

if __name__ == "__main__":
    db = xml2sqlite(PATH)
    df = sqlite2pandas(PATH,
                       db,
                       'expertise',
                       min_listen_time=0.5,
                       cursor_moved=False)
    __import__('ipdb').set_trace()
