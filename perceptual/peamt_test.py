import os
import pandas
from . import subjective_eval
from . import objective_eval
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np

ANSWER_PATH = './peamt_results/answers_data.csv'
MIDI_PATH = './peamt_results/all_midi_cut'


def load_pmt_data():
    print("Loading csv...")
    df = pandas.read_csv(ANSWER_PATH, sep=';')

    # selecting difficulty 1 or 2
    def check_pooled_std():
        for i in range(1, 5):
            A = df.loc[(df['difficulty'] == i)]
            print(f"pooled std for difficulty {i}: {std_pooled(A):.4f}")
    print("Before than filtering...")
    check_pooled_std()

    df = df.loc[(df['difficulty'] == 1) | (df['difficulty'] == 2)]

    # taking only questions with more than 4 answer
    df = df.loc[
        df.groupby('question_id')['question_id'].transform('size') >= 4]

    # counts = df.groupby('user_id')['answer'].count()
    # count = counts.loc[counts < 10]
    # df = df.loc[df['user_id'].isin(count.index)]
    # print("After filtering...")
    # check_pooled_std()
    return df


def evaluate_measure(df,
                     eval_func=subjective_eval.get_peamt().evaluate_from_midi):
    def process(df, example):
        agreement = 0
        root = os.path.join(MIDI_PATH, example)
        ex_df = df.loc[df['example'] == example]
        for sys1 in ex_df['system1'].unique():
            ex_df_s1 = ex_df.loc[ex_df['system1'] == sys1]
            for sys2 in ex_df_s1['system2'].unique():
                targ = os.path.join(root, 'target.mid')
                midi1 = os.path.join(root, sys1 + '.mid')
                midi2 = os.path.join(root, sys2 + '.mid')

                m1 = eval_func(targ, midi1)
                m2 = eval_func(targ, midi2)
                pred = np.argmax([m1, m2])
                ex_df_s1_s2 = ex_df_s1.loc[ex_df_s1['system2'] == sys2]

                # count how many values were rated in the predicted way
                agreement += (ex_df_s1_s2['answer'] == pred).sum()

        return agreement

    print("Evaluating...")
    examples = df['example'].unique()
    ret = Parallel(n_jobs=-2)(delayed(process)(df, example)
                              for example in tqdm(examples))

    # returning the average
    return np.sum(ret) / df.shape[0]


def std_pooled(df):
    counts = df.groupby('question_id')['answer'].count()
    stds = df.groupby('question_id')['answer'].std()
    std_pooled = np.sqrt(np.sum((counts - 1) * (stds**2)) / np.sum(counts - 1))
    return std_pooled


def error_margin(df):

    counts = df.groupby('question_id')['answer'].count()
    stds = df.groupby('question_id')['answer'].std()
    error_average = np.sqrt((1.96**2) * (stds.mean()**2) / counts.mean())

    var_pooled = np.sum((counts - 1) * (stds**2)) / np.sum(counts - 1)
    error_pooled = np.sqrt((1.96**2) * var_pooled / counts.mean())

    return error_average, error_pooled


if __name__ == '__main__':
    import sys
    if 'our_eval' in sys.argv:
        eval_func = subjective_eval.evaluate
    elif 'obj_eval' in sys.argv:

        def eval_func(x, y):
            return objective_eval.evaluate(x, y)[1, 2]
    else:
        eval_func = subjective_eval.get_peamt().evaluate_from_midi

    df = load_pmt_data()
    print(
        f"Error margin with 95% of confidence (average, pooled): {error_margin(df)}"
    )
    ret = evaluate_measure(df, eval_func=eval_func)

    print(f"Agreement {ret:.4f}")
