import os
import sys

import streamlit as st

from perceptual import subjective_eval

st.write("""
# A Perceptual Measure for Evaluating the Resynthesis of Automatic Music Transcriptions
_Federico Simonetta, Federico Avanzini, Stavros Ntalampiras_
""")

st.write("To generate this file in html format use the following command: `" +
         "streamlit run " + os.path.basename(__file__) +
         " ".join(sys.argv[1:]) + "`")

args = ['dummy_arg']

variable = st.selectbox(
    "Chose control group:",
    ('expertise', 'headphones', 'habits_classical', 'habits_general', 'all'),
    index=4)
args.append(variable)

measure = st.selectbox("Chose measure to compare:",
                       ('mir_eval', 'peamt', 'our_eval'),
                       index=0)
args.append(measure)

average = st.checkbox("Use average of all the excerpts", value=True)
if average:
    args.append("average")

args += sys.argv

order = st.radio(
    "Chose the variable that is iterated first (external loop) and next (inner loop):",
    ('1. task, 2. method', '1. method, 2. task'),
    index=0)

if order == '1. task, 2. method':
    var1, var2 = 'task', 'method'
else:
    var2, var1 = 'task', 'method'

print(f"Using args: {args}")
subjective_eval.main(args, var1=var1, var2=var2)
print("Ended computing!")
print("This script stays open to leave the server up!")
