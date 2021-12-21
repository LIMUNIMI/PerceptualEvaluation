import os
import sys

import streamlit as st

from perceptual import subjective_eval

st.write("""
# A Perceptual Measure for Evaluating the Resynthesis of Automatic Music Transcriptions
_Federico Simonetta, Federico Avanzini, Stavros Ntalampiras_
""")

st.write("To generate this file in html format use the following command: `" +
         "streamlit run " + os.path.basename(__file__) + " -- " +
         " ".join(sys.argv[1:]) + "`")
subjective_eval.main()
print("Ended computing!")
print("This script stays open to leave the server up!")
