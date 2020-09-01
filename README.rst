Cite us
=======

This is the repo of all codes used in our paper [...].

Setup
=====

Install ``pyenv`` and ``poetry``. You can also setup your environment with other
tools, if the dependencies listed in ``pyproject.toml`` are installed.

Environment
-----------

#. ``pyenv install 3.6.9``
#. ``git clone <this-repo-url>``
#. ``cd <repo>``
#. ``pyenv local 3.6.9``
#. ``poetry install``
#. ``pip install git+https://github.com/CPJKU/madmom -c constraints.txt`` (this
   is needed because google sucks and has an exact dependency for an old mido
   version in its requirements.txt)

Then, you'll need ``vienna_corpus``, ``SMD`` and ``Maestro`` datasets from
``asmd`` package:

    ``poetry run python -m asmd.install``

Chose excerpts
--------------

#. Download our pretrained vienna model on Maestro and put it in your working
   dir from our mega_

#. Train our proposed model or download the pretrained ones from our mega_:

#. You will need the template matrix provided in this repo. To rebuild it
   run ``poetry run python -m perceptual.make_template``. You will need
   the synthesized scale and the corresponding midi in the ``scales``
   and ``audio`` folder. You can download them from our mega_

   #. ``poetry run python -m perceptual.proposed create_mini_specs`` to create
      the dataset of mini-specs or download it from our mega_.

   #. dataset size: 474.429 notes (831 batches in test, 178 in train))

   #. ``poetry run python -m perceptual.proposed train`` to train our model
      for velocity estimation and test it. I obtained the following
      absolute error (avg, std) on the test set: 15.11, 10.94 (251 epochs)

   #. redo everything with vienna model (use ``--vienna`` for
      ``create_mini_specs`` and ``train``)

#. Run ``poetry run python -m perceptual.excerpt_search``

This will analyze ``vienna_corpus`` in search of excerpts, will transcribe the
original performances and will create a new directory ``audio`` with all
extracted excerpts audio files and a directory ``to_be_synthesized`` with all
midi files that you have to synthesize and put in ``audio``

Chose vsts
----------

#. Synthesize the chosen excerpts with vsts or download our
   synthesized midis from our mega_; extract them in the ``audio`` directory.
   You should have a directory for each vst in ``audio`` and for each vst you
   should have 5 different audio. In the root of ``audio`` you should also have
   the original recordings.
#. Install ``sox`` in your path for post-processing to add reverb or run with
   ``--no-postprocess``
#. Analyze chosen excerpts:
   ``poetry run python -m perceptual.chose_vst``

This will copy the excerpts relative to the chosen vsts to the folder
``excerpts``.

Chosen vsts:
- q0: ./audio/salamander
- q1: ['./audio/pianoteq1', './audio/salamander-norm_-20_reverb_50_norm']
- q2: ./audio/pianoteq1-norm_-20_reverb_100_norm

Build tests
-----------

Set up your server (Python or PHP) and download WAET_.

#. place the directory ``excerpts`` in the root of WAET
#. place the directory ``reveal.js-3.9.2`` into the root of WAET
#. place the file ``index.html`` in the root of WAET (if you want, you can
   regenerate the ``index.html`` by running ``pandoc --to revealjs -V
   revealjs-url=reveal.js-3.9.2 --output index.html --standalone
   index.md``)
#. place the file ``listening_test.xml`` in ``[WAET root]/tests/pool.xml``
#. place the file ``core.css`` in ``[WAET root]/css/core.css``

You should be able to access your test at ``/test.html?url=php/pool.php``.
More info in the WAET wiki_

``index.html`` contains the instructions for the test, so that you can
distribute the url to the root of WAET to your partecipants.

.. _WAET: https://github.com/BrechtDeMan/WebAudioEvaluationTool
.. _wiki: https://github.com/BrechtDeMan/WebAudioEvaluationTool/wiki/Pooling-tests


Answer analysis
---------------

To plot tests you should use ``perceptual.subjective_eval``, which also prints
correlations with the objective measure from ``mir_eval``.

The results that we collected are available in our mega_

You can see all options by running ``poetry run python -m
perceptual.subjective_eval``. Before of running you should change the settings
according to your system: open the script and change the initial global
variables:

#. ``PATH`` is the path to the ``saves`` dir of WAET_
#. ``DISCARD_BEFORE_THAN`` defines a date before of which the answers whould be
   discarded; this is useful for removing debug answers
#. ``MAP_VALUES`` defines the mapping for creating the control groups according
   to the answer of the users

Also note that all answers in which users listened to for less than 5 seconds
or didn't move the cursor are completely discarded. This is hard-coded in final
section of the script.

At each run, violin plots are created for each control group and each method.
One plot is created for each question type and excerpt or for each question
type if ``average`` option is used.  Under each plot, there are the p-values
computed with Wilcoxon test for each combination of groups or methods. Then the
error margins and correlations are shown.

Linear regression
~~~~~~~~~~~~~~~~~

To compute the linear regressions of the perceptual values, you should run
``poetry run python perceptual.eval_regression``. It will plot the regression
predictions for various model and weights for the case with and without MFCC
features. Than, it will also plots the weights with only the selected features.

If you want, you can test the selected features by using ``our_eval`` as option
to the ``subjective_eval`` script.

Alignment
---------

#. Install fluidsynth and download SalamanderGrandPianoV3 soundfont in sf2 format
   from our mega_ folder and put it in your working dir
#. run ``poetry run python -m perceptual.alignment.dtw_tuning`` to check the
   FastDTW tuning in midi2midi over ``MusicNet`` solo piano songs
#. run ``poetry run python -m perceptual.alignment.align amt`` to perform our
   amt-based alignment over SMD dataset with the best parameters found in the
   previous step
#. run ``poetry run python -m perceptual.alignment.align ewert`` to perform our
   baseline alignment over SMD dataset
#. run ``poetry run python -m perceptual.alignment.analysis results/ewert.csv
   results/amt.csv`` to plot the results of alignment


.. _mega: https://mega.nz/folder/KVExwayZ#TrXTvHleVhzBfBXt0FaOAA
