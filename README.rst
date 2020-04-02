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

Then, you'll need ``vienna_corpus``, ``SMD`` and ``Maestro`` datasets from ``asmd``
package: 

    ``poetry run python -m asmd.install``

Chose excerpts
--------------

#. ``poetry run python -m perceptual.excerpt_search``

This will analyze ``vienna_corpus`` in search of excerpts and will create a new
directory ``excerpts`` with all extracted excerpts and midi aligned. You still
need to create the corresponding scores for testing the score-informed method.
You can find original scores in ``my_scores`` directory, but you have to segment
them by hand. Name them with the same name of audio file but extension
'-score.mid'

Transcribe excerpts
-------------------

#. ``poetry run python -m perceptual.transcribe``

This will transcribe excerpts and will store them as midi file in the ``excerpt``
folder.

Chose vsts
----------

#. Build scales: ``poetry run python -m perceptual.midi_test_creation``
#. Synthesize scales and chosen excerpts with vsts. You can download our
   synthesized midis from [url]; extract them in the ``excerpts`` directory.
#. Analyze scales and chosen excerpts: 
   ``poetry run python -m perceptual.vst_search``

This will copy the excerpts relative to the chosen vsts to the folder ``chosen``.

Build tests
-----------

[no idea of how to do this...]

Objective evaluation
--------------------

Alignment
_________

#. Install fluidsynth and download SalamanderGrandPiano soundfont in sf2 format
   from our mega_ folder and put it in your working dir
#. run ``python -m perceptual.alignment.dtw_tuning`` to check the DTW tuning in
   midi2midi over `MusicNet` solo piano songs
#. run ``python -m perceptual.alignment.align amt`` to perform our amt-based
   alignment over SMD dataset
#. run ``python -m perceptual.alignment.align ewert`` to perform our baseline
   alignment over SMD dataset
#. run ``python -m perceptual.alignment.analysis results/ewert.csv
   results/amt.csv`` to plot the results of alignment

Transcription
_____________

#. Download our pretrained vienna model on Maestro and put it in your working dir from our mega_

To run the objective evaluation use

    ``poetry run python -m perceptual.objective_eval``

To compare with subjective evaluation, store percptual test results in
``results`` directory with name ``perceptual.csv`` and run

    ``poetry run python -m perceptual.compare_eval``

This will create csv files in ``results`` directory.

To plot them use ``poetry run python -m perceptual.plots``.


Perceptual tests
================

Introduction: dictionary and principles
---------------------------------------

Dictionary to the purpose of this text: 

#. *interpretation*: the symbolic idea about how a music piece should be
   performed: that is, the ideal performance that the player wants to achieve 
#. *performance*: the pyhisical act of playing a music piece: that is, the set
   of movements through which the musician realizes the *interpretation*, which
   changes depending on the context (piano, room, environmnet)

The listening tests are based on the following principles that are
supposed to be true: 

#. All musicians have a target *interpretation*, that is, an idea of the target
   sound (at least at professional levels)
#. The MIDI is able to record all the characteristics of a piano *performance* 
#. (reformulation of the above) a musician will change its way of playing
   according to the context (room, environment, piano), that is: using the same
   *interpretation*, a musician will create different *performances* if the
   context changes 
#. during the audio recording process, some information is lost and some other
   is introduced, due to the context (including microphones). Consequently, it
   is not possible to extract the exact MIDI *performance* from the audio

Aim
---

The aim of the listening tests is to answer to the following questions:

#. Is the MIDI format able to represent an *interpretation*? (principle
   2. states that MIDI is able to represent a *performance*)
#. Is the *interpretation* still identifiable when changing the context
   but keeping the *performance* (MIDI recording)?
#. Which transcription system is better for an up-quality resynthesis
   target?
#. [Which transcription system is better for a same-quality resynthesis
   target?] (really useful???)
#. Which audio-to-score alignment system is better?

Question 1, actually depends upon question 2: if 2 is answered
positively, then 1 is also ok. If 2 is answered negatively, we have a
hint suggesting that 1 is false (it’s just a hint since we could have
used *bad* instrument).

Method
------

Likert test with 6 items. Eventually, we can use a MUSHRA test (likert
with 5 items) and a meta-question asking “how much do you feel confident
with your answer?”. If the middle category shows an endorsement by low
confidence, we normalize data so that low-confidence answer are equally
distributed among categories [really useful?].

Generic
~~~~~~~

-  Age?
-  Sex?
-  Years of study of music?
-  Hobbistic or academic study?
-  How many hours per week do you played music in the last month?
-  How many hours per week do you listened to music in the last month?

Introduction
~~~~~~~~~~~~

Explain difference between *interpretation* and *performance* with
examples [to be done].

-  simple explanation of what is Standard MIDI Format.
-  simple explanation of how microphones and environment change the
   sound.

Examples: 

- same performer in different concerts (same interpretation, different
  performance) 
- same performer with different interpretations (same condition, different
  interpretation - and performance) 
- different performers in different concerts (different interpretations,
  different conditions - and performances)

Question type 1
~~~~~~~~~~~~~~~

::

   Listen to this target audio recording: [original audio]
   For each of the following recordings, rate how much you think the _interpretation_ is similar to the target audio? 
   Note that these are different performances because the piano, the microphones, and the environment changed.
   [possible answers: scale 1 to 6]
   - original midi recording resynthesized with instrument 3 (hidden reference)
   - another _interpretation_ resynthesized with instrument 3 (negative reference)
   - transcribed performance with method 1 with instrument 3
   - transcribed performance with method 2 with instrument 3
   - score-informed transcription (auto-alignment + velocity estimation) with instrument 3

Each score can be computed with the *“Absolute Category Rating with
Hidden Reference”* (ACR-HR): score - score_hr + 6

This poll wants to answer to questions 2 and 3. Question 2 is answered
positively if the hr has high scores and the nr has low scores.
Otherwise it is answered negatively. Question 3 is answered by comparing.

Question type 2
~~~~~~~~~~~~~~~

::

   Listen to this target audio recording: [exact performance resynthesized with instrument 1]
   For each of the following recordings, rate how much you think the _interpretation_ is similar to the target audio? 
   Note that these are different performances because the piano, the microphones, and the environment changed.
   [possible answers: scale 1 to 6]
   - original midi recording resynthesized with instrument 2 (hidden reference)
   - another _interpretation_ resynthesized with instrument 2 (negative reference)
   - transcribed performance with method 1 with instrument 2
   - transcribed performance with method 2 with instrument 2
   - score-informed transcription (auto-alignment + velocity estimation) with instrument 2

Each score can be computed with the *“Absolute Category Rating with
Hidden Reference”* (ACR-HR): score - score_hr + 6

This poll wants to answer to questions 2 and 3. Question 2 is answered
positively if the hr has high scores and the nr has low scores.
Otherwise it is answered negatively. Question 3 is answered by comparing
the 3 different transcription systems.

Question type 3
~~~~~~~~~~~~~~~

::

   Listen to this target audio recording: [exact performance resynthesized with instrument 3]
   For each of the following recordings, rate how much you think the _interpretation_ is similar to the target audio?
   [possible answers: scale 1 to 6]
   - original midi recording resynthesized with instrument 3 (hidden reference)
   - another _interpretation_ resynthesized with instrument 3 (negative reference)
   - transcribed performance with method 1 with instrument 3
   - transcribed performance with method 2 with instrument 3
   - score-informed transcription (auto-alignment + velocity estimation) with instrument 3

With this question, we want to compare various transcription systems.
Each score can be computed with the *“Absolute Category Rating with
Hidden Reference”* (ACR-HR): score - score_hr + 6

This poll wants to answer to question 4.

.. _mega: https://mega.nz/folder/KVExwayZ#TrXTvHleVhzBfBXt0FaOAA
