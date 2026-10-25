# Lixian (#2 TF-IDF + SVM): narration script

One section per slide; record one audio file per slide (audio/01.m4a, ...).
Slides: Claude Slides deck "B1 TF-IDF + SVM Baseline" (export to .pptx or PDF from the deck's Share menu).
Estimated total: about 103 s.

## 01 · TF-IDF + linear SVM

Hi, I'm Lixian. My workstream is B1, the traditional machine-learning baseline: TF-IDF features with a linear SVM. It reads one utterance at a time, with no dialogue context and no speaker information, so it tells us how far the words alone can get, and sets the bar every context-aware model has to beat. [about 10 s]

## 02 · How B1 works

Each utterance becomes a sparse TF-IDF vector over word unigrams and bigrams. I keep exclamation and question marks as tokens, because they carry emotion. A linear SVM then scores each of the seven emotions, one versus rest. Two settings matter: C, which trades a wide margin against training errors, and balanced class weights, which make mistakes on rare emotions like fear and disgust cost more. [about 15 s]

## 03 · The core in scikit-learn

In code it is a scikit-learn pipeline: a TF-IDF vectorizer feeding a linear SVM. Data comes from our shared loader, so B1 uses exactly the same splits, labels and shift definition as every other model in the project. The full run, with the grid search, five seeds and all tables, is one command; the notebook walks through it step by step. It is also cheap: under one megabyte, under a second to train on MELD, and well under a millisecond per utterance, so it costs our demo nothing. [about 15 s — switch to the notebook here if showing code live]

## 04 · Tuned on dev only

I tuned on the dev set only: 32 settings over classifier, C, class weights and features. The best on MELD is a linear SVM with C equal to 0.1 and balanced class weights. Small C means strong regularisation, which suits short utterances with a large vocabulary. Character n-grams did not help. The test set was scored once and never used to choose anything. [about 13 s]

## 05 · Test results: a strong floor

On the test sets, B1 reaches 56.8 weighted-F1 on MELD and 53.0 micro-F1, excluding neutral, on the de-duplicated DailyDialog test set. On MELD that is far above always predicting neutral, and above copying the speaker's previous emotion, and only about one point below DialogueGCN, a published model that uses context. So a bag of n-grams is already a strong floor for the transformer models. [about 15 s]

## 06 · Words alone fail at the turning points

Here is the key finding. Splitting MELD test by emotion shift, B1 scores 62 on stable turns but only 49 on shift turns, where the speaker's emotion changes. That is a thirteen-point gap with no context involved at all. Offsets look easy only because they end in neutral, which B1 predicts often. Within-negative shifts, like anger to sadness, are the hardest. This is exactly the gap our context and ShiftAware models are designed to close. [about 18 s]

## 07 · What the model learned

Because the model is linear, we can read its weights. Surprise is driven by "wow", "whoa" and "really?"; disgust by "ugh" and "ew"; sadness by "sorry"; fear by "scared" and "oh no". So B1 captures emotion that is marked in the words. What it misses: a one-word turn like "Yeah." is always neutral, because tone, sarcasm and the conversation are invisible to it. That is where the context-aware models should gain. [about 17 s]

## 08 · References

References for this section; they will be merged into the group's references slide.
