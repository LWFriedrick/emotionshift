"""Build notebooks/b1_tfidf_svm.ipynb (run once; then execute with nbconvert)."""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
md("""# B1 · TF-IDF + linear SVM baseline

**EE6405 group project · EmotionShift · workstream #2 (Lixian)**

This notebook walks through the traditional-ML baseline step by step:

1. Load MELD and DailyDialog with the shared data rules from the proposal
2. Turn each utterance into TF-IDF features
3. Tune a linear classifier on the **dev** set only
4. Score once on **test**, overall and split by emotion shift
5. Look inside the model: which n-grams drive each emotion

B1 reads **one utterance at a time** (no dialogue context, no speaker), so it is the
reference point the context-aware models (#4 to #6) must beat. The full 5-seed run
with every output file is `scripts/train_b1.py`; this notebook reproduces its numbers
(LinearSVC is deterministic, so every seed gives the same model)."""),
code("""import sys, time, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from emotionshift.data.labels import EMOTIONS
from emotionshift.data.loaders import load, dailydialog_leakage
from emotionshift.models.tfidf import TfidfBaseline, TfidfConfig, _features, _classifier
from emotionshift.eval import metrics as M
from emotionshift.eval import plots as P

FIG = ROOT / "reports" / "b1" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
pd.set_option("display.precision", 3)"""),

md("""## 1. Data

Two datasets, one table schema (`dataset, split, dialogue_id, turn, speaker, text, emotion, stratum, shift_type`).
Rules shared with the whole group (checked by `tests/test_data.py`):

* official splits; 7 shared labels (DailyDialog `happiness` = joy, `no_emotion` = neutral)
* a **shift** = emotion differs from the *same speaker's* previous utterance; a speaker's first turn is neither
* DailyDialog headline test set = **`test_dedup`** (dialogues that also appear verbatim in train removed)"""),
code("""meld, dd = load("meld"), load("dailydialog")
sizes = pd.DataFrame({
    "MELD": meld.groupby("split").size(),
    "DailyDialog": dd.groupby("split").size(),
}).reindex(["train", "dev", "test", "test_dedup"])
sizes"""),
code("""print("DailyDialog leakage:", dailydialog_leakage())
dist = pd.DataFrame({
    "MELD train": meld[meld.split == "train"].emotion.value_counts(normalize=True),
    "DailyDialog train": dd[dd.split == "train"].emotion.value_counts(normalize=True),
}).reindex(EMOTIONS) * 100
dist.round(1)"""),
md("""Both datasets are dominated by *neutral* (MELD 47%, DailyDialog 83%), and fear / disgust
are rare (about 3% each in MELD, under 0.5% in DailyDialog). That imbalance is why we report macro-F1 next to the standard metrics
and try class-weighted training below.

An example MELD dialogue with the shift labels every model is scored on:"""),
code("""ex = meld[(meld.split == "test") & (meld.dialogue_id == 3)]
ex[["turn", "speaker", "text", "emotion", "stratum", "shift_type"]]"""),
code("""t = meld[meld.split == "test"]
print("MELD test strata:", t.stratum.value_counts().to_dict())
print("MELD test shift types:", t.shift_type.value_counts().to_dict())"""),

md(r"""## 2. Method

**TF-IDF.** Each utterance $d$ becomes a sparse vector with one weight per n-gram $t$:

$$\mathrm{tfidf}(t,d) = \big(1+\log \mathrm{tf}(t,d)\big)\cdot \Big(\log\frac{1+N}{1+\mathrm{df}(t)} + 1\Big),$$

then L2-normalised. Word 1–2-grams keep short cues ("oh my", "i love"); `!` and `?` are kept as
tokens because they matter for emotion. Optionally character 2–5-grams are added (robust to
spelling like "sooo" and "ugh").

**Linear SVM.** One-vs-rest: for each emotion $k$ learn $w_k$ minimising

$$\tfrac12\lVert w_k\rVert^2 + C\sum_i c_{y_i}\,\max\big(0,\,1 - y_{ik}\, w_k^\top x_i\big)^2,$$

and predict $\arg\max_k w_k^\top x$. $C$ trades margin against training errors; the class weight
$c_y = N / (7\,N_y)$ ("balanced") makes mistakes on rare emotions cost more.
Logistic regression (multinomial, L2) is tried as the alternative linear classifier."""),
code("""vec = _features(TfidfConfig(features="word"))
tr = meld[meld.split == "train"]
X = vec.fit_transform(tr.text)
print("MELD train matrix:", X.shape, f"(density {X.nnz / np.prod(X.shape):.5f})")

row = X[tr.text.tolist().index("Oh my God!")] if "Oh my God!" in set(tr.text) else X[0]
names = vec.get_feature_names_out()
top = np.argsort(row.toarray().ravel())[::-1][:6]
print("Top TF-IDF weights for one utterance:", [(names[j], round(row[0, j], 3)) for j in top])"""),

md("""## 3. Tuning on dev (MELD)

Grid: classifier × C × class weight × features, every point fitted on train and scored on **dev**.
Test is never used for choosing. (DailyDialog has 87K training utterances, so its grid ran in
`scripts/train_b1.py`; its dev results are loaded from `reports/b1/dailydialog/dev_grid.csv`.)"""),
code("""dev = meld[meld.split == "dev"]
rows = []
t0 = time.time()
for feats in ["word", "word+char"]:
    v = _features(TfidfConfig(features=feats)); Xtr, Xdev = v.fit_transform(tr.text), v.transform(dev.text)
    for clf in ["svm", "logreg"]:
        for C in [0.1, 0.3, 1.0, 3.0]:
            for cw in [None, "balanced"]:
                m = _classifier(TfidfConfig(classifier=clf, C=C, class_weight=cw)).fit(Xtr, tr.emotion)
                s = M.scores(dev.emotion, m.predict(Xdev))
                rows.append(dict(classifier=clf, C=C, class_weight=cw or "none", features=feats, **s))
grid = pd.DataFrame(rows).sort_values("weighted_f1", ascending=False)
print(f"{len(grid)} configurations in {time.time() - t0:.0f}s")
grid.head(8)[["classifier", "C", "class_weight", "features", "weighted_f1", "macro_f1"]]"""),
code("""fig = P.tuning_chart(grid, "weighted_f1", "word", "MELD dev: stronger regularisation and class weights help")
fig.savefig(FIG / "meld_tuning.png", dpi=200); plt.show()"""),
md("""Reading the curve: the best MELD setting is a **linear SVM, C = 0.1, balanced class weights,
word n-grams**. Small C (strong regularisation) wins because MELD utterances are short and the
vocabulary is large relative to 10K training examples. Balanced class weights help a lot:
at C = 0.1 they raise dev weighted-F1 from 47.5 to 54.9 and macro-F1 from 29.1 to 39.9. Adding character n-grams
did not improve dev weighted-F1."""),
code("""dd_grid = pd.read_csv(ROOT / "reports/b1/dailydialog/dev_grid.csv")
dd_grid.head(5)[["classifier", "C", "class_weight", "features", "micro_f1_no_neutral", "macro_f1"]]"""),

md("""## 4. Final models, scored once on test"""),
code("""best = {
    "meld": dict(classifier="svm", C=0.1, class_weight="balanced", features="word"),
    "dailydialog": dict(classifier="svm", C=0.3, class_weight="balanced", features="word"),
}
assert grid.iloc[0][["classifier", "C", "class_weight", "features"]].tolist() == ["svm", 0.1, "balanced", "word"]

results, preds, models = [], {}, {}
for name, df, split in [("meld", meld, "test"), ("dailydialog", dd, "test_dedup"), ("dailydialog", dd, "test")]:
    if name not in models:
        train = df[df.split == "train"]
        models[name] = TfidfBaseline(TfidfConfig(**best[name], seed=0)).fit(train.text, train.emotion)
    t = df[df.split == split].copy()
    t["pred"] = models[name].predict(t.text)
    preds[(name, split)] = t
    s = M.scores(t.emotion, t.pred)
    head = M.HEADLINE[name]
    lo, hi = M.bootstrap_ci(t, head, n_boot=1000)
    results.append({"dataset": name, "split": split, "n": s["n"], "headline metric": head,
                    "headline": 100 * s[head], "95% CI": f"[{100*lo:.1f}, {100*hi:.1f}]",
                    "weighted_f1": 100 * s["weighted_f1"], "macro_f1": 100 * s["macro_f1"],
                    "micro_f1_no_neutral": 100 * s["micro_f1_no_neutral"]})
pd.DataFrame(results).round(2)"""),
md("""For context on MELD (weighted-F1): always-neutral 31.3, copying the speaker's previous
gold emotion 44.9, published context models DialogueGCN 58.1 and COSMIC 65.2. An
utterance-only bag of n-grams lands at **56.8**, close to DialogueGCN."""),

md("""## 5. Where does B1 fail? Shift vs stable turns"""),
code("""mt = preds[("meld", "test")]
strat = M.stratified(mt)
strat.assign(weighted_f1=100 * strat.weighted_f1, macro_f1=100 * strat.macro_f1)[["group", "n", "weighted_f1", "macro_f1"]].round(1)"""),
code("""fig = P.strata_chart(strat, "MELD test: B1 is 13 points worse when the speaker's emotion shifts")
fig.savefig(FIG / "meld_strata.png", dpi=200); plt.show()"""),
md("""* **Shift turns 49.1 vs stable 62.1.** Even with no context at all, the turning points are where
  the model fails, which is exactly the gap the project studies (RQ1/RQ2).
* **Offsets look easy (85.9)** only because the gold label is neutral and B1 predicts neutral
  often; their macro-F1 is very low.
* **Within-negative shifts are hardest (32.1)**: anger → sadness or disgust → anger need more
  than surface words to tell apart."""),
code("""print("Derived shift detection (predicted label vs speaker's previous predicted label):")
M.derived_shift_detection(mt)"""),
code("""fig = P.confusion_chart(M.confusion(mt.emotion, mt.pred), "MELD test: recall per gold emotion (%)")
fig.savefig(FIG / "meld_confusion.png", dpi=200); plt.show()
fig = P.per_class_chart(M.per_class_f1(mt.emotion, mt.pred), "MELD test: F1 per emotion")
fig.savefig(FIG / "meld_per_class.png", dpi=200); plt.show()"""),
md("""Fear and disgust (about 2–3% of MELD each) remain the weakest classes; most of their errors go to *neutral*."""),

md("""## 6. Inside the model: top n-grams per emotion"""),
code("""top = models["meld"].top_features(10)
pd.DataFrame({e: top[e] for e in EMOTIONS})"""),
md("""The weights are readable: surprise ← *wow, whoa, really ?*; disgust ← *ugh, ew, disgusting*;
sadness ← *sorry*; fear ← *scared, oh no*; joy ← *!, great, love*. Short, lexically marked
emotions are what a bag of n-grams captures, and what it cannot see (sarcasm, tone, context)
is where the context-aware models should gain."""),

md("""## 7. Try it"""),
code("""m = models["meld"]
lines = ["Oh my God! You got the job?!", "Yeah.", "I'm so sorry about your dad.",
         "Get out of my apartment!", "Ew, what is that smell?", "I'm scared, what if he finds out?"]
proba = m.predict_proba(lines)
pd.DataFrame({"utterance": lines, "prediction": m.predict(lines),
              "confidence": proba.max(axis=1).round(2)})"""),
md("""Note "Yeah." gets neutral regardless of tone: one-word turns are labelled from voice and context
in MELD, which a text-only, utterance-only model cannot see (the proposal's text-only limitation)."""),

md("""## 8. Efficiency (E9)

Parameters, training time, model size and the per-utterance CPU latency the demo will see
(one utterance per call; the proposal's demo budget is 300 ms per turn)."""),
code("""from emotionshift.eval import efficiency as E
rows = []
for name in ["meld", "dailydialog"]:
    df = meld if name == "meld" else dd
    train = df[df.split == "train"]
    t0 = time.perf_counter()
    mdl = TfidfBaseline(TfidfConfig(**best[name], seed=0)).fit(train.text, train.emotion)
    fit_s = time.perf_counter() - t0
    path = ROOT / "models" / f"_nb_b1_{name}.joblib"; path.parent.mkdir(exist_ok=True); mdl.save(path)
    clf = mdl.pipe.named_steps["clf"]
    lat = E.latency_ms(mdl.predict, df[df.split == "test"].text)
    rows.append({"dataset": name, "features": clf.coef_.shape[1], "parameters": clf.coef_.size + clf.intercept_.size,
                 "model file (MB)": round(E.file_size_mb(path), 2), "train (s)": round(fit_s, 1),
                 "latency median (ms)": round(lat["median_ms"], 2), "latency p95 (ms)": round(lat["p95_ms"], 2)})
    path.unlink()
print(E.machine())
pd.DataFrame(rows)"""),
md("""B1 is tiny: under 80K parameters on MELD, a model file under 1 MB, seconds to train, and
well under a millisecond per utterance, so it costs the demo essentially nothing."""),

md("""## Takeaways

* B1 reaches **56.8 weighted-F1 on MELD** and **53.0 micro-F1 (excl. neutral) on de-duplicated
  DailyDialog**, a strong floor for the transformer models.
* Tuning on dev chose a **linear SVM with balanced class weights**; strong regularisation suits
  short utterances.
* B1 is **13 points worse on shift turns than on stable turns** on MELD, the gap the
  context and ShiftAware models are designed to close.
* It is **cheap**: under 1 MB, seconds to train, under 1 ms per utterance on CPU.

Sources: Poria et al. (2019) MELD; Li et al. (2017) DailyDialog; Joachims (1998) text
categorisation with SVMs; scikit-learn (Pedregosa et al., 2011)."""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
out = ROOT / "notebooks" / "b1_tfidf_svm.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
