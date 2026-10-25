# Team guide: what each workstream hands in

Lixian integrates the slides, code walkthroughs, demo and final video. To make
that possible in the last week, every member hands in the same things in the
same formats. The rule behind all of it: **send source files** (text, data,
separate images, separate audio), not screenshots of tables or view-only links.

Course requirements this follows (EE6405 group project brief): video **10 min
max**, showing **Python code**, **slides for theory** and **references**; **no
AI-generated voice**; code as `.py` or `.ipynb`; a signed group reflection with
each member's contribution; the GAI declaration form; peer evaluation (mandatory).

---

## 1. Slides (2–3 slides per person)

| Item | Format | Why |
|---|---|---|
| Your slides | Use the **shared team template**, 16:9 | One consistent deck; the text can be read and merged directly |
| Script for each slide | In the slide's **speaker notes**, or a `script.md` with one section per slide | Used to sync narration with slides and to check timing |
| Result figures | Each figure as its own **PNG, at least 1600 px wide**, plus the **CSV of the numbers** behind it | Figures can be redrawn in one consistent style if needed |
| Result numbers | A `summary.json` or CSV using the shared metrics (below) | The group comparison table is generated, not typed by hand |
| References | A plain-text list in **APA style** | Merged into one References slide |

**Time budget:** 10 minutes total, about **1 min 30 s per person**; ~40 s
reserved for the intro and conclusion.

Shared metrics: MELD weighted-F1; DailyDialog micro-F1 excluding neutral (on
`test_dedup`); macro-F1 for both; scores per stratum (first / stable / shift);
95% bootstrap CI. `emotionshift/eval/metrics.py` computes all of them.

## 2. Code (in this GitHub repo)

1. **Your own folder or module** under the package layout (for example
   `emotionshift/models/<your_model>.py`), using the **shared data loader**:
   `from emotionshift.data.loaders import load` (same splits, labels and shift
   definition for everyone). Run `bash scripts/fetch_data.sh` once after cloning.
2. **One executed notebook with outputs** (`notebooks/<your_model>.ipynb`), used
   to show code in the video. See `notebooks/b1_tfidf_svm.ipynb` as an example.
3. **Prediction CSVs in the shared format**, one per test split and seed:
   `predictions/<model>_<dataset>_<split>_seed<k>.csv` with columns
   `dataset, split, dialogue_id, turn, speaker, text, emotion, pred, stratum, shift_type, model, seed`.
4. **The code to show in the video**: name the notebook cell(s) or file and
   line range, at most 15 lines per snippet.
5. **How to run it**: the command, roughly how long it takes, whether it needs a
   GPU, and any extra packages (add them to `pyproject.toml`).
6. **For the demo** (models #3–#6): the saved model file and a function
   `predict(list_of_utterances) -> list_of_emotions` (plus `predict_proba` if
   possible), and the model file size. Labels in this fixed order:
   `neutral, joy, surprise, anger, sadness, disgust, fear`.

## 3. Narration (no AI voice allowed)

- **Record in your own voice**, one audio file per slide, named like
  `03_ruizhi_slide2.m4a`.
- **M4A or WAV** (a phone voice memo is fine). Quiet room, no background music.
- If you want to show code running live: a **1080p MP4 screen recording** with
  your own voice.
- Read what is in your script, and check the length of each clip yourself.

## 4. Admin items

- **Contribution note**: 3–5 lines on exactly what you did (for the group
  reflection document).
- **AI declaration form: one per person.** The "Academic Integrity and AI
  Disclosure Form" has one student name, one matriculation number and one
  signature, so **each member fills in and signs their own**; all forms go in
  our single team submission. On it you tick the phases where you used AI
  (ideation, planning, generation/execution, editing) and add a shareable link
  to your **complete, unedited** AI chat. Note: the form allows either the NTU AI
  Enterprise Suite (Option A) or a third-party tool **explicitly approved by the
  instructor** (Option B), so we are confirming with the instructor which tools
  are allowed.
- **Signature** on the group reflection: once it is final.

---

## Where to put it

One folder per person (in the repo's `deliverables/` folder, or the shared
drive if files are too large for Git):

```
deliverables/
  02_lixian/
    slides.pptx           (or the template's format)
    script.md
    figures/              PNG + matching CSV
    audio/                01.m4a, 02.m4a, ...
    refs.txt
    contribution.txt
```

## Suggested deadlines

| When | What |
|---|---|
| End of Week 12 | Code, prediction CSVs, result numbers |
| Tue 10 Nov | Slides, script, figures |
| Wed 11 Nov | Narration audio |
| 12–14 Nov | Integration and editing |
| **Sun 15 Nov, 11:59 pm** | Submission |
