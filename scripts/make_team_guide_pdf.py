"""Render docs/TEAM_GUIDE.pdf: the hand-in standards as a short, readable PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "TEAM_GUIDE.pdf"

INK, MUTED, ACCENT, BAND, RULE = (colors.HexColor(c) for c in
                                  ["#16202e", "#4a5568", "#256abf", "#eef2f7", "#d5dce6"])
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24,
                    textColor=INK, alignment=TA_LEFT, spaceAfter=4)
SUB = ParagraphStyle("Sub", fontName="Helvetica", fontSize=10.5, leading=14, textColor=MUTED, spaceAfter=10)
H2 = ParagraphStyle("H2", fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=ACCENT,
                    spaceBefore=12, spaceAfter=6)
BODY = ParagraphStyle("Body", fontName="Helvetica", fontSize=10, leading=14, textColor=INK, spaceAfter=4)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=9.5, leading=12.5, spaceAfter=0)
CELLB = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")
BUL = ParagraphStyle("Bul", parent=BODY, leftIndent=12, bulletIndent=2)
MONO = ParagraphStyle("Mono", parent=BODY, fontName="Courier", fontSize=9, leading=12,
                      backColor=BAND, borderPadding=6, leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=8)


def table(rows, widths, header=True):
    data = [[Paragraph(c, CELLB if (header and i == 0) else CELL) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0), (-1, -1), 0.5, RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
             ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), BAND))
    t.setStyle(TableStyle(style))
    return t


def bullets(items):
    return [Paragraph(i, BUL, bulletText="•") for i in items]


W = 174 * mm
story = [
    Paragraph("EmotionShift: hand-in standards", H1),
    Paragraph("EE6405 Group Project, Table 4 · for the final video, slides, code and demo · "
              "integration: Lixian · full version in the repo: docs/TEAM_GUIDE.md", SUB),
    Paragraph("Please hand in your part <b>in these formats</b>, so everything can be merged into one video "
              "in the last week. The rule behind all of it: <b>send source files</b> (text, data, separate "
              "images, separate audio), not screenshots of tables or view-only links.", BODY),
    Spacer(1, 4),
    table([["Course requirement (EE6405 brief)", ""],
           ["Video", "<b>10 minutes max</b>; must show <b>Python code</b>, <b>slides for theory</b> and <b>references</b>"],
           ["Voice", "<b>No AI-generated voice</b>: everyone records in their own voice"],
           ["Code", "<b>.py</b> or <b>.ipynb</b>"],
           ["Also", "Signed group reflection with each member's contribution; AI declaration form; "
                    "peer evaluation on NTULearn (mandatory, otherwise 5% deduction)"]],
          [38 * mm, W - 38 * mm]),

    Paragraph("1. Slides (2–3 per person, about 1 min 30 s)", H2),
    table([["Item", "Format"],
           ["Your slides", "The shared team template, 16:9"],
           ["Script", "In each slide's <b>speaker notes</b> (or a script.md, one section per slide)"],
           ["Figures", "Each figure as its own <b>PNG, at least 1600 px wide</b>, plus the <b>CSV</b> of its numbers"],
           ["Result numbers", "A <b>summary.json or CSV</b> with the shared metrics (below)"],
           ["References", "Plain-text list in <b>APA style</b>"]],
          [38 * mm, W - 38 * mm]),
    Spacer(1, 4),
    Paragraph("<b>Shared metrics:</b> MELD weighted-F1; DailyDialog micro-F1 excluding neutral on "
              "<i>test_dedup</i>; macro-F1 for both; scores per stratum (first / stable / shift); 95% bootstrap CI. "
              "<font face='Courier'>emotionshift/eval/metrics.py</font> computes all of them.", BODY),

    Paragraph("2. Code (in the GitHub repo)", H2),
    *bullets([
        "Your <b>own folder or module</b>, using the <b>shared data loader</b> so we all have the same splits, "
        "labels and shift definition. Run <font face='Courier'>bash scripts/fetch_data.sh</font> once after cloning.",
        "<b>One executed notebook with outputs</b> (used to show code in the video); "
        "see <font face='Courier'>notebooks/b1_tfidf_svm.ipynb</font>.",
        "<b>Prediction CSVs</b>, one per test split and seed, in the shared format below.",
        "The <b>code to show in the video</b>: notebook cell(s) or file + line range, max 15 lines each.",
        "<b>How to run it</b>: command, rough run time, GPU or not, extra packages.",
        "<b>For the demo (#3–#6)</b>: the saved model and "
        "<font face='Courier'>predict(list_of_utterances) -&gt; list_of_emotions</font>, labels in the order "
        "neutral, joy, surprise, anger, sadness, disgust, fear.",
    ]),
    Paragraph("from emotionshift.data.loaders import load<br/>"
              "df = load(\"meld\")   # or \"dailydialog\"<br/><br/>"
              "# prediction CSV columns:<br/>"
              "dataset, split, dialogue_id, turn, speaker, text, emotion, pred,<br/>"
              "stratum, shift_type, model, seed", MONO),
    Paragraph("<b>How to add your work (no invite needed):</b> Fork the repo → add your files in your fork "
              "(git push, or Add file › Upload files) → Contribute › Open pull request. Lixian reviews and merges.", BODY),

    KeepTogether([
        Paragraph("3. Narration (no AI voice)", H2),
        *bullets([
            "Record in <b>your own voice</b>, <b>one audio file per slide</b>, named like "
            "<font face='Courier'>03_name_slide2.m4a</font>.",
            "<b>M4A or WAV</b> (a phone voice memo is fine); quiet room, no background music.",
            "Want to show code running live? A <b>1080p MP4 screen recording</b>, also in your own voice.",
            "Read what is in your script and check each clip's length.",
        ]),
    ]),

    KeepTogether([
        Paragraph("4. Admin items", H2),
        *bullets([
            "<b>Contribution note</b>: 3–5 lines on exactly what you did (for the group reflection).",
            "<b>AI declaration form: one per person.</b> It has one name, one matriculation number and one "
            "signature, so each member fills in and signs their own; all forms go in our single team submission. "
            "Tick the phases where you used AI and add a shareable link to your <b>complete, unedited</b> AI chat.",
            "Note: the form allows the <b>NTU AI Enterprise Suite</b> (Option A) or a third-party tool "
            "<b>explicitly approved by the instructor</b> (Option B). We are confirming with the instructor "
            "which tools are allowed.",
            "<b>Signature</b> on the group reflection, once it is final.",
        ]),
    ]),

    KeepTogether([
        Paragraph("Where to put it", H2),
        Paragraph("One folder per person, in the repo's <font face='Courier'>deliverables/</font> folder "
                  "(or the shared drive if files are too large for Git). It is only for collecting material for the video "
                  "and is removed before the final submission.", BODY),
        Paragraph("deliverables/02_lixian/<br/>"
                  "&nbsp;&nbsp;slides.pptx&nbsp;&nbsp;&nbsp;&nbsp;script.md&nbsp;&nbsp;&nbsp;&nbsp;refs.txt"
                  "&nbsp;&nbsp;&nbsp;&nbsp;contribution.txt<br/>"
                  "&nbsp;&nbsp;figures/&nbsp;&nbsp;(PNG + matching CSV)<br/>"
                  "&nbsp;&nbsp;audio/&nbsp;&nbsp;&nbsp;&nbsp;(01.m4a, 02.m4a, ...)", MONO),
    ]),

    KeepTogether([
        Paragraph("Deadlines", H2),
        table([["When", "What"],
               ["End of Week 12 (Sun 8 Nov)", "Code, prediction CSVs, result numbers (pull request merged)"],
               ["Tue 10 Nov", "Slides, script, figures"],
               ["Wed 11 Nov", "Narration audio"],
               ["12–14 Nov", "Integration and editing"],
               ["<b>Sun 15 Nov, 11:59 pm</b>", "<b>Submission</b>"]],
              [52 * mm, W - 52 * mm]),
    ]),
]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "EmotionShift · hand-in standards · github.com/qkuku1117-lgtm/emotionshift")
    canvas.drawRightString(192 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                  bottomMargin=18 * mm, title="EmotionShift: hand-in standards", author="Lixian").build(
    story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
