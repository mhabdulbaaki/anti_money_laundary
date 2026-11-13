"""Step 7: fill the DSCI 6601 course template with our results -> report/AML_Project_Report.docx.

Results and exploration figures are read from results/*.json, so re-running earlier steps updates
the report; only the fixed design facts (grids, windows, feature counts) are written into the text.
"""

from __future__ import annotations

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.table import Table
from docx.text.paragraph import Paragraph

from aml import config

TEMPLATE = config.ROOT / "DSCI6601_Course_Project_Template.docx"
OUT = config.ROOT / "report" / "AML_Project_Report.docx"
BODY_PT = 10
FAMILY = {"xgboost": "XGBoost", "lightgbm": "LightGBM", "logreg": "Logistic regression"}


def load() -> dict[str, Any]:
    r = config.RESULTS
    out = {
        "q": json.loads((r / "data_quality.json").read_text()),
        "eda": json.loads((r / "eda_summary.json").read_text()),
        "final": json.loads((r / "final_metrics.json").read_text()),
        "exp": json.loads((r / "explanations.json").read_text()),
        "acc": json.loads((r / "accounts.json").read_text()),
    }
    llm = r / "reports_summary_llm.json"
    tpl = r / "reports_summary_template.json"
    out["drafts"] = json.loads(llm.read_text()) if llm.exists() else None
    out["drafts_template"] = json.loads(tpl.read_text()) if tpl.exists() else None
    return out


# ---------------------------------------------------------------- docx helpers
def _style_run(run: Any, size: float = BODY_PT, bold: bool = False, italic: bool = False, color: str | None = None) -> None:
    run.font.name = "Calibri"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Calibri")
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def _fill(paragraph: Paragraph, text: str, size: float = BODY_PT, italic: bool = False, color: str | None = None) -> None:
    """Write text into a paragraph; **bold** spans become bold runs."""
    for part in re.split(r"(\*\*[^*]+\*\*)", text):
        if not part:
            continue
        bold = part.startswith("**")
        run = paragraph.add_run(part.strip("*") if bold else part)
        _style_run(run, size, bold=bold, italic=italic, color=color)


def replace_placeholder(placeholder: Paragraph, paragraphs: list[str]) -> None:
    for text in paragraphs:
        p = placeholder.insert_paragraph_before()
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        _fill(p, text)
    placeholder._element.getparent().remove(placeholder._element)


def set_cell(cell: Any, text: str, bold: bool = False, size: float = 9) -> None:
    p = cell.paragraphs[0]
    for r in list(p.runs):
        r._element.getparent().remove(r._element)
    for extra in cell.paragraphs[1:]:
        extra._element.getparent().remove(extra._element)
    _fill(p, f"**{text}**" if bold else text, size=size)


def widen_table(tbl: Table, widths_dxa: list[int]) -> None:
    """Give a deep-copied 3-column table the requested number of columns."""
    grid = tbl._tbl.tblGrid
    cols = grid.findall(qn("w:gridCol"))
    while len(cols) < len(widths_dxa):
        grid.append(deepcopy(cols[-1]))
        cols = grid.findall(qn("w:gridCol"))
    for c, w in zip(cols, widths_dxa):
        c.set(qn("w:w"), str(w))
    for tr in tbl._tbl.findall(qn("w:tr")):
        tcs = tr.findall(qn("w:tc"))
        while len(tcs) < len(widths_dxa):
            tr.append(deepcopy(tcs[-1]))
            tcs = tr.findall(qn("w:tc"))
        for tc, w in zip(tcs, widths_dxa):
            tc.find(qn("w:tcPr")).find(qn("w:tcW")).set(qn("w:w"), str(w))


def ensure_rows(tbl: Table, n_body: int) -> None:
    while len(tbl.rows) - 1 < n_body:
        tbl._tbl.append(deepcopy(tbl.rows[-1]._tr))
    while len(tbl.rows) - 1 > n_body:
        tbl._tbl.remove(tbl.rows[-1]._tr)


# ---------------------------------------------------------------- content
def f2(x: float) -> str:
    return f"{x:.2f}"


def pct(x: float) -> str:
    return f"{x:.0%}"


def eda_facts(eda: dict[str, Any]) -> dict[str, float]:
    """The handful of exploration numbers the report quotes, taken from results/eda_summary.json."""

    def signal(prefix: str) -> dict[str, Any]:
        return next(s for s in eda["signal_lift"] if s["signal"].startswith(prefix))

    hubs = eda["hubs"]
    return {
        "ach_laund": eda["ach_share_of_laundering"],
        "ach_tx": eda["ach_share_of_transfers"],
        "hub_top": hubs["in_degree_top_involved_rate"],
        "hub_rest": hubs["in_degree_rest_involved_rate"],
        "loop_rate": signal("Transfer closes a three-step loop")["laundering_rate"],
        "loop_lift": signal("Transfer closes a three-step loop")["lift"],
        "first_cover": signal("Sender pays this receiver for the first time")["coverage"],
        "fan_lift": signal("Sender already paid")["lift"],
    }


def content(R: dict[str, Any]) -> dict[str, Any]:
    q, fm, ex, ac = R["q"], R["final"], R["exp"], R["acc"]
    e = eda_facts(R["eda"])
    win = fm["winner_family"]
    name = FAMILY[win]
    W = fm["families"][win]
    core, allp = fm["test"]["core_days"], fm["test"]["all"]
    rules_core = fm["rules"]["test"]["core_days"]
    abl = fm["ablation_no_shortcuts"]["val"]
    p_model = ac["model"]["core_days_mean_precision_at_100"]
    p_rules = ac["rules"]["core_days_mean_precision_at_100"]
    tail_share = q["tail_laundering"] / q["tail_rows"]
    target_met = core["f1"] >= 0.5
    p_text = "p < 0.001" if ex["permutation_p_value"] < 0.001 else f"p = {ex['permutation_p_value']:.3f}"
    beats_random = ex["hit_rate"] > ex["random_reason_baseline"]
    runner_key = max((k for k in fm["families"] if k != win), key=lambda k: fm["families"][k]["cv_f1_mean"])
    close = fm["families"][runner_key]["cv_f1_mean"] >= W["cv_f1_mean"] - W["cv_f1_std"]
    drafts = R["drafts"] or R["drafts_template"]
    per_day = drafts["reports"] // max(len({c["case_id"][4:12] for c in drafts["cases"]}), 1)
    if R["drafts"]:
        draft_text = (
            f"**AI drafts:** for each test day’s top {per_day} accounts, Claude Opus 5.5 turned an evidence packet of computed facts "
            f"into a FINTRAC-style draft: {pct(drafts['format_ok_rate'])} were well-formed and {pct(drafts['numbers_ok_rate'])} used only "
            f"evidence numbers ({drafts['reports']} drafts)."
        )
    else:
        draft_text = (
            f"**AI drafts:** evidence packets (top {per_day} accounts a day) and checks are built, but the Claude Opus 5.5 run "
            "awaits API credentials: target 3 is unmeasured."
        )

    sections = {
        1: [
            "**Problem and decision.** Financial institutions must find money laundering hidden among millions of legitimate "
            "transfers; fixed rules miss organised schemes and every alert needs a hand-written report. Our model answers one "
            "daily question: which transactions and accounts should an analyst review first, and why? **Objective.** (1) Predict "
            "whether each transaction is laundering (Is Laundering = 1); (2) roll scores up into a daily account ranking; (3) "
            "explain each top alert and have an AI assistant draft a suspicious transaction report (STR). **Success metric and "
            "baselines.** With 0.1% positives accuracy is discarded (never flagging scores 99.9%); our main metric is F1 on the "
            "laundering class, plus PR-AUC and the share of each day’s top 100 accounts truly involved. "
            "Baselines: do-nothing (F1 = 0), warning-sign rules, and the published LightGBM without graph information "
            "(F1 ≈ 0.21; Blanuša et al., 2024). **Targets:** test F1 ≥ 0.50; reasons matching the true scheme more often than "
            "random guessing; ≥ 95% of AI drafts well-formed and using only evidence numbers.",
        ],
        2: [
            f"**Source and size.** IBM’s synthetic AMLworld HI-Small data (Altman et al., 2023): {q['rows_raw']:,} transfers among "
            f"{q['accounts']:,} accounts, 1–18 September 2022, {q['laundering']:,} ({q['laundering_rate']:.2%}) laundering; rows hold "
            f"accounts, amounts, currencies, payment format and label; extra files list {q['pattern_attempts']} schemes in eight "
            "typologies. **Domain.** Being simulated, every laundering transfer is "
            f"labelled, and quirks invite “cheating”: {pct(e['ach_laund'])} of laundering uses ACH and none changes currency. "
            "**Governance and ethics.** CDLA-Sharing-1.0; no real people; data not "
            "redistributed. IDs, bank country and owner type are excluded so the model judges behaviour, not identity; currency "
            "may stand in for country, so we test without it. False alarms freeze innocent money and misses let crime through, "
            "so we report both; AI drafts are advisory.",
        ],
        3: [
            "Each finding drove a decision (EDA notebook; its label statistics use all 18 days, the models only training labels). (i) No missing values or non-positive amounts, so no "
            f"missingness (MCAR or MAR) to impute; {q['exact_duplicates_dropped']} exact duplicates → dropped. (ii) Exchange rates are "
            f"fixed: both sides of {q['cross_currency_rows']:,} currency-changing transfers agree within {q['max_conversion_error_rel']:.3%} "
            f"→ amounts in US dollars. (iii) Amounts span US$0.0001–{q['amount_usd_max'] / 1e9:.0f} billion → log scale. (iv) Days "
            f"1–10 hold 99.98% of transfers; the last eight hold {q['tail_rows']:,}, {pct(tail_share)} laundering → time split, tail "
            f"reported apart. (v) ACH carries {pct(e['ach_tx'])} of transfers but {pct(e['ach_laund'])} of laundering → shortcut test. "
            f"(vi) The top 0.1% of accounts by payers are {e['hub_top'] / e['hub_rest']:.0f} times more often involved "
            f"({pct(e['hub_top'])} vs {e['hub_rest']:.1%}), but whole-period centrality sees the future → rolling fan-in and fan-out "
            f"from past transfers. (vii) Closing a three-step loop means laundering {pct(e['loop_rate'])} of the time "
            f"({e['loop_lift']:.0f}× the base rate) and new counterparties cover {pct(e['first_cover'])} of it, but “already paid 5+ "
            f"accounts today” is only {e['fan_lift']:.1f}× → loop, path and first-payment features in a learned model.",
        ],
        4: [
            "A transfer at minute t is described only by transfers in [t − w, t), so nothing leaks from the future: sender and "
            "receiver counts, US-dollar totals and distinct counterparties, in and out, over 1 h, 24 h and 7 days (36); "
            "pass-through ratios (4); loop, path and first-payment flags and time since the sender was last paid (5); transfer "
            "facts such as log amount and hour (7); payment format and both currencies (3). Encoders "
            "sit in a scikit-learn Pipeline fitted on training rows only (one-hot and scaling for logistic regression, ordinal "
            "codes for trees). Dropped: IDs, owner type, bank country, raw time. Split: chronological 60/20/20 by minute.",
        ],
        5: [
            "Candidates: logistic regression (L1/L2, C 0.01–10, balanced weights), LightGBM (31/63/127 leaves, learning rate "
            "0.05/0.1, is_unbalance) and XGBoost (depth 3/6/9, learning rate 0.01/0.1, scale_pos_weight 1/500/1000; 1 added as "
            "a no-reweighting reference), all on the same three expanding-window folds of the training days (seed 42). Each fit "
            "holds back its latest 15% for early stopping and to set its alert threshold, so fold F1 never peeks at fold labels. "
            "We chose the highest mean CV F1 (Table 1). F1 grows with training history "
            f"(winner: {', '.join(f2(v) for v in R['winner_folds'])}), hence the wide SD"
            + (f"; {FAMILY[runner_key]} was within one SD, so the choice is close. " if close else ". ")
            + "Re-weighting positives hurt both tree models, and LightGBM needed min_child_weight = 1 to stay stable."
            + R["lr_note"],
        ],
        6: [
            f"Refit on all training days with its threshold tuned on validation days (F1 {W['val']['f1']:.2f}), {name} reaches "
            f"**F1 = {core['f1']:.3f}** on the normal test days, "
            + ("meeting" if target_met else "just short of")
            + f" the 0.50 target, and {allp['f1']:.3f} with the tail days (published: 0.21 without and 0.63 with graph features), "
            f"against {rules_core['f1']:.3f} for the rules (Table 2, Figure 1), a gap far beyond the CV spread (±{W['cv_f1_std']:.2f}). "
            "Without payment format and currency, validation F1 "
            f"falls to {abl['f1']:.2f}: part of the score rests on simulator shortcuts. **Accounts:** {pct(p_model)} of each test day’s top 100 "
            f"accounts are truly involved (rules: {pct(p_rules)}). **Explanations:** the top three reasons match the true scheme’s "
            f"warning-sign family for {pct(ex['hit_rate'])} of caught laundering, vs {pct(ex['permutation_baseline'])} for randomly "
            f"guessed schemes ({p_text}), "
            + (
                f"and {pct(ex['random_reason_baseline'])} for three random features. "
                if beats_random
                else f"but {pct(ex['random_reason_baseline'])} for three random features: target 2 is only partly met, as the model "
                "leans on generic signals (payment format, first-time payee). "
            )
            + draft_text,
        ],
        7: [
            "Synthetic, fully labelled data with clean shortcuts flatter every score; we miss "
            f"{pct(1 - core['recall'])} of normal-day laundering; features drift while week-long windows fill; currency may "
            "proxy for country. **Next step:** multi-hop graph features or a GNN, re-tested on LI-Small.",
        ],
        8: [
            "Code and notebook: project repository (scripts 01–07); seed 42; Python 3.12, scikit-learn 1.5.1, XGBoost 2.1.1, "
            "LightGBM 4.5.0. • Altman et al. (2023), NeurIPS D&B. • Blanuša et al. (2024), ICAIF. • Egressy et al. (2024), AAAI.",
        ],
    }
    table2 = [
        [
            "Warning-sign rules (tuned on validation)",
            f"{rules_core['f1']:.3f}",
            f"{rules_core['precision']:.3f}",
            f"{rules_core['recall']:.3f}",
            f"{rules_core['pr_auc']:.3f}",
        ],
        [f"**{name}, normal days**", f"**{core['f1']:.3f}**", f"{core['precision']:.2f}", f"{core['recall']:.2f}", f"{core['pr_auc']:.2f}"],
        [f"{name}, incl. tail days", f"{allp['f1']:.3f}", f"{allp['precision']:.2f}", f"{allp['recall']:.2f}", f"{allp['pr_auc']:.2f}"],
    ]
    return {"sections": sections, "table1": R["table1"], "table2": table2, "name": name}


def table1_rows(fm: dict[str, Any], cv: list[dict[str, Any]]) -> list[list[str]]:
    """Best setting of each family, plus the best re-weighted XGBoost, in a fixed order."""
    win = fm["winner_family"]

    def row(fam: str, params: dict[str, Any], mean: float, sd: float, pr: float, chosen: bool) -> list[str]:
        cells = [f"{FAMILY[fam]}: {_params(params)}", f"{mean:.3f} ± {sd:.3f} (PR-AUC {pr:.3f})", "Yes" if chosen else "No"]
        return [f"**{c}**" for c in cells[:2]] + cells[2:] if chosen else cells

    rows = []
    for fam in ("logreg", "lightgbm"):
        if fam in fm["families"]:
            f = fm["families"][fam]
            rows.append(row(fam, f["params"], f["cv_f1_mean"], f["cv_f1_std"], f["cv_pr_auc_mean"], fam == win))
    weighted = [c for c in cv if c["model"] == "xgboost" and json.loads(c["params"]).get("scale_pos_weight", 1) != 1]
    if weighted:
        c = max(weighted, key=lambda r: r["f1_mean"])
        rows.append(row("xgboost", json.loads(c["params"]), c["f1_mean"], c["f1_std"], c["pr_auc_mean"], False))
    if "xgboost" in fm["families"]:
        f = fm["families"]["xgboost"]
        rows.append(row("xgboost", f["params"], f["cv_f1_mean"], f["cv_f1_std"], f["cv_pr_auc_mean"], win == "xgboost"))
    return rows


def _params(p: dict[str, Any]) -> str:
    names = {"max_depth": "depth", "learning_rate": "lr", "scale_pos_weight": "scale_pos_weight", "num_leaves": "leaves", "is_unbalance": "is_unbalance", "penalty": "penalty", "C": "C"}
    return ", ".join(f"{names.get(k, k)} {v}" for k, v in p.items())


# ---------------------------------------------------------------- build
def build() -> Path:
    import pandas as pd

    R = load()
    cv: list[dict[str, Any]] = [{str(k): v for k, v in r.items()} for r in pd.read_csv(config.RESULTS / "cv_summary.csv").to_dict("records")]
    folds = pd.concat(pd.read_csv(p) for p in (config.RESULTS / "cv").glob("*.csv"))
    win = R["final"]["winner_family"]
    wparams = json.dumps(R["final"]["families"][win]["params"], sort_keys=True)
    R["winner_folds"] = folds[(folds.model == win) & (folds.params == wparams)].sort_values("fold")["f1"].tolist()
    R["lr_note"] = (
        " Logistic regression saw all laundering and 10% of normal rows, for speed."
        if "logreg" in R["final"]["families"]
        else ""
    )
    R["table1"] = table1_rows(R["final"], cv)
    C = content(R)

    doc = Document(str(TEMPLATE))
    paras = doc.paragraphs
    # title
    for p in paras:
        if p.text.startswith("Course Project Report"):
            p.runs[0].text = "Course Project Report"
            for r in p.runs[1:]:
                r.text = ""
    # header table
    head = doc.tables[0]
    set_cell(head.cell(0, 1), "Finding Money Laundering in Bank Transfers with Explainable Risk Scores and Automated AI-Drafted Reports")
    set_cell(head.cell(0, 3), "Mohammed Hardi Abdul Baaki, 202680274; Chris Dwamena, 202680635")
    set_cell(head.cell(1, 1), "IBM Transactions for Anti-Money Laundering (HI-Small), kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml; licence CDLA-Sharing-1.0")
    set_cell(head.cell(1, 3), "November 12, 2026")

    # section bars: drop the word-count hints; delete the grey guidance lines; fill placeholders
    section = 0
    for p in list(doc.paragraphs):
        t = p.text
        m = re.match(r"^\s*(\d)\.\s", t)
        if m and p._element.xpath(".//w:shd"):
            section = int(m.group(1))
            for r in p.runs[1:]:
                r.text = ""
            continue
        if "INSTRUCTIONS" in t:
            break
        italic_hint = p.runs and p.runs[0].italic and not t.startswith(("Table 1.", "Figure"))
        if italic_hint:
            p._element.getparent().remove(p._element)
        elif t.startswith("[") and section in C["sections"]:
            replace_placeholder(p, C["sections"][section])

    # Table 1
    t1 = doc.tables[1]
    ensure_rows(t1, len(C["table1"]))
    for i, row in enumerate(C["table1"], start=1):
        for j, val in enumerate(row):
            set_cell(t1.cell(i, j), val)
    set_cell(t1.cell(0, 1), "CV F1, laundering class (mean ± SD, 3 folds)", bold=True)

    # Table 2 (test results), placed after Section 6 text, before the figure
    fig_tbl = doc.tables[2]
    t2_el = deepcopy(t1._tbl)
    fig_tbl._tbl.addprevious(t2_el)
    t2 = Table(t2_el, fig_tbl._parent)
    widen_table(t2, [4080, 1500, 1500, 1500, 1500])
    ensure_rows(t2, len(C["table2"]))
    for j, h in enumerate(["Test days (used once)", "F1", "Precision", "Recall", "PR-AUC"]):
        set_cell(t2.cell(0, j), h, bold=True)
    for i, row in enumerate(C["table2"], start=1):
        for j, val in enumerate(row):
            set_cell(t2.cell(i, j), val)
    caption = deepcopy([p for p in doc.paragraphs if p.text.startswith("Table 1.")][0]._element)
    t2_el.addprevious(caption)
    cap = Paragraph(caption, fig_tbl._parent)
    for r in cap.runs[1:]:
        r._element.getparent().remove(r._element)
    cap.runs[0].text = "Table 2. Test-day results at the alert threshold tuned on validation days (normal days unless stated)."
    spacer = deepcopy(caption)
    for r in spacer.findall(qn("w:r")):
        spacer.remove(r)
    fig_tbl._tbl.addprevious(spacer)

    # Figure 1
    cell = fig_tbl.cell(0, 0)
    for p in cell.paragraphs[1:]:
        p._element.getparent().remove(p._element)
    p0 = cell.paragraphs[0]
    for r in list(p0.runs):
        r._element.getparent().remove(r._element)
    p0.add_run().add_picture(str(config.FIGURES / "model_result_test.png"), width=Inches(5.2))
    capp = cell.add_paragraph()
    _fill(
        capp,
        f"Figure 1. Selected {C['name']}, normal test days: (a) precision–recall curve, validation-tuned threshold marked (rules at "
        "the origin); (b) confusion matrix at that threshold.",
        size=8.5,
        italic=True,
        color="595959",
    )

    # delete the instructions page
    body = doc.element.body
    started = False
    for el in list(body.iterchildren()):
        if el.tag == qn("w:sectPr"):
            continue
        txt = "".join(el.itertext())
        if "INSTRUCTIONS" in txt:
            started = True
            prev = el.getprevious()
            if prev is not None and prev.tag == qn("w:p") and not "".join(prev.itertext()).strip():
                body.remove(prev)
        if started:
            body.remove(el)

    OUT.parent.mkdir(exist_ok=True)
    doc.save(str(OUT))
    words = sum(len(p.split()) for s in C["sections"].values() for p in s)
    print(f"wrote {OUT} ({words} words of section text)")
    return OUT


if __name__ == "__main__":
    build()
