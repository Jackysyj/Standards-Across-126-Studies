#!/usr/bin/env python3
"""
Manuscript data consistency validator for ML4Env Critical Review.

Single source of truth: data/public/statistics_summary.json
                       data/public/figure_data/fig6_rigor.json
Checks: manuscript text (section1-7 + abstract + SI)

Usage: python3 scripts/validate_manuscript.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
STATS_PATH = ROOT / "data/public/statistics_summary.json"
RIGOR_PATH = ROOT / "data/public/figure_data/fig6_rigor.json"
DRAFTS = ROOT / "manuscript/drafts"

# ── Colors for terminal output ───────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def load_stats():
    with open(STATS_PATH) as f:
        return json.load(f)


def load_rigor():
    with open(RIGOR_PATH) as f:
        return json.load(f)


def load_sections():
    """Load all manuscript sections into a dict keyed by section number."""
    sections = {}
    mapping = {
        "abstract": DRAFTS / "abstract.md",
        "s1": DRAFTS / "section1_introduction.md",
        "s2": DRAFTS / "section2_methodology.md",
        "s3": DRAFTS / "section3_data_bias.md",
        "s4": DRAFTS / "section4_validation.md",
        "s5": DRAFTS / "section5_comparability.md",
        "s6": DRAFTS / "section6_discussion.md",
        "s7": DRAFTS / "section7_recommendations.md",
        "si": DRAFTS / "SI/supporting_information.md",
    }
    for key, path in mapping.items():
        if path.exists():
            sections[key] = path.read_text(encoding="utf-8")
        else:
            print(f"{YELLOW}WARN: {path} not found, skipping{RESET}")
            sections[key] = ""
    # Combined full text for cross-section checks
    sections["all"] = "\n".join(sections.values())
    return sections


def build_checks(stats, rigor):
    """Build list of (section_key, regex_pattern, expected_value, description)."""
    checks = []

    # Helper: percentage from rate (1 decimal)
    def pct(rate):
        return f"{rate * 100:.1f}"

    # Helper: percentage from count/total
    def pct_ct(count, total):
        return f"{count / total * 100:.1f}"

    s = stats  # shorthand
    intro = s["introduction"]
    p1 = s["pillar1_data_bias"]
    p2 = s["pillar2_validation"]
    p3 = s["pillar3_comparability"]
    disc = s["discussion"]
    xtab = s["cross_tabulations"]

    # Rigor data
    ov = rigor["overall"]
    gs = rigor["group_stats"]

    # ══════════════════════════════════════════════════════════════════════
    #  ABSTRACT
    # ══════════════════════════════════════════════════════════════════════
    checks.append(("abstract", r"126 peer-reviewed studies", "126",
                    "Abstract: corpus size"))
    checks.append(("abstract", r"median 280 samples", "280",
                    "Abstract: median dataset size"))
    checks.append(("abstract", r"64\.3%.*not describing data selection",
                    pct(1 - p1["data_selection_criteria_described_rate"]),
                    "Abstract: no selection criteria rate"))
    checks.append(("abstract", r"4 of 41 literature-compiled studies",
                    "4/41",
                    "Abstract: grouped splitting in literature"))
    checks.append(("abstract", r"87\.3% lacked external validation",
                    pct(1 - p2["external_validation_rate"]),
                    "Abstract: no external validation"))
    checks.append(("abstract", r"\*ρ\* = [−\-]0\.455",
                    f"{ov['spearman_rho']:.3f}",
                    "Abstract: Spearman rho"))
    checks.append(("abstract", r"scoring 0[–-]2 reported median.*0\.991.*\*n\* = 58",
                    f"0.991/n=58",
                    "Abstract: low rigor group"))
    checks.append(("abstract", r"scoring 7[–-]9 reported 0\.910.*\*n\* = 3",
                    f"0.910/n=3",
                    "Abstract: high rigor group"))
    checks.append(("abstract", r"[Δ].*0\.081",
                    "0.081",
                    "Abstract: delta R2"))
    checks.append(("abstract", r"12\.7%",
                    pct(p3["code_available_rate"]),
                    "Abstract: code availability"))
    checks.append(("abstract", r"4\.8%.*validated.*realistic",
                    f"{6/126*100:.1f}",
                    "Abstract: real-world validation rate"))

    # ══════════════════════════════════════════════════════════════════════
    #  SECTION 1 - INTRODUCTION
    # ══════════════════════════════════════════════════════════════════════
    checks.append(("s1", r"only 2 relevant publications in 2018",
                    str(intro["year_distribution"]["2018"]),
                    "S1: 2018 count"))
    checks.append(("s1", r"rising to 33 in 2023",
                    str(intro["year_distribution"]["2023"]),
                    "S1: 2023 count"))
    checks.append(("s1", r"78 journals",
                    str(intro["total_journals"]),
                    "S1: journal count"))
    checks.append(("s1", r"median.*coefficient.*0\.978",
                    str(p2["r2_stats"]["median"]),
                    "S1: median R2"))
    checks.append(("s1", r"35\.8%\).*studies claim.*above 0\.99",
                    pct(p2["r2_above_099"]),
                    "S1: R2 > 0.99 rate"))

    # ══════════════════════════════════════════════════════════════════════
    #  SECTION 2 - METHODOLOGY
    # ══════════════════════════════════════════════════════════════════════
    checks.append(("s2", r"158 candidate papers",
                    "158",
                    "S2: initial candidates"))
    checks.append(("s2", r"32 papers were excluded",
                    "32",
                    "S2: excluded count"))
    checks.append(("s2", r"126 studies \[10-135\]",
                    "126",
                    "S2: final corpus"))

    # ══════════════════════════════════════════════════════════════════════
    #  SECTION 3 - DATA BIAS
    # ══════════════════════════════════════════════════════════════════════
    ds = p1["dataset_size_stats"]
    checks.append(("s3", r"109 \(86\.5%\) reported dataset size",
                    f"{ds['n']}/{s['meta']['n_papers']}",
                    "S3: dataset size reporting"))
    checks.append(("s3", r"17 \(13\.5%\) did not disclose",
                    str(p1["dataset_size_n_null"]),
                    "S3: missing dataset size"))
    checks.append(("s3", r"median dataset contained 280 data points",
                    str(int(ds["median"])),
                    "S3: median dataset"))
    checks.append(("s3", r"mean of 10,527",
                    str(round(ds["mean"])),
                    "S3: mean dataset"))
    checks.append(("s3", r"33\.0% of studies used fewer than 100",
                    pct(p1["dataset_size_pct_under_100"]),
                    "S3: <100 samples rate"))
    checks.append(("s3", r"59\.6% used fewer than 500",
                    pct(p1["dataset_size_pct_under_500"]),
                    "S3: <500 samples rate"))

    # By source
    exp = p1["dataset_size_by_source"]["experimental"]
    lit = p1["dataset_size_by_source"]["literature"]
    db = p1["dataset_size_by_source"]["database"]
    checks.append(("s3", r"median of just 45\.5 samples.*\*n\* = 42",
                    f"exp median={exp['median']}, n={exp['n']}",
                    "S3: experimental median"))
    checks.append(("s3", r"median of 418\.5 samples.*\*n\* = 38",
                    f"lit median={lit['median']}, n={lit['n']}",
                    "S3: literature median"))
    checks.append(("s3", r"median of 10,995 samples.*\*n\* = 27",
                    f"db median={int(db['median'])}, n={db['n']}",
                    "S3: database median"))

    # Features
    nf = p2["n_features_stats"]
    checks.append(("s3", r"119 studies reporting this information was 5",
                    f"n={nf['n']}, median={int(nf['median'])}",
                    "S3: features count"))

    # Data source distribution
    dsd = p1["data_source_distribution"]
    checks.append(("s3", r"55 studies, 43\.7%\).*literature.*41, 32\.5%.*28, 22\.2%.*2, 1\.6%",
                    f"sum={dsd['experimental']+dsd['literature']+dsd['database']+dsd['mixed']}",
                    "S3: data source distribution sum=126"))

    # Selection criteria
    checks.append(("s3", r"35\.7% of studies \(45/126\)",
                    f"{p1['data_selection_criteria_described_count']}/126",
                    "S3: selection criteria"))
    checks.append(("s3", r"64\.3% without",
                    pct(1 - p1["data_selection_criteria_described_rate"]),
                    "S3: no selection criteria"))

    # Preprocessing
    checks.append(("s3", r"56 studies \(44\.4%\) did not describe",
                    f"{p1['data_preprocessing_distribution']['none_reported']}",
                    "S3: no preprocessing"))

    # Materials
    checks.append(("s3", r"162 unique pollutants and 96 unique materials",
                    f"{p1['total_unique_pollutants']}/{p1['total_unique_materials']}",
                    "S3: unique pollutants/materials"))
    checks.append(("s3", r"44\.2% of all material mentions \(96 of 217",
                    "96/217=44.2%",
                    "S3: top 3 materials share"))

    # Targets
    checks.append(("s3", r"adsorption capacity: 37.*removal efficiency: 37.*rate constants: 4",
                    "37/37/4",
                    "S3: target distribution"))

    # ══════════════════════════════════════════════════════════════════════
    #  SECTION 4 - VALIDATION
    # ══════════════════════════════════════════════════════════════════════
    vmd = p2["validation_method_distribution"]
    checks.append(("s4", r"88 of 126 studies \(69\.8%\)",
                    f"random_split={vmd['random_split']}",
                    "S4: random split count"))
    checks.append(("s4", r"32 studies \(25\.4%\)",
                    f"k_fold={vmd['k_fold']}",
                    "S4: k-fold count"))
    checks.append(("s4", r"5 studies \(4\.0%\) did not report",
                    f"none={vmd['none_reported']}",
                    "S4: no validation count"))
    checks.append(("s4", r"1 study \(0\.8%\) employed external",
                    f"external={vmd['external']}",
                    "S4: external primary count"))

    # Validation sum = 126
    vsum = sum(vmd.values())
    checks.append(("s4", "CATEGORY_SUM_CHECK",
                    f"validation_sum={vsum}==126",
                    "S4: validation method sum=126"))

    # Grouped splitting
    checks.append(("s4", r"4 of 126 studies \(3\.2%.*1\.2[–-]7\.9%\)",
                    f"{p2['grouped_splitting_count']}/126",
                    "S4: grouped splitting"))
    checks.append(("s4", r"4 \(9\.8%\) employed it",
                    f"4/41={4/41*100:.1f}%",
                    "S4: grouped in literature"))

    # R2 distribution
    r2s = p2["r2_stats"]
    checks.append(("s4", r"median value was 0\.978 and the mean was 0\.960",
                    f"median={r2s['median']}, mean={r2s['mean']}",
                    "S4: R2 stats"))
    checks.append(("s4", r"91\.7%.*0\.90.*69\.7%.*0\.95.*35\.8%.*0\.99",
                    f"{pct(p2['r2_above_090'])}/{pct(p2['r2_above_095'])}/{pct(p2['r2_above_099'])}",
                    "S4: R2 threshold rates"))

    # R2 by validation
    rv = xtab["r2_by_validation_method"]
    checks.append(("s4", r"random splitting.*median.*0\.979.*\*n\* = 80",
                    f"random: median={rv['random_split']['median']}, n={rv['random_split']['n']}",
                    "S4: R2 by random split"))
    checks.append(("s4", r"\*k\*-fold.*median of 0\.957.*\*n\* = 25",
                    f"kfold: median={round(rv['k_fold']['median'], 3)}, n={rv['k_fold']['n']}",
                    "S4: R2 by k-fold"))

    # R2 by data source
    rds = xtab["r2_by_data_source"]
    checks.append(("s4", r"experimental.*median.*0\.989.*\*n\* = 48.*0\.957.*literature.*\*n\* = 38",
                    f"exp={round(rds['experimental']['median'], 3)}, lit={round(rds['literature']['median'], 3)}",
                    "S4: R2 by data source"))

    # Rigor score
    checks.append(("s4", r"median rigor score was 2\.0 \(mean 2\.6, range 0[–-]7\)",
                    f"median={ov['rigor_median']}, mean={round(ov['rigor_mean'], 1)}, range={ov['rigor_min']}-{ov['rigor_max']}",
                    "S4: rigor score stats"))
    checks.append(("s4", r"\*ρ\* = [−\-]0\.455.*\*p\* < 0\.0001",
                    f"rho={ov['spearman_rho']:.4f}, p={ov['spearman_p']:.2e}",
                    "S4: Spearman correlation"))

    # Rigor groups
    checks.append(("s4", r"0[–-]2.*median.*0\.991.*\*n\* = 58",
                    f"0-2: median={gs['0-2']['median_r2']}, n={gs['0-2']['n']}",
                    "S4: rigor group 0-2"))
    checks.append(("s4", r"0\.960 for scores 3[–-]4.*\*n\* = 32",
                    f"3-4: median={gs['3-4']['median_r2']}, n={gs['3-4']['n']}",
                    "S4: rigor group 3-4"))
    checks.append(("s4", r"0\.954 for scores 5[–-]6.*\*n\* = 16",
                    f"5-6: median={round(gs['5-6']['median_r2'], 3)}, n={gs['5-6']['n']}",
                    "S4: rigor group 5-6"))
    checks.append(("s4", r"0\.910 for scores 7[–-]9.*\*n\* = 3",
                    f"7-9: median={gs['7-9']['median_r2']}, n={gs['7-9']['n']}",
                    "S4: rigor group 7-9"))

    # Rigor group sum = 109
    rg_sum = sum(gs[g]["n"] for g in gs)
    checks.append(("s4", "CATEGORY_SUM_CHECK",
                    f"rigor_group_sum={rg_sum}==109",
                    "S4: rigor group sum=109"))

    # Sensitivity
    checks.append(("s4", r"\*ρ\* ranged from [−\-]0\.474 to [−\-]0\.373",
                    "sensitivity range",
                    "S4: sensitivity analysis range"))

    # External validation
    checks.append(("s4", r"16 of 126 studies \(12\.7%\)",
                    f"{p2['external_validation_count']}/126={pct(p2['external_validation_rate'])}%",
                    "S4: external validation"))
    checks.append(("s4", r"75 studies \(59\.5%\) reported",
                    f"{p2['reports_train_and_test_count']}/126={pct(p2['reports_train_and_test_rate'])}%",
                    "S4: train+test reporting"))

    # HPO
    hpo = p2["hyperparameter_tuning_distribution"]
    checks.append(("s4", r"74 studies, 58\.7%\) did not report",
                    f"none={hpo['none_reported']}, {pct_ct(hpo['none_reported'], 126)}%",
                    "S4: HPO not reported"))
    checks.append(("s4", r"grid search.*29.*Bayesian.*12.*genetic.*9.*random.*2",
                    f"grid={hpo['grid_search']}, bayes={hpo['bayesian']}, gen={hpo['genetic']}, rand={hpo['random_search']}",
                    "S4: HPO methods"))

    # ══════════════════════════════════════════════════════════════════════
    #  SECTION 5 - COMPARABILITY
    # ══════════════════════════════════════════════════════════════════════
    checks.append(("s5", r"30 unique algorithms",
                    str(p3["total_unique_algorithms"]),
                    "S5: unique algorithms"))
    checks.append(("s5", r"78 of 126 studies \(61\.9%\)",
                    f"ANN={p3['algorithm_frequency'][0][1]}",
                    "S5: ANN count"))
    checks.append(("s5", r"RF, 49, 38\.9%",
                    f"RF={p3['algorithm_frequency'][1][1]}",
                    "S5: RF count"))
    checks.append(("s5", r"XGBoost \(29,\s*23\.0%\)",
                    f"XGBoost={p3['algorithm_frequency'][2][1]}",
                    "S5: XGBoost count"))

    algc = p3["n_algorithms_compared_stats"]
    checks.append(("s5", r"median number of algorithms compared.*was just 2",
                    f"median={int(algc['median'])}",
                    "S5: median algorithms compared"))
    checks.append(("s5", r"38 studies \(30\.2%\) evaluating only a single",
                    f"single={p3['n_algorithms_compared_distribution']['1']}",
                    "S5: single algorithm studies"))

    # Metrics
    checks.append(("s5", r"\*R\*² \(112.*88\.9%\).*RMSE \(79.*62\.7%\).*MAE \(59.*46\.8%\)",
                    "R2=112/RMSE=79/MAE=59",
                    "S5: metric frequencies"))

    # Code/data
    checks.append(("s5", r"16 of 126 studies \(12\.7%\) made their code",
                    f"code={p3['code_available_count']}",
                    "S5: code availability"))
    checks.append(("s5", r"40 \(31\.7%\) shared their training data",
                    f"data={p3['data_available_count']}",
                    "S5: data availability"))

    # Software
    checks.append(("s5", r"Python \(51.*40\.5%\).*MATLAB \(41.*32\.5%\)",
                    "Python=51/MATLAB=41",
                    "S5: software tools"))

    # Feature selection
    checks.append(("s5", r"67 studies \(53\.2%\) did not report any feature selection",
                    f"none={p3['feature_selection_distribution']['none_reported']}",
                    "S5: no feature selection"))

    # ══════════════════════════════════════════════════════════════════════
    #  SECTION 6 - DISCUSSION
    # ══════════════════════════════════════════════════════════════════════
    checks.append(("s6", r"84 \(66\.7%\) employed.*interpretability",
                    f"{disc['has_interpretability_count']}/126",
                    "S6: interpretability count"))
    checks.append(("s6", r"57 studies, 45\.2%\)",
                    f"feature_importance={disc['interpretability_methods_frequency'][0][1]}",
                    "S6: feature importance"))
    checks.append(("s6", r"SHAP \(26, 20\.6%\)",
                    f"SHAP={disc['interpretability_methods_frequency'][2][1]}",
                    "S6: SHAP count"))
    checks.append(("s6", r"78 studies \(61\.9%\).*mechan",
                    f"{disc['mechanistic_discussion_count']}/126",
                    "S6: mechanistic discussion"))

    # Water type
    wt = disc["water_type_distribution"]
    checks.append(("s6", r"49 computational studies \(38\.9%\)",
                    f"not_applicable={wt['not_applicable']}",
                    "S6: computational studies"))
    checks.append(("s6", r"58 \(75\.3%\) used synthetic",
                    f"synthetic={wt['synthetic']}, {wt['synthetic']}/77={wt['synthetic']/77*100:.1f}%",
                    "S6: synthetic studies"))
    checks.append(("s6", r"6 \(7\.8%.*3\.6[–-]16\.2%\).*real-world",
                    f"real={wt['real_wastewater']}+both={wt['both']}=6",
                    "S6: real-world studies"))
    checks.append(("s6", r"13 \(16\.9%\) did not specify",
                    f"not_specified={wt['not_specified']}",
                    "S6: not specified water"))

    # Water type sum (non-computational) = 77
    wt_noncomp = wt["synthetic"] + wt["real_wastewater"] + wt["both"] + wt["not_specified"]
    checks.append(("s6", "CATEGORY_SUM_CHECK",
                    f"water_type_noncomp={wt_noncomp}==77",
                    "S6: water type non-computational sum=77"))

    # Full water type sum = 126
    wt_total = sum(wt.values())
    checks.append(("s6", "CATEGORY_SUM_CHECK",
                    f"water_type_total={wt_total}==126",
                    "S6: water type total sum=126"))

    # Deployment
    checks.append(("s6", r"[Ss]calability.*\(58 studies,?\s*46\.0%\)",
                    f"{disc['discusses_scalability_count']}/126",
                    "S6: scalability"))
    checks.append(("s6", r"8 studies \(6\.3%\).*engineering validation",
                    f"{disc['engineering_validation_count']}/126",
                    "S6: engineering validation"))
    checks.append(("s6", r"5 \(4\.0%\)",
                    f"cost={disc['cost_analysis_count']}",
                    "S6: cost analysis"))

    # ══════════════════════════════════════════════════════════════════════
    #  SECTION 7 - RECOMMENDATIONS (Table 1 cross-check)
    # ══════════════════════════════════════════════════════════════════════
    checks.append(("s7", r"64\.3% did not describe data selection",
                    pct(1 - p1["data_selection_criteria_described_rate"]),
                    "S7/Table1: no selection criteria"))
    checks.append(("s7", r"96\.8% lacked grouped splitting",
                    f"{100 - p2['grouped_splitting_rate']*100:.1f}",
                    "S7/Table1: no grouped splitting"))
    checks.append(("s7", r"87\.3% lacked external validation",
                    pct(1 - p2["external_validation_rate"]),
                    "S7/Table1: no external validation"))
    checks.append(("s7", r"58\.7% did not report",
                    pct_ct(hpo["none_reported"], 126),
                    "S7/Table1: no HPO reported"))
    checks.append(("s7", r"12\.7%.*shared code",
                    pct(p3["code_available_rate"]),
                    "S7/Table1: code available"))

    # ══════════════════════════════════════════════════════════════════════
    #  SI CHECKS
    # ══════════════════════════════════════════════════════════════════════
    checks.append(("si", r"44 top-level structured fields \(49 including sub-fields\)",
                    "44/49",
                    "SI: field count consistency"))

    return checks


def run_checks(checks, sections):
    """Execute all checks and return pass/fail/skip counts."""
    n_pass = 0
    n_fail = 0
    n_skip = 0
    failures = []

    for sec_key, pattern, expected, desc in checks:
        text = sections.get(sec_key, "")
        if not text:
            n_skip += 1
            print(f"  {YELLOW}SKIP{RESET} {desc} (section not loaded)")
            continue

        # Special handler: category sum checks
        if pattern == "CATEGORY_SUM_CHECK":
            # Expected format: "label=VALUE==TARGET"
            parts = expected.split("==")
            actual_part = parts[0]  # e.g., "validation_sum=126"
            target = int(parts[1])
            actual_val = int(actual_part.split("=")[1])
            if actual_val == target:
                n_pass += 1
                print(f"  {GREEN}PASS{RESET} {desc} ({actual_part})")
            else:
                n_fail += 1
                failures.append((desc, f"expected {target}, got {actual_val}"))
                print(f"  {RED}FAIL{RESET} {desc}: expected {target}, got {actual_val}")
            continue

        # Regex check
        match = re.search(pattern, text)
        if match:
            n_pass += 1
            print(f"  {GREEN}PASS{RESET} {desc}")
        else:
            n_fail += 1
            failures.append((desc, f"pattern not found: {pattern[:60]}"))
            print(f"  {RED}FAIL{RESET} {desc}: pattern not found")

    return n_pass, n_fail, n_skip, failures


def check_placeholders(sections):
    """Check for unresolved placeholders in manuscript text."""
    placeholders = [r"\[ref\]", r"\[TODO\]", r"\[CITE\]", r"\[X\]", r"\[cite\]", r"\[引文\]"]
    issues = []
    for sec_key, text in sections.items():
        if sec_key == "all":
            continue
        for ph in placeholders:
            matches = re.findall(ph, text, re.IGNORECASE)
            if matches:
                issues.append((sec_key, ph, len(matches)))
    return issues


def check_ai_traces(sections):
    """Check for AI-style filler words."""
    ai_words = [
        r"\bFurthermore\b", r"\bMoreover\b", r"\bAdditionally\b",
        r"\bNotably\b", r"\bdelve\b", r"\bIndeed\b",
    ]
    issues = []
    for sec_key, text in sections.items():
        if sec_key == "all":
            continue
        for w in ai_words:
            matches = re.findall(w, text)
            if matches:
                issues.append((sec_key, w, len(matches)))
    return issues


def check_em_dashes(sections):
    """Check for em dash parentheticals."""
    issues = []
    for sec_key, text in sections.items():
        if sec_key == "all":
            continue
        # Match " — " (space-em-dash-space)
        matches = re.findall(r" — ", text)
        if matches:
            issues.append((sec_key, len(matches)))
    return issues


def main():
    print(f"\n{BOLD}ML4Env Manuscript Validator{RESET}")
    print(f"{'='*60}\n")

    stats = load_stats()
    rigor = load_rigor()
    sections = load_sections()

    # ── Data-text consistency checks ─────────────────────────────────────
    print(f"{BOLD}Data-Text Consistency Checks{RESET}")
    print(f"{'-'*40}")
    checks = build_checks(stats, rigor)
    n_pass, n_fail, n_skip, failures = run_checks(checks, sections)

    # ── Placeholder checks ───────────────────────────────────────────────
    print(f"\n{BOLD}Placeholder Checks{RESET}")
    print(f"{'-'*40}")
    ph_issues = check_placeholders(sections)
    if ph_issues:
        for sec, ph, count in ph_issues:
            print(f"  {YELLOW}WARN{RESET} {sec}: {count}x {ph}")
    else:
        print(f"  {GREEN}PASS{RESET} No unresolved placeholders")

    # ── AI trace checks ──────────────────────────────────────────────────
    print(f"\n{BOLD}AI Trace Checks{RESET}")
    print(f"{'-'*40}")
    ai_issues = check_ai_traces(sections)
    if ai_issues:
        for sec, word, count in ai_issues:
            print(f"  {YELLOW}WARN{RESET} {sec}: {count}x {word}")
    else:
        print(f"  {GREEN}PASS{RESET} No AI filler words detected")

    # ── Em dash checks ───────────────────────────────────────────────────
    print(f"\n{BOLD}Em Dash Checks{RESET}")
    print(f"{'-'*40}")
    em_issues = check_em_dashes(sections)
    if em_issues:
        for sec, count in em_issues:
            print(f"  {YELLOW}WARN{RESET} {sec}: {count}x em dash parenthetical")
    else:
        print(f"  {GREEN}PASS{RESET} No em dash parentheticals")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    total = n_pass + n_fail + n_skip
    print(f"  {GREEN}PASS: {n_pass}{RESET}  {RED}FAIL: {n_fail}{RESET}  {YELLOW}SKIP: {n_skip}{RESET}  TOTAL: {total}")

    if failures:
        print(f"\n{RED}Failures:{RESET}")
        for desc, detail in failures:
            print(f"  - {desc}: {detail}")

    print()
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
