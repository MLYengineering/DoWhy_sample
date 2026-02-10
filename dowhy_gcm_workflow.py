#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DoWhy + GCM (Graphical Causal Model) Workflow – Lern-/Template-Skript
====================================================================

Ziel dieses Skripts
-------------------
Du kannst damit (1) einen inhaltlich angenommenen DAG (gerichteten Graph) definieren,
(2) daraus mit DoWhy einen kausalen Effekt (ATE) schätzen (Backdoor),
(3) mit DoWhy-GCM ein strukturelles Kausalmodell (SCM) fitten,
(4) synthetische Daten generieren und Interventionen ("do(X=...)") simulieren,
(5) Root-Cause-Ranking (Driver-Ranking) per Interventionen durchführen
    und (6) einfache Checks machen, ob deine angenommenen Zusammenhänge "gesund" wirken.

WICHTIGER KONZEPT-HINWEIS (bitte einmal wirklich lesen)
-------------------------------------------------------
- Der DAG ist NICHT "von der KI bewiesen".
  Der DAG ist deine Annahme / dein Modell der Welt (E/E-Architektur, Prozesslogik, etc.).
- DoWhy kann unter dieser Annahme Effekte schätzen und testen, ob das Ergebnis plausibel/robust wirkt.
- "Gut" heißt hier: (a) synthetische Daten sind verteilungsmäßig plausibel,
  (b) Interventionen verhalten sich in der erwarteten Richtung,
  (c) Refuter-Checks schlagen nicht direkt Alarm,
  (d) die wichtigsten Effekte sind stabil und nicht komplett Artefakt.

Dieses Skript ist absichtlich stark kommentiert – damit du es in 10 Tagen noch verstehst.

Voraussetzungen
---------------
pip install pandas numpy scikit-learn matplotlib networkx dowhy

Aufruf (Beispiele)
------------------
python dowhy_gcm_workflow.py --csv ai4i2020.csv
python dowhy_gcm_workflow.py --csv ai4i2020.csv --treatment tool_wear --outcome failure
python dowhy_gcm_workflow.py --csv ai4i2020.csv --do_var tool_wear --do_low 50 --do_high 200

Du kannst den DAG unten in DAG_EDGES ändern (Kantenliste).
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.linear_model import LogisticRegression

from dowhy import CausalModel
from dowhy import gcm


# =============================================================================
# 1) KONFIG: Hier passt du deinen DAG (die Struktur) an
# =============================================================================
#
# Du definierst den DAG als Liste von gerichteten Kanten (Ursache -> Wirkung).
# Das ist der "Schaltplan" deines Systems.
#
# Für AI4I (Demo) nutzen wir:
# air_temp -> proc_temp
# air_temp/proc_temp/rpm/torque -> tool_wear
# air_temp/proc_temp/rpm/torque/tool_wear -> failure
#
# Du kannst:
# - Kanten hinzufügen/entfernen
# - Variablennamen an deine echten Spalten anpassen (z.B. IDEX: ECU_A_SW -> ECU_A_State -> CENTRAL_State -> Symptom)
#
DAG_EDGES: List[Tuple[str, str]] = [
    ("air_temp", "proc_temp"),

    ("air_temp", "tool_wear"),
    ("proc_temp", "tool_wear"),
    ("rpm", "tool_wear"),
    ("torque", "tool_wear"),

    ("air_temp", "failure"),
    ("proc_temp", "failure"),
    ("rpm", "failure"),
    ("torque", "failure"),
    ("tool_wear", "failure"),
]


# =============================================================================
# 2) Daten-Preprocessing (AI4I-spezifisch, aber leicht anpassbar)
# =============================================================================

RENAME_MAP_DEFAULT = {
    "Air temperature [K]": "air_temp",
    "Process temperature [K]": "proc_temp",
    "Rotational speed [rpm]": "rpm",
    "Torque [Nm]": "torque",
    "Tool wear [min]": "tool_wear",
    "Failure": "failure",
    "Machine failure": "failure",
}


def load_and_prepare_ai4i(csv_path: str) -> pd.DataFrame:
    """
    Lädt AI4I CSV und macht Minimal-Preprocessing:
    - Spaltennamen trimmen
    - typische Spalten umbenennen
    - Type one-hot encoden (Type_L, Type_M ...)
    - failure als int (0/1)
    - NaNs droppen
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Umbenennen, soweit vorhanden:
    rename_map = {k: v for k, v in RENAME_MAP_DEFAULT.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    if "failure" not in df.columns:
        fail_candidates = [c for c in df.columns if "fail" in c.lower()]
        raise KeyError(
            f"Spalte 'failure' nicht gefunden. Kandidaten: {fail_candidates}. "
            f"Bitte rename mapping anpassen."
        )

    # Type dummy encoding, falls Type existiert:
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], drop_first=True)

    df["failure"] = df["failure"].astype(int)
    df = df.dropna().reset_index(drop=True)
    return df


# =============================================================================
# 3) Hilfsfunktionen: DOT-Graph String + Rollen (Confounder/Mediator/Collider)
# =============================================================================

def build_dot_graph(edges: List[Tuple[str, str]]) -> str:
    """Baut einen DOT-String für DoWhy CausalModel(graph=...)."""
    lines = ["digraph {"]
    for a, b in edges:
        lines.append(f"{a} -> {b};")
    lines.append("}")
    return "\n".join(lines)


def roles_for_query(G: nx.DiGraph, treatment: str, outcome: str) -> Dict[str, List[str]]:
    """
    Rollen bezogen auf eine konkrete Frage X=treatment -> Y=outcome.

    Confounder: Z -> X und Z -> Y (gemeinsame Ursache)
    Mediator:   liegt auf Pfad X -> ... -> Y
    Collider:   X -> C <- Z (vereinfachte Erkennung: Kind von X mit indegree>=2)
    """
    if treatment not in G.nodes or outcome not in G.nodes:
        raise ValueError("treatment/outcome müssen im Graph sein.")

    # Mediatoren: alle Knoten auf irgendeinem Pfad X -> ... -> Y
    mediators = set()
    try:
        for path in nx.all_simple_paths(G, source=treatment, target=outcome):
            for n in path[1:-1]:
                mediators.add(n)
    except nx.NetworkXNoPath:
        pass

    parents_X = set(G.predecessors(treatment))
    parents_Y = set(G.predecessors(outcome))
    confounders = parents_X & parents_Y

    descendants_X = nx.descendants(G, treatment)
    confounders = {z for z in confounders if z not in descendants_X}

    colliders = set()
    for c in G.successors(treatment):
        if G.in_degree(c) >= 2:
            colliders.add(c)

    return {
        "confounders": sorted(confounders),
        "mediators": sorted(mediators),
        "colliders": sorted(colliders),
    }


# =============================================================================
# 4) DoWhy ATE-Schätzung + Refuter
# =============================================================================

def estimate_ate_with_dowhy(df: pd.DataFrame, dot_graph: str, treatment: str, outcome: str):
    """ATE via DoWhy: identify_effect + estimate_effect."""
    model = CausalModel(data=df, treatment=treatment, outcome=outcome, graph=dot_graph)
    identified_estimand = model.identify_effect()
    estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
    return model, identified_estimand, estimate


def run_refuters(
    model: CausalModel,
    identified_estimand,
    estimate,
    num_simulations: int = 20,
    random_seed: int = 42,
    show_progress_bar: bool = False,
):
    """Quick Checks: Placebo + Random Common Cause.

    Hinweis: Refuter können je nach Dataset/Estimator dauern (Resampling/Permutation).
    Darum: num_simulations klein halten, wenn du nur einen schnellen Plausibilitätscheck willst.

    Interpretation:
    - placebo_treatment_refuter: ersetzt Treatment durch Zufall -> Effekt sollte ~ 0 sein.
    - random_common_cause: fügt Zufalls-Confounder hinzu -> Effekt sollte nicht komplett kippen.
    """
    placebo = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="placebo_treatment_refuter",
        num_simulations=num_simulations,
        random_seed=random_seed,
        show_progress_bar=show_progress_bar,
    )
    random_cc = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="random_common_cause",
        num_simulations=num_simulations,
        random_seed=random_seed,
        show_progress_bar=show_progress_bar,
    )
    return {"placebo": placebo, "random_common_cause": random_cc}


# =============================================================================
# 5) GCM/SCM fitten + Synthetische Daten + Interventionen
# =============================================================================

def fit_gcm_scm(df: pd.DataFrame, G: nx.DiGraph, cols: List[str]) -> gcm.StructuralCausalModel:
    """
    Baut ein StructuralCausalModel (SCM) und fittet Mechanismen.
    Inhaltlich: Jeder Knoten wird als f(Eltern)+Rauschen gelernt.
    """
    scm = gcm.StructuralCausalModel(G)
    gcm.auto.assign_causal_mechanisms(scm, df[cols])
    gcm.fit(scm, df[cols])
    return scm


def draw_synthetic(scm: gcm.StructuralCausalModel, n: int = 5000) -> pd.DataFrame:
    """Generiert synthetische Samples aus dem gefitteten SCM."""
    return gcm.draw_samples(scm, num_samples=n)


def interventional_delta(
    scm: gcm.StructuralCausalModel,
    var: str,
    low_value,
    high_value,
    n: int = 5000,
    outcome_col: str = "failure",
) -> Dict[str, float]:
    """do(var=low) vs do(var=high): Failure-Raten + Delta."""
    base = gcm.draw_samples(scm, num_samples=n)

    low = gcm.interventional_samples(
        scm,
        interventions={var: (lambda _x, val=low_value: val)},
        num_samples_to_draw=n,
    )
    high = gcm.interventional_samples(
        scm,
        interventions={var: (lambda _x, val=high_value: val)},
        num_samples_to_draw=n,
    )

    return {
        "baseline": float(base[outcome_col].mean()),
        "low": float(low[outcome_col].mean()),
        "high": float(high[outcome_col].mean()),
        "delta_high_minus_low": float(high[outcome_col].mean() - low[outcome_col].mean()),
    }


# =============================================================================
# 6) Root-Cause Ranking (Driver-Ranking) per Interventionen
# =============================================================================


def driver_ranking_numeric(
    df: pd.DataFrame,
    scm: gcm.StructuralCausalModel,
    drivers: List[str],
    outcome_col: str = "failure",
    n: int = 5000,
    q_low: float = 0.1,
    q_high: float = 0.9,
) -> pd.DataFrame:
    """Interventions-Ranking: für jeden Driver q10 vs q90 und Delta in failure."""
    quant = df[drivers].quantile([q_low, q_high])
    low_vals = quant.loc[q_low].to_dict()
    high_vals = quant.loc[q_high].to_dict()

    base = gcm.draw_samples(scm, num_samples=n)
    base_rate = float(base[outcome_col].mean())

    rows = []
    for v in drivers:
        # DoWhy-GCM API Hinweis:
        # In manchen DoWhy-Versionen erwartet interventional_samples als 2. Positionsargument
        # beobachtete Daten (observed_data). Wenn man dort versehentlich "n" (int) übergibt,
        # kommt der Fehler: AttributeError: 'int' object has no attribute 'copy'.
        # Daher rufen wir interventional_samples hier immer mit KEYWORD-Argumenten auf.

        low = gcm.interventional_samples(
            scm,
            interventions={v: (lambda _x, val=low_vals[v]: val)},
            num_samples_to_draw=n,
        )
        high = gcm.interventional_samples(
            scm,
            interventions={v: (lambda _x, val=high_vals[v]: val)},
            num_samples_to_draw=n,
        )

        rows.append({
            "driver": v,
            "baseline_failure": base_rate,
            "low_value(q10)": float(low_vals[v]),
            "low_failure": float(low[outcome_col].mean()),
            "high_value(q90)": float(high_vals[v]),
            "high_failure": float(high[outcome_col].mean()),
            "delta(high-low)": float(high[outcome_col].mean() - low[outcome_col].mean()),
        })

    return pd.DataFrame(rows).sort_values("delta(high-low)", ascending=False).reset_index(drop=True)


# =============================================================================
# 6b) Auto-Scan: mögliche Treatments automatisch durchgehen
# =============================================================================

def infer_low_high_for_var(df: pd.DataFrame, var: str, q_low: float = 0.1, q_high: float = 0.9):
    """Wählt sinnvolle (low, high) Werte für Interventionen.

    - Für 0/1-Variablen (Dummy): low=0, high=1
    - Für numerische Variablen: low=q10, high=q90

    Rückgabe: (low, high)
    """
    s = df[var]
    uniq = pd.unique(s.dropna())
    if len(uniq) <= 2 and set(uniq).issubset({0, 1}):
        return 0.0, 1.0
    # numerisch: Quantile
    return float(s.quantile(q_low)), float(s.quantile(q_high))


def auto_scan_treatments_dowhy(
    df: pd.DataFrame,
    G: nx.DiGraph,
    edges: List[Tuple[str, str]],
    outcome: str,
    method_name: str = "backdoor.linear_regression",
) -> pd.DataFrame:
    """Geht automatisch alle möglichen Treatments durch und schätzt ATE (DoWhy).

    Kandidaten = alle Graph-Knoten, die im DataFrame existieren und nicht das Outcome sind.

    Wichtiger Lernpunkt:
    - Das ist ein *Screening*: Du siehst schnell, welche Variablen unter deinem DAG stark mit dem Outcome wirken.
    - Für manche Variablen ist eine "Intervention" in der Realität nicht sinnvoll (z.B. fixe Klassenlabels).

    Ergebnis: Tabelle mit Treatment, ATE, Rollen-Hinweisen (Confounder/Mediator/Collider).
    """
    dot_graph = build_dot_graph(edges)

    candidates = [n for n in G.nodes() if (n in df.columns and n != outcome)]

    rows = []
    for t in sorted(candidates):
        try:
            model, identified_estimand, estimate = estimate_ate_with_dowhy(
                df=df,
                dot_graph=dot_graph,
                treatment=t,
                outcome=outcome,
            )
            ate = float(estimate.value)
            roles = roles_for_query(G, treatment=t, outcome=outcome)
            rows.append({
                "treatment": t,
                "ate": ate,
                "abs_ate": abs(ate),
                "n_confounders": len(roles["confounders"]),
                "n_mediators": len(roles["mediators"]),
                "n_colliders": len(roles["colliders"]),
                "confounders": ",".join(roles["confounders"]),
                "mediators": ",".join(roles["mediators"]),
                "colliders": ",".join(roles["colliders"]),
            })
        except Exception as e:
            # Nicht jedes Treatment ist schätzbar (z.B. wenn DoWhy intern Probleme hat).
            rows.append({
                "treatment": t,
                "ate": np.nan,
                "abs_ate": np.nan,
                "n_confounders": np.nan,
                "n_mediators": np.nan,
                "n_colliders": np.nan,
                "confounders": "",
                "mediators": "",
                "colliders": "",
                "error": str(e),
            })

    out = pd.DataFrame(rows)
    if "abs_ate" in out.columns:
        out = out.sort_values("abs_ate", ascending=False, na_position="last").reset_index(drop=True)
    return out


def auto_scan_interventions_gcm(
    df: pd.DataFrame,
    scm: gcm.StructuralCausalModel,
    drivers: List[str],
    outcome_col: str = "failure",
    n: int = 5000,
    q_low: float = 0.1,
    q_high: float = 0.9,
) -> pd.DataFrame:
    """Auto-Scan im GCM-Stil: do(low) vs do(high) pro Variable und Delta im Outcome.

    Das ist sehr nah am Root-Cause/Driver-Ranking.
    Unterschied zum DoWhy-ATE:
    - Hier misst du konkret "high vs low" (q90 vs q10 oder 1 vs 0), also einen *Hebel-Effekt*.
    """
    base = gcm.draw_samples(scm, num_samples=n)
    base_rate = float(base[outcome_col].mean())

    rows = []
    for v in drivers:
        if v not in df.columns:
            continue
        low_val, high_val = infer_low_high_for_var(df, v, q_low=q_low, q_high=q_high)

        low = gcm.interventional_samples(
            scm,
            interventions={v: (lambda _x, val=low_val: val)},
            num_samples_to_draw=n,
        )
        high = gcm.interventional_samples(
            scm,
            interventions={v: (lambda _x, val=high_val: val)},
            num_samples_to_draw=n,
        )

        rows.append({
            "var": v,
            "baseline_failure": base_rate,
            "low": low_val,
            "high": high_val,
            "low_failure": float(low[outcome_col].mean()),
            "high_failure": float(high[outcome_col].mean()),
            "delta(high-low)": float(high[outcome_col].mean() - low[outcome_col].mean()),
        })

    return pd.DataFrame(rows).sort_values("delta(high-low)", ascending=False).reset_index(drop=True)


# =============================================================================
# 7) Simple Checks: "Ist das Ergebnis gut?"
# =============================================================================

def sanity_checks_real_vs_synth(df_real: pd.DataFrame, df_syn: pd.DataFrame, cols: List[str]) -> None:
    """Plausibilitätschecks: failure-rate, describe, corr."""
    print("\n=== SANITY CHECKS: Real vs Synthetic ===")
    print("Real failure rate:", float(df_real["failure"].mean()))
    print("Syn  failure rate:", float(df_syn["failure"].mean()))

    print("\nReal describe():")
    print(df_real[cols].describe())

    print("\nSynthetic describe():")
    print(df_syn[cols].describe())

    print("\nReal corr (numeric):")
    print(df_real[cols].corr(numeric_only=True))

    print("\nSyn corr (numeric):")
    print(df_syn[cols].corr(numeric_only=True))


def logit_naive_vs_adjusted(df: pd.DataFrame, treatment: str, adjust_cols: List[str], outcome: str = "failure") -> None:
    """
    Bias-Check: NAIV vs ADJUSTED (Logistic Regression).
    Wenn sich der treatment-Koeffizient stark ändert, beeinflussen adjust_cols die Schätzung stark.
    """
    def coef(cols: List[str]) -> float:
        X = df[cols].to_numpy()
        y = df[outcome].to_numpy()
        clf = LogisticRegression(max_iter=3000)
        clf.fit(X, y)
        coef_map = dict(zip(cols, clf.coef_[0]))
        return float(coef_map[treatment])

    naive = coef([treatment])
    adj = coef([treatment] + adjust_cols)
    print("\n=== Bias-Check (Logit): NAIV vs ADJUSTED ===")
    print(f"NAIV coef({treatment}): {naive}")
    print(f"ADJ  coef({treatment}): {adj}")
    print(f"Delta (ADJ-NAIV): {adj - naive}")


# =============================================================================
# 8) Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True, help="Pfad zur CSV (AI4I 2020)")
    parser.add_argument("--treatment", type=str, default="tool_wear", help="Treatment/Action Variable")
    parser.add_argument("--outcome", type=str, default="failure", help="Outcome Variable")
    parser.add_argument("--n_samples", type=int, default=5000, help="Anzahl Samples für synthetic/interventions")
    parser.add_argument("--do_var", type=str, default="tool_wear", help="Variable für Demo-Intervention")
    parser.add_argument("--do_low", type=float, default=50.0, help="Low Wert für Intervention")
    parser.add_argument("--do_high", type=float, default=200.0, help="High Wert für Intervention")
    parser.add_argument("--auto_scan", action="store_true", help="Scan: alle Treatments im DAG automatisch durchgehen (DoWhy-ATE + Rollen).")
    parser.add_argument("--auto_topk", type=int, default=10, help="Top-K Treatments (nach |ATE|) für kurze Ausgabe/Checks.")
    parser.add_argument("--auto_refuters", action="store_true", help="Optional: Refuter für die Top-K Treatments laufen lassen (kann dauern).")
    args = parser.parse_args()

    # --- Daten laden ---
    df = load_and_prepare_ai4i(args.csv)

    # --- Type Dummy-Spalten automatisch erkennen und als Confounder-Kanten ergänzen ---
    type_cols = [c for c in df.columns if c.startswith("Type_")]
    edges = list(DAG_EDGES)
    for t in type_cols:
        edges.append((t, "tool_wear"))
        edges.append((t, "failure"))

    # --- Graph bauen ---
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # --- Rollen im DAG für die konkrete Fragestellung ausgeben ---
    print("\n=== Rollen im DAG für X=treatment -> Y=outcome ===")
    roles = roles_for_query(G, treatment=args.treatment, outcome=args.outcome)
    print(f"Treatment: {args.treatment} | Outcome: {args.outcome}")
    print("Confounders:", roles["confounders"])
    print("Mediators:  ", roles["mediators"])
    print("Colliders:  ", roles["colliders"])

    # --- DOT für DoWhy ---
    dot_graph = build_dot_graph(edges)

    # --- ATE schätzen ---
    print("\n=== DoWhy: ATE Schätzung ===")
    model, identified_estimand, estimate = estimate_ate_with_dowhy(df, dot_graph, args.treatment, args.outcome)
    print(identified_estimand)
    print("ATE:", estimate.value)

    # --- Refuter ---
    print("\n=== DoWhy: Refuter Checks ===")
    ref = run_refuters(model, identified_estimand, estimate, num_simulations=20)
    print(ref["placebo"])
    print(ref["random_common_cause"])

    # --- GCM Fit ---
    cols = sorted([n for n in G.nodes() if n in df.columns])
    missing = [n for n in G.nodes() if n not in df.columns]
    if missing:
        print("\nWARNUNG: Diese Graph-Knoten fehlen im DataFrame und werden ignoriert:", missing)

    print("\n=== GCM: Fit SCM ===")
    scm = fit_gcm_scm(df, G, cols)

    # --- Synthetic ---
    synthetic = draw_synthetic(scm, n=args.n_samples)
    print("\nSynthetic head():")
    print(synthetic.head())

    # --- Checks ---
    sanity_checks_real_vs_synth(df, synthetic, cols)

    # --- Intervention Demo ---
    print("\n=== Intervention Demo ===")
    if args.do_var not in df.columns:
        print(f"do_var '{args.do_var}' ist nicht im df.")
    else:
        res = interventional_delta(scm, args.do_var, args.do_low, args.do_high, n=args.n_samples, outcome_col=args.outcome)
        print(f"Baseline {args.outcome} rate:", res["baseline"])
        print(f"do({args.do_var}={args.do_low}) rate:", res["low"])
        print(f"do({args.do_var}={args.do_high}) rate:", res["high"])
        print("Delta (high-low):", res["delta_high_minus_low"])


    # --- Driver Ranking ---
    print("\n=== Root-cause Driver Ranking ===")
    drivers = [c for c in ["air_temp", "proc_temp", "rpm", "torque", "tool_wear"] if c in df.columns]
    if len(drivers) >= 2:
        ranking = driver_ranking_numeric(df, scm, drivers=drivers, outcome_col=args.outcome, n=args.n_samples)
        print(ranking)


    # --- Auto-Scan: alle Treatments im DAG (DoWhy ATE) ---
    if args.auto_scan:
        print("\n=== AUTO-SCAN (DoWhy): alle Treatments im DAG ===")
        scan = auto_scan_treatments_dowhy(
            df=df,
            G=G,
            edges=edges,
            outcome=args.outcome,
            method_name="backdoor.linear_regression",
        )
        # Kurze Top-K Ausgabe:
        print(scan.head(args.auto_topk))

        # Optional: Refuter nur für Top-K laufen lassen (kann dauern)
        if args.auto_refuters:
            print("\n=== AUTO-SCAN: Refuter für Top-K Treatments ===")
            dot_graph_all = build_dot_graph(edges)
            for t in scan["treatment"].head(args.auto_topk).tolist():
                print(f"\n--- Treatment: {t} ---")
                try:
                    m = CausalModel(data=df, treatment=t, outcome=args.outcome, graph=dot_graph_all)
                    ie = m.identify_effect()
                    est = m.estimate_effect(ie, method_name="backdoor.linear_regression")
                    print("ATE:", float(est.value))
                    rr = run_refuters(m, ie, est, num_simulations=10)
                    print(rr["placebo"])
                    print(rr["random_common_cause"])
                except Exception as e:
                    print("Refuter/ATE Fehler:", e)

        # Zusätzlich (GCM): Hebel-Scan als Root-cause Ranking über alle Treiber
        print("\n=== AUTO-SCAN (GCM): Hebel-Effekte do(low) vs do(high) ===")
        all_vars = [c for c in cols if c != args.outcome]
        gcm_scan = auto_scan_interventions_gcm(df=df, scm=scm, drivers=all_vars, outcome_col=args.outcome, n=args.n_samples)
        print(gcm_scan.head(args.auto_topk))

    # --- Bias-Check: NAIV vs ADJ (Logit) ---
    adjust_cols = [c for c in roles["confounders"] if c in df.columns and c != args.treatment]
    if adjust_cols:
        logit_naive_vs_adjusted(df, args.treatment, adjust_cols, outcome=args.outcome)

    print("\nFERTIG. Tipp: Ändere DAG_EDGES und wiederhole – dann siehst du, wie sich alles verschiebt.")


if __name__ == "__main__":
    main()
