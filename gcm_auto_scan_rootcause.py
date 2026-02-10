#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GCM Auto-Scan (Root-Cause / Driver-Ranking) – Minimal-Skript
===========================================================

Was macht das Skript?
---------------------
1) Du definierst einen DAG (gerichteten Graph) als "Physik-/Systemmodell".
2) DoWhy-GCM lernt daraus ein Structural Causal Model (SCM): pro Knoten ein Modell f(Eltern)+Noise.
3) Danach wird für *jede* Variable X ein Interventionstest gemacht:
      do(X=low) vs do(X=high)
   und wir messen, wie stark sich die Failure-Rate (Outcome) verändert.

Das Ergebnis ist eine Tabelle:
- var: getestete Variable
- low/high: Intervention-Werte (q10/q90 oder 0/1 bei Dummy)
- low_failure/high_failure: Failure-Rate nach Intervention
- delta(high-low): Hebelwirkung (Root-cause Kandidat)

Wichtig:
--------
- Das ist "interventional feature ranking" unter deinem DAG.
- Es beweist NICHT, dass der DAG wahr ist – es zeigt nur, was *unter dem DAG* ein Hebel wäre.
- Nutze es als Priorisierung: welche Variablen schaue ich mir als erstes an?

Install:
--------
pip install pandas numpy networkx dowhy scikit-learn

Run:
----
python gcm_auto_scan_rootcause.py --csv ai4i2020.csv
python gcm_auto_scan_rootcause.py --csv ai4i2020.csv --outcome failure --n_samples 5000
python gcm_auto_scan_rootcause.py --csv ai4i2020.csv --topk 15 --save_csv scan_out.csv
python gcm_auto_scan_rootcause.py --csv ai4i2020.csv --plot_dag --plot_dir . --save_strength_csv arrow_strength.csv
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

import pandas as pd
import networkx as nx
from dowhy import gcm

# Optional: schöne DAG-Visualisierung (Graphviz/pygraphviz), mit Fallback auf matplotlib.
import os


# =============================================================================
# 1) DAG: HIER ANPASSEN
# =============================================================================
# Für AI4I Demo:

'''
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
'''

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
# 2) AI4I Preprocessing (leicht austauschbar)
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
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    rename_map = {k: v for k, v in RENAME_MAP_DEFAULT.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    if "failure" not in df.columns:
        raise KeyError("Spalte 'failure' nicht gefunden. Bitte RENAME_MAP_DEFAULT anpassen.")

    # Type -> Dummies (Type_L, Type_M). drop_first=True -> Referenz ist z.B. Type_H (implizit)
    if "Type" in df.columns:
        df = pd.get_dummies(df, columns=["Type"], drop_first=True)

    df["failure"] = df["failure"].astype(int)
    df = df.dropna().reset_index(drop=True)
    return df


# =============================================================================
# 3) Helpers: low/high Auswahl + Auto-Scan
# =============================================================================
def infer_low_high_for_var(df: pd.DataFrame, var: str, q_low: float = 0.1, q_high: float = 0.9):
    """
    Wählt (low, high) für Interventionen:
    - Dummy 0/1 -> (0,1)
    - sonst q10/q90
    """
    s = df[var]
    uniq = pd.unique(s.dropna())
    if len(uniq) <= 2 and set(uniq).issubset({0, 1}):
        return 0.0, 1.0
    return float(s.quantile(q_low)), float(s.quantile(q_high))


def auto_scan_interventions_gcm(
    df: pd.DataFrame,
    scm: gcm.StructuralCausalModel,
    vars_to_scan: List[str],
    outcome_col: str,
    n: int,
    q_low: float = 0.1,
    q_high: float = 0.9,
) -> pd.DataFrame:
    """
    Für jede Variable X:
      do(X=low) vs do(X=high) -> delta in outcome (Failure-Rate)
    """
    base = gcm.draw_samples(scm, num_samples=n)
    base_rate = float(base[outcome_col].mean())

    rows = []
    for v in vars_to_scan:
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
# 4) Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Pfad zur AI4I CSV (oder später IDEX CSV).")
    ap.add_argument("--outcome", default="failure", help="Outcome-Spalte (Default: failure).")
    ap.add_argument("--n_samples", type=int, default=5000, help="Samples pro Intervention (Default: 5000).")
    ap.add_argument("--topk", type=int, default=10, help="Top-K Zeilen ausgeben (Default: 10).")
    ap.add_argument("--save_csv", default="", help="Optional: Pfad, um die komplette Tabelle als CSV zu speichern.")

    # Plot-Optionen
    ap.add_argument("--plot_dag", action="store_true", help="Erzeuge dag.png (nur Struktur) und dag_strength.png (mit Kantenstärken).")
    ap.add_argument("--plot_dir", default=".", help="Ausgabeordner für DAG-Plots (Default: aktuelles Verzeichnis).")
    ap.add_argument("--save_strength_csv", default="", help="Optional: Speichere Arrow-Strengths als CSV (edge,strength).")

    args = ap.parse_args()

    df = load_and_prepare_ai4i(args.csv)

    # Graph bauen
    edges = list(DAG_EDGES)

    # Optional: Type Dummy-Knoten in Graph einhängen (wenn im DF vorhanden)
    type_cols = [c for c in df.columns if c.startswith("Type_")]
    for t in type_cols:
        edges.append((t, "tool_wear"))
        edges.append((t, args.outcome))

    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Welche Spalten werden im SCM gefittet?
    cols = sorted([n for n in G.nodes() if n in df.columns])
    if args.outcome not in cols:
        raise KeyError(f"Outcome '{args.outcome}' ist nicht im DataFrame/Graph enthalten.")

    # SCM fitten
    scm = gcm.StructuralCausalModel(G)
    gcm.auto.assign_causal_mechanisms(scm, df[cols])
    gcm.fit(scm, df[cols])

    # -------------------------------------------------------------------------
    # OPTIONAL: DAG visualisieren + "gefundene Abhängigkeiten" als Kantenstärke
    # -------------------------------------------------------------------------
    # Das ist KEIN "DAG Discovery". Wir zeichnen deinen *angenommenen* DAG.
    # Zusätzlich schätzen wir aus dem gefitteten GCM pro Kante eine Arrow-Strength
    # (direkter Einfluss Parent -> Child im Sinne des Modells).
    #
    # Ergebnisdateien:
    # - dag.png            (nur Struktur)
    # - dag_strength.png   (Struktur + Kantenstärken)
    # - optional CSV       (edge,strength)
    if args.plot_dag:
        os.makedirs(args.plot_dir, exist_ok=True)

        dag_png = os.path.join(args.plot_dir, "dag.png")
        dag_strength_png = os.path.join(args.plot_dir, "dag_strength.png")

        # 1) Struktur plotten (ohne Stärken)
        try:
            from dowhy.utils.plotting import plot as dowhy_plot
            dowhy_plot(G, filename=dag_png, display_plot=False)
            print(f"DAG (Struktur) gespeichert: {dag_png}")
        except Exception as e:
            print("Konnte DAG nicht mit dowhy.utils.plotting.plot zeichnen.")
            print("Tipp: Graphviz + pygraphviz installieren, oder nutze den Fallback.")
            print("Fehler:", e)
            # Fallback: matplotlib
            try:
                import matplotlib.pyplot as plt
                plt.figure()
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=True, node_size=1200, font_size=8, arrows=True)
                plt.tight_layout()
                plt.savefig(dag_png, dpi=200)
                plt.close()
                print(f"DAG (Fallback) gespeichert: {dag_png}")
            except Exception as e2:
                print("Fallback-Plot fehlgeschlagen:", e2)

        # 2) Kantenstärken (Arrow Strength) berechnen
        causal_strengths = {}
        for node in G.nodes():
            # arrow_strength liefert i.d.R. ein dict: {(parent, node): strength, ...}
            # Root-Nodes haben keine Eltern -> kann leer sein.
            try:
                strengths_in = gcm.arrow_strength(scm, node)
                if strengths_in:
                    causal_strengths.update(strengths_in)
            except Exception:
                pass

        # Optional als CSV speichern
        if args.save_strength_csv:
            try:
                rows = []
                for (src, dst), s in causal_strengths.items():
                    rows.append({"edge": f"{src}->{dst}", "src": src, "dst": dst, "strength": float(s)})
                pd.DataFrame(rows).sort_values("strength", ascending=False).to_csv(args.save_strength_csv, index=False)
                print(f"Arrow-Strength CSV gespeichert: {args.save_strength_csv}")
            except Exception as e:
                print("Konnte Arrow-Strength CSV nicht schreiben:", e)

        # 3) Plot mit Kantenstärken
        # (Wenn Graphviz vorhanden: sehr schön. Sonst Fallback mit Edge-Labels.)
        try:
            from dowhy.utils.plotting import plot as dowhy_plot
            dowhy_plot(G, causal_strengths=causal_strengths, filename=dag_strength_png, display_plot=False)
            print(f"DAG (mit Kantenstärken) gespeichert: {dag_strength_png}")
        except Exception as e:
            print("Konnte dag_strength.png nicht mit dowhy_plot erzeugen, nutze Fallback.")
            print("Fehler:", e)
            try:
                import matplotlib.pyplot as plt
                plt.figure()
                pos = nx.spring_layout(G, seed=42)
                nx.draw(G, pos, with_labels=True, node_size=1200, font_size=8, arrows=True)
                # Edge labels: Stärke auf 3 Dezimalstellen
                edge_labels = { (u, v): f"{float(causal_strengths.get((u, v), 0.0)):.3f}" for (u, v) in G.edges() }
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
                plt.tight_layout()
                plt.savefig(dag_strength_png, dpi=200)
                plt.close()
                print(f"DAG (mit Kantenstärken, Fallback) gespeichert: {dag_strength_png}")
            except Exception as e2:
                print("Fallback-Plot für dag_strength.png fehlgeschlagen:", e2)

    # Auto-Scan über alle Variablen außer Outcome
    vars_to_scan = [c for c in cols if c != args.outcome]
    scan = auto_scan_interventions_gcm(
        df=df,
        scm=scm,
        vars_to_scan=vars_to_scan,
        outcome_col=args.outcome,
        n=args.n_samples,
    )

    print("\n=== GCM Auto-Scan (Root-cause Kandidaten) ===")
    print(scan.head(args.topk))

    if args.save_csv:
        scan.to_csv(args.save_csv, index=False)
        print(f"\nGesamte Tabelle gespeichert: {args.save_csv}")


if __name__ == "__main__":
    main()
