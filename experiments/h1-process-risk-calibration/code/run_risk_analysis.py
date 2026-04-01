#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.revise_trace_analysis import build_trace_features, load_revise_traces


def load_synthetic(path: str | Path) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "sample_id": obj["sample_id"],
                    "text": f"Question: {obj['question']}\nChain: {obj['chain']}",
                    "label": int(bool(obj.get("has_error", False))),
                    "source": obj.get("source", ""),
                    "error_types": ",".join(obj.get("error_types") or []),
                }
            )
    return pd.DataFrame(rows)


def metric_bundle(labels: pd.Series, scores: pd.Series) -> dict[str, float]:
    labels = labels.astype(int)
    out = {
        "positive_rate": float(labels.mean()),
        "mean_score": float(scores.mean()),
    }
    if labels.nunique() < 2:
        out["auroc"] = float("nan")
        out["auprc"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["auprc"] = float(average_precision_score(labels, scores))
    return out


def add_topk_slice(df: pd.DataFrame, score_col: str, *, frac: float = 0.1) -> dict[str, float]:
    top_k = max(1, int(len(df) * frac))
    ranked = df.sort_values(score_col, ascending=False).head(top_k)
    return {
        "top_fraction": frac,
        "top_k": top_k,
        "incorrect_rate_topk": float(ranked["incorrect"].mean()),
        "incorrect_rate_all": float(df["incorrect"].mean()),
        "lift_over_random": float(ranked["incorrect"].mean() / max(df["incorrect"].mean(), 1e-8)),
    }


def fit_numeric_logistic(train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols])
    x_test = scaler.transform(test_df[feature_cols])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_train, train_df["incorrect"])
    scores = pd.Series(clf.predict_proba(x_test)[:, 1], index=test_df.index)
    scored_df = test_df.assign(outcome_supervised_risk=scores)
    return {
        **metric_bundle(scored_df["incorrect"], scored_df["outcome_supervised_risk"]),
        **add_topk_slice(scored_df, "outcome_supervised_risk"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-log", required=True)
    parser.add_argument("--sft-log", required=True)
    parser.add_argument("--synthetic-train", required=True)
    parser.add_argument("--synthetic-test", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    synthetic_train = load_synthetic(args.synthetic_train)
    synthetic_test = load_synthetic(args.synthetic_test)
    text_clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    text_clf.fit(synthetic_train["text"], synthetic_train["label"])

    synthetic_test_scores = text_clf.predict_proba(synthetic_test["text"])[:, 1]
    synthetic_eval = metric_bundle(synthetic_test["label"], pd.Series(synthetic_test_scores))

    datasets = {
        "base": load_revise_traces(args.base_log),
        "sft": load_revise_traces(args.sft_log),
    }

    summary: dict[str, object] = {
        "synthetic_eval": synthetic_eval,
        "datasets": {},
    }
    feature_cols = [
        "num_turns",
        "answer_round",
        "forced_answer_used",
        "invalid_call_frac",
        "retry_frac",
        "stale_summary_frac",
        "duplicate_request_frac",
        "mean_summary_similarity",
        "final_summary_len",
        "final_uncertainty_len",
        "final_reason_len",
        "final_observation_len",
        "final_uncertainty_hits",
        "answer_with_uncertainty",
    ]
    frames: dict[str, pd.DataFrame] = {}

    for name, traces in datasets.items():
        frame = pd.DataFrame(build_trace_features(trace) for trace in traces)
        frame["synthetic_risk"] = text_clf.predict_proba(frame["trace_text"])[:, 1]
        frame["combined_risk"] = 0.5 * frame["heuristic_risk"] + 0.5 * frame["synthetic_risk"]
        frames[name] = frame

        frame.to_csv(output_dir / f"{name}_trace_features.csv", index=False)

        summary["datasets"][name] = {
            "n_samples": int(len(frame)),
            "correct_rate": float(frame["correct"].mean()),
            "heuristic": {
                **metric_bundle(frame["incorrect"], frame["heuristic_risk"]),
                **add_topk_slice(frame, "heuristic_risk"),
            },
            "synthetic_transfer": {
                **metric_bundle(frame["incorrect"], frame["synthetic_risk"]),
                **add_topk_slice(frame, "synthetic_risk"),
            },
            "combined": {
                **metric_bundle(frame["incorrect"], frame["combined_risk"]),
                **add_topk_slice(frame, "combined_risk"),
            },
            "feature_means_by_correctness": frame.groupby("correct")[
                [
                    "invalid_call_frac",
                    "retry_frac",
                    "stale_summary_frac",
                    "duplicate_request_frac",
                    "final_uncertainty_hits",
                    "heuristic_risk",
                    "synthetic_risk",
                    "combined_risk",
                ]
            ]
            .mean()
            .round(4)
            .to_dict(),
        }

    summary["outcome_supervised_transfer"] = {
        "train_base_test_sft": fit_numeric_logistic(frames["base"], frames["sft"], feature_cols),
        "train_sft_test_base": fit_numeric_logistic(frames["sft"], frames["base"], feature_cols),
    }

    with open(output_dir / "risk_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
