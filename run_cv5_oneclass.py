import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import OneClassSVM
from transformers import AutoModel

from cv5_metrics import (
    compute_eer,
    metrics_at_threshold,
    summarize_metric_rows,
    threshold_for_far_zero,
)
from cv5_reve_cache import embed_with_cache


SCENARIOS = {
    "PC->PC": {"train": "pc", "val": "pc", "test": "pc"},
    "VR->VR": {"train": "vr", "val": "vr", "test": "vr"},
    "PC->VR": {"train": "pc", "val": "pc", "test": "vr"},
    "VR->PC": {"train": "vr", "val": "vr", "test": "pc"},
    "Mixed->PC": {"train": "mixed", "val": "mixed", "test": "pc"},
    "Mixed->VR": {"train": "mixed", "val": "mixed", "test": "vr"},
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_reve(model_dir, device):
    model_dir = Path(model_dir)
    reve = AutoModel.from_pretrained(
        str(model_dir / "reve-base"),
        trust_remote_code=True,
        torch_dtype="auto",
        local_files_only=True,
    )
    pos_bank = AutoModel.from_pretrained(
        str(model_dir / "reve-positions"),
        trust_remote_code=True,
        torch_dtype="auto",
        local_files_only=True,
    )
    reve.eval().to(device)
    pos_bank.eval().to(device)
    for p in reve.parameters():
        p.requires_grad = False
    for p in pos_bank.parameters():
        p.requires_grad = False
    return reve, pos_bank


def subject_data(oneclass, env, fold, subject):
    env_data = oneclass[env]
    fold_data = env_data[fold] if fold in env_data else env_data[str(fold)]
    return fold_data[subject] if subject in fold_data else fold_data[str(subject)]


def concat_sets(sets, key):
    arrays = [np.asarray(s[key]) for s in sets]
    return np.concatenate(arrays, axis=0)


def oneclass_splits(oneclass, scenario, fold, subject):
    spec = SCENARIOS[scenario]
    test_set = subject_data(oneclass, spec["test"], fold, subject)

    if spec["train"] == "mixed":
        src_sets = [
            subject_data(oneclass, "pc", fold, subject),
            subject_data(oneclass, "vr", fold, subject),
        ]
        Xtr = concat_sets(src_sets, "Xtr")
        Xva = concat_sets(src_sets, "Xva")
        yva = concat_sets(src_sets, "yva")
    else:
        src_set = subject_data(oneclass, spec["train"], fold, subject)
        Xtr = np.asarray(src_set["Xtr"])
        Xva = np.asarray(src_set["Xva"])
        yva = np.asarray(src_set["yva"])

    return Xtr, Xva, yva.astype(int), np.asarray(test_set["Xte"]), np.asarray(test_set["yte"]).astype(int)


def fit_model(model_name, Ztr, args):
    if model_name == "OCSVM":
        model = OneClassSVM(kernel=args.kernel, nu=args.nu, gamma=args.gamma)
        model.fit(Ztr)
        return model
    if model_name == "OCkNN":
        model = NearestNeighbors(n_neighbors=args.n_neighbors, metric=args.metric)
        model.fit(Ztr)
        return model
    raise ValueError(f"Unknown model: {model_name}")


def score_model(model_name, model, Z):
    if model_name == "OCSVM":
        return model.decision_function(Z).ravel()
    if model_name == "OCkNN":
        distances, _ = model.kneighbors(Z)
        return -distances.mean(axis=1)
    raise ValueError(f"Unknown model: {model_name}")


def run_subject(data, oneclass, reve, pos_bank, model_name, scenario, fold, subject, args):
    Xtr, Xva, yva, Xte, yte = oneclass_splits(oneclass, scenario, fold, subject)

    if set(np.unique(yva)) != {0, 1}:
        raise ValueError(f"Validation labels for subject {subject}, fold {fold}, {scenario} must contain 0 and 1.")
    if set(np.unique(yte)) != {0, 1}:
        raise ValueError(f"Test labels for subject {subject}, fold {fold}, {scenario} must contain 0 and 1.")

    common = f"{model_name}_{scenario}_fold{fold}_subj{subject}".replace("->", "_to_")
    Ztr = embed_with_cache(
        reve,
        pos_bank,
        Xtr,
        data["ch_names"],
        args.device,
        args.bs,
        args.cache_dir,
        args.model_dir,
        f"{common}_train",
        not args.no_cache,
    )
    Zva = embed_with_cache(
        reve,
        pos_bank,
        Xva,
        data["ch_names"],
        args.device,
        args.bs,
        args.cache_dir,
        args.model_dir,
        f"{common}_val",
        not args.no_cache,
    )
    Zte = embed_with_cache(
        reve,
        pos_bank,
        Xte,
        data["ch_names"],
        args.device,
        args.bs,
        args.cache_dir,
        args.model_dir,
        f"{common}_test",
        not args.no_cache,
    )

    clf = fit_model(model_name, Ztr, args)
    val_scores = score_model(model_name, clf, Zva)
    test_scores = score_model(model_name, clf, Zte)

    threshold = threshold_for_far_zero(val_scores, yva)
    metrics = metrics_at_threshold(test_scores, yte, threshold)
    metrics["EER"] = compute_eer(test_scores, yte)
    metrics["threshold"] = threshold
    return metrics


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser(description="Run corrected 5-fold one-class REVE verification.")
    parser.add_argument("--npz", default="./out/cv5/cv5_verification_dataset_lphp10_50.npz")
    parser.add_argument("--model_dir", default="./models")
    parser.add_argument("--out_dir", default="./out/cv5/results")
    parser.add_argument("--model", choices=["OCSVM", "OCkNN", "all"], default="all")
    parser.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bs", type=int, default=128)
    parser.add_argument("--cache_dir", default="./out/cv5/cache/reve_oneclass")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--nu", type=float, default=0.1)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--kernel", default="rbf")
    parser.add_argument("--n_neighbors", type=int, default=5)
    parser.add_argument("--metric", default="euclidean")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz = np.load(args.npz, allow_pickle=True)
    meta = json.loads(str(npz["meta_json"]))
    data = {
        "ch_names": meta["ch_names"],
    }
    oneclass = npz["oneclass"].item()
    subjects = sorted(int(s) for s in meta["subject_map"].values())
    folds = list(range(int(meta["n_folds"])))
    models = ["OCSVM", "OCkNN"] if args.model == "all" else [args.model]
    scenarios = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    reve, pos_bank = load_reve(args.model_dir, args.device)

    subject_rows = []
    for model_name in models:
        for scenario in scenarios:
            for fold in folds:
                for subject in subjects:
                    metrics = run_subject(data, oneclass, reve, pos_bank, model_name, scenario, fold, subject, args)
                    subject_rows.append(
                        {
                            "model": model_name,
                            "scenario": scenario,
                            "fold": fold,
                            "subject": subject,
                            "threshold": metrics["threshold"],
                            "accuracy": metrics["accuracy"],
                            "balanced_accuracy": metrics["balanced_accuracy"],
                            "FAR": metrics["FAR"],
                            "FRR": metrics["FRR"],
                            "EER": metrics["EER"],
                            "seed": args.seed,
                        }
                    )
                    write_csv(out_dir / "oneclass_cv5_subject_results.csv", subject_rows)
                fold_done = len([r for r in subject_rows if r["model"] == model_name and r["scenario"] == scenario and r["fold"] == fold])
                print(f"{model_name} {scenario} fold {fold}: completed {fold_done} subjects")

    subject_df = pd.DataFrame(subject_rows)
    metric_cols = ["accuracy", "balanced_accuracy", "FAR", "FRR", "EER", "threshold"]
    fold_df = (
        subject_df.groupby(["model", "scenario", "fold", "seed"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["model", "scenario", "fold"])
    )
    fold_df.to_csv(out_dir / "oneclass_cv5_per_fold.csv", index=False)

    summary_rows = summarize_metric_rows(
        fold_df.to_dict("records"),
        group_cols=["model", "scenario", "seed"],
        metric_cols=["accuracy", "balanced_accuracy", "FAR", "FRR", "EER"],
    )
    write_csv(out_dir / "oneclass_cv5_summary.csv", summary_rows)
    print(f"Saved one-class results to {out_dir}")


if __name__ == "__main__":
    main()
