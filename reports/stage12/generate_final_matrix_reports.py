"""Generate the final federated matrix JSON + CSV from the six result artifacts.

READ-ONLY with respect to all experiment artifacts: reads the six *_result.json
files (Stage 10B, Stage 11, four Stage 12 cells) and writes exactly two
documentation files (final_federated_matrix.json / .csv). No training, no
checkpoint writes, no artifact modification.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CELLS = [
    {
        "key": "nossl_a10",
        "alpha": 1.0,
        "ssl": False,
        "experiment": "stage12_fed_nossl_alpha_1_0_seed42",
        "result": "artifacts/federated/stage12/stage12_fed_nossl_alpha_1_0_seed42/stage12_fed_nossl_alpha_1_0_seed42_result.json",
        "output_dir": "artifacts/federated/stage12/stage12_fed_nossl_alpha_1_0_seed42",
        "stage": "Stage 12",
        "model_family": "Transformer + MMoE (supervised click+install, no SSL)",
        "model_type": "transformer_mmoe",
        "expected_params": 2_502_930, "expected_entries": 138,
    },
    {
        "key": "ssl_a10",
        "alpha": 1.0,
        "ssl": True,
        "experiment": "stage12_fed_ssl_alpha_1_0_seed42",
        "result": "artifacts/federated/stage12/stage12_fed_ssl_alpha_1_0_seed42/stage12_fed_ssl_alpha_1_0_seed42_result.json",
        "output_dir": "artifacts/federated/stage12/stage12_fed_ssl_alpha_1_0_seed42",
        "stage": "Stage 12",
        "model_family": "Transformer + MMoE + SSL (joint 0.6*sup + 0.4*NT-Xent)",
        "model_type": "ssl_transformer_mmoe",
        "expected_params": 2_527_698, "expected_entries": 142,
    },
    {
        "key": "nossl_a05",
        "alpha": 0.5,
        "ssl": False,
        "experiment": "stage12_fed_nossl_alpha_0_5_seed42",
        "result": "artifacts/federated/stage10/stage10b_result.json",
        "output_dir": "artifacts/federated/stage10",
        "stage": "Stage 10B",
        "model_family": "Transformer + MMoE (supervised click+install, no SSL)",
        "model_type": "transformer_mmoe",
        "expected_params": 2_502_930, "expected_entries": 138,
    },
    {
        "key": "ssl_a05",
        "alpha": 0.5,
        "ssl": True,
        "experiment": "stage12_fed_ssl_alpha_0_5_seed42",
        "result": "artifacts/federated/stage11/stage11_result.json",
        "output_dir": "artifacts/federated/stage11",
        "stage": "Stage 11",
        "model_family": "Transformer + MMoE + SSL (joint 0.6*sup + 0.4*NT-Xent)",
        "model_type": "ssl_transformer_mmoe",
        "expected_params": 2_527_698, "expected_entries": 142,
    },
    {
        "key": "nossl_a01",
        "alpha": 0.1,
        "ssl": False,
        "experiment": "stage12_fed_nossl_alpha_0_1_seed42",
        "result": "artifacts/federated_stage12_placeholder.json",  # replaced below
        "output_dir": "artifacts/federated/stage12/stage12_fed_nossl_alpha_0_1_seed42",
        "stage": "Stage 12",
        "model_family": "Transformer + MMoE (supervised click+install, no SSL)",
        "model_type": "transformer_site_mmoe",  # placeholder, replaced below
        "expected_params": 2_502_930, "expected_entries": 138,
    },
    {
        "key": "ssl_a01",
        "alpha": 0.1,
        "ssl": True,
        "experiment": "stage12_fed_ssl_alpha_0_1_seed42",
        "result": "artifacts/federated_stage12_placeholder.json",  # replaced below
        "output_dir": "artifacts/federated/stage12/stage12_fed_ssl_alpha_0_1_seed42",
        "stage": "Stage 12",
        "model_family": "Transformer + MMoE + SSL (joint 0.5*sup + 0.5*NT-Xent)",  # placeholder, replaced below
        "model_type": "ssl_transformer_mmoe",
        "expected_params": 2_527_698, "expected_entries": 142,
    },
]
# fix the two placeholder entries
CELLS[4]["result"] = ("artifacts/federated/stage12/stage12_fed_nossl_alpha_0_1_seed42/"
                      "stage12_fed_nossl_alpha_0_1_seed42_result.json")
CELLS[4]["model_type"] = "transformer_mmoe"
CELLS[5]["result"] = ("artifacts/federated/stage12/stage12_fed_ssl_alpha_0_1_seed42/"
                      "stage12_fed_ssl_alpha_0_1_seed42_result.json")
CELLS[5]["model_family"] = "Transformer + MMoE + SSL (joint 0.6*sup + 0.4*NT-Xent)"

PARTITIONS = {
    "1.0": {"npz": "artifacts/federated/partitions/stage12_partition_alpha_1_0_seed42.npz",
            "hash": "c382d47f5824f64a"},
    "0.5": {"npz": "artifacts/federated/partition_alpha_0_5_seed42.npz",
            "hash": "9603ddc9facc973d"},
    "0.1": {"npz": "artifacts/federated/partitions/stage12_partition_alpha_0_1_seed42.npz",
            "hash": "23146aebf184345b"},
}

PROTOCOL = {
    "num_clients": 10,
    "participation": "all_clients_every_round",
    "rounds": 10,
    "local_epochs": 1,
    "local_batch_size": 1024,
    "optimizer": "adamw",
    "learning_rate": 0.0003,
    "weight_decay": 0.01,
    "lambda_click": 1.0,
    "lambda_install": 1.0,
    "aggregation": "sample_weighted_fedavg",
    "selection_metric": "global_validation_install_logloss",
    "local_optimizer_state_persistence": False,
    "local_sampler_epoch_mode": "round_minus_one",
    "seed": 42,
    "partition_seed": 42,
    "initialization": "cold seed-42 (deterministic, no checkpoint load)",
    "validation_rows": 348586,
    "federated_train_rows": 3137266,
    "official_test": "never loaded, never evaluated, by any federated stage",
}

SSL_IDENTITY = {
    "model": "SSLTransformerMMoE",
    "parameter_count": 2527698,
    "state_entries": 142,
    "supervised_weight": 0.6,
    "contrastive_weight": 0.4,
    "corruption_rate": 0.15,
    "temperature": 0.2,
    "projection_dim": 64,
    "augmentation": "FeatureCorruptionAugmentation (Stage 7, two-view, CPU generators)",
    "contrastive_loss": "symmetric NT-Xent (nt_xent_loss)",
    "joint_loss": "ssl_joint_loss (Stage 7)",
    "evaluation_path": "supervised_forward (Stage 6 behavior, no augmentation)",
}

NOSSL_IDENTITY = {
    "model": "Transformer + MMoE (supervised click+install)",
    "parameter_count": 2502930,
    "state_entries": 138,
    "evaluation_path": "supervised_forward (Stage 6 behavior)",
}


def effective_use_ssl(d: dict, cell_ssl: bool) -> bool:
    """Effective SSL flag. Stage 12 cells record it in stage12.use_ssl; older
    cells expose the ssl config dict (present iff SSL was used). The recorded
    protocol.use_ssl field is a runner default in ALL result JSONs and is not
    the effective value."""
    s12 = d.get("stage12")
    if isinstance(s12, dict) and "use_ssl" in s12:
        return bool(s12["use_ssl"])
    return (d.get("ssl") is not None) or cell_ssl


def effective_alpha(d: dict, cell_alpha: float) -> float:
    """Effective partition alpha. Stage 12 cells record it in stage12.alpha;
    for Stage 10B / Stage 11 the cell constant (0.5, verified against the
    partition hash and experiment identifier) is used."""
    s12 = d.get("stage12")
    if isinstance(s12, dict) and "alpha" in s12:
        return s12["alpha"]
    return cell_alpha


def best_metrics(d: dict) -> dict:
    """Best-round metrics from the artifact's own history at best_round."""
    br = int(d["best_round"])
    row = next(r for r in d["history"] if int(r["round"]) == br)
    vm = row["validation_metrics"]
    return {
        "best_round": br,
        "install_logloss": vm["val_install_logloss"],
        "install_auc": vm["val_install_auc"],
        "click_logloss": vm["val_click_logloss"],
        "click_auc": vm["val_click_auc"],
    }


def per_round(d: dict) -> list[dict]:
    out = []
    for r in d["history"]:
        vm = r["validation_metrics"]
        out.append({
            "round": int(r["round"]),
            "install_logloss": vm["val_install_logloss"],
            "install_auc": vm["val_install_auc"],
            "click_logloss": vm["val_click_logloss"],
            "click_auc": vm["val_click_auc"],
            "validation_rows": vm.get("rows"),
            "round_wall_seconds": r.get("round_wall_seconds"),
        })
    return out


def main() -> None:
    records = []
    for cell in CELLS:
        path = ROOT / cell["result"]
        if not path.is_file():
            records.append({
                **{k: cell[k] for k in ("key", "alpha", "ssl", "experiment", "stage",
                                        "output_dir", "result")},
                "status": "missing", "notes": f"result artifact not found: {cell['result']}",
            })
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        bm = best_metrics(d)
        rec = {
            "key": cell["key"],
            "alpha": cell["alpha"],
            "ssl": cell["ssl"],
            "experiment": cell["experiment"],
            "stage": cell["stage"],
            "model_family": cell["model_family"],
            "effective_use_ssl": effective_use_ssl(d, cell["ssl"]),
            "effective_alpha": effective_alpha(d, cell["alpha"]),
            "recorded_ssl_config": d.get("ssl"),
            "model_identity": (SSL_IDENTITY if cell["ssl"] else NOSSL_IDENTITY),
            "status": d.get("status"),
            "rounds_completed": d.get("rounds_completed"),
            "protocol_rounds": d.get("protocol_rounds"),
            "complete": (d.get("status") == "complete"
                         and d.get("rounds_completed") == d.get("protocol_rounds") == 10),
            "seed": d.get("seed"),
            "partition_hash": d.get("partition_hash"),
            "partition_npz": PARTITIONS[str(cell["alpha"])]["npz"],
            "expected_partition_hash": PARTITIONS[str(cell["alpha"])]["hash"],
            "partition_hash_matches_expected": (d.get("partition_hash")
                                                == PARTITIONS[str(cell["alpha"])]["hash"]),
            "split_train_index_hash": d.get("split_train_index_hash"),
            "split_validation_index_hash": d.get("split_validation_index_hash"),
            "preprocessing_hash": d.get("preprocessing_hash"),
            "best": bm,
            "per_round": per_round(d),
            "runtime_seconds_total": d.get("total_elapsed_seconds"),
            "peak_rss_mib_overall": d.get("peak_rss_mib_overall"),
            "started_at": d.get("started_at"),
            "completed_at": d.get("completed_at"),
            "reload_verified": d.get("reload_verified"),
            "reloaded_metrics_match": d.get("reloaded_metrics_match"),
            "device": d.get("device"),
            "source_result_json": cell["result"],
            "output_dir": cell["output_dir"],
            "protocol_recorded": d.get("protocol"),
        }
        records.append(rec)

    # integrity status per cell (from the recorded artifacts themselves)
    integrity = {
        "all_six_cells_complete": all(r.get("complete") for r in records),
        "all_partition_hashes_match_expected": all(
            r.get("partition_hash_matches_expected") for r in records),
        "all_reload_verified": all(r.get("reload_verified") is True
                                   and r.get("reloaded_metrics_match") is True
                                   for r in records),
        "protocol_consistent_across_cells": all(
            r.get("protocol_recorded") and all(
                r["protocol_recorded"].get(k) == v
                for k, v in (("num_clients", 10), ("rounds", 10), ("local_epochs", 1),
                             ("local_batch_size", 1024), ("optimizer", "adamw"),
                             ("learning_rate", 0.0003), ("weight_decay", 0.01),
                             ("aggregation", "sample_weighted_fedavg"),
                             ("participation", "all_clients_every_round"),
                             ("selection_metric", "global_validation_install_logloss"),
                             ("seed", 42)))
            for r in records),
        "stage12_cells": {
            r["experiment"]: r.get("complete") for r in records if r["stage"] == "Stage 12"
        },
        "stage11_alpha_0_5_ssl_intact": next(
            r.get("complete") and r.get("reload_verified") and r.get("reloaded_metrics_match")
            for r in records if r["key"] == "ssl_a05"),
        "stage10b_alpha_0_5_nossl_intact": next(
            r.get("complete") and r.get("reload_verified") and r.get("reloaded_metrics_match")
            for r in records if r["key"] == "nossl_a05"),
        "checkpoint_identity_checks": {
            "ssl_cells_2527698_params_142_entries": True,
            "nossl_cells_2502930_params_138_entries": True,
            "note": ("verified directly against the best.pt payloads on 2026-09-24 "
                     "(state-entry count and parameter sum); result JSONs and audit "
                     "script assert the same constants"),
        },
    }

    out = {
        "generated": "2026-09-24",
        "generated_by": "reports/stage12/generate_final_matrix_reports.py (read-only extraction)",
        "protocol": PROTOCOL,
        "partitions": PARTITIONS,
        "ssl_identity": SSL_IDENTITY,
        "nossl_identity": NOSSL_IDENTITY,
        "experiments": records,
        "integrity": integrity,
        "notes_unavailable_fields": [
            "Recorded-protocol metadata quirk (documented, not corrected): every result "
            "JSON's protocol.use_ssl / protocol.alpha fields hold the runner's default "
            "values (False / 0.5) rather than the effective cell settings - including "
            "Stage 11. The effective identity is recorded elsewhere in each artifact: "
            "Stage 12 cells carry stage12.use_ssl / stage12.alpha plus the ssl config "
            "dict; Stage 11 carries the ssl config dict. These effective values were "
            "cross-verified against each cell's partition hash, experiment identifier, "
            "and (earlier in this audit) the checkpoint payloads. The training-relevant "
            "protocol fields (clients, rounds, epochs, batch, optimizer, lr, weight "
            "decay, aggregation, participation, selection metric, seed) are recorded "
            "correctly and are identical across all six cells.",
            "Stage 10B runtime metadata anomaly: total_elapsed_seconds (5,226.76 s) is "
            "inconsistent with the sum of its per-round wall times (~25,494 s); it matches "
            "the sum of rounds 1-2 only (5,225.6 s), suggesting the elapsed-time accumulator "
            "stopped updating after round 2. Its started_at/completed_at span (34 s) is also "
            "inconsistent with the training log (log header start 2026-09-19T10:32:28+00:00). "
            "The per-round metrics are internally consistent and reload-verified; the anomaly "
            "affects runtime metadata only and was not corrected (read-only audit).",
            "The four Stage 12 runtime values equal the sums of their per-round wall times "
            "within rounding (internally consistent).",
            "alpha=0.5 partition artifacts live at artifacts/federated/partition_alpha_0_5_seed42.npz "
            "(Stage 9 convention); alpha=1.0 / 0.1 partitions live at artifacts/federated/partitions/.",
            "All six result JSONs record reload_verified=true and reloaded_metrics_match=true.",
            "No official-test metrics exist in any artifact by design; every metric reported "
            "here is on the frozen 348,586-row validation split.",
        ],
    }
    outdir = ROOT / "reports" / "stage12"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "final_federated_matrix.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    # ---- CSV: one row per cell -------------------------------------------
    csv_path = outdir / "final_federated_matrix.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["alpha", "ssl", "experiment", "best_round", "install_logloss",
                    "install_auc", "click_logloss", "click_auc", "runtime_seconds",
                    "status"])
        for r in sorted(records, key=lambda x: (-x["alpha"], x["ssl"])):
            b = r.get("best") or {}
            w.writerow([
                r["alpha"], r["ssl"], r["experiment"], b.get("best_round"),
                b.get("install_logloss"), b.get("install_auc"),
                b.get("click_logloss"), b.get("click_auc"),
                r.get("runtime_seconds_total"), r.get("status"),
            ])

    print("wrote:", outdir / "final_federated_matrix.json")
    print("wrote:", csv_path)
    print("all complete:", integrity["all_six_cells_complete"])
    print("all partition hashes match:", integrity["all_partition_hashes_match_expected"])


if __name__ == "__main__":
    main()
