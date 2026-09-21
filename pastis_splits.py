"""Canonical PASTIS split handling.

Single source of truth: `pastis_official_splits_v1.json`, which carries all
2433 patch records (id, fold, acquisition dates) plus the five official
train/val/test rotations. Everything downstream keys off this file so that
every run — pretraining, fine-tuning, evaluation — sees the same partition.

The validator here is stricter than a duplicate/overlap check: it also verifies
that every ID in a split actually exists in the dataset, which is the failure
mode that silently shrinks a test set.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "pastis_official_splits_v1"


# --------------------------------------------------------------------------

@dataclass
class Manifest:
    patches: dict[str, dict]      # id -> {"fold": int, "dates": [YYYYMMDD, ...]}
    splits: dict[str, dict]       # "1".."5" -> {"train": [...], "val": [...], "test": [...]}
    labels: dict
    expected_hw: tuple[int, int]
    sha256: str

    @property
    def ids(self) -> list[str]:
        return list(self.patches.keys())

    def fold_of(self, pid: str) -> int:
        return self.patches[pid]["fold"]

    def fold_sizes(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for p in self.patches.values():
            out[p["fold"]] = out.get(p["fold"], 0) + 1
        return dict(sorted(out.items()))

    def split(self, fold: int | str) -> dict[str, list[str]]:
        return self.splits[str(fold)]


def load_manifest(path: str | Path) -> Manifest:
    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw)
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"expected schema {SCHEMA!r}, got {doc.get('schema')!r}")
    patches = {
        str(p["id"]): {"fold": int(p["fold"]), "dates": [str(d) for d in p["dates"]]}
        for p in doc["patches"]
    }
    return Manifest(
        patches=patches,
        splits={str(k): v for k, v in doc["splits"].items()},
        labels=doc.get("labels", {"background": 0, "crop_ids": list(range(1, 19)), "void": 19}),
        expected_hw=tuple(doc.get("expected_hw", [128, 128])),
        sha256=hashlib.sha256(raw).hexdigest(),
    )


# --------------------------------------------------------------------------
# normalisation of ID spellings
# --------------------------------------------------------------------------

def normalize_id(x) -> str:
    """'S2_10000' / 10000 / '10000' -> '10000'."""
    s = str(x)
    for pre in ("S2_", "TARGET_", "S1A_", "S1D_"):
        if s.startswith(pre):
            s = s[len(pre):]
    return s.removesuffix(".npy")


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def validate_split(split: dict, manifest: Manifest | None = None,
                   name: str = "split") -> dict:
    """Return a report dict. Checks duplicates, cross-set overlap, membership,
    coverage, and — when a manifest is given — whether the split respects the
    official geographic folds."""
    sets = {}
    report = {"name": name, "errors": [], "warnings": [], "counts": {}}

    for key in ("train", "val", "test"):
        raw = [normalize_id(i) for i in split.get(key, [])]
        uniq = set(raw)
        sets[key] = uniq
        report["counts"][key] = len(raw)
        if len(raw) != len(uniq):
            report["errors"].append(
                f"{len(raw) - len(uniq)} duplicate id(s) inside '{key}'"
            )

    report["total"] = sum(report["counts"].values())

    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        inter = sets[a] & sets[b]
        if inter:
            report["errors"].append(
                f"{len(inter)} id(s) shared between '{a}' and '{b}': "
                f"{sorted(inter)[:5]}"
            )

    if manifest is not None:
        known = set(manifest.ids)
        for key in ("train", "val", "test"):
            unknown = sets[key] - known
            if unknown:
                report["errors"].append(
                    f"{len(unknown)} id(s) in '{key}' do not exist in PASTIS: "
                    f"{sorted(unknown)[:5]}"
                )
        union = sets["train"] | sets["val"] | sets["test"]
        missing = known - union
        if missing:
            report["warnings"].append(
                f"{len(missing)} PASTIS patch(es) appear in no split"
            )
        report["covers_dataset"] = not missing and not (union - known)

        # fold purity: a split that mixes official folds leaks spatial context
        mix = {}
        for key in ("train", "val", "test"):
            folds = {manifest.fold_of(i) for i in sets[key] if i in known}
            mix[key] = sorted(folds)
        report["fold_mix"] = mix
        if any(len(v) > 1 for v in (mix["val"], mix["test"])):
            report["warnings"].append(
                "val/test draw from several official folds — patches adjacent "
                "in the field can straddle the boundary, which inflates scores "
                "and makes results incomparable to the PASTIS leaderboard"
            )

    report["ok"] = not report["errors"]
    report["sets"] = sets
    return report


def print_report(report: dict) -> None:
    """Integrity report, in the format used in the project's existing checks."""
    c, total = report["counts"], max(report["total"], 1)
    print("=" * 60)
    print(f"SPLIT: {report['name']}")
    print(f"TỔNG SỐ PATCH: {report['total']}")
    for key, label in (("train", "Train"), ("val", "Val  "), ("test", "Test ")):
        print(f"- {label} : {c[key]:>6} files ({c[key] / total * 100:.2f}%)")
    if "fold_mix" in report:
        print(f"- Fold mix: {report['fold_mix']}")
    print("-" * 60)
    for e in report["errors"]:
        print(f"[LỖI] {e}")
    for w in report["warnings"]:
        print(f"[CẢNH BÁO] {w}")
    if report["ok"] and not report["warnings"]:
        print("[XÁC NHẬN] Dữ liệu hoàn toàn sạch: không trùng lặp, không rò rỉ, "
              "mọi ID đều tồn tại trong PASTIS.")
    elif report["ok"]:
        print("[XÁC NHẬN] Không có lỗi chặn — xem cảnh báo ở trên.")
    print("=" * 60)


# --------------------------------------------------------------------------
# selecting folds to run
# --------------------------------------------------------------------------

def folds_to_run(manifest: Manifest, folds=(1, 2)) -> dict[int, dict[str, list[str]]]:
    """Pick a subset of the five official rotations, e.g. the first two."""
    out = {}
    for f in folds:
        sp = manifest.split(f)
        out[int(f)] = {k: [normalize_id(i) for i in sp[k]] for k in ("train", "val", "test")}
    return out


def required_ids(runs: dict[int, dict[str, list[str]]]) -> set[str]:
    """Every patch touched by the chosen folds — what you actually need on disk."""
    out: set[str] = set()
    for sp in runs.values():
        for key in ("train", "val", "test"):
            out.update(sp[key])
    return out
