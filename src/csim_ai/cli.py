"""`csim-ai {report,group,info,setup}` -- CLI entry point (see
[project.scripts] in pyproject.toml). Mirrors csim's CLI shape (a
single command, an action positional, `--path` pointing at a directory
of source files compared exhaustively) rather than a two-file-only
interface, for consistency with the predecessor tool.
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

from . import Scorer


def _primary_score(result: dict) -> float | None:
    return result["fusion"] if result["fusion"] is not None else result["biencoder_cosine"]


def _iter_py_files(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.suffix == ".py")


def _score_all_pairs(files: list[Path], scorer: Scorer) -> dict[tuple[Path, Path], dict]:
    results = {}
    for a, b in itertools.combinations(files, 2):
        code_a = a.read_text(encoding="utf-8", errors="ignore")
        code_b = b.read_text(encoding="utf-8", errors="ignore")
        results[(a, b)] = scorer.score(code_a, code_b)
    return results


def cmd_report(args: argparse.Namespace) -> None:
    scorer = Scorer(args.model_path, fusion_model_path=args.fusion_model, use_fusion=args.use_fusion)
    files = _iter_py_files(args.path)
    results = _score_all_pairs(files, scorer)
    for (a, b), r in results.items():
        extra = f" (biencoder_cosine={r['biencoder_cosine']:.4f}"
        if r["csim_ted"] is not None:
            extra += f", csim_ted={r['csim_ted']:.4f}"
        if r["fusion"] is not None:
            extra += f", fusion={r['fusion']:.4f}"
        extra += ")"
        print(f"{b} is similar to {a} with similarity index: {_primary_score(r):.4f}{extra}")


def cmd_group(args: argparse.Namespace) -> None:
    scorer = Scorer(args.model_path, fusion_model_path=args.fusion_model, use_fusion=args.use_fusion)
    files = _iter_py_files(args.path)
    results = _score_all_pairs(files, scorer)

    parent = {f: f for f in files}

    def find(f: Path) -> Path:
        while parent[f] != f:
            parent[f] = parent[parent[f]]
            f = parent[f]
        return f

    def union(a: Path, b: Path) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for (a, b), r in results.items():
        if _primary_score(r) >= args.threshold:
            union(a, b)

    groups: dict[Path, list[Path]] = {}
    for f in files:
        groups.setdefault(find(f), []).append(f)

    print(f"Threshold: {args.threshold}")
    print(f"Total files processed: {len(files)}")

    group_num = 0
    unique_files = []
    for members in groups.values():
        if len(members) == 1:
            unique_files.append(members[0])
            continue
        group_num += 1
        pair_scores = [
            _primary_score(results[(a, b)] if (a, b) in results else results[(b, a)])
            for a, b in itertools.combinations(members, 2)
        ]
        avg = sum(pair_scores) / len(pair_scores)
        print(f"Group {group_num} (Average Similarity: {avg:.2f}):")
        for f in members:
            print(f)

    if unique_files:
        print("Unique Files (similarity below threshold):")
        for f in unique_files:
            print(f)


def cmd_info(args: argparse.Namespace) -> None:
    print("csim-ai backends")
    print()

    def check(label: str, module: str) -> None:
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "?")
            print(f"  {label:<24} available (v{version})")
        except ImportError:
            print(f"  {label:<24} not installed")

    check("onnxruntime (base)", "onnxruntime")
    check("tokenizers (base)", "tokenizers")
    check("huggingface_hub (base)", "huggingface_hub")
    check("csim (ast extra)", "csim")
    check("scikit-learn (scorer extra)", "sklearn")
    check("torch (export extra)", "torch")

    if args.model_path:
        model_path = Path(args.model_path)
        onnx_ok = (model_path / "model.onnx").exists()
        tok_ok = (model_path / "tokenizer.json").exists()
        print()
        print(f"  model-path: {model_path}")
        print(f"    model.onnx: {'found' if onnx_ok else 'MISSING'}")
        print(f"    tokenizer.json: {'found' if tok_ok else 'MISSING'}")


def cmd_setup(args: argparse.Namespace) -> None:
    if args.export_from:
        try:
            from ._export import export
        except ImportError:
            print("The `export` extra is required: pip install csim-ai[export]", file=sys.stderr)
            raise SystemExit(1)

        export(args.export_from, args.out, opset=args.opset, verify_after=not args.no_verify)
        return

    # Default action: download the pre-trained weights from HF Hub, so
    # `pip install csim-ai && csim-ai setup` is all that's needed before
    # `report`/`group` work with the full hybrid score, no flags needed
    # after that (Scorer() auto-detects the cache -- see __init__.py).
    from ._hub import download_fusion_model, download_model

    model_path = download_model()
    print(f"bi-encoder cached at: {model_path}")
    fusion_path = download_fusion_model()
    print(f"fusion model cached at: {fusion_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Python files for plagiarism similarity.")
    sub = parser.add_subparsers(dest="action", required=True)

    def add_path_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--path", "-p", type=Path, required=True, help="Directory of .py files to compare exhaustively.")
        p.add_argument("--model-path", default=None, help="Directory with model.onnx/tokenizer.json. Default: download from Hugging Face Hub.")
        p.add_argument("--fusion-model", default=None, help="Path to a fusion_model.joblib (requires the [ast,scorer] extras).")
        p.add_argument("--use-fusion", action="store_true", help="Use the fusion score, downloading the fusion model from Hugging Face Hub if --fusion-model isn't given.")

    report_p = sub.add_parser("report", help="Pairwise similarity report over a directory.")
    add_path_args(report_p)
    report_p.set_defaults(func=cmd_report)

    group_p = sub.add_parser("group", help="Group files by similarity threshold.")
    add_path_args(group_p)
    group_p.add_argument("--threshold", "-t", type=float, required=True, help="Similarity threshold (0.0-1.0) for grouping.")
    group_p.set_defaults(func=cmd_group)

    info_p = sub.add_parser("info", help="Show which optional backends (onnxruntime, csim, scikit-learn, torch) are available.")
    info_p.add_argument("--model-path", default=None, help="Optionally check a model directory for model.onnx/tokenizer.json.")
    info_p.set_defaults(func=cmd_info)

    setup_p = sub.add_parser("setup", help="Download the pre-trained weights from Hugging Face Hub (default), or export a local checkpoint to ONNX instead.")
    setup_p.add_argument("--export-from", default=None, help="Local torch checkpoint directory to export to ONNX instead of downloading (requires the [export] extra). No network access -- see docs/DEVELOPMENT.md, Fase 5.")
    setup_p.add_argument("--out", default="onnx_model", help="Output directory for --export-from (default: ./onnx_model).")
    setup_p.add_argument("--opset", type=int, default=17)
    setup_p.add_argument("--no-verify", action="store_true", help="Skip the PyTorch-vs-ONNX parity check after --export-from.")
    setup_p.set_defaults(func=cmd_setup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
