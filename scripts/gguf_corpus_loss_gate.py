#!/usr/bin/env python3
"""Evaluate teacher-forced GGUF corpus loss against a reference runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gguf_generation_quality_gate as logits_io


CAMPAIGN_SCHEMA = "litenn.corpus_loss_campaign.v1"
MANIFEST_SCHEMA = "litenn.teacher_forced_logits.v1"
REPORT_SCHEMA = "litenn.corpus_loss_report.v1"
DEFAULT_THRESHOLDS: dict[str, int | float] = {
    "minimumCaseCount": 3,
    "minimumTokenCount": 128,
    "maximumCrossEntropyRegressionNats": 0.02,
    "maximumRelativePerplexityRegression": 0.02,
    "maximumWorstSampleCrossEntropyRegressionNats": 0.10,
}


class CorpusLossError(RuntimeError):
    pass


def load_json(path: Path, schema: str) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CorpusLossError(f"failed to read JSON {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != schema:
        actual = document.get("schema") if isinstance(document, dict) else type(document).__name__
        raise CorpusLossError(f"unsupported schema in {path}: {actual!r}")
    return document


def resolve_path(raw: object, base: Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else base / path


def token_ids(document: dict[str, object], key: str, source: Path) -> list[int]:
    raw = document.get(key)
    if not isinstance(raw, list) or not raw or any(not isinstance(token, int) or token < 0 for token in raw):
        raise CorpusLossError(f"{source} has invalid or empty {key}")
    return raw


def resolve_logits(
    document: dict[str, object], source: Path, prompt_count: int, targets: list[int]
) -> dict[int, Path]:
    artifacts = document.get("logitsArtifacts")
    if not isinstance(artifacts, list):
        raise CorpusLossError(f"{source} is missing logitsArtifacts")
    result: dict[int, Path] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise CorpusLossError(f"{source} contains an invalid logits artifact")
        step = artifact.get("decisionStep")
        position = artifact.get("position")
        target = artifact.get("targetTokenId")
        if not isinstance(step, int) or step < 0 or step in result:
            raise CorpusLossError(f"{source} contains invalid or duplicate decision step {step!r}")
        if step >= len(targets):
            raise CorpusLossError(f"{source} contains out-of-range decision step {step}")
        if position != prompt_count + step:
            raise CorpusLossError(f"{source} decision step {step} has a shifted position")
        if target != targets[step]:
            raise CorpusLossError(f"{source} decision step {step} has a shifted target token")
        path = resolve_path(artifact.get("path"), source.parent)
        if not path.is_file():
            raise CorpusLossError(f"logits artifact does not exist: {path}")
        result[step] = path
    expected = set(range(len(targets)))
    if set(result) != expected:
        missing = sorted(expected - set(result))
        extra = sorted(set(result) - expected)
        raise CorpusLossError(f"{source} logits coverage mismatch: missing={missing}, extra={extra}")
    return result


def load_manifest(path: Path) -> tuple[dict[str, object], list[int], list[int], dict[int, Path]]:
    document = load_json(path, MANIFEST_SCHEMA)
    if document.get("captureBoundary") != "pre-target":
        raise CorpusLossError(f"{path} must declare captureBoundary=pre-target")
    if document.get("fallbackUsed") is not False:
        raise CorpusLossError(f"{path} must explicitly declare fallbackUsed=false")
    prompt = token_ids(document, "promptTokenIds", path)
    targets = token_ids(document, "targetTokenIds", path)
    artifacts = resolve_logits(document, path, len(prompt), targets)
    return document, prompt, targets, artifacts


def target_nll(logits: list[float], target: int) -> float:
    if target < 0 or target >= len(logits):
        raise CorpusLossError(f"target token {target} is outside vocabulary size {len(logits)}")
    maximum = max(logits)
    normalizer = maximum + math.log(math.fsum(math.exp(value - maximum) for value in logits))
    result = normalizer - logits[target]
    if not math.isfinite(result) or result < 0.0:
        raise CorpusLossError(f"target token {target} produced invalid NLL {result}")
    return result


def optional_exp(value: float) -> float | None:
    try:
        result = math.exp(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def evaluate_case(name: str, reference_path: Path, candidate_path: Path) -> dict[str, object]:
    report: dict[str, object] = {
        "name": name,
        "referenceManifest": str(reference_path),
        "candidateManifest": str(candidate_path),
        "passedIntegrity": False,
        "errors": [],
    }
    try:
        reference, reference_prompt, reference_targets, reference_artifacts = load_manifest(reference_path)
        candidate, candidate_prompt, candidate_targets, candidate_artifacts = load_manifest(candidate_path)
        if reference_prompt != candidate_prompt:
            raise CorpusLossError("reference and candidate prompt token ids differ")
        if reference_targets != candidate_targets:
            raise CorpusLossError("reference and candidate target token ids differ")

        steps: list[dict[str, object]] = []
        for step, target in enumerate(reference_targets):
            try:
                reference_logits = logits_io.parse_logits(reference_artifacts[step])
                candidate_logits = logits_io.parse_logits(candidate_artifacts[step])
            except logits_io.QualityError as error:
                raise CorpusLossError(str(error)) from error
            if len(reference_logits) != len(candidate_logits):
                raise CorpusLossError(
                    f"decision step {step} vocabulary mismatch: "
                    f"reference={len(reference_logits)}, candidate={len(candidate_logits)}"
                )
            reference_nll = target_nll(reference_logits, target)
            candidate_nll = target_nll(candidate_logits, target)
            steps.append(
                {
                    "decisionStep": step,
                    "position": len(reference_prompt) + step,
                    "targetTokenId": target,
                    "referenceNegativeLogLikelihoodNats": reference_nll,
                    "candidateNegativeLogLikelihoodNats": candidate_nll,
                    "negativeLogLikelihoodDeltaNats": candidate_nll - reference_nll,
                }
            )

        reference_cross_entropy = statistics.fmean(
            float(step["referenceNegativeLogLikelihoodNats"]) for step in steps
        )
        candidate_cross_entropy = statistics.fmean(
            float(step["candidateNegativeLogLikelihoodNats"]) for step in steps
        )
        delta = candidate_cross_entropy - reference_cross_entropy
        worst = max(steps, key=lambda step: float(step["negativeLogLikelihoodDeltaNats"]))
        report.update(
            {
                "referenceProducer": reference.get("producer"),
                "candidateProducer": candidate.get("producer"),
                "promptTokenCount": len(reference_prompt),
                "targetTokenCount": len(reference_targets),
                "referenceCrossEntropyNats": reference_cross_entropy,
                "candidateCrossEntropyNats": candidate_cross_entropy,
                "crossEntropyDeltaNats": delta,
                "referencePerplexity": optional_exp(reference_cross_entropy),
                "candidatePerplexity": optional_exp(candidate_cross_entropy),
                "relativePerplexityDelta": optional_exp(delta) - 1.0 if optional_exp(delta) is not None else None,
                "maximumTokenNegativeLogLikelihoodRegressionNats": float(
                    worst["negativeLogLikelihoodDeltaNats"]
                ),
                "worstToken": worst,
                "fallbackUsed": False,
                "finiteLogits": True,
                "passedIntegrity": True,
                "steps": steps,
            }
        )
    except CorpusLossError as error:
        report["errors"] = [str(error)]
        report["fallbackUsed"] = "fallbackUsed=false" in str(error)
        report["finiteLogits"] = False if "non-finite logit" in str(error) else None
    return report


def merged_thresholds(campaign: dict[str, object]) -> dict[str, int | float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    raw = campaign.get("thresholds", {})
    if not isinstance(raw, dict):
        raise CorpusLossError("campaign thresholds must be an object")
    for key in thresholds:
        if key not in raw:
            continue
        value = raw[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise CorpusLossError(f"campaign threshold {key} must be numeric")
        thresholds[key] = value
    for key in ("minimumCaseCount", "minimumTokenCount"):
        if not float(thresholds[key]).is_integer() or int(thresholds[key]) <= 0:
            raise CorpusLossError(f"{key} must be a positive integer")
    for key in (
        "maximumCrossEntropyRegressionNats",
        "maximumRelativePerplexityRegression",
        "maximumWorstSampleCrossEntropyRegressionNats",
    ):
        if float(thresholds[key]) < 0.0:
            raise CorpusLossError(f"{key} must be non-negative")
    return thresholds


def evidence_digest(cases: list[dict[str, object]]) -> str:
    evidence = [
        {
            "name": case["name"],
            "promptTokenCount": case.get("promptTokenCount"),
            "targetTokenCount": case.get("targetTokenCount"),
            "steps": case.get("steps", []),
            "errors": case.get("errors", []),
        }
        for case in cases
    ]
    payload = json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def evaluate_campaign(campaign_path: Path) -> dict[str, object]:
    campaign = load_json(campaign_path, CAMPAIGN_SCHEMA)
    thresholds = merged_thresholds(campaign)
    raw_cases = campaign.get("cases")
    if not isinstance(raw_cases, list):
        raise CorpusLossError("campaign is missing cases")
    cases: list[dict[str, object]] = []
    names: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise CorpusLossError(f"campaign case {index} must be an object")
        name = raw_case.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise CorpusLossError(f"campaign case {index} has an invalid or duplicate name")
        names.add(name)
        if "referenceManifest" not in raw_case or "candidateManifest" not in raw_case:
            raise CorpusLossError(f"campaign case {name} is missing a manifest path")
        cases.append(
            evaluate_case(
                name,
                resolve_path(raw_case["referenceManifest"], campaign_path.parent),
                resolve_path(raw_case["candidateManifest"], campaign_path.parent),
            )
        )

    valid = [case for case in cases if case.get("passedIntegrity") is True]
    total_tokens = sum(int(case["targetTokenCount"]) for case in valid)
    reference_nll_sum = sum(
        float(step["referenceNegativeLogLikelihoodNats"])
        for case in valid
        for step in case["steps"]  # type: ignore[union-attr]
    )
    candidate_nll_sum = sum(
        float(step["candidateNegativeLogLikelihoodNats"])
        for case in valid
        for step in case["steps"]  # type: ignore[union-attr]
    )
    reference_cross_entropy = reference_nll_sum / total_tokens if total_tokens else None
    candidate_cross_entropy = candidate_nll_sum / total_tokens if total_tokens else None
    cross_entropy_delta = (
        candidate_cross_entropy - reference_cross_entropy
        if candidate_cross_entropy is not None and reference_cross_entropy is not None
        else None
    )
    relative_perplexity_delta = (
        optional_exp(cross_entropy_delta) - 1.0
        if cross_entropy_delta is not None and optional_exp(cross_entropy_delta) is not None
        else None
    )
    worst_sample_delta = max((float(case["crossEntropyDeltaNats"]) for case in valid), default=None)
    checks = [
        {
            "name": "caseCoverage",
            "passed": len(cases) >= int(thresholds["minimumCaseCount"]),
            "actual": len(cases),
            "required": thresholds["minimumCaseCount"],
        },
        {
            "name": "tokenCoverage",
            "passed": total_tokens >= int(thresholds["minimumTokenCount"]),
            "actual": total_tokens,
            "required": thresholds["minimumTokenCount"],
        },
        {
            "name": "integrity",
            "passed": len(valid) == len(cases),
            "actual": len(valid),
            "required": len(cases),
        },
        {
            "name": "crossEntropyRegression",
            "passed": cross_entropy_delta is not None
            and cross_entropy_delta <= float(thresholds["maximumCrossEntropyRegressionNats"]),
            "actual": cross_entropy_delta,
            "required": thresholds["maximumCrossEntropyRegressionNats"],
        },
        {
            "name": "relativePerplexityRegression",
            "passed": relative_perplexity_delta is not None
            and relative_perplexity_delta <= float(thresholds["maximumRelativePerplexityRegression"]),
            "actual": relative_perplexity_delta,
            "required": thresholds["maximumRelativePerplexityRegression"],
        },
        {
            "name": "worstSampleCrossEntropyRegression",
            "passed": worst_sample_delta is not None
            and worst_sample_delta <= float(thresholds["maximumWorstSampleCrossEntropyRegressionNats"]),
            "actual": worst_sample_delta,
            "required": thresholds["maximumWorstSampleCrossEntropyRegressionNats"],
        },
    ]
    return {
        "schema": REPORT_SCHEMA,
        "campaign": str(campaign_path),
        "thresholds": thresholds,
        "passed": all(bool(check["passed"]) for check in checks),
        "checks": checks,
        "summary": {
            "caseCount": len(cases),
            "validCaseCount": len(valid),
            "targetTokenCount": total_tokens,
            "referenceCrossEntropyNats": reference_cross_entropy,
            "candidateCrossEntropyNats": candidate_cross_entropy,
            "crossEntropyDeltaNats": cross_entropy_delta,
            "referencePerplexity": optional_exp(reference_cross_entropy) if reference_cross_entropy is not None else None,
            "candidatePerplexity": optional_exp(candidate_cross_entropy) if candidate_cross_entropy is not None else None,
            "relativePerplexityDelta": relative_perplexity_delta,
            "maximumSampleCrossEntropyRegressionNats": worst_sample_delta,
            "fallbackCaseCount": sum(case.get("fallbackUsed") is True for case in cases),
            "nonFiniteCaseCount": sum(case.get("finiteLogits") is False for case in cases),
            "evidenceDigestSha256": evidence_digest(cases),
        },
        "cases": cases,
    }


def markdown_report(report: dict[str, object]) -> str:
    summary = report["summary"]
    assert isinstance(summary, dict)

    def number(value: object, precision: int = 6) -> str:
        return "n/a" if value is None else f"{float(value):.{precision}g}"

    lines = [
        "# GGUF Corpus Loss Gate",
        "",
        f"Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Cases | {summary['validCaseCount']} / {summary['caseCount']} valid |",
        f"| Target tokens | {summary['targetTokenCount']} |",
        f"| Reference cross-entropy | {number(summary['referenceCrossEntropyNats'])} nats |",
        f"| Candidate cross-entropy | {number(summary['candidateCrossEntropyNats'])} nats |",
        f"| Cross-entropy delta | {number(summary['crossEntropyDeltaNats'])} nats |",
        f"| Reference perplexity | {number(summary['referencePerplexity'])} |",
        f"| Candidate perplexity | {number(summary['candidatePerplexity'])} |",
        f"| Relative perplexity delta | {number(summary['relativePerplexityDelta'])} |",
        f"| Fallback / non-finite cases | {summary['fallbackCaseCount']} / {summary['nonFiniteCaseCount']} |",
        "",
        "| Case | Tokens | Reference CE | Candidate CE | Delta | Relative PPL delta | Worst token delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in report["cases"]:  # type: ignore[union-attr]
        assert isinstance(case, dict)
        if case.get("passedIntegrity") is not True:
            lines.append(f"| {case['name']} | invalid | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {case['name']} | {case['targetTokenCount']} | {number(case['referenceCrossEntropyNats'])} | "
            f"{number(case['candidateCrossEntropyNats'])} | {number(case['crossEntropyDeltaNats'])} | "
            f"{number(case['relativePerplexityDelta'])} | "
            f"{number(case['maximumTokenNegativeLogLikelihoodRegressionNats'])} |"
        )
    lines.extend(["", f"Evidence digest: `{summary['evidenceDigestSha256']}`", ""])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = evaluate_campaign(args.campaign.resolve())
    except CorpusLossError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
