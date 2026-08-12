#!/usr/bin/env python3
"""Evaluate natural GGUF generation trajectories against a reference runtime.

The campaign points at pairs of ``litenn.natural_generation.v1`` manifests.
Decision step zero is the prefill distribution that selects the first generated
token. Top-k comparisons through the first divergence use identical contexts;
later comparisons are reported separately as trajectory-level diagnostics.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import statistics
import sys
from pathlib import Path


CAMPAIGN_SCHEMA = "litenn.natural_generation_campaign.v1"
MANIFEST_SCHEMA = "litenn.natural_generation.v1"
REPORT_SCHEMA = "litenn.natural_generation_quality.v1"
DEFAULT_THRESHOLDS: dict[str, int | float] = {
    "topK": 10,
    "minimumCaseCount": 2,
    "minimumTotalReferenceTokens": 128,
    "minimumPrefixAgreement": 0.95,
    "minimumSameContextTopKOverlap": 0.90,
}


class QualityError(RuntimeError):
    pass


def load_json(path: Path, schema: str) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualityError(f"failed to read JSON {path}: {error}") from error
    if not isinstance(document, dict) or document.get("schema") != schema:
        actual = document.get("schema") if isinstance(document, dict) else type(document).__name__
        raise QualityError(f"unsupported schema in {path}: {actual!r}")
    return document


def resolve_path(raw: object, base: Path) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else base / path


def token_ids(document: dict[str, object], key: str, source: Path) -> list[int]:
    raw = document.get(key)
    if not isinstance(raw, list) or any(not isinstance(token, int) or token < 0 for token in raw):
        raise QualityError(f"{source} has invalid {key}")
    return raw


def resolve_logits(document: dict[str, object], source: Path, generated_count: int) -> dict[int, Path]:
    raw_artifacts = document.get("logitsArtifacts")
    if not isinstance(raw_artifacts, list):
        raise QualityError(f"{source} is missing logitsArtifacts")
    artifacts: dict[int, Path] = {}
    for raw in raw_artifacts:
        if not isinstance(raw, dict) or "decisionStep" not in raw or "path" not in raw:
            raise QualityError(f"{source} contains an invalid logits artifact")
        step = raw["decisionStep"]
        if not isinstance(step, int) or step < 0 or step in artifacts:
            raise QualityError(f"{source} contains invalid or duplicate decision step {step!r}")
        path = resolve_path(raw["path"], source.parent)
        if not path.is_file():
            raise QualityError(f"logits artifact does not exist: {path}")
        artifacts[step] = path
    expected = set(range(generated_count))
    if set(artifacts) != expected:
        missing = sorted(expected - set(artifacts))
        extra = sorted(set(artifacts) - expected)
        raise QualityError(f"{source} logits coverage mismatch: missing={missing}, extra={extra}")
    return artifacts


def parse_logits(path: Path) -> list[float]:
    values: dict[int, float] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise QualityError(f"failed to read logits {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        index_text, separator, value_text = line.partition(":")
        if not separator:
            raise QualityError(f"malformed logit at {path}:{line_number}")
        try:
            index = int(index_text.strip())
            value = float(value_text.strip())
        except ValueError as error:
            raise QualityError(f"malformed logit at {path}:{line_number}") from error
        if index < 0 or index in values:
            raise QualityError(f"invalid or duplicate logit index {index} in {path}")
        if not math.isfinite(value):
            raise QualityError(f"non-finite logit at index {index} in {path}")
        values[index] = value
    if not values or len(values) != max(values) + 1:
        raise QualityError(f"logits file is empty or sparse: {path}")
    return [values[index] for index in range(len(values))]


def top_indices(values: list[float], count: int) -> list[int]:
    return heapq.nlargest(min(count, len(values)), range(len(values)), key=lambda index: (values[index], -index))


def token_rank(values: list[float], token: int) -> int:
    if token < 0 or token >= len(values):
        raise QualityError(f"selected token {token} is outside vocabulary size {len(values)}")
    selected = values[token]
    return 1 + sum(value > selected or (value == selected and index < token) for index, value in enumerate(values))


def distribution_metrics(reference: list[float], candidate: list[float], top_k: int) -> dict[str, object]:
    if len(reference) != len(candidate):
        raise QualityError(
            f"logit vocabulary mismatch: reference={len(reference)}, candidate={len(candidate)}"
        )
    reference_top = top_indices(reference, top_k)
    candidate_top = top_indices(candidate, top_k)
    overlap_count = len(set(reference_top) & set(candidate_top))
    denominator = min(top_k, len(reference))
    return {
        "vocabularySize": len(reference),
        "referenceTopTokenId": reference_top[0],
        "candidateTopTokenId": candidate_top[0],
        "referenceTopMargin": reference[reference_top[0]] - reference[reference_top[1]] if len(reference_top) > 1 else None,
        "candidateTopMargin": candidate[candidate_top[0]] - candidate[candidate_top[1]] if len(candidate_top) > 1 else None,
        "referenceTopKTokenIds": reference_top,
        "candidateTopKTokenIds": candidate_top,
        "topKOverlapCount": overlap_count,
        "topKOverlap": overlap_count / denominator,
    }


def common_prefix_length(reference: list[int], candidate: list[int]) -> int:
    result = 0
    for expected, actual in zip(reference, candidate):
        if expected != actual:
            break
        result += 1
    return result


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def evaluate_case(name: str, reference_path: Path, candidate_path: Path, top_k: int) -> dict[str, object]:
    report: dict[str, object] = {
        "name": name,
        "referenceManifest": str(reference_path),
        "candidateManifest": str(candidate_path),
        "passedIntegrity": False,
        "fallbackUsed": None,
        "finiteLogits": None,
        "errors": [],
    }
    try:
        reference_document = load_json(reference_path, MANIFEST_SCHEMA)
        candidate_document = load_json(candidate_path, MANIFEST_SCHEMA)
        reference_prompt = token_ids(reference_document, "promptTokenIds", reference_path)
        candidate_prompt = token_ids(candidate_document, "promptTokenIds", candidate_path)
        if reference_prompt != candidate_prompt:
            raise QualityError("reference and candidate prompt token ids differ")
        reference_tokens = token_ids(reference_document, "generatedTokenIds", reference_path)
        candidate_tokens = token_ids(candidate_document, "generatedTokenIds", candidate_path)
        reference_fallback = reference_document.get("fallbackUsed")
        candidate_fallback = candidate_document.get("fallbackUsed")
        report["fallbackUsed"] = reference_fallback is not False or candidate_fallback is not False
        if reference_fallback is not False or candidate_fallback is not False:
            raise QualityError("both manifests must explicitly report fallbackUsed=false")
        reference_artifacts = resolve_logits(reference_document, reference_path, len(reference_tokens))
        candidate_artifacts = resolve_logits(candidate_document, candidate_path, len(candidate_tokens))

        prefix_length = common_prefix_length(reference_tokens, candidate_tokens)
        denominator = max(len(reference_tokens), len(candidate_tokens), 1)
        first_divergence = prefix_length if prefix_length < max(len(reference_tokens), len(candidate_tokens)) else None
        common_steps = range(min(len(reference_tokens), len(candidate_tokens)))
        steps: list[dict[str, object]] = []
        same_context_overlaps: list[float] = []
        trajectory_overlaps: list[float] = []
        selected_top1_mismatches: list[dict[str, object]] = []
        disputed: dict[str, object] | None = None
        for step in common_steps:
            reference_logits = parse_logits(reference_artifacts[step])
            candidate_logits = parse_logits(candidate_artifacts[step])
            metrics = distribution_metrics(reference_logits, candidate_logits, top_k)
            same_context = first_divergence is None or step <= first_divergence
            overlap = float(metrics["topKOverlap"])
            trajectory_overlaps.append(overlap)
            if same_context:
                same_context_overlaps.append(overlap)
            reference_token = reference_tokens[step]
            candidate_token = candidate_tokens[step]
            if metrics["referenceTopTokenId"] != reference_token:
                selected_top1_mismatches.append(
                    {"decisionStep": step, "producer": "reference", "selectedTokenId": reference_token}
                )
            if metrics["candidateTopTokenId"] != candidate_token:
                selected_top1_mismatches.append(
                    {"decisionStep": step, "producer": "candidate", "selectedTokenId": candidate_token}
                )
            step_report = {
                "decisionStep": step,
                "sameContext": same_context,
                "referenceSelectedTokenId": reference_token,
                "candidateSelectedTokenId": candidate_token,
                **metrics,
            }
            steps.append(step_report)
            if first_divergence == step and reference_token != candidate_token:
                disputed = {
                    "decisionStep": step,
                    "sameContext": True,
                    "referenceSelectedTokenId": reference_token,
                    "candidateSelectedTokenId": candidate_token,
                    "referenceDistribution": {
                        "referenceTokenRank": token_rank(reference_logits, reference_token),
                        "candidateTokenRank": token_rank(reference_logits, candidate_token),
                        "preferenceMargin": reference_logits[reference_token] - reference_logits[candidate_token],
                    },
                    "candidateDistribution": {
                        "candidateTokenRank": token_rank(candidate_logits, candidate_token),
                        "referenceTokenRank": token_rank(candidate_logits, reference_token),
                        "preferenceMargin": candidate_logits[candidate_token] - candidate_logits[reference_token],
                    },
                }

        if selected_top1_mismatches:
            raise QualityError(f"selected token does not match greedy top-1: {selected_top1_mismatches[:4]}")
        report.update(
            {
                "referenceProducer": reference_document.get("producer"),
                "candidateProducer": candidate_document.get("producer"),
                "promptTokenCount": len(reference_prompt),
                "referenceGeneratedTokenCount": len(reference_tokens),
                "candidateGeneratedTokenCount": len(candidate_tokens),
                "commonPrefixTokenCount": prefix_length,
                "prefixAgreement": prefix_length / denominator,
                "firstDivergenceDecisionStep": first_divergence,
                "sameContextComparedStepCount": len(same_context_overlaps),
                "sameContextTopKOverlapMean": mean(same_context_overlaps),
                "sameContextTopKOverlapMinimum": min(same_context_overlaps) if same_context_overlaps else None,
                "trajectoryComparedStepCount": len(trajectory_overlaps),
                "trajectoryTopKOverlapMean": mean(trajectory_overlaps),
                "disputedToken": disputed,
                "referenceStoppedOnEos": reference_document.get("stoppedOnEos"),
                "candidateStoppedOnEos": candidate_document.get("stoppedOnEos"),
                "fallbackUsed": False,
                "finiteLogits": True,
                "passedIntegrity": True,
                "steps": steps,
            }
        )
    except QualityError as error:
        report["errors"] = [str(error)]
        if "non-finite logit" in str(error):
            report["finiteLogits"] = False
    return report


def merged_thresholds(campaign: dict[str, object], overrides: dict[str, int | float | None]) -> dict[str, int | float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    raw = campaign.get("thresholds", {})
    if not isinstance(raw, dict):
        raise QualityError("campaign thresholds must be an object")
    for key in thresholds:
        if key in raw:
            value = raw[key]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise QualityError(f"campaign threshold {key} must be numeric")
            thresholds[key] = value
        if overrides.get(key) is not None:
            thresholds[key] = overrides[key]  # type: ignore[assignment]
    for key in ("topK", "minimumCaseCount", "minimumTotalReferenceTokens"):
        if not float(thresholds[key]).is_integer():
            raise QualityError(f"{key} must be an integer")
    if int(thresholds["topK"]) <= 0 or int(thresholds["minimumCaseCount"]) <= 0:
        raise QualityError("topK and minimumCaseCount must be positive")
    if int(thresholds["minimumTotalReferenceTokens"]) <= 0:
        raise QualityError("minimumTotalReferenceTokens must be positive")
    for key in ("minimumPrefixAgreement", "minimumSameContextTopKOverlap"):
        if not 0.0 <= float(thresholds[key]) <= 1.0:
            raise QualityError(f"{key} must be in [0, 1]")
    return thresholds


def evaluate_campaign(
    campaign_path: Path, overrides: dict[str, int | float | None] | None = None
) -> dict[str, object]:
    campaign = load_json(campaign_path, CAMPAIGN_SCHEMA)
    thresholds = merged_thresholds(campaign, overrides or {})
    raw_cases = campaign.get("cases")
    if not isinstance(raw_cases, list):
        raise QualityError("campaign is missing cases")
    cases: list[dict[str, object]] = []
    seen_names: set[str] = set()
    for index, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise QualityError(f"campaign case {index} must be an object")
        name = raw.get("name")
        if not isinstance(name, str) or not name or name in seen_names:
            raise QualityError(f"campaign case {index} has an invalid or duplicate name")
        seen_names.add(name)
        if "referenceManifest" not in raw or "candidateManifest" not in raw:
            raise QualityError(f"campaign case {name} is missing a manifest path")
        cases.append(
            evaluate_case(
                name,
                resolve_path(raw["referenceManifest"], campaign_path.parent),
                resolve_path(raw["candidateManifest"], campaign_path.parent),
                int(thresholds["topK"]),
            )
        )

    valid_cases = [case for case in cases if case.get("passedIntegrity") is True]
    total_reference_tokens = sum(int(case["referenceGeneratedTokenCount"]) for case in valid_cases)
    total_candidate_tokens = sum(int(case["candidateGeneratedTokenCount"]) for case in valid_cases)
    prefix_numerator = sum(int(case["commonPrefixTokenCount"]) for case in valid_cases)
    prefix_denominator = sum(
        max(int(case["referenceGeneratedTokenCount"]), int(case["candidateGeneratedTokenCount"]), 1)
        for case in valid_cases
    )
    weighted_prefix = prefix_numerator / prefix_denominator if prefix_denominator else 0.0
    same_context_values = [
        float(step["topKOverlap"])
        for case in valid_cases
        for step in case.get("steps", [])
        if isinstance(step, dict) and step.get("sameContext") is True
    ]
    trajectory_values = [
        float(step["topKOverlap"])
        for case in valid_cases
        for step in case.get("steps", [])
        if isinstance(step, dict)
    ]
    divergences = [
        int(case["firstDivergenceDecisionStep"])
        for case in valid_cases
        if case.get("firstDivergenceDecisionStep") is not None
    ]
    same_context_overlap = mean(same_context_values) or 0.0
    checks = [
        {
            "name": "caseCoverage",
            "passed": len(cases) >= int(thresholds["minimumCaseCount"]),
            "actual": len(cases),
            "required": thresholds["minimumCaseCount"],
        },
        {
            "name": "tokenCoverage",
            "passed": total_reference_tokens >= int(thresholds["minimumTotalReferenceTokens"]),
            "actual": total_reference_tokens,
            "required": thresholds["minimumTotalReferenceTokens"],
        },
        {
            "name": "integrity",
            "passed": len(valid_cases) == len(cases),
            "actual": len(valid_cases),
            "required": len(cases),
        },
        {
            "name": "prefixAgreement",
            "passed": weighted_prefix >= float(thresholds["minimumPrefixAgreement"]),
            "actual": weighted_prefix,
            "required": thresholds["minimumPrefixAgreement"],
        },
        {
            "name": "sameContextTopKOverlap",
            "passed": bool(same_context_values)
            and same_context_overlap >= float(thresholds["minimumSameContextTopKOverlap"]),
            "actual": same_context_overlap,
            "required": thresholds["minimumSameContextTopKOverlap"],
        },
    ]
    passed = all(bool(check["passed"]) for check in checks)
    return {
        "schema": REPORT_SCHEMA,
        "campaign": str(campaign_path),
        "thresholds": thresholds,
        "passed": passed,
        "checks": checks,
        "summary": {
            "caseCount": len(cases),
            "validCaseCount": len(valid_cases),
            "totalReferenceGeneratedTokens": total_reference_tokens,
            "totalCandidateGeneratedTokens": total_candidate_tokens,
            "weightedPrefixAgreement": weighted_prefix,
            "sameContextComparedStepCount": len(same_context_values),
            "sameContextTopKOverlapMean": same_context_overlap,
            "sameContextTopKOverlapMinimum": min(same_context_values) if same_context_values else None,
            "trajectoryComparedStepCount": len(trajectory_values),
            "trajectoryTopKOverlapMean": mean(trajectory_values),
            "firstDivergenceDecisionStepMedian": statistics.median(divergences) if divergences else None,
            "noDivergenceCaseCount": len(valid_cases) - len(divergences),
            "fallbackCaseCount": sum(case.get("fallbackUsed") is True for case in cases),
            "nonFiniteCaseCount": sum(case.get("finiteLogits") is False for case in cases),
        },
        "cases": cases,
    }


def percent(value: object) -> str:
    return "n/a" if value is None else f"{float(value) * 100.0:.2f}%"


def markdown_report(report: dict[str, object]) -> str:
    summary = report["summary"]
    assert isinstance(summary, dict)
    lines = [
        "# Natural Generation Quality Gate",
        "",
        f"Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Cases | {summary['validCaseCount']} / {summary['caseCount']} valid |",
        f"| Reference tokens | {summary['totalReferenceGeneratedTokens']} |",
        f"| Weighted prefix agreement | {percent(summary['weightedPrefixAgreement'])} |",
        f"| Same-context top-k overlap | {percent(summary['sameContextTopKOverlapMean'])} |",
        f"| Trajectory top-k overlap | {percent(summary['trajectoryTopKOverlapMean'])} |",
        f"| Median first divergence | {summary['firstDivergenceDecisionStepMedian']} |",
        f"| Fallback / non-finite cases | {summary['fallbackCaseCount']} / {summary['nonFiniteCaseCount']} |",
        "",
        "| Case | Ref / candidate tokens | Prefix | First divergence | Same-context top-k | Trajectory top-k | Integrity |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    cases = report["cases"]
    assert isinstance(cases, list)
    for case in cases:
        assert isinstance(case, dict)
        if case.get("passedIntegrity") is not True:
            errors = case.get("errors", [])
            lines.append(f"| {case['name']} | n/a | n/a | n/a | n/a | n/a | FAIL: {'; '.join(errors)} |")
            continue
        lines.append(
            f"| {case['name']} | {case['referenceGeneratedTokenCount']} / {case['candidateGeneratedTokenCount']} "
            f"| {percent(case['prefixAgreement'])} | {case['firstDivergenceDecisionStep']} "
            f"| {percent(case['sameContextTopKOverlapMean'])} | {percent(case['trajectoryTopKOverlapMean'])} | PASS |"
        )
    lines.extend(
        [
            "",
            "Same-context metrics include the first divergent decision because both runtimes consumed an identical token prefix.",
            "Post-divergence top-k overlap is trajectory-level evidence and is not interpreted as same-input numerical parity.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--minimum-case-count", type=int)
    parser.add_argument("--minimum-total-reference-tokens", type=int)
    parser.add_argument("--minimum-prefix-agreement", type=float)
    parser.add_argument("--minimum-same-context-top-k-overlap", type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    overrides = {
        "topK": args.top_k,
        "minimumCaseCount": args.minimum_case_count,
        "minimumTotalReferenceTokens": args.minimum_total_reference_tokens,
        "minimumPrefixAgreement": args.minimum_prefix_agreement,
        "minimumSameContextTopKOverlap": args.minimum_same_context_top_k_overlap,
    }
    try:
        report = evaluate_campaign(args.campaign, overrides)
    except QualityError as error:
        raise SystemExit(str(error)) from error
    output = args.output if args.output is not None else args.campaign.parent / "generation_quality_report.json"
    markdown = args.markdown if args.markdown is not None else output.with_suffix(".md")
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"quality_gate={'PASS' if report['passed'] else 'FAIL'} report={output} markdown={markdown}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
