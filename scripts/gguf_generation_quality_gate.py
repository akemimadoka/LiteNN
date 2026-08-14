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
    "minimumFixedTrajectoryTop1Agreement": 0.95,
    "minimumFixedTrajectoryTopKOverlap": 0.95,
    "minimumFixedTrajectoryCenteredCosine": 0.999,
    "maximumFixedTrajectoryJensenShannon": 0.001,
}

NATURAL_COMPARISON = "natural"
FIXED_REFERENCE_COMPARISON = "fixed-reference-trajectory"


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


def centered_logit_metrics(reference: list[float], candidate: list[float]) -> dict[str, float]:
    reference_mean = statistics.fmean(reference)
    candidate_mean = statistics.fmean(candidate)
    squared_error = 0.0
    reference_energy = 0.0
    candidate_energy = 0.0
    dot = 0.0
    maximum_absolute_error = 0.0
    for reference_value, candidate_value in zip(reference, candidate, strict=True):
        centered_reference = reference_value - reference_mean
        centered_candidate = candidate_value - candidate_mean
        difference = centered_candidate - centered_reference
        squared_error += difference * difference
        reference_energy += centered_reference * centered_reference
        candidate_energy += centered_candidate * centered_candidate
        dot += centered_reference * centered_candidate
        maximum_absolute_error = max(maximum_absolute_error, abs(difference))
    normalized_rms_error = (
        math.sqrt(squared_error / reference_energy)
        if reference_energy > 0.0
        else (0.0 if squared_error == 0.0 else math.inf)
    )
    cosine_denominator = math.sqrt(reference_energy * candidate_energy)
    cosine = dot / cosine_denominator if cosine_denominator > 0.0 else (1.0 if squared_error == 0.0 else 0.0)
    return {
        "centeredLogitNormalizedRmsError": normalized_rms_error,
        "centeredLogitCosineSimilarity": max(-1.0, min(1.0, cosine)),
        "centeredLogitMaximumAbsoluteError": maximum_absolute_error,
        "logitMeanOffset": candidate_mean - reference_mean,
    }


def probability_metrics(reference: list[float], candidate: list[float]) -> dict[str, float]:
    reference_maximum = max(reference)
    candidate_maximum = max(candidate)
    reference_exp_sum = math.fsum(math.exp(value - reference_maximum) for value in reference)
    candidate_exp_sum = math.fsum(math.exp(value - candidate_maximum) for value in candidate)
    reference_log_normalizer = reference_maximum + math.log(reference_exp_sum)
    candidate_log_normalizer = candidate_maximum + math.log(candidate_exp_sum)
    reference_entropy = 0.0
    cross_entropy = 0.0
    kl_divergence = 0.0
    jensen_shannon = 0.0
    total_variation = 0.0
    for reference_value, candidate_value in zip(reference, candidate, strict=True):
        reference_log_probability = reference_value - reference_log_normalizer
        candidate_log_probability = candidate_value - candidate_log_normalizer
        reference_probability = math.exp(reference_log_probability)
        candidate_probability = math.exp(candidate_log_probability)
        if reference_probability > 0.0:
            reference_entropy -= reference_probability * reference_log_probability
            cross_entropy -= reference_probability * candidate_log_probability
            kl_divergence += reference_probability * (
                reference_log_probability - candidate_log_probability
            )
        midpoint = 0.5 * (reference_probability + candidate_probability)
        if midpoint > 0.0:
            if reference_probability > 0.0:
                jensen_shannon += 0.5 * reference_probability * math.log(reference_probability / midpoint)
            if candidate_probability > 0.0:
                jensen_shannon += 0.5 * candidate_probability * math.log(candidate_probability / midpoint)
        total_variation += abs(reference_probability - candidate_probability)
    return {
        "referenceEntropyNats": reference_entropy,
        "referenceCrossEntropyCandidateNats": cross_entropy,
        "referenceToCandidateKLDivergenceNats": max(0.0, kl_divergence),
        "jensenShannonDivergenceNats": max(0.0, jensen_shannon),
        "totalVariationDistance": 0.5 * total_variation,
    }


def distribution_metrics(
    reference: list[float],
    candidate: list[float],
    top_k: int,
    include_full_distribution: bool = False,
) -> dict[str, object]:
    if len(reference) != len(candidate):
        raise QualityError(
            f"logit vocabulary mismatch: reference={len(reference)}, candidate={len(candidate)}"
        )
    reference_top = top_indices(reference, top_k)
    candidate_top = top_indices(candidate, top_k)
    overlap_count = len(set(reference_top) & set(candidate_top))
    denominator = min(top_k, len(reference))
    result: dict[str, object] = {
        "vocabularySize": len(reference),
        "referenceTopTokenId": reference_top[0],
        "candidateTopTokenId": candidate_top[0],
        "referenceTopMargin": reference[reference_top[0]] - reference[reference_top[1]] if len(reference_top) > 1 else None,
        "candidateTopMargin": candidate[candidate_top[0]] - candidate[candidate_top[1]] if len(candidate_top) > 1 else None,
        "referenceTopKTokenIds": reference_top,
        "candidateTopKTokenIds": candidate_top,
        "topKOverlapCount": overlap_count,
        "topKOverlap": overlap_count / denominator,
        "top1Agreement": reference_top[0] == candidate_top[0],
    }
    if include_full_distribution:
        result.update(
            {
                "referenceTopTokenCandidateRank": token_rank(candidate, reference_top[0]),
                "candidateTopTokenReferenceRank": token_rank(reference, candidate_top[0]),
                **centered_logit_metrics(reference, candidate),
                **probability_metrics(reference, candidate),
            }
        )
    return result


def common_prefix_length(reference: list[int], candidate: list[int]) -> int:
    result = 0
    for expected, actual in zip(reference, candidate):
        if expected != actual:
            break
        result += 1
    return result


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def fixed_step_excerpt(step: dict[str, object]) -> dict[str, object]:
    keys = (
        "decisionStep",
        "referenceTopTokenId",
        "candidateTopTokenId",
        "top1Agreement",
        "topKOverlap",
        "referenceTopMargin",
        "candidateTopMargin",
        "referenceTopTokenCandidateRank",
        "centeredLogitNormalizedRmsError",
        "centeredLogitCosineSimilarity",
        "referenceToCandidateKLDivergenceNats",
        "jensenShannonDivergenceNats",
        "totalVariationDistance",
    )
    return {key: step[key] for key in keys}


def fixed_trajectory_summary(steps: list[dict[str, object]]) -> dict[str, object]:
    if not steps:
        return {"comparedStepCount": 0}
    top1_values = [1.0 if step["top1Agreement"] else 0.0 for step in steps]
    top_k_values = [float(step["topKOverlap"]) for step in steps]
    nrmse_values = [float(step["centeredLogitNormalizedRmsError"]) for step in steps]
    cosine_values = [float(step["centeredLogitCosineSimilarity"]) for step in steps]
    kl_values = [float(step["referenceToCandidateKLDivergenceNats"]) for step in steps]
    jensen_shannon_values = [float(step["jensenShannonDivergenceNats"]) for step in steps]
    total_variation_values = [float(step["totalVariationDistance"]) for step in steps]
    rank_values = [float(step["referenceTopTokenCandidateRank"]) for step in steps]
    return {
        "comparedStepCount": len(steps),
        "top1Agreement": statistics.fmean(top1_values),
        "topKOverlapMean": statistics.fmean(top_k_values),
        "topKOverlapMinimum": min(top_k_values),
        "centeredLogitNormalizedRmsErrorMean": statistics.fmean(nrmse_values),
        "centeredLogitNormalizedRmsErrorMaximum": max(nrmse_values),
        "centeredLogitCosineSimilarityMean": statistics.fmean(cosine_values),
        "centeredLogitCosineSimilarityMinimum": min(cosine_values),
        "referenceToCandidateKLDivergenceNatsMean": statistics.fmean(kl_values),
        "referenceToCandidateKLDivergenceNatsMaximum": max(kl_values),
        "jensenShannonDivergenceNatsMean": statistics.fmean(jensen_shannon_values),
        "jensenShannonDivergenceNatsMaximum": max(jensen_shannon_values),
        "totalVariationDistanceMean": statistics.fmean(total_variation_values),
        "totalVariationDistanceMaximum": max(total_variation_values),
        "referenceTokenCandidateRankMean": statistics.fmean(rank_values),
        "referenceTokenCandidateRankMaximum": max(rank_values),
        "top1Mismatches": [fixed_step_excerpt(step) for step in steps if not step["top1Agreement"]],
        "worstCenteredCosineStep": fixed_step_excerpt(
            min(steps, key=lambda step: float(step["centeredLogitCosineSimilarity"]))
        ),
        "worstJensenShannonStep": fixed_step_excerpt(
            max(steps, key=lambda step: float(step["jensenShannonDivergenceNats"]))
        ),
    }


def evaluate_case(
    name: str,
    reference_path: Path,
    candidate_path: Path,
    top_k: int,
    comparison_mode: str = NATURAL_COMPARISON,
) -> dict[str, object]:
    report: dict[str, object] = {
        "name": name,
        "comparisonMode": comparison_mode,
        "referenceManifest": str(reference_path),
        "candidateManifest": str(candidate_path),
        "passedIntegrity": False,
        "fallbackUsed": None,
        "finiteLogits": None,
        "errors": [],
    }
    try:
        if comparison_mode not in (NATURAL_COMPARISON, FIXED_REFERENCE_COMPARISON):
            raise QualityError(f"unsupported comparison mode {comparison_mode!r}")
        reference_document = load_json(reference_path, MANIFEST_SCHEMA)
        candidate_document = load_json(candidate_path, MANIFEST_SCHEMA)
        if reference_document.get("sampling") != "greedy":
            raise QualityError("the reference manifest must use sampling=greedy")
        expected_candidate_sampling = (
            "greedy" if comparison_mode == NATURAL_COMPARISON else "forced-reference-trajectory"
        )
        if candidate_document.get("sampling") != expected_candidate_sampling:
            raise QualityError(
                f"candidate sampling must be {expected_candidate_sampling!r} for {comparison_mode} comparison"
            )
        reference_prompt = token_ids(reference_document, "promptTokenIds", reference_path)
        candidate_prompt = token_ids(candidate_document, "promptTokenIds", candidate_path)
        if reference_prompt != candidate_prompt:
            raise QualityError("reference and candidate prompt token ids differ")
        reference_tokens = token_ids(reference_document, "generatedTokenIds", reference_path)
        candidate_tokens = token_ids(candidate_document, "generatedTokenIds", candidate_path)
        if comparison_mode == FIXED_REFERENCE_COMPARISON and reference_tokens != candidate_tokens:
            raise QualityError("fixed-reference candidate context tokens differ from the reference trajectory")
        reference_fallback = reference_document.get("fallbackUsed")
        candidate_fallback = candidate_document.get("fallbackUsed")
        report["fallbackUsed"] = reference_fallback is not False or candidate_fallback is not False
        if reference_fallback is not False or candidate_fallback is not False:
            raise QualityError("both manifests must explicitly report fallbackUsed=false")
        reference_artifacts = resolve_logits(reference_document, reference_path, len(reference_tokens))
        candidate_artifacts = resolve_logits(candidate_document, candidate_path, len(candidate_tokens))

        prefix_length = common_prefix_length(reference_tokens, candidate_tokens)
        denominator = max(len(reference_tokens), len(candidate_tokens), 1)
        first_divergence = (
            prefix_length
            if comparison_mode == NATURAL_COMPARISON
            and prefix_length < max(len(reference_tokens), len(candidate_tokens))
            else None
        )
        common_steps = range(min(len(reference_tokens), len(candidate_tokens)))
        steps: list[dict[str, object]] = []
        same_context_overlaps: list[float] = []
        trajectory_overlaps: list[float] = []
        selected_top1_mismatches: list[dict[str, object]] = []
        disputed: dict[str, object] | None = None
        for step in common_steps:
            reference_logits = parse_logits(reference_artifacts[step])
            candidate_logits = parse_logits(candidate_artifacts[step])
            metrics = distribution_metrics(
                reference_logits,
                candidate_logits,
                top_k,
                comparison_mode == FIXED_REFERENCE_COMPARISON,
            )
            same_context = (
                comparison_mode == FIXED_REFERENCE_COMPARISON
                or first_divergence is None
                or step <= first_divergence
            )
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
            if comparison_mode == NATURAL_COMPARISON and metrics["candidateTopTokenId"] != candidate_token:
                selected_top1_mismatches.append(
                    {"decisionStep": step, "producer": "candidate", "selectedTokenId": candidate_token}
                )
            step_report = {
                "decisionStep": step,
                "sameContext": same_context,
                "referenceSelectedTokenId": reference_token,
                "candidateSelectedTokenId": (
                    candidate_token if comparison_mode == NATURAL_COMPARISON else metrics["candidateTopTokenId"]
                ),
                "candidateContextTokenId": candidate_token,
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
                "comparisonMode": comparison_mode,
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
    for key in (
        "minimumPrefixAgreement",
        "minimumSameContextTopKOverlap",
        "minimumFixedTrajectoryTop1Agreement",
        "minimumFixedTrajectoryTopKOverlap",
        "minimumFixedTrajectoryCenteredCosine",
    ):
        if not 0.0 <= float(thresholds[key]) <= 1.0:
            raise QualityError(f"{key} must be in [0, 1]")
    if float(thresholds["maximumFixedTrajectoryJensenShannon"]) < 0.0:
        raise QualityError("maximumFixedTrajectoryJensenShannon must be non-negative")
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
        comparison_mode = raw.get("comparisonMode", NATURAL_COMPARISON)
        if not isinstance(comparison_mode, str):
            raise QualityError(f"campaign case {name} has an invalid comparisonMode")
        cases.append(
            evaluate_case(
                name,
                resolve_path(raw["referenceManifest"], campaign_path.parent),
                resolve_path(raw["candidateManifest"], campaign_path.parent),
                int(thresholds["topK"]),
                comparison_mode,
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
    natural_cases = [case for case in valid_cases if case.get("comparisonMode") == NATURAL_COMPARISON]
    fixed_cases = [case for case in valid_cases if case.get("comparisonMode") == FIXED_REFERENCE_COMPARISON]
    for case in fixed_cases:
        case_steps = [step for step in case.get("steps", []) if isinstance(step, dict)]
        case["fixedTrajectorySummary"] = fixed_trajectory_summary(case_steps)
    fixed_steps = [
        step
        for case in fixed_cases
        for step in case.get("steps", [])
        if isinstance(step, dict)
    ]
    fixed_top1 = [1.0 if step["top1Agreement"] else 0.0 for step in fixed_steps]
    fixed_top_k = [float(step["topKOverlap"]) for step in fixed_steps]
    fixed_centered_nrmse = [float(step["centeredLogitNormalizedRmsError"]) for step in fixed_steps]
    fixed_centered_cosine = [float(step["centeredLogitCosineSimilarity"]) for step in fixed_steps]
    fixed_kl = [float(step["referenceToCandidateKLDivergenceNats"]) for step in fixed_steps]
    fixed_jensen_shannon = [float(step["jensenShannonDivergenceNats"]) for step in fixed_steps]
    fixed_total_variation = [float(step["totalVariationDistance"]) for step in fixed_steps]
    fixed_reference_ranks = [float(step["referenceTopTokenCandidateRank"]) for step in fixed_steps]
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
    ]
    if natural_cases:
        checks.extend(
            [
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
        )
    if fixed_cases:
        checks.extend(
            [
                {
                    "name": "fixedTrajectoryTop1Agreement",
                    "passed": bool(fixed_top1)
                    and float(mean(fixed_top1)) >= float(thresholds["minimumFixedTrajectoryTop1Agreement"]),
                    "actual": mean(fixed_top1),
                    "required": thresholds["minimumFixedTrajectoryTop1Agreement"],
                },
                {
                    "name": "fixedTrajectoryTopKOverlap",
                    "passed": bool(fixed_top_k)
                    and float(mean(fixed_top_k)) >= float(thresholds["minimumFixedTrajectoryTopKOverlap"]),
                    "actual": mean(fixed_top_k),
                    "required": thresholds["minimumFixedTrajectoryTopKOverlap"],
                },
                {
                    "name": "fixedTrajectoryCenteredCosine",
                    "passed": bool(fixed_centered_cosine)
                    and min(fixed_centered_cosine) >= float(thresholds["minimumFixedTrajectoryCenteredCosine"]),
                    "actual": min(fixed_centered_cosine) if fixed_centered_cosine else None,
                    "required": thresholds["minimumFixedTrajectoryCenteredCosine"],
                },
                {
                    "name": "fixedTrajectoryJensenShannon",
                    "passed": bool(fixed_jensen_shannon)
                    and max(fixed_jensen_shannon) <= float(thresholds["maximumFixedTrajectoryJensenShannon"]),
                    "actual": max(fixed_jensen_shannon) if fixed_jensen_shannon else None,
                    "required": thresholds["maximumFixedTrajectoryJensenShannon"],
                },
            ]
        )
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
            "fixedTrajectoryCaseCount": len(fixed_cases),
            "fixedTrajectoryComparedStepCount": len(fixed_steps),
            "fixedTrajectoryTop1Agreement": mean(fixed_top1),
            "fixedTrajectoryTopKOverlapMean": mean(fixed_top_k),
            "fixedTrajectoryTopKOverlapMinimum": min(fixed_top_k) if fixed_top_k else None,
            "fixedTrajectoryCenteredLogitNormalizedRmsErrorMean": mean(fixed_centered_nrmse),
            "fixedTrajectoryCenteredLogitNormalizedRmsErrorMaximum": (
                max(fixed_centered_nrmse) if fixed_centered_nrmse else None
            ),
            "fixedTrajectoryCenteredLogitCosineSimilarityMean": mean(fixed_centered_cosine),
            "fixedTrajectoryCenteredLogitCosineSimilarityMinimum": (
                min(fixed_centered_cosine) if fixed_centered_cosine else None
            ),
            "fixedTrajectoryReferenceToCandidateKLDivergenceNatsMean": mean(fixed_kl),
            "fixedTrajectoryReferenceToCandidateKLDivergenceNatsMaximum": max(fixed_kl) if fixed_kl else None,
            "fixedTrajectoryJensenShannonDivergenceNatsMean": mean(fixed_jensen_shannon),
            "fixedTrajectoryJensenShannonDivergenceNatsMaximum": (
                max(fixed_jensen_shannon) if fixed_jensen_shannon else None
            ),
            "fixedTrajectoryTotalVariationDistanceMean": mean(fixed_total_variation),
            "fixedTrajectoryTotalVariationDistanceMaximum": (
                max(fixed_total_variation) if fixed_total_variation else None
            ),
            "fixedTrajectoryReferenceTokenCandidateRankMean": mean(fixed_reference_ranks),
            "fixedTrajectoryReferenceTokenCandidateRankMaximum": (
                max(fixed_reference_ranks) if fixed_reference_ranks else None
            ),
        },
        "cases": cases,
    }


def percent(value: object) -> str:
    return "n/a" if value is None else f"{float(value) * 100.0:.2f}%"


def markdown_report(report: dict[str, object]) -> str:
    summary = report["summary"]
    assert isinstance(summary, dict)
    cases = report["cases"]
    assert isinstance(cases, list)
    fixed_only = int(summary.get("fixedTrajectoryCaseCount", 0)) == int(summary["validCaseCount"])
    prefix_label = "Context trajectory agreement" if fixed_only else "Weighted prefix agreement"
    lines = [
        "# Natural Generation Quality Gate",
        "",
        f"Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Cases | {summary['validCaseCount']} / {summary['caseCount']} valid |",
        f"| Reference tokens | {summary['totalReferenceGeneratedTokens']} |",
        f"| {prefix_label} | {percent(summary['weightedPrefixAgreement'])} |",
        f"| Same-context top-k overlap | {percent(summary['sameContextTopKOverlapMean'])} |",
        f"| Trajectory top-k overlap | {percent(summary['trajectoryTopKOverlapMean'])} |",
        f"| Median first divergence | {summary['firstDivergenceDecisionStepMedian']} |",
        f"| Fallback / non-finite cases | {summary['fallbackCaseCount']} / {summary['nonFiniteCaseCount']} |",
        "",
    ]
    if int(summary.get("fixedTrajectoryCaseCount", 0)) > 0:
        lines.extend(
            [
                "## Fixed-Reference Trajectory",
                "",
                "| Metric | Mean | Worst |",
                "| --- | ---: | ---: |",
                f"| Top-1 agreement | {percent(summary['fixedTrajectoryTop1Agreement'])} | n/a |",
                f"| Top-k overlap | {percent(summary['fixedTrajectoryTopKOverlapMean'])} "
                f"| {percent(summary['fixedTrajectoryTopKOverlapMinimum'])} |",
                f"| Centered logit NRMSE | {summary['fixedTrajectoryCenteredLogitNormalizedRmsErrorMean']:.6g} "
                f"| {summary['fixedTrajectoryCenteredLogitNormalizedRmsErrorMaximum']:.6g} |",
                f"| Centered logit cosine | {summary['fixedTrajectoryCenteredLogitCosineSimilarityMean']:.9f} "
                f"| {summary['fixedTrajectoryCenteredLogitCosineSimilarityMinimum']:.9f} |",
                f"| KL(reference || candidate), nats | "
                f"{summary['fixedTrajectoryReferenceToCandidateKLDivergenceNatsMean']:.6g} "
                f"| {summary['fixedTrajectoryReferenceToCandidateKLDivergenceNatsMaximum']:.6g} |",
                f"| Jensen-Shannon, nats | {summary['fixedTrajectoryJensenShannonDivergenceNatsMean']:.6g} "
                f"| {summary['fixedTrajectoryJensenShannonDivergenceNatsMaximum']:.6g} |",
                f"| Total variation | {summary['fixedTrajectoryTotalVariationDistanceMean']:.6g} "
                f"| {summary['fixedTrajectoryTotalVariationDistanceMaximum']:.6g} |",
                f"| Reference top-token rank in candidate | "
                f"{summary['fixedTrajectoryReferenceTokenCandidateRankMean']:.3f} "
                f"| {summary['fixedTrajectoryReferenceTokenCandidateRankMaximum']:.0f} |",
                "",
                "| Case | Top-1 | Top-k mean / min | NRMSE mean / max | Cosine min | JS mean / max | TV max |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for case in cases:
            if not isinstance(case, dict) or case.get("fixedTrajectorySummary") is None:
                continue
            fixed = case["fixedTrajectorySummary"]
            assert isinstance(fixed, dict)
            lines.append(
                f"| {case['name']} | {percent(fixed['top1Agreement'])} "
                f"| {percent(fixed['topKOverlapMean'])} / {percent(fixed['topKOverlapMinimum'])} "
                f"| {fixed['centeredLogitNormalizedRmsErrorMean']:.6g} / "
                f"{fixed['centeredLogitNormalizedRmsErrorMaximum']:.6g} "
                f"| {fixed['centeredLogitCosineSimilarityMinimum']:.9f} "
                f"| {fixed['jensenShannonDivergenceNatsMean']:.6g} / "
                f"{fixed['jensenShannonDivergenceNatsMaximum']:.6g} "
                f"| {fixed['totalVariationDistanceMaximum']:.6g} |"
            )
        lines.append("")
    lines.extend(
        [
            "| Case | Mode | Ref / candidate tokens | Context prefix | First divergence | Same-context top-k | Trajectory top-k | Integrity |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for case in cases:
        assert isinstance(case, dict)
        if case.get("passedIntegrity") is not True:
            errors = case.get("errors", [])
            lines.append(
                f"| {case['name']} | {case.get('comparisonMode')} | n/a | n/a | n/a | n/a | n/a "
                f"| FAIL: {'; '.join(errors)} |"
            )
            continue
        lines.append(
            f"| {case['name']} | {case['comparisonMode']} "
            f"| {case['referenceGeneratedTokenCount']} / {case['candidateGeneratedTokenCount']} "
            f"| {percent(case['prefixAgreement'])} | {case['firstDivergenceDecisionStep']} "
            f"| {percent(case['sameContextTopKOverlapMean'])} | {percent(case['trajectoryTopKOverlapMean'])} | PASS |"
        )
    lines.extend(
        [
            "",
            "Same-context metrics include the first divergent decision because both runtimes consumed an identical token prefix.",
            "Post-divergence top-k overlap is trajectory-level evidence and is not interpreted as same-input numerical parity.",
            "Fixed-reference metrics replay the reference tokens as context; candidate top-1 remains an observation, not the fed token.",
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
    parser.add_argument("--minimum-fixed-trajectory-top1-agreement", type=float)
    parser.add_argument("--minimum-fixed-trajectory-top-k-overlap", type=float)
    parser.add_argument("--minimum-fixed-trajectory-centered-cosine", type=float)
    parser.add_argument("--maximum-fixed-trajectory-jensen-shannon", type=float)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    overrides = {
        "topK": args.top_k,
        "minimumCaseCount": args.minimum_case_count,
        "minimumTotalReferenceTokens": args.minimum_total_reference_tokens,
        "minimumPrefixAgreement": args.minimum_prefix_agreement,
        "minimumSameContextTopKOverlap": args.minimum_same_context_top_k_overlap,
        "minimumFixedTrajectoryTop1Agreement": args.minimum_fixed_trajectory_top1_agreement,
        "minimumFixedTrajectoryTopKOverlap": args.minimum_fixed_trajectory_top_k_overlap,
        "minimumFixedTrajectoryCenteredCosine": args.minimum_fixed_trajectory_centered_cosine,
        "maximumFixedTrajectoryJensenShannon": args.maximum_fixed_trajectory_jensen_shannon,
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
