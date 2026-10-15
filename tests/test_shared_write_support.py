"""Tests for the offline S3 support diagnostic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from online_rl.shared_write_grouping import DynamicsGrouper, empty_profile  # noqa: E402
from scripts.diagnose_shared_write_support import (  # noqa: E402
    EXPECTED_GROUPER_CONFIG,
    audit_local_context_references,
    pairwise_family_audit,
)


def _profile(cells: list[tuple[int, int, int]]) -> np.ndarray:
    profile = empty_profile()
    for band, action, direction in cells:
        profile[band, action, direction] = 2
    return profile


def test_explicit_equal_prototypes_are_reported_as_ambiguous() -> None:
    cells = [(band, action, 0) for band in range(2) for action in range(2)]
    reference = _profile(cells)
    query = _profile(cells)

    # Direct state construction is intentional: this is a synthetic tied-
    # prototype pressure test, not a grouper-discovered training state.
    state = DynamicsGrouper(**EXPECTED_GROUPER_CONFIG).state_dict()
    state["group_counts"] = np.stack((reference, reference)).astype(np.uint16)
    state["nbytes"] = int(state["group_counts"].nbytes)
    tied = DynamicsGrouper.from_state_dict(state)
    result = tied.assign(query, commit=False)

    assert result.group == -1
    assert result.reason == "ambiguous_match"
    assert result.scores[0] == result.scores[1] == {"overlap": 4, "agreement": 1.0}


def test_pairwise_audit_keeps_unresolved_pairs_out_of_accuracy_denominator() -> None:
    assignment = {"A0": 0, "B0": 1, "A2": -1, "B2": 1, "A1": 0, "B1": -1}

    audit, pairs = pairwise_family_audit(assignment)

    assert len(pairs) == 15
    assert audit["resolved_pairs"] == 6
    assert audit["unresolved_pairs"] == 9
    assert audit["resolved_pair_coverage"] == 6 / 15
    assert audit["pairwise_accuracy_over_resolved_pairs"] == 1.0
    assert audit["pairwise_error_rate_over_resolved_pairs"] == 0.0
    assert sum(row["pair_status"] == "unresolved" for row in pairs) == 9


def test_action_censored_profile_can_match_family_refs_but_hide_local_context() -> None:
    full_profiles = {}
    censored_profiles = {}
    decisions = []
    for context in ("A0", "B0", "A2", "B2", "A1", "B1"):
        profile = empty_profile()
        family_direction = 0 if context.startswith("A") else 1
        for band in range(3):
            for action in (0, 1):
                profile[band, action, family_direction] = 2
            local_direction = (int(context[1]) + band) % 4
            for action in (2, 3):
                profile[band, action, local_direction] = 2
        full_profiles[context] = profile
        censored = profile.copy()
        censored[:, 2:, :] = 0
        censored_profiles[context] = censored
        decisions.append(
            {
                "context": context,
                "assigned_group": 0 if context.startswith("A") else 1,
                "reason": "matched",
            }
        )

    summary, rows = audit_local_context_references(
        827, full_profiles, censored_profiles, decisions
    )
    a0 = next(row for row in rows if row["query_context_audit_label_only"] == "A0")

    assert summary["new_environment_interactions"] == 0
    assert a0["fully_covered_no_conflict_reference_names"] == ["A0", "A2", "A1"]
    assert a0["fully_covered_no_conflict_reference_count"] == 3
    assert a0["local_context_ambiguity_present"] is True
    assert a0["local_context_identifiable_from_censored_profile"] is False
    assert a0["unprobed_actions_with_full_profile_support"] == [2, 3]
    assert set(a0["candidate_references_with_differences_hidden_on_unprobed_cells"]) == {"A2", "A1"}
