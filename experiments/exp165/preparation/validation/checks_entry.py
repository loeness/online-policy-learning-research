"""Remote-only CHECKS worker for the frozen E165 validation stage."""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Any

import phase_common as common


PROPERTY_FIELDS = {
    "full_training_oracle_evaluations": "synthetic_full_training_oracle_evaluations",
    "complete_score_calls": "synthetic_complete_score_calls",
}
REAL_HELPER_PROPERTIES = {
    "training_oracle": "e165.real_budgeted_training_oracle_calls",
    "protected_objective_and_gradient": "e165.real_protected_objective_and_gradient_calls",
    "owner_kl_diagnostics": "e165.real_owner_kl_diagnostics_calls",
    "stage0_scorer": "e165.real_stage0_scorer_calls",
}
MOCK_HELPER_PROPERTIES = {
    "training_oracle": "e165.mock_training_oracle_calls",
    "stage0_scorer": "e165.mock_stage0_scorer_calls",
}
REAL_FIT_PROPERTY_FIELDS = {
    "protected": "real_protected_fit_property",
    "behavioral": "real_behavioral_fit_property",
    "total": "real_fit_total_property",
}
METRIC_HELPERS = {
    "full_training_oracle_evaluations": "training_oracle",
    "complete_score_calls": "stage0_scorer",
}
TEARDOWN_PROPERTIES = (
    *PROPERTY_FIELDS.values(), *REAL_HELPER_PROPERTIES.values(),
    *MOCK_HELPER_PROPERTIES.values(),
    *(common.CHECKS_REAL_CALL_LIMITS[field] for field in REAL_FIT_PROPERTY_FIELDS.values()),
)


class _ChecksPlugin:
    """Collect exact pytest execution outcomes and declared synthetic counters."""

    def __init__(self) -> None:
        self.collected: list[str] = []
        self.executed: list[str] = []
        self._called: set[str] = set()
        self.passed = 0
        self.failed = 0
        self.error = 0
        self.skipped = 0
        self.xfail = 0
        self.collection_errors = 0
        self._xfail_nodes: set[str] = set()
        self._teardown_properties: dict[str, dict[str, int]] = {}
        self.property_errors: list[dict[str, str]] = []
        self.property_consistency_errors: list[dict[str, Any]] = []
        self.failure_details: list[dict[str, str]] = []
        self.exit_status: int | None = None

    def pytest_collection_finish(self, session: Any) -> None:
        self.collected = [item.nodeid for item in session.items]

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self.collection_errors += 1
            self.error += 1
            self.failure_details.append({
                "nodeid": str(report.nodeid), "phase": "collection", "outcome": "failed",
            })
        elif report.skipped:
            self.collection_errors += 1
            self.skipped += 1
            self.failure_details.append({
                "nodeid": str(report.nodeid), "phase": "collection", "outcome": "skipped",
            })

    def pytest_internalerror(self, excrepr: Any, excinfo: Any) -> None:
        self.error += 1
        self.failure_details.append({
            "nodeid": "", "phase": "internal", "outcome": "error",
        })

    def pytest_runtest_logreport(self, report: Any) -> None:
        nodeid = str(report.nodeid)
        if report.when == "call":
            if nodeid in self._called:
                self.error += 1
                self.failure_details.append({
                    "nodeid": nodeid, "phase": "call", "outcome": "duplicate_call_report",
                })
            else:
                self._called.add(nodeid)
                self.executed.append(nodeid)
        was_xfail = getattr(report, "wasxfail", None) is not None
        if was_xfail and nodeid not in self._xfail_nodes:
            self._xfail_nodes.add(nodeid)
            self.xfail += 1

        if report.failed:
            self.failed += 1
            self.failure_details.append({
                "nodeid": nodeid, "phase": str(report.when), "outcome": "failed",
            })
        elif report.skipped:
            self.skipped += 1
            self.failure_details.append({
                "nodeid": nodeid, "phase": str(report.when), "outcome": "skipped",
            })
        elif report.when == "call" and report.passed and not was_xfail:
            self.passed += 1

        # The frozen contract consumes only teardown report properties. The
        # e165.real_* and e165.mock_* counters remain separate fields.
        if report.when == "teardown":
            counts = self._teardown_properties.setdefault(nodeid, {})
            properties = list(getattr(report, "user_properties", ()) or ())
            for property_name in TEARDOWN_PROPERTIES:
                values = [value for key, value in properties if key == property_name]
                if len(values) > 1:
                    counts.pop(property_name, None)
                    self.property_errors.append({
                        "nodeid": nodeid, "property": property_name,
                        "phase": str(report.when), "reason": "duplicate user_property",
                    })
                    self.error += 1
                    continue
                if not values:
                    continue
                value = values[0]
                if type(value) is not int or value < 0:
                    counts.pop(property_name, None)
                    self.property_errors.append({
                        "nodeid": nodeid, "property": property_name,
                        "phase": str(report.when),
                        "reason": "value must be a nonnegative integer",
                    })
                    self.error += 1
                    continue
                counts[property_name] = value

    def pytest_sessionfinish(self, session: Any, exitstatus: Any) -> None:
        self.exit_status = int(exitstatus)

    def synthetic_counts(self, expected: list[str], envelope: dict[str, Any],
                         mock_call_declaration: dict[str, Any],
                         real_call_limits: dict[str, Any]) -> dict[str, Any]:
        source = envelope["actual_count_source"]
        result: dict[str, Any] = {
            "declared_envelope": dict(envelope),
            "mock_call_declaration": dict(mock_call_declaration),
            "real_call_limits": dict(real_call_limits),
            "actual_count_source": source,
            "property_capture_phase": "teardown",
        }
        upper_fields = {
            "full_training_oracle_evaluations": "full_training_oracle_evaluations_upper_bound",
            "complete_score_calls": "complete_score_calls_upper_bound",
        }

        def total(property_name: str) -> tuple[int | None, list[str], list[str]]:
            missing = [nodeid for nodeid in expected
                       if property_name not in self._teardown_properties.get(nodeid, {})]
            missing_calls = [nodeid for nodeid in expected if nodeid not in self._called]
            known = source == "pytest_user_properties" and not missing and not missing_calls
            value = (sum(self._teardown_properties[nodeid][property_name] for nodeid in expected)
                     if known else None)
            return value, missing, missing_calls

        def per_test(property_name: str) -> dict[str, int | None]:
            return {
                nodeid: self._teardown_properties.get(nodeid, {}).get(property_name)
                for nodeid in expected
            }

        def unknown_reason(missing_properties: list[str], missing_calls: list[str]) -> str | None:
            if source == "pytest_user_properties" and not missing_properties and not missing_calls:
                return None
            if source == "UNAVAILABLE":
                return "source_declared_unavailable"
            if missing_properties:
                return "teardown_property_missing_or_invalid"
            if missing_calls:
                return "expected_test_call_not_observed"
            return "source_not_recognized"

        for metric, upper_key in upper_fields.items():
            property_name = PROPERTY_FIELDS[metric]
            helper = METRIC_HELPERS[metric]
            real_property = REAL_HELPER_PROPERTIES[helper]
            mock_property = MOCK_HELPER_PROPERTIES[helper]
            actual, missing, missing_calls = total(property_name)
            real_actual, missing_real, missing_real_calls = total(real_property)
            mock_actual, missing_mock, missing_mock_calls = total(mock_property)
            consistency_unknown = (
                source != "pytest_user_properties"
                or bool(missing or missing_real or missing_real_calls or missing_calls)
            )
            mismatches = [
                nodeid for nodeid in expected
                if property_name in self._teardown_properties.get(nodeid, {})
                and real_property in self._teardown_properties.get(nodeid, {})
                and self._teardown_properties[nodeid][property_name]
                != self._teardown_properties[nodeid][real_property]
            ]
            for nodeid in mismatches:
                detail = {
                    "nodeid": nodeid,
                    "property": property_name,
                    "real_helper_property": real_property,
                    "property_value": self._teardown_properties[nodeid][property_name],
                    "real_helper_value": self._teardown_properties[nodeid][real_property],
                    "reason": "synthetic total differs from its real helper counter",
                }
                if detail not in self.property_consistency_errors:
                    self.property_consistency_errors.append(detail)
                    self.error += 1
            real_status = "KNOWN" if real_actual is not None else "UNKNOWN"
            mock_status = "KNOWN" if mock_actual is not None else "UNKNOWN"
            consistency_status = (
                "MISMATCH" if mismatches else
                "UNKNOWN" if consistency_unknown else
                "MATCH"
            )
            result[metric] = {
                "counter_kind": "synthetic_junit_teardown_property",
                "test_property_total": actual,
                "test_property_by_nodeid": per_test(property_name),
                "test_property_status": "KNOWN" if actual is not None else "UNKNOWN",
                "real_helper_calls_actual": real_actual,
                "real_helper_calls_by_nodeid": per_test(real_property),
                "real_helper_calls_status": real_status,
                "real_helper_property": real_property,
                "mock_helper_calls_actual": mock_actual,
                "mock_helper_calls_by_nodeid": per_test(mock_property),
                "mock_helper_calls_status": mock_status,
                "mock_helper_property": mock_property,
                "property_consistency_status": consistency_status,
                "property_consistency_mismatch_nodeids": mismatches,
                "unknown_reason": unknown_reason(missing, missing_calls),
                "missing_property_nodeids": missing,
                "missing_call_nodeids": missing_calls,
                "missing_real_helper_property_nodeids": missing_real,
                "missing_mock_helper_property_nodeids": missing_mock,
                "missing_real_helper_call_nodeids": missing_real_calls,
                "missing_mock_helper_call_nodeids": missing_mock_calls,
                "real_helper_unknown_reason": unknown_reason(missing_real, missing_real_calls),
                "mock_helper_unknown_reason": unknown_reason(missing_mock, missing_mock_calls),
                "property_name": property_name,
                "upper_bound": envelope[upper_key],
                "within_upper_bound": None if actual is None else actual <= envelope[upper_key],
            }
        additional_real = {}
        for helper, property_name in REAL_HELPER_PROPERTIES.items():
            value, missing, missing_calls = total(property_name)
            additional_real[helper] = {
                "property_name": property_name,
                "calls_actual": value,
                "calls_by_nodeid": per_test(property_name),
                "status": "KNOWN" if value is not None else "UNKNOWN",
                "missing_property_nodeids": missing,
                "missing_call_nodeids": missing_calls,
                "unknown_reason": unknown_reason(missing, missing_calls),
            }
        additional_mock = {}
        mock_upper_bound_fields = {
            "training_oracle": "training_oracle_calls_upper_bound",
            "stage0_scorer": "complete_score_calls_upper_bound",
        }
        for helper, property_name in MOCK_HELPER_PROPERTIES.items():
            value, missing, missing_calls = total(property_name)
            upper_bound = mock_call_declaration[mock_upper_bound_fields[helper]]
            additional_mock[helper] = {
                "property_name": property_name,
                "calls_actual": value,
                "calls_by_nodeid": per_test(property_name),
                "status": "KNOWN" if value is not None else "UNKNOWN",
                "missing_property_nodeids": missing,
                "missing_call_nodeids": missing_calls,
                "unknown_reason": unknown_reason(missing, missing_calls),
                "upper_bound": upper_bound,
                "within_upper_bound": None if value is None else value <= upper_bound,
            }
        real_fit_calls = {}
        for label, field in REAL_FIT_PROPERTY_FIELDS.items():
            property_name = real_call_limits[field]
            value, missing, missing_calls = total(property_name)
            upper_bound = real_call_limits["fit_calls_upper_bound"]
            real_fit_calls[label] = {
                "property_name": property_name,
                "calls_actual": value,
                "calls_by_nodeid": per_test(property_name),
                "status": "KNOWN" if value is not None else "UNKNOWN",
                "missing_property_nodeids": missing,
                "missing_call_nodeids": missing_calls,
                "unknown_reason": unknown_reason(missing, missing_calls),
                "upper_bound": upper_bound,
                "within_upper_bound": None if value is None else value <= upper_bound,
            }
        result["additional_real_helper_calls"] = additional_real
        result["mock_helper_calls"] = additional_mock
        result["real_fit_calls"] = real_fit_calls
        result["property_consistency_errors"] = list(self.property_consistency_errors)
        return result


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run only the authorized E165 CHECKS suite")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--authorization", required=True)
    return parser.parse_args(argv)


def _suite_record(context: common.PhaseContext, plugin: _ChecksPlugin,
                  exit_code: int, synthetic_counts: dict[str, Any],
                  basetemp_preflight_passed: bool) -> dict[str, Any]:
    expected = list(context.spec["expected_test_nodeids"])
    nodeids_match = plugin.collected == expected and plugin.executed == expected
    error_count = plugin.error
    if not nodeids_match:
        error_count += 1
    required_counter_unknowns: list[str] = []
    required_counter_violations: list[str] = []
    for metric in PROPERTY_FIELDS:
        row = synthetic_counts[metric]
        for field in ("test_property_status", "real_helper_calls_status",
                      "mock_helper_calls_status"):
            if row[field] != "KNOWN":
                required_counter_unknowns.append(f"{metric}.{field}")
        if row["property_consistency_status"] == "UNKNOWN":
            required_counter_unknowns.append(f"{metric}.property_consistency_status")
        elif row["property_consistency_status"] == "MISMATCH":
            required_counter_violations.append(f"{metric}.property_consistency_status")
        if row["within_upper_bound"] is None:
            required_counter_unknowns.append(f"{metric}.within_upper_bound")
        elif row["within_upper_bound"] is False:
            required_counter_violations.append(f"{metric}.within_upper_bound")
    for helper, row in synthetic_counts["mock_helper_calls"].items():
        if row["status"] != "KNOWN":
            required_counter_unknowns.append(f"mock_helper_calls.{helper}")
        if row["within_upper_bound"] is None:
            required_counter_unknowns.append(f"mock_helper_calls.{helper}.within_upper_bound")
        elif row["within_upper_bound"] is False:
            required_counter_violations.append(f"mock_helper_calls.{helper}.within_upper_bound")
    for helper, row in synthetic_counts["real_fit_calls"].items():
        if row["status"] != "KNOWN":
            required_counter_unknowns.append(f"real_fit_calls.{helper}")
        if row["within_upper_bound"] is None:
            required_counter_unknowns.append(f"real_fit_calls.{helper}.within_upper_bound")
        elif row["within_upper_bound"] is False:
            required_counter_violations.append(f"real_fit_calls.{helper}.within_upper_bound")
    synthetic_over_bound = bool(required_counter_violations)
    if required_counter_unknowns:
        error_count += 1
    if synthetic_over_bound:
        error_count += 1
    if (exit_code != 0 and error_count == plugin.error and plugin.failed == 0
            and plugin.skipped == 0 and plugin.collection_errors == 0):
        error_count += 1
    return {
        "format": "e165_checks_result_v1",
        "stage": "CHECKS",
        "source_identity_sha256": context.source_identity_sha256,
        "pytest_target": list(context.spec["pytest_target"]),
        "expected_test_nodeids": expected,
        "collected_test_nodeids": list(plugin.collected),
        "executed_test_nodeids": list(plugin.executed),
        "nodeids_match_exactly": nodeids_match,
        "pytest_exit_code": int(exit_code),
        "pytest_plugin_autoload_disabled": True,
        "pytest_cacheprovider_disabled": True,
        "pytest_noconftest": True,
        "pytest_config": str(context.spec["pytest_config"]),
        "pytest_basetemp_path": str(common.PYTEST_BASETEMP),
        "pytest_basetemp_preflight_passed": basetemp_preflight_passed,
        "junit_xml_path": str(context.run_root / "junit.xml"),
        "rootdir": str(context.source_cwd),
        "confcutdir": str(context.source_cwd),
        "passed": plugin.passed,
        "failed": plugin.failed,
        "error": error_count,
        "errors": error_count,
        "skipped": plugin.skipped,
        "xfail": plugin.xfail,
        "collection_errors": plugin.collection_errors,
        "failure_details": list(plugin.failure_details),
        "synthetic_property_errors": list(plugin.property_errors),
        "synthetic_property_consistency_errors": list(plugin.property_consistency_errors),
        "synthetic_calls": synthetic_counts,
        "required_call_counters_unknown": required_counter_unknowns,
        "required_call_counter_violations": required_counter_violations,
        "required_call_counters_status": (
            "VIOLATION" if required_counter_violations else
            "UNKNOWN" if required_counter_unknowns else
            "CHECKED_WITHIN_LIMITS"
        ),
        "synthetic_upper_bound_status": (
            "VIOLATION" if synthetic_over_bound else
            "CHECKED_WITHIN_UPPER_BOUNDS" if not required_counter_unknowns else
            "UNKNOWN_DECLARED_UPPER_BOUND_ONLY"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        context = common.load_context("CHECKS", args.spec, args.authorization)
    except common.PhaseError as exc:
        print(f"E165 CHECKS precondition failed: {exc}", file=sys.stderr)
        return 2

    try:
        environment = common.environment_record(context)
        common.write_json(context, "environment.json", environment)
    except common.PhaseError as exc:
        print(f"E165 CHECKS environment validation failed: {exc}", file=sys.stderr)
        return 2

    # pytest reads this before import; no auto-loaded third-party plugin runs.
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    plugin = _ChecksPlugin()
    pytest_exception: str | None = None
    exit_code = 1
    basetemp_preflight_passed = False
    try:
        import pytest

        targets = list(context.spec["pytest_target"])
        pytest_config = context.spec["pytest_config"]
        junit_path = common.safe_project_path(
            str(context.run_root / "junit.xml"), root=context.run_root,
            must_exist=False, directory=False,
        )
        if junit_path.exists():
            raise common.PhaseError("refusing to overwrite an existing CHECKS junit.xml")
        basetemp_path = common.check_pytest_basetemp()
        basetemp_preflight_passed = True
        pytest_args = [
            "-p", "no:cacheprovider",
            "-c", pytest_config,
            f"--rootdir={context.source_cwd}",
            f"--confcutdir={context.source_cwd}",
            "--noconftest",
            f"--basetemp={basetemp_path}",
            f"--junitxml={junit_path}",
            *targets,
        ]
        exit_code = int(pytest.main(pytest_args, plugins=[plugin]))
    except Exception as exc:
        pytest_exception = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        plugin.error += 1
        plugin.failure_details.append({
            "nodeid": "", "phase": "pytest_entry", "outcome": pytest_exception,
        })

    expected = list(context.spec["expected_test_nodeids"])
    envelope = dict(context.spec["synthetic_call_envelope"])
    synthetic = plugin.synthetic_counts(
        expected, envelope,
        dict(context.spec["mock_call_declaration"]),
        dict(context.spec["real_call_limits"]),
    )
    suite = _suite_record(context, plugin, exit_code, synthetic,
                          basetemp_preflight_passed)
    if pytest_exception is not None:
        suite["pytest_exception"] = pytest_exception

    state = common.base_state(context)
    state.update({
        "collected_test_nodeids": list(plugin.collected),
        "executed_test_nodeids": list(plugin.executed),
        "passed": plugin.passed,
        "failed": plugin.failed,
        "error": suite["error"],
        "errors": suite["errors"],
        "skipped": plugin.skipped,
        "xfail": plugin.xfail,
        "collection_errors": plugin.collection_errors,
        "synthetic_calls": synthetic,
    })
    complete = (
        exit_code == 0
        and suite["nodeids_match_exactly"]
        and suite["failed"] == 0
        and suite["error"] == 0
        and suite["errors"] == 0
        and suite["skipped"] == 0
        and suite["xfail"] == 0
        and suite["collection_errors"] == 0
        and suite["passed"] == len(expected)
        and suite["required_call_counters_status"] == "CHECKED_WITHIN_LIMITS"
        and suite["synthetic_upper_bound_status"] != "VIOLATION"
    )
    status = "COMPLETE" if complete else "INVALID"
    try:
        common.write_json(context, "checks.json", suite)
        common.finish_state(context, state, status)
    except common.PhaseError as exc:
        print(f"E165 CHECKS output validation failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps({
        "stage": "CHECKS", "status": status,
        "collected": len(plugin.collected), "executed": len(plugin.executed),
        "passed": plugin.passed, "failed": plugin.failed,
        "errors": suite["errors"], "skipped": plugin.skipped,
        "xfail": plugin.xfail,
    }, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
