from __future__ import annotations

import unittest

import numpy as np

from scripts import run_knowledge_causal_ablation as harness
from scripts import run_knowledge_loop as loop
from scripts import knowledge_v2_components


class KnowledgeCausalAblationTests(unittest.TestCase):
    def test_oracle_route_uses_external_context_and_not_observation(self) -> None:
        oracle = harness.OracleHarness()
        with self.assertRaises(RuntimeError):
            oracle.route_before_action(np.zeros(4, dtype=np.float32))

        context_names = tuple(loop.C1_CONTEXTS)
        oracle.set_external_context(context_names[2], step=0)
        first = oracle.route_before_action(np.zeros(4, dtype=np.float32))
        second = oracle.route_before_action(np.ones(4, dtype=np.float32))
        self.assertEqual(first, second)
        self.assertEqual(first["slot"], 2)
        self.assertEqual(first["confidence"], 1.0)
        self.assertEqual(first["status"], "privileged_oracle")

        oracle.set_external_context(context_names[4], step=1)
        changed = oracle.route_before_action(np.zeros(4, dtype=np.float32))
        self.assertEqual(changed["slot"], 4)
        self.assertEqual(changed["reason"], "external_context_before_action")

    def test_learned_router_receives_observation_without_oracle_label(self) -> None:
        class ProbeRouter:
            def __init__(self) -> None:
                self.received = None

            def route(self, observation):
                self.received = np.asarray(observation).copy()
                return {"slot": -1, "confidence": 0.0, "status": "unresolved"}

        router = ProbeRouter()
        probe = type("Probe", (), {"router": router})()
        oracle = harness.OracleHarness()
        oracle.set_external_context(tuple(loop.C1_CONTEXTS)[3], step=0)
        observation = np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

        route = loop.OnlineArm.route_before_action(probe, observation)

        np.testing.assert_array_equal(router.received, observation)
        self.assertEqual(route["slot"], -1)
        self.assertEqual(route["status"], "unresolved")
        self.assertEqual(oracle.route_calls, 0)

    def test_scoped_runtime_restores_globals_after_failure(self) -> None:
        original_arm = loop.OnlineArm
        original_lookup = loop.hidden_context_at
        original_budget = loop.ACTOR_BUDGET_BYTES
        original_codecs = loop.CODEC_FOR_ARM

        with self.assertRaisesRegex(RuntimeError, "deliberate scope exit"):
            with harness.scoped_runtime("oracle_dense") as oracle:
                self.assertIsNotNone(oracle)
                self.assertIsNot(loop.OnlineArm, original_arm)
                self.assertIsNot(loop.hidden_context_at, original_lookup)
                self.assertEqual(loop.ACTOR_BUDGET_BYTES, harness.DENSE_BUDGET_BYTES)
                self.assertEqual(loop.CODEC_FOR_ARM["int8"], "dense_f32")
                raise RuntimeError("deliberate scope exit")

        self.assertIs(loop.OnlineArm, original_arm)
        self.assertIs(loop.hidden_context_at, original_lookup)
        self.assertEqual(loop.ACTOR_BUDGET_BYTES, original_budget)
        self.assertIs(loop.CODEC_FOR_ARM, original_codecs)

    def test_trial_preservation_condition_selects_shared_fix_only(self) -> None:
        original_lookup = loop.hidden_context_at
        original_budget = loop.ACTOR_BUDGET_BYTES
        original_codecs = loop.CODEC_FOR_ARM

        with harness.scoped_runtime("learned_int8_preserve_trial") as oracle:
            self.assertIsNone(oracle)
            self.assertIs(loop.OnlineArm, knowledge_v2_components.PreservingTrialArm)
            self.assertIs(loop.hidden_context_at, original_lookup)
            self.assertEqual(loop.ACTOR_BUDGET_BYTES, harness.INT8_BUDGET_BYTES)
            self.assertIs(loop.CODEC_FOR_ARM, original_codecs)

        self.assertIs(loop.OnlineArm, knowledge_v2_components.PreservingTrialArm.__bases__[0])
        self.assertIs(loop.hidden_context_at, original_lookup)
        self.assertEqual(loop.ACTOR_BUDGET_BYTES, original_budget)
        self.assertIs(loop.CODEC_FOR_ARM, original_codecs)

    def test_codec_sizes_match_declared_diagnostic_conditions(self) -> None:
        zeros = np.zeros((6, 27), dtype=np.float32)
        int8 = loop.encode(zeros, "int8", harness.INT8_BUDGET_BYTES)
        dense = loop.encode(zeros, "dense_f32", harness.DENSE_BUDGET_BYTES)

        self.assertEqual(int8.nbytes, 202)
        self.assertEqual(dense.nbytes, 664)
        self.assertEqual(harness.condition_patch("learned_dense")["actor_budget_bytes"], 664)
        self.assertEqual(harness.condition_patch("oracle_int8")["actor_budget_bytes"], 320)
        self.assertFalse(harness.condition_patch("oracle_int8")["resource_comparison_fair"])


if __name__ == "__main__":
    unittest.main()
