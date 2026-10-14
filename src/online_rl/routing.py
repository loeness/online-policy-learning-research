"""Causal, bounded router based only on observed action responses.

This pilot exploits a planar actuator-response statistic: dot and cross products
between source-data action responses and observed movement. This is a specific
inductive bias, not a universal POMDP history encoder or a hidden-context label.
"""
from __future__ import annotations

from collections import deque
import numpy as np

from .policy import HISTORY_DIM, MAX_NODES


class HistoryRouter:
    def __init__(self, source_responses, window=16, novelty_threshold=.3,
                 minimum_events=5, confidence_margin=.12, max_nodes=MAX_NODES):
        self.source_responses = np.asarray(source_responses, np.float32)
        self.window = window
        self.events = deque(maxlen=window)
        self.threshold = novelty_threshold
        self.minimum_events = minimum_events
        self.margin = confidence_margin
        self.max_nodes = max_nodes
        self.prototypes = [np.array([1., 0.], np.float32)]
        self.counts = [1]
        self.recurs = [0]
        self.active = 0
        self.confidence = 0.
        self.novel_streak = 0
        self.creation_count = 0
        self.last_entered = 0
        self.pending_reset = None
        self.boundary_events = 0

    def clear_working_memory(self):
        self.events.clear()
        self.active = 0
        self.confidence = 0.
        self.novel_streak = 0
        self.last_entered = 0

    def observe(self, before, action, after):
        displacement = (np.asarray(after[:2]) - np.asarray(before[:2])) * 8
        norm = float(np.linalg.norm(displacement))
        if norm < .5:  # A wall collision has no directional identification signal.
            self.boundary_events += 1
            return
        movement = displacement / norm
        reference = self.source_responses[int(action)]
        reference = reference / max(float(np.linalg.norm(reference)), 1e-6)
        feature = np.array([reference @ movement,
                            reference[0] * movement[1] - reference[1] * movement[0]], np.float32)
        self.events.append((feature, int(action)))
        self._select()

    def _select(self):
        if len(self.events) < self.minimum_events:
            self.confidence = 0.
            return
        features = np.stack([e[0] for e in self.events])
        mean = features.mean(0)
        scores = np.array([float(np.square(mean - p).sum() / 2) for p in self.prototypes])
        order = np.argsort(scores)
        best = int(order[0])
        gap = float(scores[order[1]] - scores[best]) if len(order) > 1 else 1.
        if scores[best] > self.threshold:
            self.novel_streak += 1
            self.confidence = 0.
            if self.novel_streak >= 3:
                # Use recent observations to limit a mixed old/new transition window.
                recent = np.stack([e[0] for e in list(self.events)[-self.minimum_events:]])
                prototype = recent.mean(0)
                if float(np.square(recent - prototype).mean()) < .18:
                    # A mixed old/new window can look novel even when its stable
                    # recent tail is a known context. Recheck the actual proposal
                    # before allocating persistent knowledge.
                    recent_scores = np.array([float(np.square(prototype - p).sum() / 2)
                                              for p in self.prototypes])
                    match = int(recent_scores.argmin())
                    if recent_scores[match] <= self.threshold:
                        best = match
                        if best != self.active:
                            self.recurs[best] += 1
                            self.pending_reset = best
                    elif len(self.prototypes) < self.max_nodes:
                        self.prototypes.append(prototype)
                        self.counts.append(len(recent))
                        self.recurs.append(0)
                        best = len(self.prototypes) - 1
                        self.creation_count += 1
                    else:
                        return
                    self.active = best
                    self.last_entered = best
                    self.events = deque(list(self.events)[-self.minimum_events:], maxlen=self.window)
                    self.novel_streak = 0
                    self.confidence = .8
            return
        self.novel_streak = 0
        if gap < self.margin:
            self.confidence = 0.
            return
        self.confidence = float(max(0., 1 - scores[best] / self.threshold))
        if best != self.active:
            self.active = best
            if self.counts[best] > 1:
                self.recurs[best] += 1
            self.pending_reset = best
            self.last_entered = best
        # Do not adapt a prototype toward ambiguous/transitional observations.
        if scores[best] < .04 and gap >= self.margin:
            self.prototypes[best] = .98 * self.prototypes[best] + .02 * mean
            self.counts[best] += 1

    def route(self):
        return int(self.active if self.confidence >= .35 else 0)

    def summary(self):
        result = np.zeros(HISTORY_DIM, np.float32)
        if self.events:
            x = np.stack([e[0] for e in self.events])
            result[:2] = x.mean(0)
            result[2:4] = x.std(0)
            for _, a in self.events:
                result[4 + a] += 1 / self.window
        return result

    def snapshot(self):
        return {"window": self.window, "events": [(x.tolist(), a) for x, a in self.events],
                "source_responses": self.source_responses.tolist(),
                "prototypes": [p.tolist() for p in self.prototypes], "counts": self.counts,
                "recurs": self.recurs, "active": self.active, "confidence": self.confidence,
                "creation_count": self.creation_count, "novel_streak": self.novel_streak,
                "pending_reset": self.pending_reset, "last_entered": self.last_entered,
                "boundary_events": self.boundary_events, "threshold": self.threshold,
                "minimum_events": self.minimum_events, "margin": self.margin,
                "max_nodes": self.max_nodes}
