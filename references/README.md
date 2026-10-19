# Citation notes

The bibliography contains 49 public references used by the baseline and related-work map in [the project note](../docs/BASELINES_AND_RELATED_WORK.md), the online-learning reading notes, and the software/workflow notes. Entries use the publisher or conference record when one is available, and the arXiv record for preprints or work without a formal proceedings entry. The original 25 records were checked on 2026-10-14; three C1 supplement entries (EWC, Progress & Compress, LoRA-XS) were checked on 2026-10-15. Three public arXiv correspondences for the twelve user-provided PDFs were checked on 2026-10-18. URLs point to the record used to verify the entry; version-specific metadata decisions are listed below.

## Metadata decisions

- **Nair2020AWAC** cites arXiv:2006.09359, the correct record for AWAC. **Ball2023RLPD** cites the ICML 2023 paper at PMLR 202:1577–1594, arXiv:2302.02948. The similar-looking arXiv identifier 2302.12639 is unrelated and is not used.
- **Nakamoto2023CalQL** follows the NeurIPS 2023 proceedings author list, which spells the second author's name “Simon Zhai.” The arXiv record (2303.05479) and the authors' project repository spell it “Yuexiang Zhai.” The proceedings spelling is used in the bibliography; the arXiv identifier is included to make the record easy to find.
- **Bossens2024LifetimePolicyReuse** gives the journal issue year, 2024; the article first appeared online in 2023.
- **Cannon2024HOP** is an arXiv preprint. Its arXiv comments say that it was accepted as a poster at the NeurIPS IMOL workshop; it is not labeled as a NeurIPS main-track paper.
- **Silver2025EraExperience** is a 2025 preprint of a chapter announced for *Designing an Intelligence* (MIT Press). It is a research agenda/position piece, not an experimental result or theorem.
- **Kassis2026ScientificAgentSkills** uses the unversioned arXiv identifier/DOI; metadata checked on 2026-10-16 identifies current version 2 (revised 2026-09-02). The project cites it only as procedural software/workflow context.
- **Hunter2007Matplotlib** credits the plotting library separately, following [Matplotlib's official citation guidance](https://matplotlib.org/stable/project/citing.html), checked 2026-10-16. It is distinct from the procedural workflow citation.

The user's notes mention an unpublished supervisor argument. It is not included in this public bibliography, and this document does not reproduce, attribute, or assess private proof material.

## Citation use

### Twelve attached online-learning manuscripts (2026-10-18)

[Synthesis](../docs/literature/20261018_SYNTHESIS.md) and [source inventory](../docs/literature/20261018_sources.json) cover all twelve attachment titles, page counts, current local locations and hashes. The public records verified in this session are **Li2026ImplicitTargetShift** (arXiv:2605.07886, current v2, 2026-10-02), **Voitovych2026SmoothedOnline** (arXiv:2610.10499, v1, 2026-10-07), and **Lu2026OPTSTTPO** (arXiv:2609.40035, v1, 2026-09-30). Analysis uses the attachments, including their appendices; public versions can differ in pagination and content. The other nine exact public correspondences were not verified in this session. They remain identified by their attachment titles rather than invented arXiv IDs or author lists. This is not evidence that no public versions exist. Anonymous review headers do not establish conference acceptance.

The linked papers provide context for algorithms and evaluation design. Their results were obtained on their own tasks and protocols. They are not performance baselines for the custom navigation runs unless the same method is implemented and evaluated under the project's frozen environment, data, interaction budget, state accounting, and seeds.

## C1 supplement attribution

`Kirkpatrick2017EWC` identifies the original importance-weighted parameter
regularizer; the experiment implements a task-free empirical-Fisher PPO
adaptation. `Schwarz2018ProgressCompress` identifies acquisition/consolidation
context; that algorithm was not run. `Balazy2024LoRAXS` identifies a close
frozen-basis small-middle-matrix adapter precedent (arXiv v3 revised 2025-08-19).
Our source-output mixers are C1 policy adapters, not its Transformer training
setup. CLEAR's correct preprint is arXiv:1811.11682; PPO behavioral replay is
only a partial inspired control. None of these custom comparisons constitutes
a faithful reproduction of published SOTA.

## Online knowledge-structure design review

Twelve additional primary references were checked on 2026-10-15 for `docs/ONLINE_KNOWLEDGE_STRUCTURE_DESIGN.md`: skill/state trees, compositional lifelong RL, task-agnostic adaptation, structure learning, rate-distortion, shared-gradient proposals and baseline-aware improvement. They are design context and proposed comparisons, not newly executed baselines. The cited primary texts show substantial overlap with individual mechanisms; this review does not certify field-wide novelty. No C2/C3 experiment was run.

## Shared contrasts and selected feedback (2026-10-19)

Three primary sources were added for the [shared-baseline and action-contrast note](../docs/literature/20261019_ACTION_CONTRAST_NOTES.md). The opening bibliography count predates these additions; the bibliography now contains 52 entries.

- **Nie and Wager (2017),** [*Quasi-Oracle Estimation of Heterogeneous Treatment Effects*](https://arxiv.org/abs/1712.04912): source for the R-learner residualization template. Its independent-sample theorem is not an online guarantee for these repeated-write streams.
- **Dudík, Langford, and Li (2011),** [*Doubly Robust Policy Evaluation and Learning*](https://icml.cc/2011/papers/554_icmlpaper.pdf), also [arXiv:1103.4601](https://arxiv.org/abs/1103.4601): source for chosen-action-only contextual-bandit feedback and doubly robust policy-value estimation. It does not supply whole-trajectory evaluation for the project setting.
- **Kallus and Zhou (2022),** [*Stateful Offline Contextual Policy Evaluation and Learning*](https://proceedings.mlr.press/v151/kallus22a.html): source for a stateful contextual-bandit reduction under specific exogenous-arrival and response-structure assumptions. The repeated-write setting has action-dependent future learner state, so that reduction is not presumed.

These papers provide design context only. No experiment or new environment interaction was run for this note.
