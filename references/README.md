# Citation notes

The bibliography contains 40 public references used by the baseline and related-work map in [the project note](../docs/BASELINES_AND_RELATED_WORK.md). Entries use the publisher or conference record when one is available, and the arXiv record for preprints or work without a formal proceedings entry. The original 25 records were checked on 2026-10-14; three C1 supplement entries (EWC, Progress & Compress, LoRA-XS) were checked on 2026-10-15. URLs point to the record used to verify the entry; arXiv eprint fields identify the cited preprint version.

## Metadata decisions

- **Nair2020AWAC** cites arXiv:2006.09359, the correct record for AWAC. **Ball2023RLPD** cites the ICML 2023 paper at PMLR 202:1577–1594, arXiv:2302.02948. The similar-looking arXiv identifier 2302.12639 is unrelated and is not used.
- **Nakamoto2023CalQL** follows the NeurIPS 2023 proceedings author list, which spells the second author's name “Simon Zhai.” The arXiv record (2303.05479) and the authors' project repository spell it “Yuexiang Zhai.” The proceedings spelling is used in the bibliography; the arXiv identifier is included to make the record easy to find.
- **Bossens2024LifetimePolicyReuse** gives the journal issue year, 2024; the article first appeared online in 2023.
- **Cannon2024HOP** is an arXiv preprint. Its arXiv comments say that it was accepted as a poster at the NeurIPS IMOL workshop; it is not labeled as a NeurIPS main-track paper.
- **Silver2025EraExperience** is a 2025 preprint of a chapter announced for *Designing an Intelligence* (MIT Press). It is a research agenda/position piece, not an experimental result or theorem.
- **Kassis2026ScientificAgentSkills** cites arXiv version 2 (revised 2026-09-02). Its authors report no task-level evaluation; the project cites it only as procedural software/workflow context.

The user's notes mention an unpublished supervisor argument. It is not included in this public bibliography, and this document does not reproduce, attribute, or assess private proof material.

## Citation use

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
