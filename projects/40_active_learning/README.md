# Phase 4 — Active Learning

## Objective
Understand how data selection impacts learning efficiency.

## Scope
- Pool-based active learning
- Applied to the detection task from Phase 1

## Key concepts to learn
- Uncertainty estimation
- Sampling strategies (entropy, margin, disagreement)
- Data efficiency vs random sampling
- Interaction between model confidence and data quality

## Done when
- Active selection outperforms random sampling in at least one setting
- Selection logic is clearly implemented and reasoned about
- Tradeoffs and failure modes are documented

## Action items
- Reuse the Phase 1 detector and dataset split.
- Implement at least two selection strategies (e.g., entropy, margin).
- Compare learning curves vs random sampling.
- Document failure cases and when selection hurts performance.
