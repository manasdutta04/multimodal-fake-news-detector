# Explainability

Explainability is a core deliverable, not a polish pass. Wire it early (Module 1–3) and **verify faithfulness** (Module 4).

## Required outputs per prediction

| Level | Technique (v1) | User-facing output |
|-------|----------------|--------------------|
| Text | SHAP on classical model; attention or SHAP on transformer | Highlighted words/phrases |
| Image | Grad-CAM on last conv layer (`pytorch-grad-cam`) | Heatmap overlay on image |
| Fusion | Cosine similarity and/or cross-attention weights | Agreement score + one short NL sentence |

## Natural-language agreement line (rule-based is OK for v1)

Examples of intent (adapt wording to real scores):

- Low agreement + image-dominant → flag mismatch between claim and image
- Low agreement + text-dominant → flag wording/style cues
- High agreement → note that modalities support the same decision

Never treat agreement score alone as the verdict — always pair with the classifier prediction and confidence.

## Faithfulness (must implement)

1. **Text deletion:** remove/mask top-k important tokens; confidence should drop if the explanation is faithful
2. **Image deletion:** blank/blur top Grad-CAM region; re-score
3. Plot deletion curves (steep early drop = better explanation)
4. Test faithfulness on **correct and incorrect** predictions

Pretty heatmaps that do not change model confidence when removed are a failure mode — call that out in reports.

## Pitfalls to avoid

- Generating Grad-CAM from the classifier layer instead of the last convolutional layer
- Different tokenization/preprocessing at explain time vs train time
- Reporting only qualitative screenshots with no deletion/ablation numbers
- Explaining each modality but never the fusion decision

## Logging

Prefer logging prediction + explanation artifacts (ids, scores, top tokens, agreement) for audit — aligns with FR7 in the PRD.
