# README Audit — tensor-spline

**Date:** 2026-05-17 | **Reviewer:** Forgemaster ⚒️

## Scores

| Criterion | Score | Notes |
|-----------|:-----:|-------|
| WHAT it is | ✅ | "Compressed neural network layers — Eisenstein lattice splines and low-rank factorization" |
| WHY you'd use it | ✅ | Honest Findings section + compression table make the tradeoffs clear |
| HOW to install | ✅ | Has `pip install tensor-spline` |
| HOW to use (code) | ✅ | Three usage options + recommend_variant() is great |
| Links / context | ❌ | No links to plato-training (primary consumer), eisenstein crate, or SuperInstance org |

**Total: 4/5**

## Issues

1. **No ecosystem links.** tensor-spline feeds into plato-training for micro model compression. Should link to eisenstein (mathematical foundation) and plato-training (primary consumer).

## Action Taken

- ✅ Links section added to README
