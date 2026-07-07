# FACTOR-Bench Data Card

**FACTOR-Bench** is a diagnostic benchmark that measures whether a
vision-language *scoring interface* executes **Boolean operators** (negation,
conjunction, disjunction, exclusion, NOR) or merely tracks which concepts
are mentioned.

- **Version:** 1.0
- **Paper:** *Similarity Is Not Logic: Factored Inference for Dual-Encoder Vision-Language Models*, ICML 2026.
- **License:** Annotations under **CC-BY-4.0**. Images are from **COCO val2017** under their original terms (not redistributed here).
- **Size:** 1,695 samples (one image + two contrastive captions each).

## Task
Each sample is a **two-alternative forced choice**: given one image and two
captions that share the same concepts but differ in logical structure, choose the
caption that is true of the image.

| field | value |
|---|---|
| `image` | `val2017/000000233825.jpg` (orange absent) |
| `caption_a` | "an orange" |
| `caption_b` | "there is no orange" |
| `correct` | `b` |

The key design is **operator-contrastive pairs**: captions with the same concepts
but different operators (e.g. "car or bicycle" vs "car and bicycle"), so a model
that scores them alike reveals bag-of-concepts behavior.

## Composition
| Split | n | Operators |
|---|---|---|
| `operator` | 1,100 | NOT (300), AND (150), OR (150), BUT_NOT (250), NEITHER (250) |
| `equivalence` | 450 | DE_MORGAN (50), DOUBLE_NEG (100), COMM_AND (150), COMM_OR (150), all `correct="both"` |
| `compound` | 145 | COMPOUND, 3–4 concepts, mixed polarity |

Operator → paper symbol: NOT=¬A, AND=A∧B, OR=A∨B, BUT_NOT=A∧¬B, NEITHER=¬A∧¬B.

## Collection & validation
- Built from **COCO val2017** over the 80 COCO object categories. The `person` category is excluded as near-ubiquitous.
- Ground-truth concept presence/absence is **validated with OWL-ViT** open-vocabulary detection. Samples where the detector contradicts COCO annotations were rejected, so **all 1,695 samples are OWL-ViT-validated**.
- Captions use **minimal templates** (5+ phrasings per operator) to isolate operator semantics from language complexity.
- Each sample embeds **oracle parses** so evaluation can be made parser-independent.

## Anti-shortcut design
- **Balanced answer position**: within the operator split `correct` is a/b ≈ 50/50 (548/552). `meta.position_swapped` records the randomization.
- **Balanced polarity** on negation (≈50% negation-correct) defeats text-only "always prefer affirmation/negation" baselines.
- **Template diversity** (5+ phrasings/operator) prevents lexical memorization.

## Intended use
Diagnostic evaluation of image–text **scoring/matching interfaces** (CLIP/SigLIP-style
dual encoders, learned re-rankers, etc.) for Boolean operator semantics. It is an
evaluation set, not a training set.

## Limitations
- Scoped to **Boolean operators over object concepts**. It does not cover attribute binding, spatial relations, or counting.
- Concepts are COCO's 80 categories, and the image domain is natural photographs (COCO).
- Ground truth depends on COCO annotations and OWL-ViT agreement.

## Getting the images
FACTOR-Bench distributes captions, labels, parses, and **COCO image IDs**,
not the images themselves. Download COCO **val2017** from
<https://cocodataset.org/#download> and point your evaluator's `--image-root` at
the directory containing the `val2017/` folder.

## Citation
```bibtex
@inproceedings{alshehri2026similarity,
  title     = {Similarity Is Not Logic: Factored Inference for Dual-Encoder Vision-Language Models},
  author    = {Alshehri, Sultan and Yang, Zhantao and Zhang, Han and Savvides, Marios},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
