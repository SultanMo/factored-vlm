# FACTOR-Bench Schema (v1.0)

`factor_bench.json` has a metadata header
(`name, version, description, paper, license, image_source, n_samples, splits,
operators, owlvit_validated`) plus a `samples` array. `factor_bench.jsonl`
contains the samples only, one JSON object per line (HuggingFace / streaming).

## Core sample fields (present in every sample)

| field | type | description |
|---|---|---|
| `sample_id` | str | unique identifier |
| `test_type` | str | `operator` \| `equivalence` \| `compound` |
| `operator` | str | logical operator (see table below) |
| `image_id` | int | COCO image id |
| `image` | str | COCO-relative path, e.g. `val2017/000000233825.jpg` |
| `caption_a` | str | first caption |
| `caption_b` | str | second caption |
| `correct` | str | `a` \| `b` \| `both` (`both` only for `equivalence`) |
| `parse_a` | object | oracle parse of `caption_a` (below) |
| `parse_b` | object | oracle parse of `caption_b` |
| `meta` | object | type-specific metadata (below) |

### `parse_a` / `parse_b`
```json
{"concepts": [{"text": "orange", "is_negated": false}], "operator": "SINGLE"}
```
The ground-truth logical reading of the caption (`operator` ∈ `SINGLE|AND|OR|NONE`).
Embedding parses lets evaluation be made **independent of any text parser**.

### `meta` (nullable per type)
| key | description |
|---|---|
| `difficulty` | `easy` \| `medium` \| `hard` |
| `position_swapped` | bool: answer-position randomization (anti-position-bias) |
| `owlvit_validated` | bool: always `true` (all samples validated) |
| `concepts` | list: all concepts involved in the sample |
| `present` / `absent` | lists: image ground-truth concept presence/absence (where determined) |
| `negation_type` | `NOT` samples: `explicit` / `implicit` / ... |
| `template_family` | template variant id |
| `polarity` | balanced-polarity label |
| `equivalence_type` | `equivalence` samples: `de_morgan` / `double_negation` / `commutativity` |
| `n_concepts`, `pattern` | `compound` samples |

## Operators

| `operator` | split | paper symbol | image condition |
|---|---|---|---|
| `NOT` | operator | ¬A | A absent |
| `AND` | operator | A∧B | one present, one absent |
| `OR` | operator | A∨B | one present, one absent |
| `BUT_NOT` | operator | A∧¬B | A present, B absent |
| `NEITHER` | operator | ¬A∧¬B | both absent |
| `DE_MORGAN` / `DOUBLE_NEG` / `COMM_AND` / `COMM_OR` | equivalence | — | two logically equivalent captions, `correct="both"` |
| `COMPOUND` | compound | mixed | 3–4 concepts, mixed polarity |

## Scoring
Two-alternative forced choice: score `caption_a` and `caption_b` for the image,
predict the higher, compare to `correct`. For the `equivalence` split
(`correct="both"`), a faithful scorer gives both captions near-equal scores,
so measure the violation rate instead (see `eval.py: evaluate_equivalence`).
