#!/usr/bin/env python3
"""
FACTOR-Bench evaluation harness.

Plug in any image-text scorer and get per-operator accuracy. Two scorer
signatures are supported (auto-detected by parameter count):

    score_fn(image_path, caption) -> float                 # simple / holistic
    score_fn(image_path, sample, which) -> float           # structured (uses parse)

`sample` is the FACTOR-Bench record (with `parse_a`/`parse_b`) and `which` is
"a" or "b". Higher score = better image-caption match.

Quickstart (holistic):
    from eval import load, evaluate, open_clip_scorer
    samples = load("data/factor_bench.jsonl")
    score = open_clip_scorer("ViT-B-32", "openai")
    print(evaluate(score, samples, image_root="/path/to/coco", split="operator"))

Quickstart (LCSE - uses the oracle parses embedded in the benchmark):
    from eval import load, evaluate, lcse_scorer
    samples = load("data/factor_bench.jsonl")
    score = lcse_scorer("ViT-B-32", "openai")
    print(evaluate(score, samples, image_root="/path/to/coco", split="operator"))

Quickstart (LCSE on SigLIP 2):
    python eval.py --image-root /path/to/coco --scorer lcse \
        --model hf-hub:timm/ViT-B-32-SigLIP2-256 --mu 0.05
"""
import os
import json
import math
import inspect
import argparse
from collections import defaultdict

OPERATOR_ORDER = ["NOT", "AND", "OR", "BUT_NOT", "NEITHER", "COMPOUND"]
PROMPT_TEMPLATES = ["a {}", "a photo of a {}", "an image of a {}", "a {} in the image"]


def load(path="data/factor_bench.jsonl"):
    """Load samples from factor_bench.jsonl (one object/line) or factor_bench.json."""
    if path.endswith(".jsonl"):
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]
    with open(path) as f:
        return json.load(f)["samples"]


def _is_structured(score_fn):
    try:
        return len(inspect.signature(score_fn).parameters) >= 3
    except (TypeError, ValueError):
        return False


def _score_pair(score_fn, image_path, sample, structured):
    if structured:
        return score_fn(image_path, sample, "a"), score_fn(image_path, sample, "b")
    return score_fn(image_path, sample["caption_a"]), score_fn(image_path, sample["caption_b"])


def evaluate(score_fn, samples, image_root="", split="operator"):
    """Two-alternative forced-choice accuracy, grouped by operator.

    Scores samples whose `correct` is "a"/"b" (the `operator` and `compound`
    splits). Returns {operator: accuracy%, ..., "Overall": accuracy%}.
    """
    structured = _is_structured(score_fn)
    per = defaultdict(lambda: [0, 0])
    for s in samples:
        if split is not None and s["test_type"] != split:
            continue
        if s["correct"] not in ("a", "b"):
            continue
        img = os.path.join(image_root, s["image"])
        sa, sb = _score_pair(score_fn, img, s, structured)
        pred = "a" if sa > sb else "b"
        per[s["operator"]][0] += int(pred == s["correct"])
        per[s["operator"]][1] += 1
    res = {op: round(100 * c / n, 1) for op, (c, n) in per.items() if n}
    tc = sum(c for c, _ in per.values())
    tn = sum(n for _, n in per.values())
    if tn:
        res["Overall"] = round(100 * tc / tn, 1)
    return res


def evaluate_equivalence(score_fn, samples, image_root="", threshold=0.001):
    """Logical-equivalence consistency for the `equivalence` split (`correct=="both"`).

    A faithful scorer assigns near-identical scores to two logically equivalent
    captions. Returns the violation rate (%) per equivalence type.
    """
    structured = _is_structured(score_fn)
    per = defaultdict(lambda: [0, 0])
    for s in samples:
        if s["correct"] != "both":
            continue
        img = os.path.join(image_root, s["image"])
        sa, sb = _score_pair(score_fn, img, s, structured)
        et = (s.get("meta") or {}).get("equivalence_type") or s["operator"]
        per[et][0] += int(abs(sa - sb) > threshold)
        per[et][1] += 1
    return {et: round(100 * v / n, 1) for et, (v, n) in per.items() if n}


# --- optional reference scorers (require: pip install open_clip_torch) ---

def _load_open_clip(model_name, pretrained, device=None):
    import torch
    import open_clip
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model_name.startswith("hf-hub:"):
        # Hub checkpoints (e.g. SigLIP 2: hf-hub:timm/ViT-B-32-SigLIP2-256)
        # carry their own weights; `pretrained` does not apply.
        model, preprocess = open_clip.create_model_from_pretrained(model_name)
        model = model.to(device).float()
        tok = open_clip.get_tokenizer(model_name)
        model.eval()
        return model, preprocess, tok, device
    # OpenAI CLIP weights use QuickGELU; open_clip's default 'ViT-B-32' config
    # uses standard GELU, which does not match the pretrained 'openai' weights.
    # Auto-select the matching quickgelu variant so scores align with the
    # pretrained activation function.
    if pretrained == "openai" and "quickgelu" not in model_name.lower():
        model_name = model_name + "-quickgelu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device=device)
    tok = open_clip.get_tokenizer(model_name)
    model.eval()
    return model, preprocess, tok, device


def _text_normalizer(model_name, lowercase=None):
    """SigLIP models are trained on lowercase text; auto-lowercase for them."""
    if lowercase is None:
        lowercase = "siglip" in model_name.lower()
    return (lambda s: s.lower()) if lowercase else (lambda s: s)


def open_clip_scorer(model_name="ViT-B-32", pretrained="openai", device=None,
                     lowercase=None):
    """Holistic cosine-similarity scorer over any open_clip model.

    Works for hub checkpoints too, e.g. SigLIP 2 via
    model_name="hf-hub:timm/ViT-B-32-SigLIP2-256" (text is auto-lowercased
    for SigLIP models). Signature: score(image_path, caption) -> float.
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    model, preprocess, tok, device = _load_open_clip(model_name, pretrained, device)
    norm = _text_normalizer(model_name, lowercase)
    cache = {}

    @torch.no_grad()
    def score(image_path, caption):
        if image_path not in cache:
            im = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
            cache[image_path] = F.normalize(model.encode_image(im), dim=-1)
        g = F.normalize(model.encode_text(tok([norm(caption)]).to(device)), dim=-1)
        return float(cache[image_path] @ g.T)
    return score


def lcse_scorer(model_name="ViT-B-32", pretrained="openai",
                mu=0.22, beta=30.0, gamma_and=-1.0, gamma_or=10.0,
                templates=None, device=None, lowercase=None):
    """LCSE scorer over any open_clip model.

    Training-free Boolean operator execution over a frozen image-text encoder.
    Uses the ORACLE PARSES embedded in factor_bench (`sample['parse_a']`,
    `sample['parse_b']`). Recommended (mu, beta) per backbone:

        CLIP ViT-B/32:  mu=0.22, beta=30  (defaults)
        SigLIP 2:       model_name="hf-hub:timm/ViT-B-32-SigLIP2-256",
                        mu=0.05, beta=30 (text auto-lowercased)

    LCSE score (similarity space):

        s_LCSE = s_hol + (1/beta) * [ logit(p_logical) - logit(p_soft) ]

    where p_logical is the polarity-adjusted aggregation of per-concept
    probabilities via power mean (gamma=-1 for AND, gamma=10 for OR,
    gamma=1 otherwise), p_soft is the arithmetic mean of the raw calibrated
    per-concept probabilities, and s_hol is recovered from the calibrated
    holistic probability so all three terms are combined in logit space.

    Signature: score(image_path, sample, which) -> float ("a"/"b").
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    model, preprocess, tok, device = _load_open_clip(model_name, pretrained, device)
    norm = _text_normalizer(model_name, lowercase)
    templates = templates or PROMPT_TEMPLATES
    img_cache = {}
    concept_cache = {}
    caption_cache = {}

    @torch.no_grad()
    def emb_image(p):
        if p not in img_cache:
            im = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            img_cache[p] = F.normalize(model.encode_image(im), dim=-1).squeeze(0)
        return img_cache[p]

    @torch.no_grad()
    def emb_caption(c):
        if c not in caption_cache:
            caption_cache[c] = F.normalize(model.encode_text(tok([c]).to(device)), dim=-1).squeeze(0)
        return caption_cache[c]

    @torch.no_grad()
    def concept_sim(v, c):
        """Image-concept similarity via template ensemble: arithmetic mean of
        the per-template cosines. (Averaging the unit-vector embeddings and
        then taking one cosine would scale the result by 1/||mean(T_i)||,
        since the average of unit vectors has norm < 1.)"""
        if c not in concept_cache:
            prompts = [t.format(c) for t in templates]
            concept_cache[c] = F.normalize(
                model.encode_text(tok(prompts).to(device)), dim=-1)  # (n_templates, dim)
        embs = concept_cache[c]
        sims = v @ embs.T
        sims_f64 = [float(x) for x in sims]
        return sum(sims_f64) / len(sims_f64)

    def sigm(x):
        return 1.0 / (1.0 + math.exp(-x))

    def logit(p):
        p = min(max(p, 1e-6), 1 - 1e-6)
        return math.log(p / (1 - p))

    def power_mean(vals, gamma):
        """Power mean: (mean(x^gamma))^(1/gamma).

        Performed in torch float32 with clipping to [1e-7, 1-1e-7] (outer) and
        [1e-6, 1-1e-6] (inner) for numerical stability when gamma is large.
        """
        if not vals:
            return 0.5
        if len(vals) == 1:
            return float(vals[0])
        if gamma == 1.0:
            return sum(vals) / len(vals)
        import numpy as _np
        clipped = _np.clip(_np.array(vals, dtype=_np.float64), 1e-7, 1 - 1e-7)
        probs_t = torch.from_numpy(clipped.astype(_np.float32))
        probs_t = probs_t.clamp(1e-6, 1 - 1e-6)
        g_t = torch.tensor(gamma, dtype=torch.float32)
        return float(probs_t.pow(g_t).mean().pow(torch.tensor(1.0, dtype=torch.float32) / g_t))

    def score(image_path, sample, which):
        caption = sample["caption_" + which]
        parse = sample["parse_" + which]
        v = emb_image(image_path)
        raw_hol = float(v @ emb_caption(norm(caption)))
        concepts = parse.get("concepts", [])
        op = parse.get("operator", "SINGLE")

        # Recover s_hol from the calibrated holistic probability so all three
        # terms (s_hol, p_logical, p_soft) are combined in logit space.
        eps = 1e-6
        p_hol = max(min(sigm(beta * (raw_hol - mu)), 1 - eps), eps)
        s_hol = math.log(p_hol / (1 - p_hol)) / beta + mu

        if not concepts:
            return s_hol

        ps = [sigm(beta * (concept_sim(v, norm(c["text"])) - mu)) for c in concepts]
        ps_tilde = [(1 - p) if c["is_negated"] else p for p, c in zip(ps, concepts)]
        if op == "AND":
            p_logic = power_mean(ps_tilde, gamma_and)
        elif op == "OR":
            p_logic = power_mean(ps_tilde, gamma_or)
        else:  # SINGLE / NONE / DESCRIPTION  ->  arithmetic mean (gamma=1)
            p_logic = sum(ps_tilde) / len(ps_tilde)
        p_soft = sum(ps) / len(ps)
        return s_hol + (logit(p_logic) - logit(p_soft)) / beta

    return score


def paper_lcse_scorer(device=None, mu=0.22, beta=30.0,
                      gamma_and=-1.0, gamma_or=10.0, templates=None):
    """LCSE scorer using OpenAI's `clip` package directly.

    Same scoring formula as `lcse_scorer` but uses CLIP ViT-B/32 loaded via
    OpenAI's `clip` package (rather than `open_clip`). Use this scorer to
    reproduce the paper's reported FACTOR-Bench numbers per-operator. The
    `lcse_scorer` (open_clip) reproduces 4 of 5 per-operator + Overall to one
    decimal; one OR sample's prediction flips due to sub-ULP precision
    differences between the two CLIP implementations at gamma=10.

    Install: `pip install git+https://github.com/openai/CLIP.git`.

    Signature: score(image_path, sample, which) -> float ("a"/"b").
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image
    try:
        import clip as openai_clip
    except ImportError:
        raise ImportError(
            "paper_lcse_scorer requires OpenAI's `clip` package:\n"
            "    pip install git+https://github.com/openai/CLIP.git\n"
            "For a portable scorer using `open_clip`, use `lcse_scorer`."
        )
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = openai_clip.load("ViT-B/32", device=device)
    model.eval()
    templates = templates or PROMPT_TEMPLATES
    img_cache, text_cache = {}, {}
    eps = 1e-6

    @torch.no_grad()
    def emb_image(p):
        if p not in img_cache:
            im = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            img_cache[p] = F.normalize(model.encode_image(im).float(), dim=-1).squeeze(0)
        return img_cache[p]

    @torch.no_grad()
    def emb_text(text):
        if text not in text_cache:
            tokens = openai_clip.tokenize([text], truncate=True).to(device)
            text_cache[text] = F.normalize(
                model.encode_text(tokens).float(), dim=-1).squeeze(0)
        return text_cache[text]

    def power_mean(vals, gamma):
        import numpy as _np
        if not vals:
            return 0.5
        if len(vals) == 1:
            return float(vals[0])
        clipped = _np.clip(_np.array(vals, dtype=_np.float64), 1e-7, 1 - 1e-7)
        probs_t = torch.from_numpy(clipped.astype(_np.float32))
        probs_t = probs_t.clamp(eps, 1 - eps)
        g_t = torch.tensor(gamma, dtype=torch.float32)
        return float(probs_t.pow(g_t).mean().pow(torch.tensor(1.0, dtype=torch.float32) / g_t))

    def logit_clip(p):
        p = max(min(p, 1 - eps), eps)
        return math.log(p / (1 - p))

    def score(image_path, sample, which):
        caption = sample["caption_" + which]
        parse = sample["parse_" + which]
        v = emb_image(image_path)
        raw_hol = float(v @ emb_text(caption))
        p_hol = 1.0 / (1.0 + math.exp(-beta * (raw_hol - mu)))
        s_hol = logit_clip(p_hol) / beta + mu

        concepts = parse.get("concepts", [])
        if not concepts:
            return s_hol
        op = parse.get("operator", "SINGLE")

        # Per-concept calibrated probabilities via template ensemble:
        # arithmetic mean of per-template cosines, then sigmoid calibration.
        ps = []
        for c in concepts:
            sims = [float(v @ emb_text(t.format(c["text"]))) for t in templates]
            raw = sum(sims) / len(sims)
            ps.append(1.0 / (1.0 + math.exp(-beta * (raw - mu))))

        ps_tilde = [(1.0 - p) if c["is_negated"] else p for p, c in zip(ps, concepts)]

        if len(ps_tilde) == 1:
            p_logic = float(ps_tilde[0])
        elif op == "AND":
            p_logic = power_mean(ps_tilde, gamma_and)
        elif op == "OR":
            p_logic = power_mean(ps_tilde, gamma_or)
        else:
            p_logic = sum(ps_tilde) / len(ps_tilde)

        p_soft = sum(ps) / len(ps)
        return s_hol + (logit_clip(p_logic) - logit_clip(p_soft)) / beta

    return score


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Evaluate a model on FACTOR-Bench.")
    ap.add_argument("--data", default="data/factor_bench.jsonl")
    ap.add_argument("--image-root", required=True,
                    help="directory containing the COCO val2017/ folder")
    ap.add_argument("--scorer", default="holistic",
                    choices=["holistic", "lcse", "paper_lcse"],
                    help="holistic = raw caption cosine (open_clip); "
                         "lcse = LCSE via open_clip; "
                         "paper_lcse = LCSE via OpenAI `clip`")
    ap.add_argument("--model", default="ViT-B-32",
                    help="open_clip model name; hf-hub checkpoints work too, "
                         "e.g. hf-hub:timm/ViT-B-32-SigLIP2-256 (SigLIP 2)")
    ap.add_argument("--pretrained", default="openai",
                    help="open_clip pretrained tag (ignored for hf-hub: models)")
    ap.add_argument("--mu", type=float, default=0.22, help="LCSE calibration center")
    ap.add_argument("--beta", type=float, default=30.0, help="LCSE calibration slope")
    ap.add_argument("--split", default="operator",
                    choices=["operator", "compound", "equivalence"])
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only the first N samples of the split "
                         "(quick smoke test; accuracies are not comparable)")
    args = ap.parse_args()
    samples = load(args.data)
    if args.limit:
        samples = [s for s in samples if s["test_type"] == args.split][:args.limit]
    if args.scorer == "paper_lcse":
        score = paper_lcse_scorer(mu=args.mu, beta=args.beta)
    elif args.scorer == "lcse":
        score = lcse_scorer(args.model, args.pretrained, mu=args.mu, beta=args.beta)
    else:
        score = open_clip_scorer(args.model, args.pretrained)
    if args.split == "equivalence":
        print("Equivalence violation rate (%):",
              evaluate_equivalence(score, samples, args.image_root))
    else:
        res = evaluate(score, samples, args.image_root, split=args.split)
        for op in OPERATOR_ORDER + ["Overall"]:
            if op in res:
                print(f"{op:10s} {res[op]:5.1f}")
