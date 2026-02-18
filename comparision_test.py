"""
evaluate_siglip2.py
====================
Evaluates Base SigLIP-2 vs Fine-tuned SigLIP-2 (LoRA) for Image-Text Retrieval.

Metrics computed:
  - Recall@1, Recall@5, Recall@10 (Image→Text and Text→Image)
  - Mean Reciprocal Rank (MRR)
  - Mean Recall (overall summary score)

Usage:
  python evaluate_siglip2.py                          # uses synthetic demo data
  python evaluate_siglip2.py --dataset /path/to/data  # your own image+caption folder
  python evaluate_siglip2.py --qdrant                 # use your existing Qdrant DB

Dataset folder structure expected (--dataset mode):
  /path/to/data/
      image1.jpg   ← image file
      image1.txt   ← caption file (same name, .txt extension)
      image2.jpg
      image2.txt
      ...
"""

import os
import argparse
import json
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModel

# ── Try to import PEFT (LoRA support) ──────────────────────────────────────
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("WARNING: 'peft' library not found. Install with: pip install peft")
    print("         Fine-tuned LoRA model will NOT be loaded.\n")


# ============================================================
# CONFIG — edit these to match your setup
# ============================================================
class Config:
    BASE_MODEL_NAME  = "google/siglip-so400m-patch14-384"
    LORA_WEIGHTS_PATH = "/workspace/models/siglip2-jewelry-lora-new/best"
    DEVICE           = "cuda" if torch.cuda.is_available() else "cpu"
    K_VALUES         = [1, 5, 10]        # Recall@K values to compute
    BATCH_SIZE       = 16                # Lower if you get OOM errors
    OUTPUT_DIR       = "./eval_results"  # Where to save charts & JSON report


# ============================================================
# MODEL LOADER
# ============================================================
def load_models():
    """
    Loads both models:
      1. Pure base SigLIP-2 (no LoRA)
      2. Fine-tuned SigLIP-2 (base + LoRA weights)
    Returns: (base_model, ft_model, processor, tokenizer)
    """
    print(f"\n{'='*60}")
    print("LOADING MODELS")
    print(f"{'='*60}")
    print(f"Device : {Config.DEVICE}")
    print(f"Base   : {Config.BASE_MODEL_NAME}")
    print(f"LoRA   : {Config.LORA_WEIGHTS_PATH}\n")

    processor = AutoProcessor.from_pretrained(Config.BASE_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME)

    # ── Base model (no LoRA) ──────────────────────────────────
    print("Loading BASE model...")
    base_model = AutoModel.from_pretrained(Config.BASE_MODEL_NAME).to(Config.DEVICE)
    base_model.eval()
    print("Base model ready.\n")

    # ── Fine-tuned model (base + LoRA) ────────────────────────
    ft_model = None
    if PEFT_AVAILABLE and os.path.exists(Config.LORA_WEIGHTS_PATH):
        print("Loading FINE-TUNED model (base + LoRA)...")
        _ft_base  = AutoModel.from_pretrained(Config.BASE_MODEL_NAME).to(Config.DEVICE)
        ft_model  = PeftModel.from_pretrained(_ft_base, Config.LORA_WEIGHTS_PATH).to(Config.DEVICE)
        ft_model.eval()
        print("Fine-tuned model ready.\n")
    else:
        if not PEFT_AVAILABLE:
            print("SKIP: peft not installed — fine-tuned model not loaded.")
        else:
            print(f"SKIP: LoRA path not found → {Config.LORA_WEIGHTS_PATH}")
        print("Only base model will be evaluated.\n")

    return base_model, ft_model, processor, tokenizer


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================
def get_text_embeddings(model, tokenizer, captions, batch_size=16, device="cpu"):
    """Extract L2-normalised text embeddings for a list of captions."""
    all_embs = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i : i + batch_size]
        inputs = tokenizer(
            text=batch,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out  = model.text_model(**inputs)
            embs = F.normalize(out.pooler_output, dim=-1)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)   # (N, D)


def get_image_embeddings(model, processor, images, batch_size=16, device="cpu"):
    """Extract L2-normalised image embeddings for a list of PIL Images."""
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = processor(images=batch, return_tensors="pt").to(device)
        with torch.no_grad():
            out  = model.vision_model(**inputs)
            embs = F.normalize(out.pooler_output, dim=-1)
        all_embs.append(embs.cpu())
    return torch.cat(all_embs, dim=0)   # (N, D)


# ============================================================
# METRICS
# ============================================================
def compute_recall_at_k(sim_matrix: torch.Tensor, k_values=[1, 5, 10]):
    """
    Computes Recall@K for both Image→Text (i2t) and Text→Image (t2i).

    Assumption: sim_matrix[i, j] is the similarity between image i and text j.
    Ground truth: image i correctly pairs with text i (diagonal = correct match).

    Returns dict: {"i2t": {"R@1": ..., "R@5": ..., ...},
                   "t2i": {"R@1": ..., "R@5": ..., ...}}
    """
    n = sim_matrix.shape[0]
    results = {}

    for direction in ["i2t", "t2i"]:
        mat = sim_matrix if direction == "i2t" else sim_matrix.T
        direction_results = {}
        for k in k_values:
            correct = 0
            for i in range(n):
                row    = mat[i]
                top_k  = torch.topk(row, min(k, n)).indices
                if i in top_k:
                    correct += 1
            direction_results[f"R@{k}"] = round(correct / n * 100, 2)
        results[direction] = direction_results

    return results


def compute_mrr(sim_matrix: torch.Tensor):
    """
    Computes Mean Reciprocal Rank for both directions.
    MRR = mean of 1/rank_of_correct_item across all queries.
    """
    n = sim_matrix.shape[0]
    mrr_scores = {"i2t": 0.0, "t2i": 0.0}

    for direction in ["i2t", "t2i"]:
        mat = sim_matrix if direction == "i2t" else sim_matrix.T
        rr_sum = 0.0
        for i in range(n):
            # argsort descending → ranks
            sorted_indices = torch.argsort(mat[i], descending=True)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            rr_sum += 1.0 / rank
        mrr_scores[direction] = round(rr_sum / n, 4)

    return mrr_scores


def mean_recall(recall_dict):
    """Flatten all Recall@K values and return their mean."""
    all_vals = [v for d in recall_dict.values() for v in d.values()]
    return round(np.mean(all_vals), 2)


# ============================================================
# DATASET LOADING
# ============================================================
def load_from_folder(folder_path):
    """
    Loads image-caption pairs from a folder.
    Expects matching files: image1.jpg + image1.txt
    Supported image formats: .jpg, .jpeg, .png, .webp
    """
    folder   = Path(folder_path)
    img_exts = {".jpg", ".jpeg", ".png", ".webp"}
    pairs    = []

    for img_file in sorted(folder.iterdir()):
        if img_file.suffix.lower() not in img_exts:
            continue
        caption_file = img_file.with_suffix(".txt")
        if caption_file.exists():
            caption = caption_file.read_text(encoding="utf-8").strip()
            try:
                image = Image.open(img_file).convert("RGB")
                pairs.append({"image": image, "caption": caption, "id": img_file.stem})
            except Exception as e:
                print(f"  Skip {img_file.name}: {e}")

    print(f"Loaded {len(pairs)} image-caption pairs from: {folder_path}")
    return pairs


def load_from_qdrant(qdrant_path, collection_name, image_base_dir, limit=500):
    """
    Loads image-caption pairs directly from your Qdrant DB.
    Uses the 'semantic_description' stored in each point's payload as the caption.
    """
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise ImportError("Install qdrant-client: pip install qdrant-client")

    client = QdrantClient(path=qdrant_path)
    print(f"Connected to Qdrant at: {qdrant_path}")

    points = client.scroll(
        collection_name=collection_name,
        limit=limit,
        with_payload=True,
        with_vectors=False
    )[0]

    pairs = []
    for pt in points:
        pl      = pt.payload
        img_path = pl.get("path", "")
        caption  = pl.get("semantic_description", "")

        if not img_path or not caption:
            continue

        if not os.path.isabs(img_path):
            img_path = os.path.join(image_base_dir, img_path)

        if not os.path.exists(img_path):
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            pairs.append({
                "image"  : image,
                "caption": caption,
                "id"     : pl.get("product_id", str(pt.id))
            })
        except Exception as e:
            print(f"  Skip {img_path}: {e}")

    print(f"Loaded {len(pairs)} image-caption pairs from Qdrant.")
    return pairs


def create_synthetic_demo_data(n=50):
    """
    Creates synthetic test data (solid colour images + matching captions).
    Used when no real dataset is provided — for smoke-testing the script.
    """
    print(f"Creating {n} synthetic image-caption pairs for demo...")
    jewelry_items = [
        "gold kundan necklace with ruby and emerald stones",
        "silver jhumka earrings with meenakari work",
        "diamond pendant with rose gold chain",
        "antique temple jewelry necklace with pearls",
        "polki choker with uncut diamonds and green enamel",
        "bridal maangtikka with kundan and pearl drops",
        "gold bangle with intricate filigree pattern",
        "ruby ring in yellow gold with floral motif",
        "emerald necklace set with matching earrings",
        "oxidised silver statement necklace with tribal motif",
    ]
    pairs = []
    for i in range(n):
        # Solid colour image as placeholder
        colour  = tuple(np.random.randint(80, 220, 3).tolist())
        img     = Image.new("RGB", (384, 384), colour)
        caption = jewelry_items[i % len(jewelry_items)] + f" (item {i+1})"
        pairs.append({"image": img, "caption": caption, "id": f"item_{i+1:03d}"})

    print(f"Synthetic demo data ready ({n} pairs).\n")
    return pairs


# ============================================================
# EVALUATION RUNNER
# ============================================================
def evaluate_model(model, processor, tokenizer, pairs, model_label, device):
    """
    Runs full evaluation for one model.
    Returns a dict with recall, MRR, mean recall, and timing info.
    """
    print(f"\n{'─'*50}")
    print(f"Evaluating: {model_label}  ({len(pairs)} pairs)")
    print(f"{'─'*50}")

    images   = [p["image"]   for p in pairs]
    captions = [p["caption"] for p in pairs]

    # Extract embeddings
    t0 = time.perf_counter()
    print("  Extracting image embeddings...")
    img_embs  = get_image_embeddings(model, processor, images,   Config.BATCH_SIZE, device)
    print("  Extracting text embeddings...")
    txt_embs  = get_text_embeddings(model, tokenizer,  captions, Config.BATCH_SIZE, device)
    embed_time = time.perf_counter() - t0
    print(f"  Embedding time: {embed_time:.2f}s")

    # Similarity matrix (N x N) — cosine sim since embeddings are normalised
    sim_matrix = torch.matmul(img_embs, txt_embs.T)

    # Recall@K
    recall = compute_recall_at_k(sim_matrix, Config.K_VALUES)

    # MRR
    mrr = compute_mrr(sim_matrix)

    # Mean Recall (single summary number)
    mr = mean_recall(recall)

    results = {
        "model"       : model_label,
        "n_pairs"     : len(pairs),
        "embed_time_s": round(embed_time, 3),
        "recall"      : recall,
        "mrr"         : mrr,
        "mean_recall" : mr,
    }

    # Pretty print
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  Results for: {model_label:<30}│")
    print(f"  ├──────────────┬──────────────────────────────┤")
    print(f"  │  Direction   │  R@1    R@5    R@10   MRR    │")
    print(f"  ├──────────────┼──────────────────────────────┤")
    for direction in ["i2t", "t2i"]:
        label = "Image→Text" if direction == "i2t" else "Text→Image"
        r     = recall[direction]
        m     = mrr[direction]
        vals  = "  ".join(f"{r[f'R@{k}']:6.2f}" for k in Config.K_VALUES)
        print(f"  │  {label:<12}│  {vals}  {m:.4f}  │")
    print(f"  ├──────────────┴──────────────────────────────┤")
    print(f"  │  Mean Recall: {mr:>5.2f}%                        │")
    print(f"  └─────────────────────────────────────────────┘")

    return results


# ============================================================
# DELTA / IMPROVEMENT SUMMARY
# ============================================================
def print_comparison(base_res, ft_res):
    """Prints a delta table showing improvement of fine-tuned over base."""
    print(f"\n{'='*60}")
    print("COMPARISON: Fine-tuned vs Base Model")
    print(f"{'='*60}")
    print(f"{'Metric':<22} {'Base':>8} {'Fine-tuned':>12} {'Delta':>8}")
    print(f"{'─'*54}")

    for direction in ["i2t", "t2i"]:
        label = "i2t" if direction == "i2t" else "t2i"
        for k in Config.K_VALUES:
            metric = f"R@{k}"
            b = base_res["recall"][direction][metric]
            f = ft_res["recall"][direction][metric]
            d = f - b
            sign = "▲" if d > 0 else ("▼" if d < 0 else "─")
            print(f"  {label} {metric:<18} {b:>8.2f} {f:>12.2f} {sign}{abs(d):>6.2f}%")

    for direction in ["i2t", "t2i"]:
        label = f"MRR ({direction})"
        b = base_res["mrr"][direction]
        f = ft_res["mrr"][direction]
        d = f - b
        sign = "▲" if d > 0 else ("▼" if d < 0 else "─")
        print(f"  {label:<20} {b:>8.4f} {f:>12.4f} {sign}{abs(d):>6.4f}")

    print(f"{'─'*54}")
    b_mr = base_res["mean_recall"]
    f_mr = ft_res["mean_recall"]
    d_mr = f_mr - b_mr
    sign = "▲" if d_mr > 0 else ("▼" if d_mr < 0 else "─")
    print(f"  {'Mean Recall':<20} {b_mr:>8.2f} {f_mr:>12.2f} {sign}{abs(d_mr):>6.2f}%")
    print(f"{'='*60}\n")


# ============================================================
# VISUALIZATION
# ============================================================
def plot_results(base_res, ft_res, output_dir):
    """Generates a grouped bar chart comparing both models."""
    os.makedirs(output_dir, exist_ok=True)

    directions = ["i2t", "t2i"]
    dir_labels = {"i2t": "Image → Text", "t2i": "Text → Image"}
    k_vals     = Config.K_VALUES

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Base vs Fine-Tuned SigLIP-2: Recall@K Comparison", fontsize=15, fontweight="bold")

    colors = {"base": "#4C72B0", "ft": "#DD8452"}
    x      = np.arange(len(k_vals))
    width  = 0.35

    for ax, direction in zip(axes, directions):
        base_vals = [base_res["recall"][direction][f"R@{k}"] for k in k_vals]
        ft_vals   = [ft_res["recall"][direction][f"R@{k}"]   for k in k_vals]

        bars1 = ax.bar(x - width/2, base_vals, width, label="Base Model",    color=colors["base"], alpha=0.85)
        bars2 = ax.bar(x + width/2, ft_vals,   width, label="Fine-Tuned",    color=colors["ft"],  alpha=0.85)

        # Value labels on top of each bar
        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, color=colors["base"])
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9, color=colors["ft"])

        ax.set_title(dir_labels[direction], fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"R@{k}" for k in k_vals])
        ax.set_ylabel("Recall (%)")
        ax.set_ylim(0, min(110, max(max(base_vals), max(ft_vals)) + 15))
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "recall_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {out_path}")
    plt.show()


def plot_mrr_comparison(base_res, ft_res, output_dir):
    """Bar chart for MRR comparison."""
    os.makedirs(output_dir, exist_ok=True)

    labels  = ["MRR i2t", "MRR t2i", "Mean Recall"]
    base_v  = [base_res["mrr"]["i2t"], base_res["mrr"]["t2i"], base_res["mean_recall"] / 100]
    ft_v    = [ft_res["mrr"]["i2t"],   ft_res["mrr"]["t2i"],   ft_res["mean_recall"] / 100]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, base_v, width, label="Base Model",  color="#4C72B0", alpha=0.85)
    ax.bar(x + width/2, ft_v,   width, label="Fine-Tuned",  color="#DD8452", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("MRR & Mean Recall: Base vs Fine-Tuned", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1.15)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "mrr_comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {out_path}")
    plt.show()


# ============================================================
# JSON REPORT
# ============================================================
def save_report(base_res, ft_res, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    report = {
        "base_model"     : base_res,
        "finetuned_model": ft_res,
        "improvement"    : {
            "mean_recall_delta": round(ft_res["mean_recall"] - base_res["mean_recall"], 2),
            "mrr_i2t_delta"    : round(ft_res["mrr"]["i2t"] - base_res["mrr"]["i2t"], 4),
            "mrr_t2i_delta"    : round(ft_res["mrr"]["t2i"] - base_res["mrr"]["t2i"], 4),
        }
    }
    path = os.path.join(output_dir, "evaluation_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"Report saved: {path}")
    return report


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Evaluate Base vs Fine-tuned SigLIP-2")
    parser.add_argument("--dataset",    type=str, help="Path to folder with image+.txt caption pairs")
    parser.add_argument("--qdrant",     action="store_true", help="Load test pairs from your Qdrant DB")
    parser.add_argument("--qdrant_path",type=str, default="/workspace/qdrant_db_rerank")
    parser.add_argument("--collection", type=str, default="jewelry_collection")
    parser.add_argument("--image_dir",  type=str, default="/workspace/multimodal-dataset/")
    parser.add_argument("--limit",      type=int, default=500,  help="Max pairs to load from Qdrant")
    parser.add_argument("--n_demo",     type=int, default=100,  help="Synthetic demo pairs (fallback)")
    parser.add_argument("--output",     type=str, default=Config.OUTPUT_DIR)
    args = parser.parse_args()

    Config.OUTPUT_DIR = args.output
    print(f"\n{'='*60}")
    print(" SigLIP-2 Evaluation: Base vs Fine-Tuned")
    print(f"{'='*60}")

    # ── Load dataset ─────────────────────────────────────────
    if args.dataset:
        pairs = load_from_folder(args.dataset)
    elif args.qdrant:
        pairs = load_from_qdrant(args.qdrant_path, args.collection, args.image_dir, args.limit)
    else:
        print("\nNo dataset provided — using synthetic demo data.")
        print("TIP: Use --dataset /path/to/data  OR  --qdrant  for real evaluation.\n")
        pairs = create_synthetic_demo_data(args.n_demo)

    if not pairs:
        print("ERROR: No valid image-caption pairs found. Exiting.")
        return

    # ── Load models ──────────────────────────────────────────
    base_model, ft_model, processor, tokenizer = load_models()

    # ── Run evaluation ────────────────────────────────────────
    base_res = evaluate_model(base_model, processor, tokenizer, pairs, "Base SigLIP-2", Config.DEVICE)

    ft_res = None
    if ft_model is not None:
        ft_res = evaluate_model(ft_model, processor, tokenizer, pairs, "Fine-tuned SigLIP-2 (LoRA)", Config.DEVICE)
    else:
        print("\nFine-tuned model not available — skipping ft evaluation.")

    # ── Comparison & Outputs ──────────────────────────────────
    if ft_res:
        print_comparison(base_res, ft_res)
        plot_results(base_res, ft_res, Config.OUTPUT_DIR)
        plot_mrr_comparison(base_res, ft_res, Config.OUTPUT_DIR)
        report = save_report(base_res, ft_res, Config.OUTPUT_DIR)

        # Final verdict
        delta = report["improvement"]["mean_recall_delta"]
        print(f"\n{'='*60}")
        if delta > 0:
            print(f"✅ Fine-tuned model is BETTER by {delta:.2f}% Mean Recall")
        elif delta < 0:
            print(f"❌ Fine-tuned model is WORSE by {abs(delta):.2f}% Mean Recall")
        else:
            print(f"➡️  Both models perform equally.")
        print(f"{'='*60}\n")
    else:
        # Only base evaluated — just save its result
        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        path = os.path.join(Config.OUTPUT_DIR, "base_model_report.json")
        with open(path, "w") as f:
            json.dump(base_res, f, indent=4)
        print(f"Base model report saved: {path}")


if __name__ == "__main__":
    main()