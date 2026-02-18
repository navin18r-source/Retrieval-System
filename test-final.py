import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModel
from peft import PeftModel
from qdrant_client import QdrantClient
from PIL import Image
import os
import json
import re
import numpy as np
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import shutil
import argparse
from pathlib import Path
import sys
from langdetect import detect, LangDetectException
from difflib import get_close_matches
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- ENVIRONMENT FIX: Monkey-patch for older transformers versions ---
try:
    import transformers.utils.import_utils
    if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
        transformers.utils.import_utils.is_torch_fx_available = lambda: False

    # Fix for XLMRobertaTokenizer (common in BGE-M3)
    from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer
    if not hasattr(XLMRobertaTokenizer, "prepare_for_model"):
        def prepare_for_model_shim(self, *args, **kwargs):
            return self.prepare_for_tokenization(*args, **kwargs)
        XLMRobertaTokenizer.prepare_for_model = prepare_for_model_shim
except ImportError:
    pass

# Ensure the script's directory is in sys.path so it can find reranking.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from reranking import Reranker
except ImportError as e:
    print(f"\nCRITICAL: Could not import 'reranking.py' or 'FlagEmbedding' library.")
    print(f"ERROR: {e}")
    print(f"FIX 1: Ensure 'reranking.py' is in the SAME folder as this script ({os.path.basename(__file__)}).")
    print(f"FIX 2: Run 'python3 -m pip install FlagEmbedding'.")
    print("RERANKING IS CURRENTLY DISABLED (Using Dummy fallback)\n")

    class Reranker:
        def __init__(self, model_name=None): pass
        def rerank(self, query, candidates, top_k=None): return candidates[:top_k] if top_k else candidates


class Config:
    MODEL_NAME        = "google/siglip-so400m-patch14-384"
    LORA_WEIGHTS_PATH = "/workspace/models/siglip2-jewelry-lora-new/best"
    COLLECTION_NAME   = "jewelry_collection"
    QDRANT_PATH       = "/workspace/qdrant_db_rerank"
    DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
    SARVAM_API_KEY    = os.getenv("SARVAM_API_KEY")
    IMAGE_BASE_DIR    = "/workspace/multimodal-dataset/"
    GEMINI_API_KEY           = os.getenv("GEMINI_API_KEY")
    GEMINI_DESCRIPTION_MODEL = "gemini-2.5-pro-preview-03-25"   # rich text description
    GEMINI_GROUNDING_MODEL   = "gemini-2.0-flash"                # fast bbox detection & crop
    OUTPUT_DIR        = "/workspace/search_results_2"

    # Vocabulary for English typo correction
    JEWELRY_VOCAB = [
        "kundan", "polki", "meenakari", "jhumka", "temple", "antique", "gold", "silver",
        "diamond", "ruby", "emerald", "necklace", "earrings", "bangles", "bracelet",
        "pendant", "choker", "mangalsutra", "maangtikka", "nosepin", "ring", "studs"
    ]

    # ------------------------------------------------------------------
    # FUSION WEIGHTS  (each case must sum to 1.0)
    #
    # Core rule: text embeddings always dominate. Even a pure image query
    # goes through Gemini which produces a text description, so the
    # semantic signal is always carried primarily by text.
    #
    # Case: text_only
    #   User typed text / audio transcribed to text / non-English translated.
    #   Everything is already text — full weight there.
    #
    # Case: image_with_ai_desc
    #   Image provided, no human text, but Gemini generated a description.
    #   AI description carries the semantic meaning (0.7).
    #   Raw image embedding adds the visual detail (0.3).
    #
    # Case: image_only
    #   Image provided, Gemini failed or is unavailable.
    #   Only the image embedding exists — give it everything.
    #
    # Case: full_hybrid
    #   Image + human text + Gemini description all present.
    #   Human text is explicit user intent          -> 0.5 (highest)
    #   Image gets real weight, user provided it    -> 0.3 (middle)
    #   AI desc is context, not competing with text -> 0.2 (lowest)
    #
    # Case: hybrid_no_ai
    #   Image + human text, Gemini description absent or suppressed.
    #   Image needs more weight here because there is no AI desc to
    #   carry semantic meaning from it.
    #   Human text                                  -> 0.6
    #   Image                                       -> 0.4
    # ------------------------------------------------------------------
    FUSION_WEIGHTS = {
        "text_only"          : {"text": 1.0},
        "image_with_ai_desc" : {"ai_desc": 0.7, "image": 0.3},
        "image_only"         : {"image": 1.0},
        "full_hybrid"        : {"text": 0.5, "image": 0.3, "ai_desc": 0.2},
        "hybrid_no_ai"       : {"text": 0.6, "image": 0.4},
    }


class LanguageHandler:
    def __init__(self):
        pass

    def correct_english_typos(self, text):
        """Simple heuristic spell checker for known jewelry terms."""
        words = text.split()
        corrected_words = []
        for word in words:
            matches = get_close_matches(word.lower(), Config.JEWELRY_VOCAB, n=1, cutoff=0.8)
            if matches:
                corrected_words.append(matches[0].title() if word.lower() != matches[0] else word)
            else:
                corrected_words.append(word)
        return " ".join(corrected_words)

    def process_query(self, text, sarvam_fn):
        # Check for known jewelry terms first — force English if found
        words = text.lower().split()
        force_english = any(get_close_matches(w, Config.JEWELRY_VOCAB, n=1, cutoff=0.7) for w in words)

        if force_english or len(text) < 5:
            lang = "en"
        else:
            try:
                lang = detect(text)
            except LangDetectException:
                lang = "en"

        if lang == "en":
            normalized = self.correct_english_typos(text)
            print(f"English Normalizer: '{text}' -> '{normalized}'")
            return normalized
        else:
            print(f"Detected Language: {lang}")
            print(f"Routing to Sarvam AI (Target: en-IN)...")
            return sarvam_fn(text)


class UnifiedJewelrySearcher:
    def __init__(self):
        print("Loading Models...")
        self.processor = AutoProcessor.from_pretrained(Config.MODEL_NAME)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        base_model = AutoModel.from_pretrained(Config.MODEL_NAME).to(Config.DEVICE)

        # Load LoRA weights if available
        if os.path.exists(Config.LORA_WEIGHTS_PATH):
            print(f"Loading LoRA weights from: {Config.LORA_WEIGHTS_PATH}")
            self.model = PeftModel.from_pretrained(base_model, Config.LORA_WEIGHTS_PATH).to(Config.DEVICE)
            print("Finetuned LoRA model loaded successfully.")
        else:
            print(f"WARNING: LoRA weights not found at {Config.LORA_WEIGHTS_PATH}. Using base model.")
            self.model = base_model

        self.model.eval()

        self.client       = QdrantClient(path=Config.QDRANT_PATH)
        self.lang_handler = LanguageHandler()
        self.reranker     = Reranker()

        if Config.SARVAM_API_KEY:
            print("Sarvam AI: ACTIVE")

        if Config.GEMINI_API_KEY:
            genai.configure(api_key=Config.GEMINI_API_KEY)
            self.gemini_description = genai.GenerativeModel(Config.GEMINI_DESCRIPTION_MODEL)
            self.gemini_grounding   = genai.GenerativeModel(Config.GEMINI_GROUNDING_MODEL)
            print(f"Gemini AI: ACTIVE (description={Config.GEMINI_DESCRIPTION_MODEL}, grounding={Config.GEMINI_GROUNDING_MODEL})")
        else:
            self.gemini_description = None
            self.gemini_grounding   = None
            print("WARNING: GEMINI_API_KEY not set — AI description will be unavailable.")

        os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
        print("Ready.")

    # ------------------------------------------------------------------ #
    #  AI Utility Methods                                                  #
    # ------------------------------------------------------------------ #

    def sarvam_translate(self, text):
        url     = "https://api.sarvam.ai/translate"
        headers = {"api-subscription-key": Config.SARVAM_API_KEY, "Content-Type": "application/json"}
        payload = {
            "input": text,
            "source_language_code": "auto",
            "target_language_code": "en-IN",
            "speaker_gender": "Female",
            "mode": "formal"
        }
        try:
            response = requests.post(url, headers=headers, json=payload)
            return response.json().get("translated_text", text)
        except Exception as e:
            print(f"Sarvam translate error: {e}")
            return text

    def sarvam_stt(self, audio_path):
        url     = "https://api.sarvam.ai/speech-to-text-translate"
        headers = {"api-subscription-key": Config.SARVAM_API_KEY}

        ext       = os.path.splitext(audio_path)[1].lower()
        mime_map  = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.m4a': 'audio/x-m4a',
            '.ogg': 'audio/ogg'
        }
        mime_type = mime_map.get(ext, 'application/octet-stream')

        try:
            with open(audio_path, 'rb') as f:
                print(f"Sending audio to Sarvam: {os.path.basename(audio_path)} ({mime_type})...")
                files    = {'file': (os.path.basename(audio_path), f, mime_type)}
                response = requests.post(url, headers=headers, files=files)

                if response.status_code != 200:
                    print(f"Sarvam API Error ({response.status_code}): {response.text}")
                    return None

                transcript = response.json().get("transcript", "")
                if not transcript:
                    print(f"WARNING: Sarvam returned empty transcript. Raw: {response.text}")
                else:
                    print(f"Transcript: {transcript}")
                return transcript
        except Exception as e:
            print(f"STT Exception: {e}")
            return None

    def detect_and_describe(self, image):
        """
        Two-stage Gemini pipeline:
          Stage 1 — Grounding (gemini-2.0-flash):
            Detects the jewelry bounding box and crops the image.
            Fast and cheap — only needs to locate the item.
          Stage 2 — Description (gemini-2.5-pro-preview):
            Generates a rich, detailed jewelry product description
            from the cropped image for semantic search embedding.
        Returns (cropped_image, description_string).
        If Gemini is unavailable or fails, returns (original_image, None).
        """
        if not self.gemini_grounding or not self.gemini_description:
            print("Gemini not initialised — skipping AI description. Check GEMINI_API_KEY in .env")
            return image, None

        # ── Stage 1: Grounding — detect bbox and crop ─────────────────────
        grounding_prompt = (
            "Locate the main jewelry item in this image. "
            "Return ONLY valid JSON with no extra text, markdown, or code fences: "
            "{\"bbox\": [y_min, x_min, y_max, x_max]} "
            "where all values are on a 0-1000 scale."
        )
        try:
            ground_response = self.gemini_grounding.generate_content([grounding_prompt, image])
            match           = re.search(r'\{.*\}', ground_response.text, re.DOTALL)

            if match:
                bbox = json.loads(match.group()).get("bbox")
                if bbox and len(bbox) == 4:
                    w, h  = image.size
                    left  = bbox[1] * w / 1000
                    upper = bbox[0] * h / 1000
                    right = bbox[3] * w / 1000
                    lower = bbox[2] * h / 1000
                    image = image.crop((left, upper, right, lower))
                    print("Grounding: jewelry cropped.")
            else:
                print(f"Grounding: no bbox returned. Using full image. Raw: {ground_response.text[:100]}")

        except Exception as e:
            print(f"Grounding failed: {e} — using full image.")

        # ── Stage 2: Description — rich text from cropped image ────────────
        description_prompt = (
            "You are an expert jewelry cataloguer. Analyze this jewelry item and write a "
            "complete, detailed product description for a jewelry catalogue. "
            "Return ONLY valid JSON with no extra text, markdown, or code fences: "
            "{\"description\": \"...\"} "
            "The description must cover all of the following in natural flowing sentences: "
            "1. Jewelry type (necklace, jhumka earrings, bangle, ring, maangtikka, pendant, etc.), "
            "2. Metal and color (yellow gold, rose gold, silver, rhodium-plated, etc.), "
            "3. Craft style (kundan, polki, meenakari, temple jewelry, antique, filigree, jadau, etc.), "
            "4. Gemstones and embellishments (uncut diamonds, rubies, emeralds, pearls, enamel work, etc.), "
            "5. Design motifs and patterns (floral, peacock, mango/paisley, geometric, deity figures, etc.), "
            "6. Finish and texture (matte, polished, oxidised, engraved, hammered, etc.), "
            "7. Occasion and aesthetic (bridal, festive, everyday wear, traditional, contemporary, etc.). "
            "Be specific, descriptive, and accurate — this text is used for semantic search retrieval."
        )
        try:
            desc_response = self.gemini_description.generate_content([description_prompt, image])
            match         = re.search(r'\{.*\}', desc_response.text, re.DOTALL)

            if not match:
                print(f"Description: no JSON returned. Raw: {desc_response.text[:200]}")
                return image, None

            desc = json.loads(match.group()).get("description")

            if desc:
                print(f"AI Description: {desc}")
            else:
                print("Description: empty string returned by Gemini.")

            return image, desc

        except Exception as e:
            print(f"Description failed: {e}")
            return image, None

    # ------------------------------------------------------------------ #
    #  Embedding Methods                                                   #
    #  NOTE: Always call self.model directly so LoRA layers are applied.  #
    #  Never use self.model.base_model — that bypasses LoRA entirely.     #
    # ------------------------------------------------------------------ #

    def get_text_embedding(self, query):
        inputs = self.tokenizer(
            text=query,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(Config.DEVICE)
        with torch.no_grad():
            out = self.model.text_model(**inputs)
            return torch.nn.functional.normalize(
                out.pooler_output, dim=-1
            ).cpu().numpy().squeeze()

    def get_image_embedding(self, image):
        inputs = self.processor(images=image, return_tensors="pt").to(Config.DEVICE)
        with torch.no_grad():
            out = self.model.vision_model(**inputs)
            return torch.nn.functional.normalize(
                out.pooler_output, dim=-1
            ).cpu().numpy().squeeze()

    # ------------------------------------------------------------------ #
    #  Fusion: detect case -> assign fixed weights -> weighted average    #
    # ------------------------------------------------------------------ #

    def _detect_fusion_case(self, has_text, has_image, has_ai_desc):
        """
        Determine which fusion case applies based on which signals are present.

        Priority order:
          full_hybrid        — image + human text + Gemini description
          hybrid_no_ai       — image + human text, no Gemini description
                               (also when AI desc suppressed by material conflict)
          text_only          — text only (includes audio->STT, translated queries)
          image_with_ai_desc — image + Gemini description, no human text
          image_only         — image only, Gemini failed or unavailable
        """
        if has_text and has_image and has_ai_desc:
            return "full_hybrid"
        elif has_text and has_image and not has_ai_desc:
            return "hybrid_no_ai"
        elif has_text and not has_image:
            return "text_only"
        elif has_image and has_ai_desc and not has_text:
            return "image_with_ai_desc"
        elif has_image and not has_ai_desc and not has_text:
            return "image_only"
        else:
            # Fallback — should never reach here with valid inputs
            return "text_only" if has_text else "image_only"

    def _fuse_embeddings(self, signal_embs: dict) -> np.ndarray:
        """
        Weighted fusion of available signal embeddings.

        signal_embs: dict of {signal_name: np.ndarray}
                     keys are a subset of: "text", "image", "ai_desc"

        Weights are looked up from Config.FUSION_WEIGHTS by detecting
        the case from which keys are present. They always sum to 1.0.
        The result is L2-normalised to unit length.
        """
        has_text    = "text"    in signal_embs
        has_image   = "image"   in signal_embs
        has_ai_desc = "ai_desc" in signal_embs

        case    = self._detect_fusion_case(has_text, has_image, has_ai_desc)
        weights = Config.FUSION_WEIGHTS[case]

        fused = np.zeros_like(next(iter(signal_embs.values())))
        for signal_name, embedding in signal_embs.items():
            fused += embedding * weights[signal_name]

        # Weights already sum to 1.0 but re-normalise to correct any
        # floating-point drift and guarantee unit-length output.
        fused = fused / np.linalg.norm(fused)
        return fused

    # ------------------------------------------------------------------ #
    #  Helper: Save result images to folder                               #
    # ------------------------------------------------------------------ #

    def save_results_to_folder(self, results, folder_path, force_initial=False):
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path, exist_ok=True)

        saved_items = []
        for i, res in enumerate(results):
            rank  = i + 1
            score = (res.get('initial_score', 0) if force_initial
                     else res.get('rerank_score', res.get('initial_score', 0)))

            prod_id  = res['product_id']
            src_path = res['path']
            ext      = os.path.splitext(src_path)[1]

            dest_filename = f"{rank:02d}_{prod_id}{ext}"
            dest_path     = os.path.join(folder_path, dest_filename)

            if os.path.exists(src_path):
                try:
                    shutil.copy2(src_path, dest_path)
                except Exception:
                    pass

            saved_items.append({
                "rank":       rank,
                "score":      float(score),
                "product_id": prod_id,
                "path":       src_path
            })
        return saved_items

    # ------------------------------------------------------------------ #
    #  Core Search                                                         #
    # ------------------------------------------------------------------ #

    def search(self, text_query=None, image_input=None, embedding_top_k=50, rerank_top_k=20):
        label            = text_query if text_query else "image_search"
        final_query_text = ""
        ai_desc          = None
        signal_embs      = {}   # {signal_name: embedding} — built up below

        # ── 1. Process Image ──────────────────────────────────────────────
        if image_input:
            if isinstance(image_input, str):
                label    = os.path.basename(image_input) if not text_query else text_query
                img_path = image_input
                if not os.path.exists(img_path) and not img_path.startswith("http"):
                    img_path = os.path.join(Config.IMAGE_BASE_DIR, img_path)
                image = Image.open(img_path).convert('RGB')
            else:
                image = image_input

            # Gemini: detect jewelry, crop image, generate text description
            cropped_img, ai_desc = self.detect_and_describe(image)

            # Always add the image embedding
            signal_embs["image"] = self.get_image_embedding(cropped_img)

            # Add AI description embedding only if Gemini returned one
            if ai_desc:
                norm_desc              = self.lang_handler.correct_english_typos(ai_desc)
                signal_embs["ai_desc"] = self.get_text_embedding(norm_desc)

        # ── 2. Process Text ───────────────────────────────────────────────
        # Audio -> STT and non-English -> translation both produce a plain
        # English string before reaching here, so they fall into the same
        # path as a direct text query and use text_only fusion weights.
        if text_query:
            processed_text      = self.lang_handler.process_query(text_query, self.sarvam_translate)
            final_query_text    = processed_text
            signal_embs["text"] = self.get_text_embedding(processed_text)

        # ── 3. Material Conflict Check ────────────────────────────────────
        # If the user explicitly states a material that contradicts what
        # Gemini detected in the image, suppress the AI description so it
        # does not pull the search toward the wrong material.
        if "text" in signal_embs and "ai_desc" in signal_embs and ai_desc:
            user_materials = [m for m in ["gold", "silver", "platinum", "diamond"]
                              if m in final_query_text.lower()]
            ai_materials   = [m for m in ["gold", "silver", "platinum", "diamond"]
                              if m in ai_desc.lower()]

            has_conflict = any(
                um in ["silver", "gold", "platinum"] and
                any(am != um for am in ai_materials if am in ["silver", "gold", "platinum"])
                for um in user_materials
            )

            if has_conflict:
                print(f"Material Conflict: User wants {user_materials} but "
                      f"image shows {ai_materials}. Suppressing AI description.")
                del signal_embs["ai_desc"]

        # ── 4. Guard: nothing to search ───────────────────────────────────
        if not signal_embs:
            print("No valid input signals found. Aborting search.")
            return

        # ── 5. Weighted Fusion ────────────────────────────────────────────
        # Detect the fusion case from which signals are present, assign
        # fixed weights that always sum to 1.0, compute and re-normalise.
        final_emb = self._fuse_embeddings(signal_embs)

        # ── 6. ANN Retrieval via Qdrant ───────────────────────────────────
        import time
        start_time = time.perf_counter()

        print(f"\nVector Search: Fetching {embedding_top_k} candidates...")
        try:
            res_obj = self.client.query_points(
                collection_name=Config.COLLECTION_NAME,
                query=final_emb.tolist(),
                limit=embedding_top_k
            )
        except Exception as e:
            print(f"Error: Qdrant query failed — {e}")
            print(f"Make sure collection '{Config.COLLECTION_NAME}' exists. Run index.py first.")
            return

        embedding_results = []
        for res in res_obj.points:
            embedding_results.append({
                "product_id":           res.payload['product_id'],
                "path":                 res.payload['path'],
                "semantic_description": res.payload.get('semantic_description', ''),
                "initial_score":        res.score
            })

        retrieval_time = time.perf_counter() - start_time
        print(f"Retrieval Time: {retrieval_time:.4f}s")

        # ── 7. Reranking ──────────────────────────────────────────────────
        # Prefer human text for reranking; fall back to Gemini description
        # for pure image queries so reranking still has a text signal.
        rerank_query = final_query_text if final_query_text else (ai_desc if ai_desc else "")

        rerank_start_time = time.perf_counter()
        if embedding_results and rerank_query:
            print(f"Reranking {len(embedding_results)} candidates using BGE-M3...")
            reranked_results = self.reranker.rerank(rerank_query, embedding_results, top_k=rerank_top_k)
        else:
            if not rerank_query:
                print("No text signal available for reranking — using embedding order.")
            reranked_results = embedding_results[:rerank_top_k]
        rerank_time = time.perf_counter() - rerank_start_time

        print(f"Reranking Time: {rerank_time:.4f}s")
        print(f"Total Time    : {retrieval_time + rerank_time:.4f}s")

        # ── 8. Save Results ───────────────────────────────────────────────
        safe_query  = re.sub(r'[^\w\s-]', '', label).strip().replace(' ', '_').lower()
        root_dir    = os.path.join(Config.OUTPUT_DIR, safe_query)
        emb_folder  = os.path.join(root_dir, "embedding_only")
        rank_folder = os.path.join(root_dir, "reranked")

        emb_metadata  = self.save_results_to_folder(embedding_results, emb_folder,  force_initial=True)
        rank_metadata = self.save_results_to_folder(reranked_results,  rank_folder, force_initial=False)

        # Save tiled overview image into the reranked folder
        self.save_tiled_image(rank_metadata, rank_folder, cols=10)

        self.update_json_results(label, {
            "embedding_only": emb_metadata,
            "reranked":       rank_metadata,
            "metrics": {
                "retrieval_time": retrieval_time,
                "reranking_time": rerank_time,
                "total_time":     retrieval_time + rerank_time
            }
        })

        print(f"\nSearch complete. Results saved to: {root_dir}")
        print(f"\nTOP 10 RERANKED RESULTS:")
        for res in rank_metadata[:10]:
            print(f"   {res['rank']}. ID: {res['product_id']}")
        print(f"\nSaved: {len(emb_metadata)} in /embedding_only, {len(rank_metadata)} in /reranked")

    def save_tiled_image(self, rank_metadata, folder_path, cols=10):
        """
        Creates a tiled grid image of all reranked results and saves it
        as 00_tiled_results.jpg in the reranked folder.

        Layout: 2 columns x N rows (fills left-to-right, top-to-bottom).
        Each tile is resized to a fixed size for a clean uniform grid.
        A rank label is drawn on each tile so results are easy to read.
        """
        from PIL import ImageDraw, ImageFont

        TILE_W    = 400
        TILE_H    = 400
        PADDING   = 6
        LABEL_H   = 28
        BG_COLOR  = (30, 30, 30)
        FONT_COLOR = (255, 255, 255)

        # Load images that exist on disk
        tiles = []
        for item in rank_metadata:
            src = item["path"]
            if os.path.exists(src):
                try:
                    img = Image.open(src).convert("RGB")
                    img.thumbnail((TILE_W, TILE_H - LABEL_H), Image.LANCZOS)
                    # Paste onto a fixed-size tile background
                    tile = Image.new("RGB", (TILE_W, TILE_H), BG_COLOR)
                    x_off = (TILE_W - img.width)  // 2
                    y_off = (TILE_H - LABEL_H - img.height) // 2
                    tile.paste(img, (x_off, y_off))
                    # Draw rank label at the bottom of the tile
                    draw = ImageDraw.Draw(tile)
                    try:
                        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                    except Exception:
                        font = ImageFont.load_default()
                    label_text = f"#{item['rank']}  {item['product_id']}"
                    draw.rectangle([(0, TILE_H - LABEL_H), (TILE_W, TILE_H)], fill=(0, 0, 0))
                    draw.text((6, TILE_H - LABEL_H + 5), label_text, fill=FONT_COLOR, font=font)
                    tiles.append(tile)
                except Exception as e:
                    print(f"Tiled image: could not load {src} — {e}")

        if not tiles:
            print("Tiled image: no valid images found, skipping.")
            return

        rows      = (len(tiles) + cols - 1) // cols
        grid_w    = cols * TILE_W + (cols + 1) * PADDING
        grid_h    = rows * TILE_H + (rows + 1) * PADDING
        grid      = Image.new("RGB", (grid_w, grid_h), (15, 15, 15))

        for idx, tile in enumerate(tiles):
            row = idx // cols
            col = idx  % cols
            x   = PADDING + col * (TILE_W + PADDING)
            y   = PADDING + row * (TILE_H + PADDING)
            grid.paste(tile, (x, y))

        out_path = os.path.join(folder_path, "00_tiled_results.jpg")
        grid.save(out_path, "JPEG", quality=92)
        print(f"Tiled image saved: {out_path}  ({cols}x{rows} grid, {len(tiles)} items)")

    # ------------------------------------------------------------------ #

    def update_json_results(self, query, metadata):
        json_path   = os.path.join(Config.OUTPUT_DIR, "search_results.json")
        all_results = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    all_results = json.load(f)
            except Exception:
                pass
        all_results[query] = metadata
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=4)

    def process_url(self, url):
        try:
            resp     = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
            soup     = BeautifulSoup(resp.content, 'html.parser')
            img_tag  = soup.find('meta', property='og:image') or soup.find('img')
            img_url  = img_tag.get('content') or img_tag.get('src')
            img_data = requests.get(img_url, stream=True).raw
            return Image.open(img_data).convert('RGB')
        except Exception as e:
            print(f"URL processing error: {e}")
            return None


# ------------------------------------------------------------------ #
#  Entry Point                                                         #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Text search query")
    parser.add_argument("--image", type=str, help="Path to local image")
    args = parser.parse_args()

    searcher = UnifiedJewelrySearcher()

    if args.query or args.image:
        searcher.search(text_query=args.query, image_input=args.image)
    else:
        print("\nUNIFIED JEWELRY SEARCH")
        print("Usage: type text | image_path | text  | URL  | audio file path | 'exit'")
        while True:
            cmd = input("\nsearch: ").strip()
            if not cmd or cmd.lower() == 'exit':
                break

            # Hybrid: /path/image.jpg | text query
            if "|" in cmd:
                parts = cmd.split("|", 1)
                img_p = parts[0].strip()
                txt_q = parts[1].strip()
                searcher.search(
                    text_query=txt_q,
                    image_input=img_p if os.path.exists(img_p) else None
                )

            # URL
            elif cmd.startswith("http"):
                img = searcher.process_url(cmd)
                if img:
                    searcher.search(image_input=img)
                else:
                    print("Could not fetch image from URL.")

            # Local file: audio or image
            elif os.path.exists(cmd) and not os.path.isdir(cmd):
                ext = Path(cmd).suffix.lower()
                if ext in ['.mp3', '.wav', '.m4a', '.ogg']:
                    # Audio -> STT -> plain English text -> text_only fusion case
                    print("Transcribing audio...")
                    text = searcher.sarvam_stt(cmd)
                    if text:
                        searcher.search(text_query=text)
                    else:
                        print("Transcription failed or returned empty.")
                else:
                    # Image file
                    searcher.search(image_input=cmd)

            # Plain text (also handles non-English — translated upstream)
            else:
                searcher.search(text_query=cmd)


if __name__ == "__main__":
    main()