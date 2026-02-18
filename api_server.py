"""
api_server.py — FastAPI wrapper for UnifiedJewelrySearcher
"""

import os
import sys
import time
import uuid
import tempfile
import traceback
import re
import json as _json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ─────────────────────────────────────────────────────────────────────────────
#  PATH
# ─────────────────────────────────────────────────────────────────────────────
SEARCH_DIR = "/workspace/Multimodal-pipeline"
_this_dir  = os.path.dirname(os.path.abspath(__file__))

for _p in [SEARCH_DIR, _this_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

print(f"[startup] sys.path includes: {SEARCH_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT DIR
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/workspace/Multimodal-pipeline/search_results_2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  App
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Jewelry Visual Search")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/results", StaticFiles(directory=OUTPUT_DIR), name="results")

# ─────────────────────────────────────────────────────────────────────────────
#  Searcher
# ─────────────────────────────────────────────────────────────────────────────
searcher = None


def get_searcher():
    global searcher
    if searcher is None:
        print("[startup] Loading UnifiedJewelrySearcher — ~30-60s first time...")
        from search import UnifiedJewelrySearcher
        searcher = UnifiedJewelrySearcher()
        print("[startup] UnifiedJewelrySearcher ready ✓")
    return searcher


@app.on_event("startup")
async def load_model_on_startup():
    import asyncio
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_searcher)


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def sanitise_label(label: str) -> str:
    """
    Mirrors exactly what search.py does to build the folder name:
      re.sub(r'[^\w\s-]', '', label).strip().replace(' ', '_').lower()

    FIX: original had r'[^ws-]' — missing backslashes stripped ALL letters,
    producing empty string → /results//reranked/ double-slash → 404s.
    """
    return re.sub(r'[^\w\s-]', '', label).strip().replace(' ', '_').lower()


def detect_search_mode(has_text: bool, has_image: bool, has_audio: bool) -> str:
    """
    Server-side mode label matching search.py fusion cases.
    Returned in JSON so the UI displays the actual pipeline mode,
    not a client-side guess.

      Voice + Image  → full_hybrid  (audio STT as text + image)
      Voice          → text_only    (after STT transcription)
      Hybrid         → full_hybrid or hybrid_no_ai
      Image          → image_only or image_with_ai_desc
      Text           → text_only
    """
    if has_audio and has_image:
        return "Voice + Image"
    if has_audio:
        return "Voice"
    if has_text and has_image:
        return "Hybrid"
    if has_image:
        return "Image"
    return "Text"


def fetch_image_from_url(url: str):
    try:
        import requests
        from bs4 import BeautifulSoup
        from PIL import Image as PILImage
        import io

        headers = {'User-Agent': 'Mozilla/5.0'}

        if re.search(r'\.(jpg|jpeg|png|webp|gif)(\?.*)?$', url, re.I):
            resp = requests.get(url, headers=headers, timeout=10)
            return PILImage.open(io.BytesIO(resp.content)).convert('RGB')

        resp    = requests.get(url, headers=headers, timeout=5)
        soup    = BeautifulSoup(resp.content, 'html.parser')
        img_tag = soup.find('meta', property='og:image') or soup.find('img')
        if not img_tag:
            return None
        img_url  = img_tag.get('content') or img_tag.get('src')
        img_resp = requests.get(img_url, headers=headers, timeout=10)
        return PILImage.open(io.BytesIO(img_resp.content)).convert('RGB')

    except Exception as e:
        print(f"[search] URL fetch error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  POST /search
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/search")
async def search(
    text:  Optional[str]        = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
):
    if not text and (not image or not image.filename) and (not audio or not audio.filename):
        raise HTTPException(
            status_code=400,
            detail="Provide at least one of: text query, image, or audio"
        )

    s              = get_searcher()
    tmp_image_path = None
    tmp_audio_path = None
    image_obj      = None
    text_query     = text.strip() if text else None

    # Capture input types BEFORE any mutation for accurate mode detection
    has_text_input  = bool(text_query)
    has_image_input = bool(image and image.filename)
    has_audio_input = bool(audio and audio.filename)

    try:
        # ── 1. URL in text field → fetch image ───────────────────────────
        if text_query and text_query.startswith("http"):
            print(f"[search] URL detected — fetching image: {text_query}")
            url_image = fetch_image_from_url(text_query)
            if url_image:
                image_obj       = url_image
                text_query      = None
                has_text_input  = False
                has_image_input = True
                print("[search] Image fetched from URL successfully")
            else:
                print("[search] URL fetch failed — treating as text query")

        # ── 2. Uploaded image ─────────────────────────────────────────────
        elif image and image.filename:
            suffix         = Path(image.filename).suffix or ".jpg"
            tmp_image_path = os.path.join(
                tempfile.gettempdir(),
                f"jewelry_query_{uuid.uuid4().hex}{suffix}"
            )
            with open(tmp_image_path, "wb") as f:
                f.write(await image.read())
            from PIL import Image as PILImage
            image_obj = PILImage.open(tmp_image_path).convert("RGB")
            print(f"[search] Image uploaded: {image.filename}")

        # ── 3. Audio → STT ────────────────────────────────────────────────
        if audio and audio.filename:
            suffix         = Path(audio.filename).suffix.lower() or ".wav"
            tmp_audio_path = os.path.join(
                tempfile.gettempdir(),
                f"jewelry_audio_{uuid.uuid4().hex}{suffix}"
            )
            with open(tmp_audio_path, "wb") as f:
                f.write(await audio.read())

            print(f"[search] Audio: {audio.filename} ({suffix}) — running STT...")

            mime_map = {
                ".mp3":  "audio/mpeg",
                ".wav":  "audio/wav",
                ".m4a":  "audio/x-m4a",
                ".ogg":  "audio/ogg",
                ".webm": "audio/webm",
            }
            mime_type = mime_map.get(suffix, "application/octet-stream")

            from search import Config as SearchConfig
            sarvam_key = SearchConfig.SARVAM_API_KEY

            if not sarvam_key:
                raise HTTPException(
                    status_code=422,
                    detail="SARVAM_API_KEY not set in .env — cannot transcribe audio"
                )

            try:
                import requests as _req
                with open(tmp_audio_path, "rb") as af:
                    stt_resp = _req.post(
                        "https://api.sarvam.ai/speech-to-text-translate",
                        headers={"api-subscription-key": sarvam_key},
                        files={"file": (Path(audio.filename).name, af, mime_type)},
                        timeout=30,
                    )
                if stt_resp.status_code != 200:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Sarvam STT error {stt_resp.status_code}: {stt_resp.text[:200]}"
                    )
                transcript = stt_resp.json().get("transcript", "").strip()
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Sarvam STT exception: {e}")

            if not transcript:
                raise HTTPException(
                    status_code=422,
                    detail="Sarvam STT returned empty transcript. Check SARVAM_API_KEY in .env"
                )

            text_query = transcript
            print(f"[search] STT transcript: {transcript}")

        # ── 4. Label + mode ───────────────────────────────────────────────
        # label must match search.py exactly:
        #   text_query set → label = text_query
        #   image only     → label = "image_search"
        label      = text_query if text_query else "image_search"
        safe_label = sanitise_label(label)
        mode_label = detect_search_mode(has_text_input, has_image_input, has_audio_input)

        print(f"[search] label='{label}' | safe='{safe_label}' | mode='{mode_label}'")

        # ── 5. Run pipeline ───────────────────────────────────────────────
        t_start = time.perf_counter()
        s.search(
            text_query=text_query,
            image_input=image_obj,
            embedding_top_k=50,
            rerank_top_k=20,
        )
        t_total = time.perf_counter() - t_start
        print(f"[search] Pipeline done in {t_total:.2f}s")

        # ── 6. Read JSON ──────────────────────────────────────────────────
        # FIX: search.py writes under raw label ("kundan necklace"),
        # NOT sanitised ("kundan_necklace"). Raw first, sanitised fallback.
        json_path = os.path.join(OUTPUT_DIR, "search_results.json")
        if not os.path.exists(json_path):
            raise HTTPException(
                status_code=500,
                detail="search_results.json not found. Check Config.OUTPUT_DIR in search.py = " + OUTPUT_DIR
            )

        with open(json_path) as f:
            all_results = _json.load(f)

        result_data = all_results.get(label) or all_results.get(safe_label)

        if not result_data:
            print(f"[search] WARNING: key '{label}' not in JSON, using last entry")
            result_data = list(all_results.values())[-1]

        reranked = result_data.get("reranked", [])
        metrics  = result_data.get("metrics", {})

        # ── 7. Build image URLs ───────────────────────────────────────────
        # Path: /results/{safe_label}/reranked/{rank:02d}_{product_id}{ext}
        results_out = []
        for item in reranked:
            src_path = item.get("path", "")
            ext      = Path(src_path).suffix or ".jpg"

            rank_filename = f"{item['rank']:02d}_{item['product_id']}{ext}"
            rank_url      = f"/results/{safe_label}/reranked/{rank_filename}"

            results_out.append({
                "rank":       item["rank"],
                "product_id": item["product_id"],
                "score":      round(float(item.get("score", 0)), 4),
                "image_url":  rank_url,
            })

        return JSONResponse({
            "results":      results_out,
            "metrics":      metrics,
            "total_time":   round(t_total, 4),
            "query_text":   text_query or "",
            "result_count": len(results_out),
            "search_mode":  mode_label,
        })

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for p in [tmp_image_path, tmp_audio_path]:
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except Exception:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
#  GET /
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    for candidate in [
        Path(__file__).parent / "index.html",
        Path("/workspace/Multimodal-pipeline/index.html"),
    ]:
        if candidate.exists():
            return HTMLResponse(content=candidate.read_text())
    raise HTTPException(status_code=404, detail="index.html not found in /workspace/Multimodal-pipeline")


# ─────────────────────────────────────────────────────────────────────────────
#  GET /health
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status":       "ok",
        "model_loaded": searcher is not None,
        "output_dir":   OUTPUT_DIR,
        "search_dir":   SEARCH_DIR,
    }


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8888))
    print(f"[startup] Starting on port {port}")
    print(f"[startup] search.py expected at : {SEARCH_DIR}/search.py")
    print(f"[startup] Results directory     : {OUTPUT_DIR}")
    uvicorn.run("api_server:app", host="0.0.0.0", port=port, reload=False)