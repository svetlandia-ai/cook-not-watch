"""
Cook not watch – main.py
Turn TikTok / Instagram recipe videos into printable recipe cards.

Stack: FastAPI + OpenAI (GPT-4o Vision + gpt-4.1-mini) + yt-dlp + ffmpeg + ReportLab
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv
import os, json, tempfile, subprocess, base64, io
import yt_dlp
from PIL import Image as PILImage

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.colors import HexColor, white

load_dotenv()

app = FastAPI(title="Cook not watch")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Colour palette  (Pantone 2026: Cloud Dancer #F4F0EC) ─────────────────────
C_BG    = HexColor("#F4F0EC")   # Cloud Dancer – page background
C_DARK  = HexColor("#1C1C1C")   # near-black text
C_BROWN = HexColor("#6B5B45")   # warm brown – section headers
C_SAGE  = HexColor("#5C8A6F")   # sage green – bullets & step circles
C_MUTED = HexColor("#9A8A7A")   # secondary / muted text
C_CARD  = HexColor("#E8E3DC")   # nutrition boxes background


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def safe_parse_json(raw: str) -> dict:
    text = raw.strip()
    for prefix in ("```json", "```"):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"GPT returned invalid JSON: {e}\n\nRaw:\n{text}")


def download_video(url: str) -> str:
    """Download video from TikTok / Instagram, return local .mp4 path."""
    tmp = tempfile.mkdtemp()
    ydl_opts = {
        "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
        "format": "mp4/bestvideo+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
    if not path.endswith(".mp4"):
        mp4 = os.path.splitext(path)[0] + ".mp4"
        if os.path.exists(mp4):
            path = mp4
    return path


def extract_frames(video_path: str) -> list:
    """Extract up to 5 frames at fixed timestamps. Returns list of existing paths."""
    base = os.path.splitext(video_path)[0]
    timestamps = ["00:00:02", "00:00:05", "00:00:08", "00:00:12", "00:00:17"]
    paths = []
    for i, ts in enumerate(timestamps):
        out = f"{base}_f{i}.jpg"
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ss", ts,
             "-vframes", "1", "-q:v", "2", out],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if os.path.exists(out):
            paths.append(out)
    return paths


def file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def bytes_to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


# ─────────────────────────────────────────────────────────────────────────────
#  GPT helpers
# ─────────────────────────────────────────────────────────────────────────────

def vision_extract_recipe(images_b64: list, source_url: str = "") -> dict:
    """Call GPT-4o Vision with one or more images to extract a recipe."""
    content = []
    for b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"},
        })

    multi = len(images_b64) > 1
    n = len(images_b64)

    extra_instruction = (
        f"I'm sending you {n} frames (indexed 0–{n-1}) from a cooking video.\n"
        f"Also identify best_frame_index – the index showing the food/dish most clearly "
        f"WITHOUT a person's face dominating the frame.\n"
        if multi else ""
    )
    schema_extra = '"best_frame_index": number,' if multi else ""

    content.append({"type": "text", "text": f"""You are a cooking assistant.
Source: {source_url}

{extra_instruction}Extract the complete recipe shown.

Return ONLY valid JSON (no markdown, no extra text):
{{
  {schema_extra}
  "title": "string",
  "description": "string",
  "source": "creator name or @handle visible in content, else empty string",
  "ingredients": [
    {{"name":"string","amount":number_or_null,"unit":"string_or_null","notes":"string_or_empty"}}
  ],
  "steps": ["string"]
}}"""})

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": content}],
        max_tokens=2000,
    )
    return safe_parse_json(resp.choices[0].message.content)


def text_extract_recipe(text: str, source_url: str = "") -> dict:
    """Extract recipe from plain text using gpt-4.1-mini."""
    prompt = f"""You are a cooking assistant.
Source URL: {source_url}

Convert the text below into a structured recipe.

Return ONLY valid JSON (no markdown):
{{
  "title": "string",
  "description": "string",
  "source": "string or empty",
  "ingredients": [
    {{"name":"string","amount":number_or_null,"unit":"string_or_null","notes":"string_or_empty"}}
  ],
  "steps": ["string"]
}}

TEXT:
{text}"""
    resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
    return safe_parse_json(resp.output_text)


def calc_nutrition(recipe: dict) -> dict:
    """Estimate КБЖУ / macros (incl. fiber) for the recipe."""
    prompt = f"""You are a nutrition assistant. Estimate nutritional values for this recipe.

Return ONLY valid JSON:
{{
  "per_100g": {{"calories":number,"protein":number,"fat":number,"carbs":number,"fiber":number}},
  "total":    {{"calories":number,"protein":number,"fat":number,"carbs":number,"fiber":number}},
  "estimated_total_weight_g": number,
  "confidence": "low|medium|high",
  "notes": "short explanation"
}}

Recipe:
{json.dumps(recipe)}"""
    resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
    return safe_parse_json(resp.output_text)


# ─────────────────────────────────────────────────────────────────────────────
#  PDF Generation  (Cloud Dancer palette · ReportLab canvas · 4-quadrant grid)
# ─────────────────────────────────────────────────────────────────────────────

def _wrap(c, text: str, x: float, y: float,
          max_w: float, font: str, size: float, lead: float,
          bot_limit: float = 0) -> float:
    """Draw word-wrapped text. Returns y after last line."""
    c.setFont(font, size)
    words = str(text).split()
    line = ""
    for word in words:
        test = (line + " " + word).strip()
        if c.stringWidth(test, font, size) <= max_w:
            line = test
        else:
            if line and y >= bot_limit:
                c.drawString(x, y, line)
                y -= lead
            line = word
    if line and y >= bot_limit:
        c.drawString(x, y, line)
        y -= lead
    return y


def _crop_to_ratio(img_bytes: bytes, target_w: float, target_h: float) -> bytes:
    """Crop image to fill target aspect ratio exactly (center crop)."""
    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
    iw, ih = img.size
    target_ratio = target_w / target_h
    current_ratio = iw / ih
    if current_ratio > target_ratio:
        new_w = int(ih * target_ratio)
        left = (iw - new_w) // 2
        img = img.crop((left, 0, left + new_w, ih))
    else:
        new_h = int(iw / target_ratio)
        top = (ih - new_h) // 2
        img = img.crop((0, top, iw, top + new_h))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=90)
    out.seek(0)
    return out.read()


def _quad_ingredients(c, recipe, x, y_top, w, h):
    """Top-right quadrant: ingredients list with printable checkboxes."""
    PAD = 4 * mm
    CB  = 5.5            # checkbox size in points
    GAP = 3.5            # gap between checkbox and text
    TX  = x + PAD + CB + GAP      # text x
    TW  = w - PAD - CB - GAP - PAD  # text max width
    BOT = y_top - h + PAD

    y = y_top - PAD

    # Section title
    c.setFillColor(C_BROWN)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + PAD, y, "INGREDIENTS")
    y -= 6 * mm

    for ing in recipe.get("ingredients", []):
        if y < BOT + CB:
            break

        amt   = ing.get("amount")
        unit  = ing.get("unit") or ""
        name  = ing.get("name", "")
        notes = ing.get("notes", "")

        # Open square checkbox (printable)
        c.setStrokeColor(C_BROWN)
        c.setFillColor(C_BG)
        c.setLineWidth(0.65)
        c.rect(x + PAD, y - CB + 2, CB, CB, fill=1, stroke=1)

        # Format amount (convert 0.5 → ½, 0.25 → ¼, 0.75 → ¾)
        FRACTIONS = {0.25: "¼", 0.5: "½", 0.75: "¾", 0.33: "⅓", 0.67: "⅔"}
        amt_val = amt
        if isinstance(amt_val, float):
            amt_val = FRACTIONS.get(round(amt_val, 2), int(amt_val) if amt_val == int(amt_val) else amt_val)
        amt_str = f"{amt_val} {unit}".strip() if amt else ""
        full_text = f"{amt_str} {name}".strip() if amt_str else name
        if notes:
            full_text += f" ({notes})"

        # Clip text
        c.setFillColor(C_DARK)
        c.setFont("Helvetica", 8)
        while len(full_text) > 3 and c.stringWidth(full_text, "Helvetica", 8) > TW:
            full_text = full_text[:-2] + "…"
        c.drawString(TX, y, full_text)
        y -= 5.2 * mm


def _quad_steps(c, recipe, x, y_top, w, h):
    """Bottom-left quadrant: numbered cooking steps."""
    PAD    = 4 * mm
    CR     = 3.5          # circle radius
    CX     = x + PAD + CR # circle centre x
    TX     = x + PAD + CR * 2 + 2.5 * mm
    TW     = w - PAD - CR * 2 - 2.5 * mm - PAD
    BOT    = y_top - h + PAD
    LEAD   = 4.3 * mm

    y = y_top - PAD

    # Section title
    c.setFillColor(C_BROWN)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + PAD, y, "STEPS")
    y -= 6 * mm

    for i, step in enumerate(recipe.get("steps", []), 1):
        if y < BOT + CR * 2:
            break

        # Numbered circle
        c.setFillColor(C_SAGE)
        c.circle(CX, y + CR - 1, CR, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 6)
        c.drawCentredString(CX, y + CR - 3, str(i))

        # Step text (wrapped)
        c.setFillColor(C_DARK)
        y = _wrap(c, step, TX, y, TW, "Helvetica", 8, LEAD, bot_limit=BOT)
        y -= 2.5 * mm


def _quad_nutrition(c, nutrition, x, y_top, w, h):
    """Bottom-right quadrant: КБЖУ boxes (Calories, Protein, Fat / Carbs, Fiber)."""
    PAD = 4 * mm
    BOT = y_top - h + PAD
    y   = y_top - PAD

    # Section title
    c.setFillColor(C_BROWN)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + PAD, y, "NUTRITION · per 100 g")
    y -= 7 * mm

    if not nutrition:
        c.setFillColor(C_MUTED)
        c.setFont("Helvetica", 8)
        c.drawString(x + PAD, y, "No data")
        return

    per100  = nutrition.get("per_100g", {})
    avail_w = w - 2 * PAD
    BH      = 17 * mm
    BG      = 3.5          # gap between boxes in points

    # Row 1: Calories · Protein · Fat
    row1 = [
        ("CALORIES", per100.get("calories", "—"), "kcal"),
        ("PROTEIN",  per100.get("protein",  "—"), "g"),
        ("FAT",      per100.get("fat",      "—"), "g"),
    ]
    bw1 = (avail_w - 2 * BG) / 3
    by1 = y - BH
    for i, (lbl, val, unit) in enumerate(row1):
        bx = x + PAD + i * (bw1 + BG)
        c.setFillColor(C_CARD)
        c.roundRect(bx, by1, bw1, BH, 3, fill=1, stroke=0)
        c.setFillColor(C_DARK)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(bx + bw1 / 2, by1 + 8.5 * mm, str(val))
        c.setFillColor(C_MUTED)
        c.setFont("Helvetica", 6.5)
        c.drawCentredString(bx + bw1 / 2, by1 + 4.5 * mm, unit)
        c.drawCentredString(bx + bw1 / 2, by1 + 1.5 * mm, lbl)
    y -= BH + 4 * mm

    # Row 2: Carbs · Fiber
    row2 = [
        ("CARBS", per100.get("carbs", "—"), "g"),
        ("FIBER", per100.get("fiber", "—"), "g"),
    ]
    bw2 = (avail_w - BG) / 2
    by2 = y - BH
    for i, (lbl, val, unit) in enumerate(row2):
        bx = x + PAD + i * (bw2 + BG)
        c.setFillColor(C_CARD)
        c.roundRect(bx, by2, bw2, BH, 3, fill=1, stroke=0)
        c.setFillColor(C_DARK)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(bx + bw2 / 2, by2 + 8.5 * mm, str(val))
        c.setFillColor(C_MUTED)
        c.setFont("Helvetica", 6.5)
        c.drawCentredString(bx + bw2 / 2, by2 + 4.5 * mm, unit)
        c.drawCentredString(bx + bw2 / 2, by2 + 1.5 * mm, lbl)
    y -= BH + 5 * mm

    # Notes (wrapped, small)
    note = nutrition.get("notes", "")
    if note and y > BOT:
        c.setFillColor(C_MUTED)
        _wrap(c, "* " + note, x + PAD, y, avail_w,
              "Helvetica-Oblique", 6.5, 3.5 * mm, bot_limit=BOT)


def build_pdf(recipe: dict, nutrition: dict = None,
              photo_b64: str = None, source_url: str = "") -> bytes:
    """
    A4 layout – 4-quadrant grid:
      ┌──────────────┬──────────────┐
      │  PHOTO       │ INGREDIENTS  │  ← top half
      ├──────────────┼──────────────┤
      │  STEPS       │ NUTRITION    │  ← bottom half
      └──────────────┴──────────────┘
    Header strip above the grid: title · author · branding · source URL
    """
    buf = io.BytesIO()
    W, H = A4          # 595.28 × 841.89 pt
    c = rl_canvas.Canvas(buf, pagesize=A4)

    # ── Dimensions ───────────────────────────────────────────────────────────
    M        = 12 * mm   # page margins (all sides)
    HEAD_H   = 19 * mm   # header strip height
    HEAD_GAP = 3  * mm   # gap between header and grid
    Q_GAP    = 5  * mm   # gap between quadrants

    QW = (W - 2 * M - Q_GAP) / 2
    QH = (H - 2 * M - HEAD_H - HEAD_GAP - Q_GAP) / 2

    # Y landmarks (y=0 at page bottom)
    head_top = H - M
    head_bot = head_top - HEAD_H
    tq_top   = head_bot - HEAD_GAP    # top of upper quadrant row
    tq_bot   = tq_top - QH            # bottom of upper quadrant row
    bq_top   = tq_bot - Q_GAP         # top of lower quadrant row
    # bq_bot = M (= bottom margin)

    # Column x positions
    lx = M
    rx = M + QW + Q_GAP

    # ── Background ───────────────────────────────────────────────────────────
    c.setFillColor(C_BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)

    # ── Grid dividers (light lines) ───────────────────────────────────────────
    c.setStrokeColor(HexColor("#CEC8BE"))
    c.setLineWidth(0.4)
    # Vertical
    mid_x = M + QW + Q_GAP / 2
    c.line(mid_x, M, mid_x, tq_top)
    # Horizontal
    mid_y = tq_bot - Q_GAP / 2
    c.line(M, mid_y, W - M, mid_y)

    # ── HEADER ───────────────────────────────────────────────────────────────
    # Bottom divider of header
    c.setStrokeColor(C_BROWN)
    c.setLineWidth(0.5)
    c.line(M, head_bot, W - M, head_bot)

    # Title (left)
    title = recipe.get("title", "Recipe")
    c.setFillColor(C_DARK)
    c.setFont("Helvetica-Bold", 15)
    max_title_w = W / 2 - M - 4 * mm
    while c.stringWidth(title, "Helvetica-Bold", 15) > max_title_w and len(title) > 3:
        title = title[:-2] + "…"
    c.drawString(M, head_top - 11 * mm, title)

    src = recipe.get("source", "")
    if src:
        c.setFillColor(C_MUTED)
        c.setFont("Helvetica-Oblique", 8.5)
        c.drawString(M, head_top - 17 * mm, f"by {src}")

    # Branding + source URL (right)
    c.setFillColor(C_SAGE)
    c.setFont("Helvetica-Bold", 9)
    c.drawRightString(W - M, head_top - 10 * mm, "Cook not watch")
    if source_url:
        disp = source_url if len(source_url) < 52 else source_url[:49] + "…"
        c.setFillColor(C_MUTED)
        c.setFont("Helvetica", 7)
        c.drawRightString(W - M, head_top - 16 * mm, disp)

    # ── Q1 TOP-LEFT: Photo ────────────────────────────────────────────────────
    if photo_b64:
        try:
            img_bytes = base64.b64decode(photo_b64)
            cropped   = _crop_to_ratio(img_bytes, QW, QH)
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.write(cropped)
            tmp.close()
            c.drawImage(tmp.name, lx, tq_bot, width=QW, height=QH)
            os.unlink(tmp.name)
        except Exception:
            c.setFillColor(C_CARD)
            c.rect(lx, tq_bot, QW, QH, fill=1, stroke=0)
    else:
        c.setFillColor(C_CARD)
        c.rect(lx, tq_bot, QW, QH, fill=1, stroke=0)

    # Author caption overlay (bottom of photo, semi-transparent strip)
    src = recipe.get("source", "")
    if src or source_url:
        STRIP_H = 9 * mm
        # Dark semi-transparent strip
        c.setFillColor(HexColor("#1C1C1C"))
        c.setFillAlpha(0.55)
        c.rect(lx, tq_bot, QW, STRIP_H, fill=1, stroke=0)
        c.setFillAlpha(1.0)
        # Author handle
        caption = f"by {src}" if src else ""
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(lx + 3 * mm, tq_bot + 5 * mm, caption)
        # Source URL (right side of strip)
        if source_url:
            disp = source_url if len(source_url) < 42 else source_url[:39] + "…"
            c.setFillColor(HexColor("#D8D2C8"))
            c.setFont("Helvetica", 6.5)
            c.drawRightString(lx + QW - 3 * mm, tq_bot + 5 * mm, disp)

    # ── Q2 TOP-RIGHT: Ingredients ─────────────────────────────────────────────
    _quad_ingredients(c, recipe, rx, tq_top, QW, QH)

    # ── Q3 BOTTOM-LEFT: Steps ────────────────────────────────────────────────
    _quad_steps(c, recipe, lx, bq_top, QW, QH)

    # ── Q4 BOTTOM-RIGHT: Nutrition ────────────────────────────────────────────
    _quad_nutrition(c, nutrition, rx, bq_top, QW, QH)

    c.save()
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

class VideoRequest(BaseModel):
    url: str

class TextRequest(BaseModel):
    text: str = ""
    url: str = ""

class PDFRequest(BaseModel):
    recipe: dict
    nutrition: Optional[dict] = None
    photo_b64: Optional[str] = None
    source_url: str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "app": "Cook not watch"}


@app.post("/extract-from-video")
def extract_from_video(req: VideoRequest):
    # 1. Download
    try:
        video_path = download_video(req.url)
    except Exception as e:
        raise HTTPException(400, f"Could not download video: {e}")

    # 2. Extract frames
    frames = extract_frames(video_path)
    if not frames:
        raise HTTPException(500, "Could not extract frames from video")

    frames_b64 = [file_to_b64(fp) for fp in frames]

    # 3. Vision → recipe
    try:
        result = vision_extract_recipe(frames_b64, req.url)
    except Exception as e:
        raise HTTPException(502, f"Vision analysis failed: {e}")

    # 4. Pick best frame for PDF photo
    best_idx = result.pop("best_frame_index", 0)
    if not isinstance(best_idx, int) or best_idx >= len(frames_b64):
        best_idx = 0
    photo_b64 = frames_b64[best_idx]

    # Cleanup temp files
    for fp in frames:
        try: os.unlink(fp)
        except: pass
    try: os.unlink(video_path)
    except: pass

    # 5. Nutrition
    nutrition = nutrition_error = None
    try:
        nutrition = calc_nutrition(result)
    except Exception as e:
        nutrition_error = str(e)

    return {
        "recipe":          result,
        "nutrition":       nutrition,
        "nutrition_error": nutrition_error,
        "photo_b64":       photo_b64,
        "source_url":      req.url,
    }


@app.post("/extract-from-image")
async def extract_from_image(
    file: UploadFile = File(...),
    source_url: str = Form(""),
):
    data = await file.read()
    b64  = bytes_to_b64(data)

    try:
        result = vision_extract_recipe([b64], source_url)
    except Exception as e:
        raise HTTPException(502, f"Vision analysis failed: {e}")

    result.pop("best_frame_index", None)

    nutrition = nutrition_error = None
    try:
        nutrition = calc_nutrition(result)
    except Exception as e:
        nutrition_error = str(e)

    return {
        "recipe":          result,
        "nutrition":       nutrition,
        "nutrition_error": nutrition_error,
        "photo_b64":       b64,
        "source_url":      source_url,
    }


@app.post("/extract-from-text")
def extract_from_text(req: TextRequest):
    try:
        result = text_extract_recipe(req.text, req.url)
    except Exception as e:
        raise HTTPException(502, f"Extraction failed: {e}")

    nutrition = nutrition_error = None
    try:
        nutrition = calc_nutrition(result)
    except Exception as e:
        nutrition_error = str(e)

    return {
        "recipe":          result,
        "nutrition":       nutrition,
        "nutrition_error": nutrition_error,
        "photo_b64":       None,
        "source_url":      req.url,
    }


@app.post("/generate-pdf")
def generate_pdf(req: PDFRequest):
    try:
        pdf_bytes = build_pdf(
            req.recipe, req.nutrition, req.photo_b64, req.source_url
        )
    except Exception as e:
        raise HTTPException(500, f"PDF generation failed: {e}")

    title    = req.recipe.get("title", "recipe").replace(" ", "_").lower()
    filename = f"{title}_cook_not_watch.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PNG Card Generation  (Pinterest / Instagram style · Pillow)
# ─────────────────────────────────────────────────────────────────────────────

# Card palette
_CRD_BG      = (255, 255, 255)
_CRD_TEXT    = (26,  26,  26 )
_CRD_MUTED   = (154, 138, 122)
_CRD_ACCENT  = (92,  138, 111)   # sage green – same as web UI
_CRD_CARD    = (247, 247, 247)
_CRD_DIVIDER = (238, 238, 238)
_CRD_WHITE   = (255, 255, 255)

_CARD_W      = 1080
_CARD_PAD    = 64
_CONTENT_W   = _CARD_W - 2 * _CARD_PAD
_PHOTO_H     = 720

# Fraction map for ingredient amounts
_FRAC = {0.25: "¼", 0.5: "½", 0.75: "¾", 0.33: "⅓", 0.67: "⅔"}

# Base directory for font lookup
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_font(bold: bool, size: int):
    """Load best available TTF font; gracefully fall back."""
    from PIL import ImageFont
    suffix = "-Bold.ttf" if bold else "-Regular.ttf"
    candidates = [
        os.path.join(_BASE_DIR, "assets", "fonts", f"Inter{suffix}"),
        "/usr/share/fonts/truetype/liberation/LiberationSans"
        + ("-Bold.ttf" if bold else "-Regular.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans"
        + ("-Bold.ttf" if bold else ".ttf"),
        "/usr/share/fonts/truetype/freefont/FreeSans"
        + ("Bold.ttf" if bold else ".ttf"),
        "/System/Library/Fonts/Supplemental/Arial"
        + (" Bold.ttf" if bold else ".ttf"),
        "/Library/Fonts/Arial" + (" Bold.ttf" if bold else ".ttf"),
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _wrap_lines(draw, text: str, font, max_w: int) -> list:
    """Split text into lines that fit within max_w pixels."""
    words = str(text).split()
    lines, current = [], []
    for word in words:
        test = " ".join(current + [word])
        bb = draw.textbbox((0, 0), test, font=font)
        if bb[2] <= max_w:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines or [""]


def _draw_wrapped(draw, text: str, font, x: int, y: int,
                  max_w: int, color, spacing: float = 1.45) -> int:
    """Draw wrapped text; return y after last line."""
    lines = _wrap_lines(draw, text, font, max_w)
    bb = draw.textbbox((0, 0), "Ay", font=font)
    lh = int((bb[3] - bb[1]) * spacing)
    for line in lines:
        draw.text((x, y), line, font=font, fill=color)
        y += lh
    return y


def _rounded_rect(draw, xy, radius: int, fill=None, outline=None, width=1):
    """Draw rounded rectangle; works on Pillow >= 8.2, falls back on older."""
    try:
        draw.rounded_rectangle(xy, radius=radius, fill=fill,
                               outline=outline, width=width)
    except AttributeError:
        draw.rectangle(xy, fill=fill, outline=outline, width=width)


def _divider(draw, y: int, pad: int = _CARD_PAD) -> int:
    draw.rectangle([pad, y, _CARD_W - pad, y + 1], fill=_CRD_DIVIDER)
    return y + 40


def build_card(recipe: dict, nutrition: dict = None,
               photo_b64: str = None, source_url: str = "") -> bytes:
    """
    Vertical PNG recipe card (1080 px wide, height auto).
    Layout top → bottom:
      photo 1080×720  |  title  |  meta  |
      ingredients (checkboxes)  |  steps (numbered)  |  КБЖУ grid  |  footer
    """
    from PIL import Image as PILImg, ImageDraw

    # Working canvas (tall; we crop at end)
    CANVAS_H = 5000
    img  = PILImg.new("RGB", (_CARD_W, CANVAS_H), _CRD_BG)
    draw = ImageDraw.Draw(img)

    # ── Fonts ─────────────────────────────────────────────────────────────────
    f_title   = _get_font(True,  52)
    f_section = _get_font(True,  30)
    f_body    = _get_font(False, 26)
    f_meta    = _get_font(False, 22)
    f_sm_bold = _get_font(True,  20)
    f_caption = _get_font(False, 18)

    y = 0   # current y cursor

    # ── Photo ─────────────────────────────────────────────────────────────────
    if photo_b64:
        try:
            raw = base64.b64decode(photo_b64)
            photo = PILImg.open(io.BytesIO(raw)).convert("RGB")
            pw, ph = photo.size
            ratio  = max(_CARD_W / pw, _PHOTO_H / ph)
            nw, nh = int(pw * ratio), int(ph * ratio)
            photo  = photo.resize((nw, nh), PILImg.LANCZOS)
            left   = (nw - _CARD_W) // 2
            top    = (nh - _PHOTO_H) // 2
            photo  = photo.crop((left, top, left + _CARD_W, top + _PHOTO_H))
            img.paste(photo, (0, 0))
        except Exception:
            draw.rectangle([0, 0, _CARD_W, _PHOTO_H], fill=(232, 227, 220))
    else:
        draw.rectangle([0, 0, _CARD_W, _PHOTO_H], fill=(232, 227, 220))

    # Author overlay strip (bottom of photo)
    src = recipe.get("source", "")
    if src or source_url:
        from PIL import Image as PILImg2
        STRIP = 60
        overlay = PILImg2.new("RGBA", (_CARD_W, STRIP), (28, 28, 28, 150))
        base    = img.convert("RGBA")
        base.paste(overlay, (0, _PHOTO_H - STRIP), overlay)
        img  = base.convert("RGB")
        draw = ImageDraw.Draw(img)
        caption = f"by {src}" if src else ""
        draw.text((_CARD_PAD, _PHOTO_H - STRIP + 18), caption,
                  font=f_sm_bold, fill=_CRD_WHITE)
        if source_url:
            disp = source_url if len(source_url) < 46 else source_url[:43] + "…"
            bb = draw.textbbox((0, 0), disp, font=f_caption)
            draw.text((_CARD_W - _CARD_PAD - (bb[2] - bb[0]),
                       _PHOTO_H - STRIP + 20),
                      disp, font=f_caption, fill=(200, 200, 200))

    y = _PHOTO_H + 52

    # ── Title ──────────────────────────────────────────────────────────────────
    y = _draw_wrapped(draw, recipe.get("title", "Recipe"),
                      f_title, _CARD_PAD, y, _CONTENT_W, _CRD_TEXT, 1.2)
    y += 14

    # ── Meta row ───────────────────────────────────────────────────────────────
    per100 = (nutrition or {}).get("per_100g", {})
    cal    = per100.get("calories")
    meta_parts = []
    if src:
        meta_parts.append(f"by {src}")
    if cal:
        meta_parts.append(f"{cal} kcal / 100 g")
    if meta_parts:
        draw.text((_CARD_PAD, y), "  ·  ".join(meta_parts),
                  font=f_meta, fill=_CRD_MUTED)
        y += 38
    y += 8

    y = _divider(draw, y)

    # ── Ingredients ─────────────────────────────────────────────────────────────
    draw.text((_CARD_PAD, y), "Ingredients", font=f_section, fill=_CRD_TEXT)
    y += 50

    CB  = 22                          # checkbox side (px)
    TXI = _CARD_PAD + CB + 16        # text x for ingredients
    TWI = _CONTENT_W - CB - 16       # text width for ingredients

    for ing in recipe.get("ingredients", []):
        amt   = ing.get("amount")
        unit  = ing.get("unit") or ""
        name  = ing.get("name", "")
        notes = ing.get("notes", "")

        # Checkbox square
        _rounded_rect(draw, [_CARD_PAD, y + 4, _CARD_PAD + CB, y + 4 + CB],
                      radius=4, outline=_CRD_ACCENT, width=2)

        # Format amount with fractions
        amt_val = amt
        if isinstance(amt_val, float):
            amt_val = _FRAC.get(round(amt_val, 2),
                                int(amt_val) if amt_val == int(amt_val) else amt_val)
        amt_str = f"{amt_val} {unit}".strip() if amt else ""
        text    = f"{amt_str} {name}".strip() if amt_str else name
        if notes:
            text += f" ({notes})"

        y = _draw_wrapped(draw, text, f_body, TXI, y, TWI, _CRD_TEXT, 1.35)
        y += 8

    y += 16
    y = _divider(draw, y)

    # ── Steps ──────────────────────────────────────────────────────────────────
    draw.text((_CARD_PAD, y), "Instructions", font=f_section, fill=_CRD_TEXT)
    y += 50

    NR  = 20                          # step circle radius
    TXS = _CARD_PAD + NR * 2 + 16    # text x for steps
    TWS = _CONTENT_W - NR * 2 - 16   # text width for steps

    for i, step in enumerate(recipe.get("steps", []), 1):
        cy = y + NR                   # circle centre y
        draw.ellipse([_CARD_PAD, y, _CARD_PAD + NR * 2, y + NR * 2],
                     fill=_CRD_ACCENT)
        num = str(i)
        nb  = draw.textbbox((0, 0), num, font=f_sm_bold)
        nw  = nb[2] - nb[0]
        nh  = nb[3] - nb[1]
        draw.text((_CARD_PAD + NR - nw // 2, cy - nh // 2 - 1),
                  num, font=f_sm_bold, fill=_CRD_WHITE)

        y_after = _draw_wrapped(draw, step, f_body, TXS, y, TWS, _CRD_TEXT, 1.5)
        y = max(y_after, y + NR * 2 + 4)
        y += 10

    y += 16
    y = _divider(draw, y)

    # ── Nutrition ──────────────────────────────────────────────────────────────
    if nutrition:
        draw.text((_CARD_PAD, y), "Nutrition  ·  per 100 g",
                  font=f_section, fill=_CRD_TEXT)
        y += 52

        nuts = [
            ("Calories", per100.get("calories", "—"), "kcal"),
            ("Protein",  per100.get("protein",  "—"), "g"),
            ("Fat",      per100.get("fat",       "—"), "g"),
            ("Carbs",    per100.get("carbs",     "—"), "g"),
            ("Fiber",    per100.get("fiber",     "—"), "g"),
        ]
        N    = len(nuts)
        GAP  = 10
        BW   = (_CONTENT_W - GAP * (N - 1)) // N
        BH   = 96

        for i, (label, val, unit) in enumerate(nuts):
            bx = _CARD_PAD + i * (BW + GAP)
            _rounded_rect(draw, [bx, y, bx + BW, y + BH],
                          radius=14, fill=_CRD_CARD)
            vs  = str(val)
            vbb = draw.textbbox((0, 0), vs, font=f_section)
            vw  = vbb[2] - vbb[0]
            draw.text((bx + (BW - vw) // 2, y + 12), vs,
                      font=f_section, fill=_CRD_TEXT)
            ubs = draw.textbbox((0, 0), unit, font=f_caption)
            uw  = ubs[2] - ubs[0]
            draw.text((bx + (BW - uw) // 2, y + 52), unit,
                      font=f_caption, fill=_CRD_MUTED)
            lbs = draw.textbbox((0, 0), label.upper(), font=f_caption)
            lw  = lbs[2] - lbs[0]
            draw.text((bx + (BW - lw) // 2, y + 70), label.upper(),
                      font=f_caption, fill=_CRD_MUTED)

        y += BH + 14
        note = nutrition.get("notes", "")
        if note:
            y = _draw_wrapped(draw, f"* {note}", f_caption,
                              _CARD_PAD, y + 4, _CONTENT_W, _CRD_MUTED, 1.3)

    y += 32

    # ── Footer ─────────────────────────────────────────────────────────────────
    draw.rectangle([_CARD_PAD, y, _CARD_W - _CARD_PAD, y + 1], fill=_CRD_DIVIDER)
    y += 20
    draw.text((_CARD_PAD, y), "Cook not watch", font=f_sm_bold, fill=_CRD_ACCENT)
    y += 52

    # Crop canvas to content height
    result = img.crop((0, 0, _CARD_W, y))

    buf = io.BytesIO()
    result.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return buf.read()


@app.post("/generate-card")
def generate_card(req: PDFRequest):
    """Generate a Pinterest/Instagram-style PNG recipe card."""
    try:
        png_bytes = build_card(
            req.recipe, req.nutrition, req.photo_b64, req.source_url
        )
    except Exception as e:
        raise HTTPException(500, f"Card generation failed: {e}")

    title    = req.recipe.get("title", "recipe").replace(" ", "_").lower()
    filename = f"{title}_cook_not_watch.png"
    return Response(
        content=png_bytes,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Frontend  (single-file HTML/CSS/JS)
# ─────────────────────────────────────────────────────────────────────────────

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Cook not watch</title>
  <style>
    :root {
      --bg:     #F4F0EC;   /* Cloud Dancer – Pantone 2026 */
      --card:   #E8E3DC;
      --dark:   #1C1C1C;
      --brown:  #6B5B45;
      --sage:   #5C8A6F;
      --muted:  #9A8A7A;
      --border: #D8D2C8;
      --white:  #FFFFFF;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--dark);
      font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
      min-height: 100vh;
    }

    /* ── Layout ───────────────────────────────────────────── */
    .wrap { max-width: 760px; margin: 0 auto; padding: 36px 20px 80px; }

    /* ── Header ───────────────────────────────────────────── */
    .header {
      display: flex; justify-content: space-between;
      align-items: center; margin-bottom: 40px;
    }
    .logo { font-size: 18px; font-weight: 800; letter-spacing: -0.5px; }
    .logo em { color: var(--sage); font-style: normal; }

    .lang-toggle { display: flex; gap: 4px; }
    .lang-btn {
      padding: 5px 13px; border-radius: 20px;
      border: 1px solid var(--border); background: transparent;
      cursor: pointer; font-size: 13px; color: var(--muted);
      transition: all .18s;
    }
    .lang-btn.active { background: var(--dark); color: #fff; border-color: var(--dark); }

    /* ── Hero ─────────────────────────────────────────────── */
    .hero { margin-bottom: 32px; }
    .hero h1 {
      font-size: 34px; font-weight: 900; line-height: 1.15;
      letter-spacing: -1.2px;
    }
    .hero p { margin-top: 9px; font-size: 15px; color: var(--muted); line-height: 1.5; }

    /* ── Tabs ─────────────────────────────────────────────── */
    .tabs { display: flex; gap: 6px; margin-bottom: 14px; }
    .tab {
      padding: 8px 18px; border-radius: 20px;
      border: 1px solid var(--border); background: transparent;
      cursor: pointer; font-size: 14px; color: var(--muted);
      transition: all .18s;
    }
    .tab.active { background: var(--dark); color: #fff; border-color: var(--dark); }

    /* ── Panels ───────────────────────────────────────────── */
    .panel { display: none; }
    .panel.active { display: block; }

    input[type=text], textarea {
      width: 100%; padding: 14px 16px;
      border: 1.5px solid var(--border); border-radius: 12px;
      background: var(--white); font-size: 15px; color: var(--dark);
      outline: none; transition: border-color .18s;
      font-family: inherit;
    }
    input[type=text]:focus, textarea:focus { border-color: var(--sage); }
    textarea { min-height: 120px; resize: vertical; }

    .file-drop {
      border: 2px dashed var(--border); border-radius: 14px;
      padding: 40px 20px; text-align: center; cursor: pointer;
      transition: border-color .18s; background: var(--white);
    }
    .file-drop:hover { border-color: var(--sage); }
    .file-drop input { display: none; }
    .drop-icon  { font-size: 36px; margin-bottom: 10px; }
    .drop-label { font-size: 14px; color: var(--muted); }
    .file-preview { margin-top: 12px; }
    .file-preview img { max-height: 180px; border-radius: 10px; object-fit: cover; }

    /* ── Buttons ──────────────────────────────────────────── */
    .btn {
      display: block; width: 100%; margin-top: 14px; padding: 15px;
      background: var(--dark); color: #fff; border: none; border-radius: 12px;
      font-size: 16px; font-weight: 700; cursor: pointer;
      transition: opacity .18s; letter-spacing: -0.2px; font-family: inherit;
    }
    .btn:hover { opacity: .82; }
    .btn:disabled { opacity: .35; cursor: not-allowed; }
    .btn.sage { background: var(--sage); margin-top: 14px; }

    /* ── Loading ──────────────────────────────────────────── */
    .loading { text-align: center; padding: 48px 0; color: var(--muted); font-size: 15px; }
    .spinner {
      width: 34px; height: 34px; margin: 0 auto 14px;
      border: 3px solid var(--border); border-top-color: var(--sage);
      border-radius: 50%; animation: spin .75s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Recipe card ──────────────────────────────────────── */
    .recipe-card {
      margin-top: 32px; background: var(--white);
      border-radius: 20px; overflow: hidden;
      box-shadow: 0 2px 24px rgba(0,0,0,.07);
    }
    .recipe-photo { width: 100%; height: 300px; object-fit: cover; display: block; }
    .recipe-body  { padding: 28px 28px 32px; }

    .recipe-title  { font-size: 26px; font-weight: 900; letter-spacing: -.7px; }
    .recipe-source { margin-top: 5px; font-size: 13px; color: var(--muted); }
    .recipe-desc   { margin-top: 12px; font-size: 14px; line-height: 1.65; color: #4A4035; }

    hr.div { border: none; border-top: 1px solid var(--border); margin: 22px 0; }

    .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }
    .col-title {
      font-size: 10px; font-weight: 800; letter-spacing: 1.2px;
      text-transform: uppercase; color: var(--brown); margin-bottom: 14px;
    }

    /* Ingredients */
    .ing-list { list-style: none; }
    .ing-list li {
      display: flex; align-items: flex-start;
      gap: 8px; margin-bottom: 8px; font-size: 14px; line-height: 1.4;
    }
    .ing-dot {
      width: 7px; height: 7px; border-radius: 50%;
      background: var(--sage); flex-shrink: 0; margin-top: 5px;
    }
    .ing-amt { color: var(--muted); white-space: nowrap; flex-shrink: 0; }

    /* Steps */
    .step-list { list-style: none; }
    .step-list li { display: flex; gap: 12px; margin-bottom: 14px; font-size: 14px; line-height: 1.55; }
    .step-num {
      width: 24px; height: 24px; border-radius: 50%;
      background: var(--sage); color: #fff;
      font-size: 11px; font-weight: 800; flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
    }

    /* Nutrition */
    .nutrition { margin-top: 22px; }
    .nut-grid  { display: grid; grid-template-columns: repeat(4,1fr); gap: 10px; margin-top: 12px; }
    .nut-box   { background: var(--card); border-radius: 12px; padding: 14px 8px; text-align: center; }
    .nut-val   { font-size: 22px; font-weight: 900; }
    .nut-unit  { font-size: 11px; color: var(--muted); margin-top: 3px; }
    .nut-lbl   { font-size: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: .5px; margin-top: 3px; }
    .nut-note  { margin-top: 10px; font-size: 12px; color: var(--muted); font-style: italic; }

    /* Error */
    .error-box {
      margin-top: 16px; padding: 14px 16px;
      background: #fff0f0; border-radius: 10px;
      font-size: 14px; color: #a33;
    }

    @media (max-width: 520px) {
      .cols     { grid-template-columns: 1fr; }
      .nut-grid { grid-template-columns: repeat(2,1fr); }
      .hero h1  { font-size: 26px; }
    }
  </style>
</head>
<body>
<div class="wrap">

  <!-- Header -->
  <div class="header">
    <div class="logo">Cook <em>not</em> watch</div>
    <div class="lang-toggle">
      <button class="lang-btn active" onclick="setLang('en')">EN</button>
      <button class="lang-btn"        onclick="setLang('ru')">RU</button>
    </div>
  </div>

  <!-- Hero -->
  <div class="hero">
    <h1 data-i18n="headline"></h1>
    <p  data-i18n="subline"></p>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab active" data-tab="url" onclick="setTab('url')" data-i18n="tab_url"></button>
    <button class="tab"        data-tab="img" onclick="setTab('img')" data-i18n="tab_img"></button>
    <button class="tab"        data-tab="txt" onclick="setTab('txt')" data-i18n="tab_txt"></button>
  </div>

  <!-- Panel: URL -->
  <div class="panel active" id="panel-url">
    <input id="url-input" type="text" data-i18n-ph="url_ph" />
    <button class="btn" onclick="submitUrl()" data-i18n="btn_go"></button>
  </div>

  <!-- Panel: Image -->
  <div class="panel" id="panel-img">
    <div class="file-drop" onclick="document.getElementById('img-file').click()">
      <div class="drop-icon">📷</div>
      <div class="drop-label" data-i18n="img_drop"></div>
      <input type="file" id="img-file" accept="image/*" onchange="onFileChange(event)" />
    </div>
    <div class="file-preview" id="img-preview"></div>
    <input id="img-src" type="text" style="margin-top:10px" data-i18n-ph="src_ph" />
    <button class="btn" onclick="submitImage()" data-i18n="btn_go"></button>
  </div>

  <!-- Panel: Text -->
  <div class="panel" id="panel-txt">
    <input id="txt-src" type="text" style="margin-bottom:10px" data-i18n-ph="src_ph" />
    <textarea id="txt-input" data-i18n-ph="txt_ph"></textarea>
    <button class="btn" onclick="submitText()" data-i18n="btn_go"></button>
  </div>

  <!-- Loading -->
  <div id="loading" style="display:none" class="loading">
    <div class="spinner"></div>
    <span data-i18n="loading"></span>
  </div>

  <!-- Error -->
  <div id="error-box" class="error-box" style="display:none"></div>

  <!-- Result -->
  <div id="result" style="display:none">
    <div class="recipe-card" id="recipe-card"></div>
    <div style="display:flex;gap:10px;margin-top:14px">
      <button class="btn sage" style="margin-top:0" onclick="downloadCard()" data-i18n="btn_card"></button>
      <button class="btn" style="margin-top:0;background:var(--card);color:var(--dark)" onclick="downloadPDF()" data-i18n="btn_pdf"></button>
    </div>
    <!-- Card preview -->
    <div id="card-preview" style="display:none;margin-top:24px">
      <p style="font-size:13px;color:var(--muted);margin-bottom:10px" data-i18n="card_preview_label"></p>
      <img id="card-preview-img" style="width:100%;border-radius:14px;box-shadow:0 2px 20px rgba(0,0,0,.1)" />
    </div>
  </div>

</div><!-- .wrap -->

<script>
// ── Translations ──────────────────────────────────────────────────────────────
const T = {
  en: {
    headline: "Turn recipe videos into recipe cards",
    subline:  "Paste a TikTok or Instagram link, upload a screenshot, or paste the recipe text — get a clean card with ingredients, steps & macros.",
    tab_url:  "Video link",
    tab_img:  "Screenshot",
    tab_txt:  "Text",
    url_ph:   "Paste TikTok or Instagram URL…",
    img_drop: "Click to choose a screenshot",
    src_ph:   "Source URL (optional)",
    txt_ph:   "Paste recipe text or caption here…",
    btn_go:   "Extract recipe →",
    btn_pdf:  "⬇ Download PDF",
    btn_card:          "⬇ Download as Image",
    card_preview_label:"Preview:",
    loading:           "Analysing… usually 20–30 seconds",
    ing:      "Ingredients",
    steps:    "Steps",
    nut:      "Nutrition · per 100 g",
    by:       "by",
    conf:     "Confidence",
  },
  ru: {
    headline: "Из видео с рецептом — в карточку рецепта",
    subline:  "Вставь ссылку на TikTok или Instagram, загрузи скриншот или текст — получи чистую карточку с ингредиентами, шагами и КБЖУ.",
    tab_url:  "Ссылка на видео",
    tab_img:  "Скриншот",
    tab_txt:  "Текст",
    url_ph:   "Вставь ссылку на TikTok или Instagram…",
    img_drop: "Нажми, чтобы выбрать скриншот",
    src_ph:   "Ссылка на источник (необязательно)",
    txt_ph:   "Вставь текст рецепта или подпись…",
    btn_go:   "Извлечь рецепт →",
    btn_pdf:  "⬇ Скачать PDF",
    btn_card:          "⬇ Скачать как картинку",
    card_preview_label:"Превью:",
    loading:           "Анализирую… обычно 20–30 секунд",
    ing:      "Ингредиенты",
    steps:    "Шаги",
    nut:      "КБЖУ · на 100 г",
    by:       "от",
    conf:     "Уверенность",
  },
};

let lang = "en";
let lastResult = null;
let imgFile = null;

// ── i18n ──────────────────────────────────────────────────────────────────────
function setLang(l) {
  lang = l;
  document.querySelectorAll(".lang-btn").forEach(b =>
    b.classList.toggle("active", b.textContent.trim() === l.toUpperCase())
  );
  applyI18n();
  if (lastResult) renderRecipe(lastResult);
}

function applyI18n() {
  const t = T[lang];
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const k = el.getAttribute("data-i18n");
    if (t[k] !== undefined) el.textContent = t[k];
  });
  document.querySelectorAll("[data-i18n-ph]").forEach(el => {
    const k = el.getAttribute("data-i18n-ph");
    if (t[k] !== undefined) el.placeholder = t[k];
  });
}

// ── Tabs ──────────────────────────────────────────────────────────────────────
function setTab(id) {
  document.querySelectorAll(".tab").forEach(b =>
    b.classList.toggle("active", b.getAttribute("data-tab") === id)
  );
  document.querySelectorAll(".panel").forEach(p =>
    p.classList.toggle("active", p.id === "panel-" + id)
  );
}

// ── File upload ───────────────────────────────────────────────────────────────
function onFileChange(e) {
  imgFile = e.target.files[0];
  if (!imgFile) return;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById("img-preview").innerHTML =
      `<img src="${ev.target.result}" />`;
  };
  reader.readAsDataURL(imgFile);
}

// ── UI helpers ────────────────────────────────────────────────────────────────
function showLoading(on) {
  document.getElementById("loading").style.display   = on ? "block" : "none";
  document.getElementById("error-box").style.display = "none";
  if (on) document.getElementById("result").style.display = "none";
}

function showError(msg) {
  const el = document.getElementById("error-box");
  el.textContent = "⚠ " + msg;
  el.style.display = "block";
  document.getElementById("loading").style.display = "none";
}

// ── Submit handlers ───────────────────────────────────────────────────────────
async function submitUrl() {
  const url = document.getElementById("url-input").value.trim();
  if (!url) return;
  showLoading(true);
  try {
    const res  = await fetch("/extract-from-video", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Server error");
    showLoading(false);
    lastResult = data;
    renderRecipe(data);
  } catch(e) { showError(e.message); }
}

async function submitImage() {
  if (!imgFile) { showError("Please choose an image first."); return; }
  showLoading(true);
  const fd = new FormData();
  fd.append("file", imgFile);
  const src = document.getElementById("img-src").value.trim();
  if (src) fd.append("source_url", src);
  try {
    const res  = await fetch("/extract-from-image", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Server error");
    showLoading(false);
    lastResult = data;
    renderRecipe(data);
  } catch(e) { showError(e.message); }
}

async function submitText() {
  const text = document.getElementById("txt-input").value.trim();
  const url  = document.getElementById("txt-src").value.trim();
  if (!text) return;
  showLoading(true);
  try {
    const res  = await fetch("/extract-from-text", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({ text, url }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Server error");
    showLoading(false);
    lastResult = data;
    renderRecipe(data);
  } catch(e) { showError(e.message); }
}

// ── Render recipe card ────────────────────────────────────────────────────────
function renderRecipe({ recipe, nutrition, photo_b64 }) {
  const t = T[lang];
  let h = "";

  if (photo_b64)
    h += `<img class="recipe-photo" src="data:image/jpeg;base64,${photo_b64}" alt="" />`;

  h += `<div class="recipe-body">`;
  h += `<div class="recipe-title">${esc(recipe.title || "")}</div>`;
  if (recipe.source)
    h += `<div class="recipe-source">${t.by} ${esc(recipe.source)}</div>`;
  if (recipe.description)
    h += `<div class="recipe-desc">${esc(recipe.description)}</div>`;

  h += `<hr class="div" /><div class="cols">`;

  // Ingredients
  h += `<div><div class="col-title">${t.ing}</div><ul class="ing-list">`;
  for (const ing of (recipe.ingredients || [])) {
    const amt = ing.amount
      ? `${Number.isInteger(ing.amount) ? ing.amount : ing.amount} ${ing.unit || ""}`.trim()
      : "";
    h += `<li>
      <span class="ing-dot"></span>
      ${amt ? `<span class="ing-amt">${esc(amt)}</span>` : ""}
      <span>${esc(ing.name)}${ing.notes ? ` <span style="color:var(--muted)">(${esc(ing.notes)})</span>` : ""}</span>
    </li>`;
  }
  h += `</ul></div>`;

  // Steps
  h += `<div><div class="col-title">${t.steps}</div><ol class="step-list">`;
  (recipe.steps || []).forEach((s, i) => {
    h += `<li><span class="step-num">${i+1}</span><span>${esc(s)}</span></li>`;
  });
  h += `</ol></div></div>`;

  // Nutrition
  if (nutrition) {
    const p = nutrition.per_100g || {};
    h += `<div class="nutrition">
      <div class="col-title">${t.nut}</div>
      <div class="nut-grid">
        <div class="nut-box"><div class="nut-val">${p.calories ?? "—"}</div><div class="nut-unit">kcal</div><div class="nut-lbl">Calories</div></div>
        <div class="nut-box"><div class="nut-val">${p.protein  ?? "—"}</div><div class="nut-unit">g</div><div class="nut-lbl">Protein</div></div>
        <div class="nut-box"><div class="nut-val">${p.fat      ?? "—"}</div><div class="nut-unit">g</div><div class="nut-lbl">Fat</div></div>
        <div class="nut-box"><div class="nut-val">${p.carbs    ?? "—"}</div><div class="nut-unit">g</div><div class="nut-lbl">Carbs</div></div>
      </div>`;
    if (nutrition.notes)
      h += `<div class="nut-note">* ${esc(nutrition.notes)}</div>`;
    h += `</div>`;
  }

  h += `</div>`;
  document.getElementById("recipe-card").innerHTML = h;
  document.getElementById("result").style.display = "block";
}

// ── PDF download ──────────────────────────────────────────────────────────────
async function downloadPDF() {
  if (!lastResult) return;
  try {
    const res = await fetch("/generate-pdf", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        recipe:     lastResult.recipe,
        nutrition:  lastResult.nutrition,
        photo_b64:  lastResult.photo_b64,
        source_url: lastResult.source_url || "",
      }),
    });
    if (!res.ok) throw new Error("PDF generation failed");
    const blob = await res.blob();
    const a    = document.createElement("a");
    a.href     = URL.createObjectURL(blob);
    a.download = (lastResult.recipe?.title || "recipe")
                   .replace(/\\s+/g, "_").toLowerCase() + "_cook_not_watch.pdf";
    a.click();
  } catch(e) { showError(e.message); }
}

// ── Image card download ───────────────────────────────────────────────────────
async function downloadCard() {
  if (!lastResult) return;
  const btn = document.querySelector('[data-i18n="btn_card"]');
  const orig = btn ? btn.textContent : "";
  if (btn) btn.textContent = "Generating…";
  try {
    const res = await fetch("/generate-card", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        recipe:     lastResult.recipe,
        nutrition:  lastResult.nutrition,
        photo_b64:  lastResult.photo_b64,
        source_url: lastResult.source_url || "",
      }),
    });
    if (!res.ok) throw new Error("Card generation failed");
    const blob    = await res.blob();
    const objUrl  = URL.createObjectURL(blob);
    // Download
    const a       = document.createElement("a");
    a.href        = objUrl;
    a.download    = (lastResult.recipe?.title || "recipe")
                      .replace(/\\s+/g, "_").toLowerCase() + "_cook_not_watch.png";
    a.click();
    // Show inline preview
    const preview = document.getElementById("card-preview");
    const img     = document.getElementById("card-preview-img");
    img.src       = objUrl;
    preview.style.display = "block";
  } catch(e) { showError(e.message); }
  finally { if (btn) btn.textContent = orig; }
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// Init
applyI18n();
</script>
</body>
</html>"""


@app.get("/app", response_class=HTMLResponse)
def app_ui():
    return HTML
