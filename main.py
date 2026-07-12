"""
main.py — PolicySense application tier (FastAPI).

Architecture (per MADT8104 3-tier model):
  Presentation (browser)  ->  THIS FILE (application tier)  ->  rules.py (deterministic logic)
                                        |
                                        v
                              Gemini API (explain-only, never decides)

IMPORTANT: The rule engine (rules.py) computes every number and every
gap/overlap decision. The LLM is called ONLY to translate that already-decided
JSON into plain Thai. If GEMINI_API_KEY is missing or the call fails,
the API still returns full results — just without the prose explanation.
This means the app is NEVER silently wrong the way the static-HTML prototype was:
the rule-based JSON in the response is always real, always computed here.
"""

import os
import json
import base64
import pathlib
import uuid
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
import pdfplumber

from rules import Profile, Policy, analyze_portfolio, calc_tax_deduction

load_dotenv()

STATIC_DIR = pathlib.Path(__file__).parent / "static"
FRONTEND_FILE = STATIC_DIR / "index.html"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

app = FastAPI(title="PolicySense API")

# Dev CORS — tighten origins before real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------

class ProfileIn(BaseModel):
    annual_income: float
    dependents: int = 0
    age: int = 35

    # optional — unlocks Needs-Approach life target (Rule B)
    debt_outstanding: Optional[float] = None
    family_monthly_expense: Optional[float] = None
    children_education_cost: Optional[float] = None
    existing_assets: Optional[float] = None

    # optional — IPD room-rate target depends on hospital tier
    hospital_tier: str = "general"  # "premium" | "general" | "economy"

    # optional — self-declared copay status (simplification; see rules.py)
    has_copayment_status: bool = False

    # optional — retirement goal (expense-replacement method)
    current_annual_expense: Optional[float] = None
    retirement_age: int = 60
    life_expectancy: int = 85

    # optional — education goal
    education_goal_amount: Optional[float] = None
    education_goal_years: Optional[int] = None


class PolicyIn(BaseModel):
    id: str
    insurer: str
    category: str
    sum_insured: float
    annual_premium: float = 0.0


class AnalyzeRequest(BaseModel):
    profile: ProfileIn
    policies: List[PolicyIn]
    explain: bool = True   # set false to skip the LLM call entirely


class TaxRequest(BaseModel):
    life_premium: float
    health_premium: float
    pension_premium: float
    gross_income: float


# ---------------------------------------------------------------------------
# Serve frontend — presentation tier, delivered by the same server as the
# application tier. Eliminates CORS entirely (same origin) and means running
# `uvicorn main:app` is the ONLY command needed — no second terminal, no
# separate `python -m http.server`, no file:// issues.
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_FILE)


# ---------------------------------------------------------------------------
# POST /extract — reads real uploaded policy PDF(s), extracts structured
# coverage data via LLM (extraction-only, not decision-making), returns it
# for the USER TO REVIEW/EDIT before it becomes input to /analyze.
#
# This is a DIFFERENT LLM function from _explain_with_llm above:
#   - explain layer: rule_output (already-decided JSON) -> Thai prose
#   - extract layer: unstructured PDF text -> structured JSON (candidate input)
# The extract layer requires a real GEMINI_API_KEY — there is no
# rule-based way to read an arbitrary PDF's coverage table.
#
# SAFETY DESIGN: the LLM is instructed to extract only what is literally
# printed, using null when uncertain, rather than guessing — because a
# hallucinated number here corrupts every downstream gap calculation.
# The frontend must show these results in an editable review step, never
# feed them straight into /analyze unconfirmed.
#
# LIMITATION: pdfplumber reads text from digital (text-layer) PDFs only.
# Scanned images or photos (e.g. saved from LINE) have no text layer and
# will return empty/near-empty text — flagged in the response so the
# frontend can tell the user to enter that policy manually instead of
# silently producing an empty result.
# ---------------------------------------------------------------------------

EXTRACT_SYSTEM_PROMPT = """\
คุณเป็นตัวสกัดข้อมูล (extractor) ไม่ใช่ที่ปรึกษา และไม่ใช่ผู้ตัดสินใจ

หน้าที่: อ่านข้อความจากกรมธรรม์ประกันภัยที่แนบมา แล้วดึงเฉพาะตัวเลขความคุ้มครอง
ที่ระบุไว้ชัดเจนในเอกสาร ออกมาเป็น JSON เท่านั้น

กฎเหล็ก:
1. ห้ามเดาตัวเลขที่ไม่ได้ระบุในเอกสาร — ถ้าไม่แน่ใจหรือหาไม่เจอ ให้ใส่ null
2. ห้ามให้คำแนะนำ ห้ามประเมินว่าพอหรือไม่พอ (นั่นเป็นหน้าที่ของ rule engine อื่น)
3. แต่ละรายการความคุ้มครองต้องจัดเข้าหมวดใดหมวดหนึ่งจาก 6 หมวดนี้เท่านั้น:
   - life: ทุนประกันชีวิต (เสียชีวิตทุกกรณี ไม่ใช่จากอุบัติเหตุ)
   - ipd_room: ค่าห้องผู้ป่วยใน ต่อคืน (บาท/คืน)
   - ipd_lumpsum: วงเงินรักษาพยาบาลผู้ป่วยในแบบเหมาจ่ายต่อปี (บาท/ปี)
   - ci: เงินก้อนโรคร้ายแรง (Critical Illness lump sum)
   - pa_medical: ค่ารักษาอุบัติเหตุต่อครั้ง (บาท/ครั้ง)
   - pa_death: ทุนเสียชีวิต/ทุพพลภาพจากอุบัติเหตุ
   ถ้าพบความคุ้มครองที่ไม่เข้าหมวดใดเลย ให้ใส่ category เป็น "other" พร้อม description
4. ตอบเป็น JSON array ล้วน ไม่มีข้อความอื่นก่อน/หลัง ไม่มี markdown code fence
   รูปแบบแต่ละ item:
   {"insurer": "...", "category": "...", "sum_insured": ตัวเลขหรือ null,
    "annual_premium": ตัวเลขหรือ null, "raw_text": "ข้อความต้นฉบับที่อ้างอิง",
    "confidence": "high" | "low"}
5. ใช้ "confidence": "low" เมื่อข้อความไม่ชัดเจนหรือกำกวม — ผู้ใช้จะเป็นคนตรวจสอบต่อ
"""


class ExtractedItem(BaseModel):
    insurer: Optional[str] = None
    category: str
    sum_insured: Optional[float] = None
    annual_premium: Optional[float] = None
    raw_text: Optional[str] = None
    confidence: str = "low"


class ExtractResponse(BaseModel):
    filename: str
    text_extracted: bool
    char_count: int
    items: List[ExtractedItem]
    warning: Optional[str] = None


IMAGE_MIME_TYPES = {"image/png", "image/jpeg", "image/jpg"}


@app.post("/extract", response_model=ExtractResponse)
async def extract_policy(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY ยังไม่ได้ตั้งค่าใน .env — ฟีเจอร์นี้ต้องใช้ LLM อ่านเอกสารจริง",
        )

    raw = await file.read()

    # --- image path: no text layer to read, send bytes straight to the LLM ---
    if file.content_type in IMAGE_MIME_TYPES:
        items = await _extract_image_with_llm(raw, file.content_type)
        return ExtractResponse(
            filename=file.filename,
            text_extracted=True,
            char_count=0,
            items=items,
            warning=None if items else "อ่านภาพได้ แต่ไม่พบตัวเลขความคุ้มครองที่จับคู่ได้ชัดเจน",
        )

    # --- extract text from digital PDF (no OCR/vision for scanned images) ---
    text = ""
    try:
        with pdfplumber.open(__import__("io").BytesIO(raw)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"อ่านไฟล์ PDF ไม่ได้: {e}")

    if len(text.strip()) < 30:
        # Almost certainly a scanned image / photo with no text layer.
        return ExtractResponse(
            filename=file.filename,
            text_extracted=False,
            char_count=len(text.strip()),
            items=[],
            warning=(
                "ไฟล์นี้ดูเหมือนเป็นรูปสแกน/ถ่ายรูป ไม่มีข้อความให้อ่าน "
                "(ระบบตอนนี้รองรับเฉพาะ PDF ที่ copy ข้อความได้ — "
                "กรุณากรอกกรมธรรม์ฉบับนี้ด้วยตนเอง)"
            ),
        )

    print("===== RAW PDF TEXT (first 2000 chars) =====")
    print(text[:2000])
    print("===== END RAW PDF TEXT =====")
    items = await _extract_with_llm(text[:15_000])  # cap to keep prompt small/cheap

    return ExtractResponse(
        filename=file.filename,
        text_extracted=True,
        char_count=len(text),
        items=items,
        warning=None if items else "อ่านข้อความได้ แต่ไม่พบตัวเลขความคุ้มครองที่จับคู่ได้ชัดเจน",
    )


async def _extract_with_llm(document_text: str) -> List[ExtractedItem]:
    payload = {
        "systemInstruction": {"parts": [{"text": EXTRACT_SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": document_text}]}],
        "generationConfig": {"maxOutputTokens": 2000},
    }
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "content-type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                headers=headers,
                json=payload,
            )
        r.raise_for_status()
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Defensive: strip stray markdown fences if the model adds them anyway
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        print("===== GEMINI RAW RESPONSE START =====")
        print(text)
        print("===== GEMINI RAW RESPONSE END =====")
        parsed = json.loads(text)
        return [ExtractedItem(**item) for item in parsed]
    except httpx.HTTPStatusError as e:
        print(f"===== HTTP ERROR {e.response.status_code}: {e.response.text} =====")
        return []
    except Exception as e:
        print(f"===== EXTRACT ERROR: {type(e).__name__}: {e} =====")
        # Extraction failed — return empty list, NOT a guess. The frontend
        # must show this as "couldn't auto-extract, please enter manually",
        # never silently substitute a fabricated number.
        return []


async def _extract_image_with_llm(image_bytes: bytes, mime_type: str) -> List[ExtractedItem]:
    payload = {
        "systemInstruction": {"parts": [{"text": EXTRACT_SYSTEM_PROMPT}]},
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode("utf-8"),
                        }
                    },
                    {"text": "อ่านภาพนี้และสกัดข้อมูลความคุ้มครองตามรูปแบบที่กำหนด"},
                ]
            }
        ],
        "generationConfig": {"maxOutputTokens": 2000},
    }
    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "content-type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                headers=headers,
                json=payload,
            )
        r.raise_for_status()
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        # Defensive: strip stray markdown fences if the model adds them anyway
        if text.startswith("```"):
            text = text.strip("`").removeprefix("json").strip()
        parsed = json.loads(text)
        return [ExtractedItem(**item) for item in parsed]
    except httpx.HTTPStatusError as e:
        print(f"===== HTTP ERROR {e.response.status_code}: {e.response.text} =====")
        return []
    except Exception as e:
        print(f"===== EXTRACT ERROR: {type(e).__name__}: {e} =====")
        # Extraction failed — return empty list, NOT a guess. The frontend
        # must show this as "couldn't auto-extract, please enter manually",
        # never silently substitute a fabricated number.
        return []


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "llm_configured": bool(GEMINI_API_KEY)}


# ---------------------------------------------------------------------------
# GET /policies — placeholder store (swap for DB in Data tier later)
# ---------------------------------------------------------------------------

MOCK_POLICIES = [
    {"id": "P1", "insurer": "เมืองไทยประกันชีวิต", "category": "pa",
     "sum_insured": 30000, "annual_premium": 3500},
    {"id": "P2", "insurer": "KTC กลุ่ม", "category": "pa",
     "sum_insured": 20000, "annual_premium": 0},
    {"id": "P3", "insurer": "AIA", "category": "life",
     "sum_insured": 1000000, "annual_premium": 45000},
]

@app.get("/policies")
def get_policies():
    return MOCK_POLICIES


# ---------------------------------------------------------------------------
# POST /analyze — the core endpoint: gap + overlap + (optional) explanation
# ---------------------------------------------------------------------------

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    profile = Profile(**req.profile.model_dump())
    policies = [Policy(**p.model_dump()) for p in req.policies]

    rule_output = analyze_portfolio(profile, policies)

    response = {"analysis": rule_output, "explanation": None}

    if req.explain:
        text = await _explain_with_llm(rule_output)
        response["explanation"] = text

    return response


# ---------------------------------------------------------------------------
# POST /tax — deterministic tax deduction calculator
# ---------------------------------------------------------------------------

@app.post("/tax")
def tax(req: TaxRequest):
    return calc_tax_deduction(
        life_premium=req.life_premium,
        health_premium=req.health_premium,
        pension_premium=req.pension_premium,
        gross_income=req.gross_income,
    )


# ---------------------------------------------------------------------------
# LLM explain layer — STRICTLY explain-only.
# The prompt forbids introducing any number not already in rule_output.
# ---------------------------------------------------------------------------

EXPLAIN_SYSTEM_PROMPT = """\
คุณเป็นผู้ช่วยอธิบายผลวิเคราะห์พอร์ตประกัน ไม่ใช่ผู้ตัดสินใจ

กฎเหล็ก:
1. ห้ามสร้างตัวเลข เป้าหมาย หรือช่องว่างใหม่ที่ไม่ได้อยู่ใน JSON ที่ให้มา
2. ใช้เฉพาะตัวเลขและสถานะ (gap/overlap/ok) ที่มีอยู่ใน JSON เท่านั้น
3. ถ้าหมวดไหนมี tier เป็น "heuristic" ต้องบอกผู้ใช้ว่าเป็นค่าอ้างอิงเบื้องต้น
   ยังไม่มี benchmark กลางที่เป็นทางการ ควรปรึกษาผู้เชี่ยวชาญเพิ่ม
4. ถ้าหมวดไหนมี tier เป็น "regulatory_backed" ให้อ้างอิงแหล่งที่มาสั้นๆ
5. ตอบเป็นภาษาไทย กระชับ ไม่เกิน 8-10 ประโยค จัดกลุ่มเป็นข้อ
6. ห้ามแนะนำให้ซื้อผลิตภัณฑ์ยี่ห้อใดยี่ห้อหนึ่ง พูดเป็นหมวดคุ้มครองเท่านั้น
"""


async def _explain_with_llm(rule_output: dict) -> Optional[str]:
    if not GEMINI_API_KEY:
        return None

    user_text = (
        "ผลวิเคราะห์จาก rule engine (ห้ามแก้ตัวเลข อธิบายเป็นภาษาคนเท่านั้น):\n"
        + json.dumps(rule_output, ensure_ascii=False)
    )

    payload = {
        "systemInstruction": {"parts": [{"text": EXPLAIN_SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": user_text}]}],
        "generationConfig": {"maxOutputTokens": 800},
    }

    headers = {
        "x-goog-api-key": GEMINI_API_KEY,
        "content-type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                headers=headers,
                json=payload,
            )
        r.raise_for_status()
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        return text or None
    except Exception:
        # Explanation is optional sugar — the deterministic JSON above
        # is already returned regardless of whether this succeeds.
        return None
