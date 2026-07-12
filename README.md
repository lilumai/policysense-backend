# PolicySense — Application Tier (FastAPI)

## รันบน localhost

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# แก้ .env ใส่ ANTHROPIC_API_KEY=sk-ant-... ของคุณ (ไม่ commit ไฟล์นี้)

uvicorn main:app --reload --port 8000
```

เปิด http://localhost:8000/docs จะเห็น Swagger UI ทดสอบ endpoint ได้เลยโดยไม่ต้องเขียน frontend ก่อน

## Endpoints

| Method | Path | หน้าที่ |
|---|---|---|
| GET | `/health` | เช็คว่า server รันอยู่ + มี API key หรือไม่ |
| GET | `/policies` | คืนกรมธรรม์ตัวอย่าง (mock — ยังไม่ต่อ DB) |
| POST | `/analyze` | **หลัก** — รับ profile+policies คืน gap/overlap ทุกหมวด + LLM อธิบาย |
| POST | `/tax` | คำนวณสิทธิ์ลดหย่อนภาษี (deterministic, cap ถูกต้องตามกรมสรรพากร) |

## สถาปัตยกรรมสำคัญ — อ่านก่อนแก้โค้ด

**`rules.py`** = deterministic core — ตัดสินใจทุกตัวเลข gap/overlap/target
**`main.py`** = เรียก LLM เพื่อ "อธิบาย" ผลจาก rules.py เท่านั้น ห้ามให้ LLM คิดตัวเลขเอง

ทุก target ใน `rules.py` มี `tier`:
- `regulatory_backed` = อ้างอิงได้ (SET, กรมสรรพากร) — มี source ติดมาด้วย
- `heuristic` = **ค่า default ที่ต้องแก้ก่อนส่ง IS จริง** ดูหมวด ipd/ci/pa ใน `TARGET_RULES`

## แก้ heuristic targets (ก่อน present)

เปิด `rules.py` หา `_ipd_target_default`, `_ci_target_default`, `_pa_target_default`
ใส่ตัวเลขที่มาจาก domain expertise ของคุณเอง แล้วอัปเดต `"source"` string ให้ตรง

## Bug ที่แก้จาก prototype เดิม

Prototype static HTML เดิมคำนวณเพดานลดหย่อนภาษีผิด — บวกประกันชีวิต (100,000) +
ประกันสุขภาพ (25,000) แยกกันเป็น 125,000 ทั้งที่จริงเพดานรวมกันต้องไม่เกิน 100,000
(ตามกรมสรรพากร). แก้แล้วใน `calc_tax_deduction()` — ดู `TAX_RULES` comment

## Deploy

- Backend: Render / Railway (ไม่ใช่ Vercel — Vercel เหมาะกับ static/serverless
  ระยะสั้น ไม่เหมาะ FastAPI ระยะยาวที่ต้อง keep process)
- ใส่ `ANTHROPIC_API_KEY` เป็น environment variable บน platform นั้น ๆ ห้าม hardcode
- Frontend (Vercel เดิม) แก้ `fetch()` ให้ชี้มาที่ backend URL แทน `api.anthropic.com` ตรง ๆ
