"""
rules.py — Deterministic rule engine for PolicySense.

CORE PRINCIPLE (per DSR framing):
  This module makes ALL numeric/decision judgments.
  The LLM layer (main.py) is only allowed to *explain* what this module
  already decided — it never invents a target or a gap on its own.

Every rule carries a `tier`:
  - "regulatory_backed": cites a government/professional body
                          (SET, กรมสรรพากร, TFPA, TLAA)
  - "industry_data":     cites real published market/industry statistics
                          (insurer rate surveys, claims-cost aggregators)
                          — real data, but not an official regulatory body
  - "heuristic":         no citable source found — internal placeholder,
                          flagged to the user, should be replaced when a
                          better source is found

v2 update: targets sourced from domain-expert-provided formulas
(TFPA/SET life needs-approach, Allianz Ayudhya room-rate survey,
รู้ใจ/AIAplanner treatment-cost stats, TLAA New Health Standard copay rule).
IPD and PA are each split into two sub-categories because they combine
values with different units (per-night room rate vs. per-year lump sum;
per-incident medical vs. lump-sum death/disability benefit) — summing
them together would be meaningless.
"""

from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# 1. USER PROFILE — minimum required inputs, plus optional inputs that
#    unlock more precise (needs-approach) targets when supplied.
# ---------------------------------------------------------------------------

@dataclass
class Profile:
    annual_income: float
    dependents: int = 0
    age: int = 35

    # --- optional: unlocks Needs-Approach life target (Rule B) ---
    debt_outstanding: Optional[float] = None
    family_monthly_expense: Optional[float] = None
    children_education_cost: Optional[float] = None
    existing_assets: Optional[float] = None

    # --- optional: IPD room-rate target depends on hospital tier ---
    hospital_tier: str = "general"  # "premium" | "general" | "economy"

    # --- optional: simplified stand-in for full claim-history tracking.
    # A real "New Health Standard copay" status requires claims history,
    # which this system does not persist yet. This flag lets the user
    # self-declare the status as an approximation. ---
    has_copayment_status: bool = False

    # --- optional: retirement goal (expense-replacement method) ---
    current_annual_expense: Optional[float] = None
    retirement_age: int = 60
    life_expectancy: int = 85

    # --- optional: education goal (kept simple, unchanged from v1) ---
    education_goal_amount: Optional[float] = None
    education_goal_years: Optional[int] = None


# ---------------------------------------------------------------------------
# 2. POLICY INPUT — one entry per policy the user holds.
#    category must be one of: life, ipd_room, ipd_lumpsum, ci,
#    pa_medical, pa_death
# ---------------------------------------------------------------------------

@dataclass
class Policy:
    id: str
    insurer: str
    category: str
    sum_insured: float      # ทุนประกัน / วงเงิน — หน่วยตามหมวด (ดู CATEGORY_UNITS)
    annual_premium: float = 0.0

    # --- optional: ties together multiple category rows extracted from the
    # SAME physical policy document (e.g. one PA policy that covers both
    # pa_death and pa_medical). All rows from one document repeat the same
    # annual_premium, so premium must be counted once per group, not once
    # per row — see analyze_portfolio's total_premium aggregation below.
    # Falls back to the row's own id (i.e. no dedup) when the caller
    # doesn't know the source document, e.g. manually-entered policies. ---
    policy_group_id: Optional[str] = None


CATEGORY_UNITS = {
    "life":        "บาท (ทุนประกัน)",
    "ipd_room":    "บาท/คืน",
    "ipd_lumpsum": "บาท/ปี",
    "ci":          "บาท (เงินก้อน)",
    "pa_medical":  "บาท/ครั้ง",
    "pa_death":    "บาท (ทุนประกัน)",
}


# ---------------------------------------------------------------------------
# 3. TARGET RULES
# ---------------------------------------------------------------------------

def _life_target(p: Profile) -> tuple[float, dict]:
    """max(Rule A: income x5, Rule B: needs approach) — TFPA + SET method."""
    rule_a = p.annual_income * 5

    rule_b = None
    if any(v is not None for v in (
        p.debt_outstanding, p.family_monthly_expense,
        p.children_education_cost, p.existing_assets
    )):
        family_5yr = (p.family_monthly_expense or 0) * 12 * 5
        education = p.children_education_cost or 0
        debt = p.debt_outstanding or 0
        existing = p.existing_assets or 0
        rule_b = debt + family_5yr + education - existing
        rule_b = max(rule_b, 0)  # never negative

    target = max(rule_a, rule_b) if rule_b is not None else rule_a
    return target, {"rule_a": rule_a, "rule_b": rule_b}


ROOM_RATE_BY_TIER = {
    "premium": 13_000,   # midpoint of 11,300–14,700 (บำรุงราษฎร์/เมดพาร์ค/กรุงเทพ)
    "general": 7_800,    # midpoint of 6,000–9,600 (พญาไท/สมิติเวช)
    "economy": 4_000,    # midpoint of 3,500–4,500 (วิภาวดี/ศิครินทร์/มงกุฎวัฒนะ)
}

def _ipd_room_target(p: Profile) -> float:
    return ROOM_RATE_BY_TIER.get(p.hospital_tier, ROOM_RATE_BY_TIER["general"])


def _ipd_lumpsum_target(p: Profile) -> float:
    # First Jobber vs 35+ tiers, per รู้ใจ/AIAplanner treatment-cost stats
    return 2_000_000.0 if p.age < 35 else 7_500_000.0


def _ci_target(p: Profile, ipd_lumpsum_target: float) -> tuple[float, dict]:
    """Rule A (targeted therapy/continued treatment) + Rule B (income
    replacement, 12 months) + optional copay uplift (+30% of IPD lump-sum
    target) if user self-declares New Health Standard copay status."""
    rule_a = 2_250_000.0  # midpoint of 1.5M–3M
    rule_b = (p.annual_income / 12) * 12  # = p.annual_income, kept explicit per source formula
    base = rule_a + rule_b
    copay_uplift = 0.30 * ipd_lumpsum_target if p.has_copayment_status else 0.0
    return base + copay_uplift, {"rule_a": rule_a, "rule_b": rule_b, "copay_uplift": copay_uplift}


def _pa_medical_target(p: Profile) -> float:
    return 40_000.0  # midpoint of 30,000–50,000 per accident


def _pa_death_target(p: Profile) -> float:
    return p.annual_income * 2


def _retirement_target(p: Profile) -> Optional[float]:
    """Expense-replacement method: total retirement corpus needed to
    fund (life_expectancy - retirement_age) years at 70% of current
    annual expenses. Only computed if current_annual_expense is given."""
    if not p.current_annual_expense:
        return None
    years = max(p.life_expectancy - p.retirement_age, 0)
    return p.current_annual_expense * 0.70 * years


def _education_target(p: Profile) -> Optional[float]:
    if not p.education_goal_amount:
        return None
    return p.education_goal_amount


TARGET_RULES = {
    "life": {
        "fn": lambda p: _life_target(p)[0],
        "tier": "regulatory_backed",
        "source": "สมาคมนักวางแผนการเงินไทย (TFPA) + ตลาดหลักทรัพย์ฯ (SET): "
                   "max(รายได้ต่อปี×5, Needs Approach: หนี้สิน+ค่าใช้จ่ายครอบครัว 5 ปี"
                   "+ทุนการศึกษาบุตร−สินทรัพย์สะสม). Rule B คำนวณเฉพาะเมื่อกรอกข้อมูลเพิ่มเติม",
        "label": "ทุนประกันชีวิต",
        "unit": CATEGORY_UNITS["life"],
    },
    "ipd_room": {
        "fn": _ipd_room_target,
        "tier": "industry_data",
        "source": "อลิอันซ์ อยุธยา — รายงานดัชนีราคาค่าห้องพักผู้ป่วยในโรงพยาบาลเอกชน 20 แห่ง (2026) "
                   "แบ่งตามระดับ รพ. ที่ผู้ใช้เลือก (premium/general/economy)",
        "label": "ค่าห้อง IPD ต่อคืน",
        "unit": CATEGORY_UNITS["ipd_room"],
    },
    "ipd_lumpsum": {
        "fn": _ipd_lumpsum_target,
        "tier": "industry_data",
        "source": "สถิติอัตราค่ารักษาหัตถการและผ่าตัดใหญ่รายโรค — รู้ใจประกันภัย, AIAplanner "
                   "แบ่งตามช่วงวัย (First Jobber vs. 35 ปีขึ้นไป)",
        "label": "วงเงิน IPD เหมาจ่ายต่อปี",
        "unit": CATEGORY_UNITS["ipd_lumpsum"],
    },
    "ci": {
        "fn": lambda p: _ci_target(p, _ipd_lumpsum_target(p))[0],
        "tier": "industry_data",
        "source": "ค่ารักษาต่อเนื่อง/ยามุ่งเป้า (เฉลี่ย 2.7–4.6 ล้านบ.) + ชดเชยรายได้ 12 เดือน. "
                   "หากมีสถานะ copay ตาม New Health Standard (TLAA) จะบวกเพิ่ม 30% ของวงเงิน IPD "
                   "— หมายเหตุ: ระบบใช้การกรอกสถานะเอง แทนการติดตามประวัติเคลมจริง (ยังไม่มี DB)",
        "label": "เงินก้อนโรคร้ายแรง (CI)",
        "unit": CATEGORY_UNITS["ci"],
    },
    "pa_medical": {
        "fn": _pa_medical_target,
        "tier": "industry_data",
        "source": "Underwriting Guideline ของบริษัทประกันวินาศภัยในไทย — วงเงินค่ารักษาอุบัติเหตุ/ครั้ง",
        "label": "ค่ารักษาอุบัติเหตุต่อครั้ง (PA)",
        "unit": CATEGORY_UNITS["pa_medical"],
    },
    "pa_death": {
        "fn": _pa_death_target,
        "tier": "industry_data",
        "source": "Underwriting Guideline ของบริษัทประกันวินาศภัยในไทย — ทุนเสียชีวิต/ทุพพลภาพจากอุบัติเหตุ "
                   "= รายได้ต่อปี × 2 เท่า",
        "label": "ทุนเสียชีวิตจากอุบัติเหตุ (PA)",
        "unit": CATEGORY_UNITS["pa_death"],
    },
}

GOAL_RULES = {
    "retirement": {
        "fn": _retirement_target,
        "tier": "regulatory_backed",
        "source": "วิธี Expense-Replacement: ค่าใช้จ่ายปัจจุบัน×70%×(อายุขัยเฉลี่ย 85 − อายุเกษียณ) "
                   "— เฉพาะเมื่อกรอกค่าใช้จ่ายต่อปีปัจจุบัน",
        "label": "เป้าหมายเกษียณ (เงินก้อนรวม)",
    },
    "education": {
        "fn": _education_target,
        "tier": "regulatory_backed",
        "source": "คำนวณจากเป้าหมายที่ผู้ใช้กรอก",
        "label": "เป้าหมายการศึกษาบุตร",
    },
}


# ---------------------------------------------------------------------------
# 4. AFFORDABILITY GUARDRAIL — separate from gap/overlap: checks whether
#    total premium is sustainable, not whether coverage is sufficient.
#    Source: TFPA / SET — total life premium should not exceed 10–15% of
#    annual income.
# ---------------------------------------------------------------------------

def check_affordability(total_annual_premium: float, annual_income: float) -> dict:
    if annual_income <= 0:
        return {"ratio": None, "status": "unknown", "note": "ไม่มีข้อมูลรายได้"}
    ratio = total_annual_premium / annual_income
    if ratio > 0.15:
        status = "over_budget"
        note = f"เบี้ยรวมอยู่ที่ {ratio*100:.1f}% ของรายได้ — เกิน 15% (เริ่มตึงมือ)"
    elif ratio > 0.10:
        status = "watch"
        note = f"เบี้ยรวมอยู่ที่ {ratio*100:.1f}% ของรายได้ — อยู่ในช่วง 10-15% ควรระวัง"
    else:
        status = "ok"
        note = f"เบี้ยรวมอยู่ที่ {ratio*100:.1f}% ของรายได้ — อยู่ในเกณฑ์ปลอดภัย (≤10%)"
    return {
        "ratio": round(ratio, 4),
        "status": status,
        "note": note,
        "source": "สมาคมนักวางแผนการเงินไทย (TFPA) / SET — เบี้ยรวมไม่ควรเกิน 10-15% ของรายได้ต่อปี",
    }


# ---------------------------------------------------------------------------
# 5. TAX DEDUCTION LIMITS (กรมสรรพากร) — deterministic, regulatory
# ---------------------------------------------------------------------------
# NOTE: life + health deduction share ONE combined cap of 100,000.
# NOTE: the commonly-circulated claim that pension deduction can "top up"
# to 300,000 by borrowing unused life/health room was NOT found in any
# verified rd.go.th source during research — deliberately NOT implemented.
# If a verified source is found, add it here with citation.

TAX_RULES = {
    "life_health_combined_cap": 100_000,
    "health_sub_cap": 25_000,
    "parent_health_cap": 15_000,
    "pension_cap": 200_000,
    "pension_pct_of_income": 0.15,
    "retirement_bucket_cap": 500_000,
    "source": "กรมสรรพากร — ประกาศอธิบดีกรมสรรพากรเกี่ยวกับภาษีเงินได้ "
              "(ฉบับที่ 172, แก้ไขเพิ่มเติมฉบับที่ 235); rd.go.th/60058.html",
}


def calc_tax_deduction(life_premium: float, health_premium: float,
                        pension_premium: float, gross_income: float) -> dict:
    health_deduct = min(health_premium, TAX_RULES["health_sub_cap"])
    life_deduct_raw = min(life_premium, TAX_RULES["life_health_combined_cap"])
    combined = min(life_deduct_raw + health_deduct,
                   TAX_RULES["life_health_combined_cap"])

    pension_deduct = min(
        pension_premium,
        TAX_RULES["pension_cap"],
        TAX_RULES["pension_pct_of_income"] * gross_income,
    )

    return {
        "life_health_deduction": combined,
        "life_health_cap": TAX_RULES["life_health_combined_cap"],
        "pension_deduction": pension_deduct,
        "pension_cap": TAX_RULES["pension_cap"],
        "note": (
            "เพดานประกันชีวิต+สุขภาพรวมกันไม่เกิน 100,000 บาท "
            "(ไม่ใช่ 100,000 + 25,000 แยกกัน)"
        ),
        "source": TAX_RULES["source"],
    }


# ---------------------------------------------------------------------------
# 6. GAP + OVERLAP ENGINE — deterministic core (N3, N4)
# ---------------------------------------------------------------------------

def analyze_portfolio(profile: Profile, policies: list[Policy]) -> dict:
    by_category: dict[str, float] = {}
    policies_by_category: dict[str, list[Policy]] = {}
    # Premium is counted once per SOURCE POLICY DOCUMENT, not once per row —
    # a single document can produce multiple category rows (e.g. one PA
    # policy split into pa_death + pa_medical) that all repeat the same
    # annual_premium. Group by policy_group_id (falls back to the row's own
    # id when unknown) and take one premium value per group.
    premium_by_group: dict[str, float] = {}
    for pol in policies:
        by_category[pol.category] = by_category.get(pol.category, 0) + pol.sum_insured
        policies_by_category.setdefault(pol.category, []).append(pol)
        group_key = pol.policy_group_id or pol.id
        premium_by_group.setdefault(group_key, pol.annual_premium)
    total_premium = sum(premium_by_group.values())

    results = []

    for cat, rule in TARGET_RULES.items():
        target = rule["fn"](profile)
        current = by_category.get(cat, 0)
        gap = target - current
        status = "gap" if gap > 0 else ("overlap" if current > target * 1.3 else "ok")

        results.append({
            "category": cat,
            "label": rule["label"],
            "unit": rule["unit"],
            "tier": rule["tier"],
            "source": rule["source"],
            "target": round(target, 2),
            "current": round(current, 2),
            "gap": round(gap, 2),
            "status": status,
            "policy_count": len(policies_by_category.get(cat, [])),
            "policies": [p.id for p in policies_by_category.get(cat, [])],
        })

    for goal_key, rule in GOAL_RULES.items():
        target = rule["fn"](profile)
        if target is None:
            continue
        results.append({
            "category": goal_key,
            "label": rule["label"],
            "unit": "บาท",
            "tier": rule["tier"],
            "source": rule["source"],
            "target": round(target, 2),
            "current": None,
            "gap": None,
            "status": "goal",
            "policy_count": 0,
            "policies": [],
        })

    affordability = check_affordability(total_premium, profile.annual_income)

    return {"results": results, "affordability": affordability, "total_premium": total_premium}

