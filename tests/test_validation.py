import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rules import (  # noqa: E402
    OVERLAP_TOLERANCE,
    Policy,
    Profile,
    analyze_portfolio,
    check_affordability,
)


def _find(results: list[dict], category: str) -> dict:
    return next(r for r in results if r["category"] == category)


def test_gap_life_below_income_x5_target():
    profile = Profile(annual_income=900_000)
    policies = [
        Policy(id="p1", insurer="X", category="life", sum_insured=2_000_000),
    ]

    output = analyze_portfolio(profile, policies)
    life = _find(output["results"], "life")

    assert life["target"] == 900_000 * 5
    assert life["status"] == "gap"


def test_ok_within_tolerance_band():
    profile = Profile(annual_income=900_000)
    target = 900_000 * 5
    # inside [target, target * OVERLAP_TOLERANCE) — must not read as gap or overlap
    coverage = target * 1.1
    policies = [
        Policy(id="p1", insurer="X", category="life", sum_insured=coverage),
    ]

    output = analyze_portfolio(profile, policies)
    life = _find(output["results"], "life")

    assert coverage <= target * OVERLAP_TOLERANCE
    assert life["status"] == "ok"


def test_overlap_pa_death_exceeds_tolerance():
    profile = Profile(annual_income=500_000)
    policies = [
        Policy(id="p1", insurer="X", category="pa_death", sum_insured=1_400_000),
    ]

    output = analyze_portfolio(profile, policies)
    pa_death = _find(output["results"], "pa_death")

    assert pa_death["target"] == 500_000 * 2
    assert pa_death["status"] == "overlap"


def test_premium_dedup_sums_unique_values_per_policy_group():
    policies = [
        Policy(id="p1", insurer="X", category="life", sum_insured=1_000_000,
               annual_premium=28_500, policy_group_id="g1"),
        Policy(id="p2", insurer="X", category="ipd_room", sum_insured=8_000,
               annual_premium=18_200, policy_group_id="g1"),
        Policy(id="p3", insurer="Y", category="ipd_lumpsum", sum_insured=2_000_000,
               annual_premium=24_550, policy_group_id="g2"),
        Policy(id="p4", insurer="Y", category="ci", sum_insured=2_000_000,
               annual_premium=7_450, policy_group_id="g2"),
        Policy(id="p5", insurer="Z", category="pa_medical", sum_insured=40_000,
               annual_premium=2_100, policy_group_id="g3"),
        Policy(id="p6", insurer="Z", category="pa_death", sum_insured=100_000,
               annual_premium=2_100, policy_group_id="g3"),
    ]
    profile = Profile(annual_income=1_000_000)

    output = analyze_portfolio(profile, policies)

    assert output["total_premium"] == 80_800


def test_affordability_ratio_and_status():
    result = check_affordability(total_annual_premium=120_000, annual_income=1_000_000)

    assert result["ratio"] == 0.12
    assert result["status"] == "watch"
