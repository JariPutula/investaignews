"""Period arithmetic, coherent valuation, evidence checks and paid-call boundaries."""

from copy import deepcopy
from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from market_data.microsoft_exercise import (
    FLOW_TAGS, BALANCE_TAGS, COMPANY, atomic_json, digest, extract_financials,
    price_summary, select_fact, validate_snapshot,
)
from market_data.microsoft_dcf import calculate_exercise, reverse_growth, value_case
from market_data.sec import SecDataError
from learning.microsoft_exercise import (
    Claim, Plan, Interpretation, ask_question, generate_exercise, load_run, source_catalog, validate_claim, validate_plan,
)


def financial_fixture():
    facts = {}
    annuals = []
    scales = {key: .01 for key in FLOW_TAGS}
    scales.update(revenue=1., operating_income=.4, net_income=.3, operating_cash_flow=.5,
                  cash_capex=.2, depreciation=.04, amortization=.01, finance_lease_additions=.02)
    periods = [(f"{year-1}-07-01", f"{year}-06-30", 1e9) for year in range(2023, 2027)]
    periods += [("2025-07-01", "2025-09-30", 2e8), ("2026-07-01", "2026-09-30", 3e8)]
    for key, tags in FLOW_TAGS.items():
        series = []
        for start, end, amount in periods:
            series.append({"start": start, "end": end, "val": amount*scales[key], "filed": "2026-10-28", "form": "10-K" if end.endswith("06-30") else "10-Q", "accn": "0001193125-26-436133"})
        facts[tags[0]] = {"units": {"USD": series}}
    balances = {"cash_and_short_term_investments": 1e8, "borrowings": 2e8, "finance_lease_debt": 1e8, "shares": 1e7}
    for key, tags in BALANCE_TAGS.items():
        facts[tags[0]] = {"units": {"shares" if key == "shares" else "USD": [{"end": "2026-09-30", "val": balances[key], "filed": "2026-10-28", "form": "10-Q", "accn": "0001193125-26-436133"}]}}
    for year in range(2023, 2027):
        annuals.append({"fiscal_year": year, "period_start": f"{year-1}-07-01", "period_end": f"{year}-06-30"})
    return {"cik": COMPANY.cik, "facts": {"us-gaap": facts}}, {"rows": annuals}


def snapshot_fixture():
    payload, annual = financial_fixture()
    finance = extract_financials(payload, annual, cutoff="2026-11-21")
    finance["balances"]["borrowings_are_reported_total"] = True
    rows = [{"date": d.date().isoformat(), "close": 80.+i/10} for i, d in enumerate(pd.bdate_range("2025-10-01", "2026-11-20"))]
    market = {"histories": {"MSFT": rows, "SPY": rows}, "summaries": {"MSFT": price_summary(rows), "SPY": price_summary(rows)},
              "source_urls": {"MSFT": "https://finance.yahoo.com/quote/MSFT/history/", "SPY": "https://finance.yahoo.com/quote/SPY/history/"}}
    snapshot = {"version": 1, "ticker": "MSFT", "fiscal_year": 2027, "as_of": "2026-11-21", "financials": finance, "market": market,
                "quarter_report": "https://www.sec.gov/fixture.htm", "guidance_url": "https://www.microsoft.com/fixture",
                "texts": {"Q01": {"text": "Revenue growth reflected strong cloud demand and higher investment.", "url": "https://www.sec.gov/fixture.htm"},
                          "G01": {"text": "We expect capital expenditures to remain elevated while demand expands.", "url": "https://www.microsoft.com/fixture"}}}
    snapshot["fingerprint"] = digest(snapshot)
    return snapshot


def claim(text="Kasvu on vahvaa, mutta investoinnit käyttävät paljon rahaa.", source="F_TTM"):
    return {"text": text, "evidence": [{"source_id": source, "quote": ""}]}


def plan_fixture():
    scenarios = []
    for key, growth, margin, capex in (("weak", 5., 35., 30.), ("base", 10., 42., 25.), ("strong", 15., 48., 20.)):
        scenarios.append({"case": key, "assumptions": {"growth_pct": [growth]*5, "operating_margin_pct": [margin]*5,
            "net_margin_pct": [margin-5]*5, "cash_capex_pct": [capex]*5, "lease_capex_pct": [3.]*5, "da_pct": [10.]*5,
            "working_capital_pct_of_sales_change": 10.},
            "drivers": [{"topic": topic, **claim()} for topic in ("growth", "profitability", "investment")], "watch": claim()})
    return Plan.model_validate({"current_view": claim(), "price_view": claim(source="F_PRICE"), "valuation_reason": claim(source="F_LIMITS"),
        "common": {"tax_pct": 20., "wacc_pct": 10., "terminal_growth_pct": 3., "terminal_roic_pct": 20.}, "scenarios": scenarios})


def interpretation_fixture():
    return Interpretation.model_validate({key: claim(source="C_base") for key in ("conclusion", "implied_growth", "counterargument", "next_question")})


def response(parsed):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(parsed=parsed, refusal=None))], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=100))


class FinancialDataTests(TestCase):
    def test_ttm_is_annual_minus_prior_ytd_plus_current_ytd(self):
        snapshot = snapshot_fixture()
        ttm = snapshot["financials"]["ttm"]
        self.assertEqual(ttm["revenue"], 1.1e9)
        self.assertEqual(ttm["period_start"], "2025-10-01")
        self.assertEqual(ttm["free_cash_flow"], 330e6)
        self.assertEqual(snapshot["financials"]["quarter_growth_pct"]["revenue"], 50)
        self.assertNotEqual(ttm["revenue"], snapshot["financials"]["quarter"]["revenue"]*4)

    def test_future_filing_quarter_and_wrong_unit_cannot_replace_exact_fact(self):
        payload, _ = financial_fixture()
        series = payload["facts"]["us-gaap"]["OperatingIncomeLoss"]["units"]["USD"]
        base = series[-1]
        series.extend([{**base, "filed": "2026-12-01", "val": 999}, {**base, "start": "2026-08-01", "val": 888}])
        selected = select_fact(payload, ("OperatingIncomeLoss",), start="2026-07-01", end="2026-09-30", cutoff="2026-11-21")
        self.assertEqual(selected["value"], 120e6)
        self.assertIsNone(select_fact(payload, ("ShortTermBorrowings",), end="2026-09-30", cutoff="2026-11-21", required=False))

    def test_missing_required_flow_is_not_zero_and_wrong_company_rejected(self):
        payload, annual = financial_fixture()
        del payload["facts"]["us-gaap"]["Depreciation"]
        with self.assertRaises(SecDataError):
            extract_financials(payload, annual, cutoff="2026-11-21")
        payload["cik"] = 1
        with self.assertRaises(SecDataError):
            extract_financials(payload, annual, cutoff="2026-11-21")

    def test_integrity_and_lookahead_protection(self):
        snapshot = snapshot_fixture()
        with patch("market_data.microsoft_exercise.today_rome", return_value=date(2026, 11, 21)):
            validate_snapshot(snapshot, require_current=True)
            altered = deepcopy(snapshot)
            altered["market"]["summaries"]["MSFT"]["price"] = 1
            with self.assertRaises(SecDataError):
                validate_snapshot(altered)
            altered = deepcopy(snapshot)
            altered["financials"]["sources"]["YTD_revenue"]["filed"] = "2026-11-25"
            altered["fingerprint"] = digest({k: v for k, v in altered.items() if k != "fingerprint"})
            with self.assertRaises(SecDataError):
                validate_snapshot(altered)
        with patch("market_data.microsoft_exercise.today_rome", return_value=date(2026, 12, 5)):
            validate_snapshot(snapshot)
            with self.assertRaises(SecDataError):
                validate_snapshot(snapshot, require_current=True)

    def test_calendar_month_and_peak_have_explicit_dates(self):
        result = snapshot_fixture()["market"]["summaries"]["MSFT"]
        self.assertEqual(result["month"]["date"], "2026-10-20")
        self.assertEqual(result["peak_52w"]["date"], "2026-11-20")
        self.assertEqual(result["peak_52w"]["change_pct"], 0)


class DcfTests(TestCase):
    def test_constant_cash_flow_matches_perpetuity_and_debt_counted_once(self):
        snapshot = snapshot_fixture()
        snapshot["financials"]["ttm"]["revenue"] = 100.
        snapshot["financials"]["balances"].update(cash_and_short_term_investments=10., borrowings=20., finance_lease_debt=10., shares=10.)
        paths = plan_fixture().scenarios[1].assumptions.model_dump()
        paths.update(growth_pct=[0.]*5, operating_margin_pct=[40.]*5, cash_capex_pct=[10.]*5, lease_capex_pct=[0.]*5, da_pct=[10.]*5)
        common = {"tax_pct": 25., "wacc_pct": 10., "terminal_growth_pct": 0., "terminal_roic_pct": 20.}
        result = value_case(snapshot, paths, common)
        self.assertAlmostEqual(result["enterprise_value"], 300.)
        self.assertAlmostEqual(result["net_debt"], 20.)
        self.assertAlmostEqual(result["value_per_share"], 28.)
        self.assertAlmostEqual(result["terminal_reinvestment"], 0.)

    def test_leased_asset_investment_reduces_cash_flow_and_negative_equity_is_preserved(self):
        snapshot = snapshot_fixture()
        paths = plan_fixture().scenarios[0].assumptions.model_dump()
        common = plan_fixture().common.model_dump()
        without = value_case(snapshot, {**paths, "lease_capex_pct": [0.]*5}, common)
        with_leases = value_case(snapshot, paths, common)
        self.assertLess(with_leases["value_per_share"], without["value_per_share"])
        snapshot["financials"]["balances"]["borrowings"] = 1e15
        self.assertLess(value_case(snapshot, paths, common)["value_per_share"], 0)

    def test_reverse_solves_price_and_sensitivity_changes_required_growth(self):
        snapshot = snapshot_fixture()
        plan = plan_fixture()
        paths, common = plan.scenarios[1].assumptions.model_dump(), plan.common.model_dump()
        known = value_case(snapshot, paths, common, growth_override=12.)["value_per_share"]
        snapshot["market"]["summaries"]["MSFT"]["price"] = known
        reverse = reverse_growth(snapshot, paths, common)
        self.assertTrue(reverse["available"])
        self.assertAlmostEqual(reverse["growth_pct"], 12., places=7)
        self.assertGreater(reverse_growth(snapshot, paths, {**common, "wacc_pct": 12.})["growth_pct"], 12.)
        snapshot["market"]["summaries"]["MSFT"]["price"] = 1e9
        self.assertFalse(reverse_growth(snapshot, paths, common)["available"])

    def test_invalid_terminal_and_missing_debt_are_blocked(self):
        snapshot = snapshot_fixture()
        plan = plan_fixture()
        with self.assertRaises((ValueError, SecDataError)):
            value_case(snapshot, plan.scenarios[1].assumptions.model_dump(), {**plan.common.model_dump(), "wacc_pct": 6., "terminal_growth_pct": 4.})
        snapshot["financials"]["balances"]["borrowings_are_reported_total"] = False
        with self.assertRaises(SecDataError):
            value_case(snapshot, plan.scenarios[1].assumptions.model_dump(), plan.common.model_dump())

    def test_scenario_pe_is_own_net_income_and_not_nopat(self):
        snapshot, plan = snapshot_fixture(), plan_fixture()
        result = calculate_exercise(snapshot, {c.case: c.assumptions.model_dump() for c in plan.scenarios}, plan.common.model_dump())
        self.assertNotEqual(result["values"]["base"]["first_year_scenario_pe"], result["values"]["base"]["first_year_market_cap_to_nopat"])


class TeacherTests(TestCase):
    def test_report_reference_selects_exact_anchor_and_product_name_is_not_financial_number(self):
        sources = source_catalog(snapshot_fixture())
        paragraph = Claim.model_validate({"text": "Microsoft 365 on mukana yhtiön pilvitarjonnassa. [Q01]", "evidence": [{"source_id": "Q01", "quote": "", "quote_index": 0}]})
        validate_claim(paragraph, sources)
        self.assertNotIn("[Q01]", paragraph.text)
        self.assertEqual(paragraph.evidence[0].quote, sources["Q01"]["quotes"][0])
        paragraph.text = "Liikevaihto kasvoi 999 prosenttia."
        with self.assertRaises(SecDataError):
            validate_claim(paragraph, sources)
        paragraph.text = "Raportti kertoo kasvusta."
        paragraph.evidence[0].quote_index = 999
        with self.assertRaises(SecDataError):
            validate_claim(paragraph, sources)

    def test_unknown_source_fabricated_quote_and_wrong_case_order_rejected(self):
        snapshot, plan = snapshot_fixture(), plan_fixture()
        for bad in ("missing", "quote", "order"):
            changed = plan.model_copy(deep=True)
            if bad == "missing":
                changed.current_view.evidence[0].source_id = "F_unknown"
            elif bad == "quote":
                changed.current_view.evidence[0].source_id = "Q01"
                changed.current_view.evidence[0].quote = "This text was never in the report."
            else:
                changed.scenarios.reverse()
            with self.assertRaises(SecDataError):
                validate_plan(changed, snapshot)

    def test_two_calls_cached_and_question_uses_same_snapshot_once(self):
        snapshot = snapshot_fixture()
        client = Mock()
        client.chat.completions.parse.side_effect = [response(plan_fixture()), response(interpretation_fixture()), response(__import__('learning.microsoft_exercise', fromlist=['Claim']).Claim.model_validate(claim(source='C_base')))]
        with TemporaryDirectory() as directory, patch("learning.microsoft_exercise.ROOT", Path(directory)), \
             patch("market_data.microsoft_exercise.today_rome", return_value=date(2026,11,21)):
            run = generate_exercise(snapshot, "test-key", client=client)
            self.assertEqual(client.chat.completions.parse.call_count, 2)
            self.assertEqual(load_run(snapshot)["integrity"], run["integrity"])
            generate_exercise(snapshot, "test-key", client=client)
            ask_question(run, "Miksi investoinnit vaikuttavat?", "test-key", client=client)
            ask_question(run, "Miksi investoinnit vaikuttavat?", "test-key", client=client)
            self.assertEqual(client.chat.completions.parse.call_count, 3)
            self.assertFalse(client.chat.completions.parse.call_args.kwargs["store"])
            self.assertNotIn("test-key", json.dumps(client.chat.completions.parse.call_args.kwargs["messages"]))

    def test_interpretation_failure_keeps_plan_checkpoint_without_automatic_retry(self):
        snapshot = snapshot_fixture()
        client = Mock()
        client.chat.completions.parse.side_effect = [response(plan_fixture()), RuntimeError("failure"), response(interpretation_fixture())]
        with TemporaryDirectory() as directory, patch("learning.microsoft_exercise.ROOT", Path(directory)), \
             patch("market_data.microsoft_exercise.today_rome", return_value=date(2026,11,21)), patch("learning.microsoft_exercise.record_api_error", return_value=SecDataError("failure")):
            with self.assertRaises(SecDataError):
                generate_exercise(snapshot, "test-key", client=client)
            self.assertEqual(client.chat.completions.parse.call_count, 2)
            self.assertEqual(len(list((Path(directory)/"plans").glob("*.json"))), 1)
            self.assertFalse((Path(directory)/"runs").exists())
            generate_exercise(snapshot, "test-key", client=client)
            self.assertEqual(client.chat.completions.parse.call_count, 3)


class ExercisePageTests(TestCase):
    def test_existing_company_page_opens_exercise_and_returns_without_external_calls(self):
        page = str(Path(__file__).resolve().parents[1]/"pages"/"01_Yhtioanalyysi.py")
        with patch("learning.company_page.load_report", return_value=None), \
             patch("learning.microsoft_exercise_page.load_exercise", return_value=None), \
             patch("learning.microsoft_exercise_page.refresh_exercise") as refresh, \
             patch("learning.microsoft_exercise_page.generate_exercise") as paid:
            app = AppTest.from_file(page).run(timeout=30)
            app.button(key="open_msft_exercise").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertIn("nykyinen hinta", app.title[0].value)
            app.button(key="close_msft_exercise").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(app.title[0].value, "Yhtiöanalyysi")
            refresh.assert_not_called()
            paid.assert_not_called()

    def test_opening_saved_exercise_does_not_fetch_or_make_paid_calls(self):
        snapshot = snapshot_fixture()
        plan = plan_fixture()
        result = calculate_exercise(snapshot, {c.case: c.assumptions.model_dump() for c in plan.scenarios}, plan.common.model_dump())
        run = {"snapshot": snapshot, "plan": plan.model_dump(), "results": result, "interpretation": interpretation_fixture().model_dump(),
               "created_at": "2026-11-21T12:00:00+00:00", "model": "test-model", "cost_estimate_usd": .01, "integrity": "fixture"}
        page = str(Path(__file__).resolve().parents[1]/"pages"/"02_Microsoft_harjoitus.py")
        with patch("learning.microsoft_exercise_page.load_exercise", return_value=snapshot), \
             patch("learning.microsoft_exercise_page.load_run", return_value=run), \
             patch("learning.microsoft_exercise_page.list_runs", return_value=[]), \
             patch("learning.microsoft_exercise_page.load_conversation", return_value=[]), \
             patch("learning.microsoft_exercise_page.refresh_exercise") as refresh, \
             patch("learning.microsoft_exercise_page.generate_exercise") as paid, \
             patch("learning.microsoft_exercise_page.ask_question") as answer:
            app = AppTest.from_file(page).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            self.assertTrue(any("edellyttäisi" in node.value for node in app.subheader))
            app.run(timeout=30)
            refresh.assert_not_called()
            paid.assert_not_called()
            answer.assert_not_called()
