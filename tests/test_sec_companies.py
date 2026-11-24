"""Multi-company periods, additional ratios and comparison semantics."""

from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from market_data.companies import get_company
from market_data.financial_metrics import calculate_metrics, cash_flow_bridge
from market_data.sec import METRICS, SecDataError, build_report, comparison_rows, extract_annual_metrics, load_report


def payload_for(ticker):
    company = get_company(ticker)
    latest = 2026 if ticker == "MSFT" else 2025
    facts = {}
    values = {"revenue": 100_000_000, "operating_income": 20_000_000,
              "operating_cash_flow": 30_000_000, "capex": 12_000_000,
              "cost_of_revenue": 40_000_000, "net_income": 15_000_000,
              "current_assets": 60_000_000, "current_liabilities": 40_000_000}
    for key, definition in METRICS.items():
        series = []
        for year in range(latest-3, latest+1):
            start, end = company.period(year)
            fact = {"end": end.isoformat(), "val": values[key], "filed": "2026-08-01",
                    "form": "10-K", "accn": "0000000001-26-000001", "fy": 2026}
            if not definition.get("instant"):
                fact["start"] = start.isoformat()
            series.append(fact)
        override = {"revenue": company.revenue_tags, "capex": company.capex_tags,
                    "net_income": company.net_income_tags, "cost_of_revenue": company.cost_of_revenue_tags}.get(key)
        tags = definition["tags"] if override is None else override
        if not tags:
            continue
        tag = tags[0]
        facts[tag] = {"units": {"USD": series}}
    return {"cik": company.cik, "entityName": company.name, "facts": {"us-gaap": facts}}


def extract(ticker, payload=None):
    return extract_annual_metrics(payload or payload_for(ticker), company=get_company(ticker), as_of=date(2026, 11, 15))


class AdditionalMetricTests(unittest.TestCase):
    def test_additional_metrics_have_correct_denominators_and_units(self):
        row = extract("KO")["rows"][-1]
        self.assertEqual(row["gross_profit"], 60_000_000)
        self.assertEqual(row["gross_margin_pct"], 60)
        self.assertEqual(row["net_margin_pct"], 15)
        self.assertEqual(row["free_cash_flow"], 18_000_000)
        self.assertEqual(row["fcf_margin_pct"], 18)
        self.assertEqual(row["capex_to_revenue_pct"], 12)
        self.assertEqual(row["current_ratio"], 1.5)

    def test_missing_optional_fields_do_not_block_core_metrics(self):
        payload = payload_for("KO")
        del payload["facts"]["us-gaap"]["AssetsCurrent"]
        report = extract("KO", payload)
        self.assertEqual(report["warnings"], [])
        self.assertEqual(len(report["data_notes"]), 4)
        self.assertIsNone(report["rows"][-1]["current_ratio"])
        self.assertEqual(report["rows"][-1]["free_cash_flow"], 18_000_000)

    def test_nonpositive_denominators_do_not_create_meaningless_ratios(self):
        row = {"revenue": 0, "operating_income": 2, "operating_cash_flow": -5,
               "capex": 3, "cost_of_revenue": 1, "net_income": -1,
               "current_assets": 4, "current_liabilities": 0}
        metrics = calculate_metrics(row)
        for key in ("operating_margin_pct", "gross_margin_pct", "net_margin_pct", "fcf_margin_pct", "capex_to_revenue_pct", "current_ratio"):
            self.assertIsNone(metrics[key])
        self.assertEqual(metrics["free_cash_flow"], -8)

    def test_loss_and_negative_free_cash_flow_remain_visible(self):
        row = {"revenue": 100, "operating_income": -10, "operating_cash_flow": 5,
               "capex": 10, "cost_of_revenue": 110, "net_income": -15,
               "current_assets": 10, "current_liabilities": 20}
        metrics = calculate_metrics(row)
        self.assertEqual(metrics["gross_margin_pct"], -10)
        self.assertEqual(metrics["net_margin_pct"], -15)
        self.assertEqual(metrics["fcf_margin_pct"], -5)
        self.assertEqual(metrics["current_ratio"], 0.5)


class CompanyPeriodTests(unittest.TestCase):
    def test_apple_and_costco_53_week_years_use_exact_reported_periods(self):
        apple, costco = get_company("AAPL"), get_company("COST")
        self.assertEqual(tuple(map(str, apple.period(2023))), ("2022-09-25", "2023-09-30"))
        self.assertEqual(tuple(map(str, costco.period(2023))), ("2022-08-29", "2023-09-03"))
        for company in (apple, costco):
            start, end = company.period(2023)
            self.assertEqual((end - start).days + 1, 371)
            payload = payload_for(company.ticker)
            tag = "RevenueFromContractWithCustomerExcludingAssessedTax"
            series = payload["facts"]["us-gaap"][tag]["units"]["USD"]
            annual = next(f for f in series if f["end"] == end.isoformat())
            series.append({**annual, "start": "2023-06-01", "val": 999})
            report = extract(company.ticker, payload)
            row = next(r for r in report["rows"] if r["fiscal_year"] == 2023)
            self.assertEqual(row["revenue"], 100_000_000)
            self.assertEqual(row["period_start"], start.isoformat())

    def test_walmart_and_energy_revenues_do_not_use_a_partial_customer_sales_line(self):
        for ticker, tag in (("WMT", "Revenues"), ("COP", "Revenues"), ("NEE", "RegulatedAndUnregulatedOperatingRevenue")):
            with self.subTest(ticker=ticker):
                payload = payload_for(ticker)
                series = payload["facts"]["us-gaap"][tag]["units"]["USD"]
                payload["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"] = {
                    "units": {"USD": [{**f, "val": 90_000_000} for f in series]}}
                row = extract(ticker, payload)["rows"][-1]
                self.assertEqual(row["revenue"], 100_000_000)
                self.assertEqual(row["sources"]["revenue"]["tag"], tag)

    def test_partial_commodity_costs_are_not_used_for_gross_margin(self):
        for ticker in ("COP", "CVX", "CAT"):
            payload = payload_for(ticker)
            template = payload["facts"]["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"]["units"]["USD"]
            payload["facts"]["us-gaap"]["CostOfRevenue"] = {"units": {"USD": [{**f, "val": 10_000_000} for f in template]}}
            row = extract(ticker, payload)["rows"][-1]
            self.assertIsNone(row["gross_margin_pct"])
            self.assertNotIn("cost_of_revenue", row["sources"])

    def test_missing_parent_income_is_not_replaced_with_income_including_minorities(self):
        payload = payload_for("MA")
        series = payload["facts"]["us-gaap"].pop("NetIncomeLoss")["units"]["USD"]
        payload["facts"]["us-gaap"]["ProfitLoss"] = {"units": {"USD": series}}
        self.assertIsNone(extract("MA", payload)["rows"][-1]["net_margin_pct"])

    def test_nvidia_52_and_53_week_years_and_actual_end_dates(self):
        company = get_company("NVDA")
        self.assertEqual(tuple(map(str, company.period(2025))), ("2024-01-29", "2025-01-26"))
        start, end = company.period(2027)
        self.assertEqual((end-start).days+1, 371)
        report = extract("NVDA")
        self.assertEqual(report["rows"][-1]["capex"], 12_000_000)
        self.assertEqual(report["rows"][-1]["sources"]["capex"]["tag"], "PaymentsToAcquireProductiveAssets")
        self.assertTrue(report["capex_includes_intangibles"])

    def test_cash_flow_bridge_explains_opposing_changes_and_negative_fcf(self):
        previous = {"operating_cash_flow": 100, "capex": 80}
        row = {"operating_cash_flow": 120, "capex": 130}
        bridge = cash_flow_bridge(row, previous)
        self.assertEqual(bridge, {"operating_change": 20, "investment_change": 50, "free_cash_flow_change": -30})
        self.assertIsNone(cash_flow_bridge({**row, "capex": None}, previous))
        self.assertIsNone(cash_flow_bridge(row, None))
        self.assertIsNone(calculate_metrics({**row, "net_income": -5})["cash_conversion_ratio"])

    def test_coca_cola_calendar_year_is_not_microsoft_fiscal_year(self):
        ko = extract("KO")
        self.assertEqual([r["fiscal_year"] for r in ko["rows"]], [2022, 2023, 2024, 2025])
        self.assertEqual(ko["rows"][-1]["period_start"], "2025-01-01")
        self.assertEqual(ko["rows"][-1]["period_end"], "2025-12-31")
        self.assertEqual(extract("MSFT")["rows"][-1]["period_start"], "2025-07-01")
        self.assertIn("/21344/", ko["rows"][-1]["sources"]["revenue"]["url"])

    def test_balance_sheet_uses_year_end_instant_and_preserves_source_period(self):
        payload = payload_for("KO")
        series = payload["facts"]["us-gaap"]["AssetsCurrent"]["units"]["USD"]
        series.extend([
            {**series[-1], "end": "2025-09-30", "val": 999, "filed": "2026-09-01"},
            {**series[-1], "start": "2025-01-01", "val": 888, "filed": "2026-09-01"},
            {**series[-1], "form": "10-Q", "val": 777, "filed": "2026-09-01"},
        ])
        row = extract("KO", payload)["rows"][-1]
        self.assertEqual(row["current_assets"], 60_000_000)
        self.assertIsNone(row["sources"]["current_assets"]["period_start"])
        self.assertEqual(row["sources"]["current_assets"]["period_end"], "2025-12-31")

    def test_wrong_company_payload_is_rejected(self):
        with self.assertRaises(SecDataError):
            extract_annual_metrics(payload_for("MSFT"), company=get_company("KO"))

    def test_comparison_keeps_actual_periods_for_same_year_label(self):
        rows = comparison_rows([extract("MSFT"), extract("KO")], 2025)
        self.assertEqual([r["period_end"] for r in rows], ["2025-06-30", "2025-12-31"])
        self.assertEqual([r["period_start"] for r in rows], ["2024-07-01", "2025-01-01"])
        latest = comparison_rows([extract("MSFT"), extract("KO")])
        self.assertEqual([r["fiscal_year"] for r in latest], [2026, 2025])

    def test_optional_balance_data_does_not_invent_a_new_income_year(self):
        payload = payload_for("KO")
        series = payload["facts"]["us-gaap"]["AssetsCurrent"]["units"]["USD"]
        series.append({**series[-1], "end": "2026-12-31", "filed": "2027-02-20"})
        report = extract_annual_metrics(payload, company=get_company("KO"), as_of=date(2027, 3, 1))
        self.assertEqual(report["rows"][-1]["fiscal_year"], 2025)

    def test_cached_company_data_are_separate(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for ticker in ("MSFT", "KO"):
                company = get_company(ticker)
                raw = json.dumps(payload_for(ticker)).encode()
                with patch("market_data.sec.download_companyfacts", return_value=raw):
                    build_report(company=company, data_dir=root/ticker, as_of=date(2026, 11, 15))
                self.assertEqual(load_report(company=company, data_dir=root/ticker)["ticker"], ticker)
            with self.assertRaises(SecDataError):
                load_report(company=get_company("KO"), data_dir=root/"MSFT")


if __name__ == "__main__":
    unittest.main()
