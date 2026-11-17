"""Multi-company periods, additional ratios and comparison semantics."""

from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from market_data.companies import get_company
from market_data.financial_metrics import calculate_metrics
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
        facts[definition["tags"][0]] = {"units": {"USD": series}}
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
