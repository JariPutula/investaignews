"""Financial period selection, missing data and local snapshot integrity."""

from datetime import date
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from market_data.sec_microsoft import (
    CIK, METRICS, SecDataError, build_report, extract_annual_metrics,
    percentage_change, revenue_cagr,
)


def example_payload():
    facts = {}
    for key, definition in METRICS.items():
        series = []
        for year in range(2023, 2027):
            value = {
                "revenue": 100_000_000 * 1.1 ** (year - 2023),
                "operating_income": 20_000_000,
                "operating_cash_flow": 30_000_000,
                "capex": 12_000_000,
            }[key]
            series.append({
                "start": f"{year - 1}-07-01", "end": f"{year}-06-30",
                "val": value, "form": "10-K", "filed": "2026-07-30",
                "accn": "0000789019-26-000001", "fy": 2026, "fp": "FY",
            })
        facts[definition["tags"][0]] = {"units": {"USD": series}}
    return {"cik": CIK, "entityName": "MICROSOFT CORP", "facts": {"us-gaap": facts}}


def observations(payload, metric):
    return payload["facts"]["us-gaap"][METRICS[metric]["tags"][0]]["units"]["USD"]


class MicrosoftAnnualSelectionTests(unittest.TestCase):
    def extract(self, payload):
        return extract_annual_metrics(payload, as_of=date(2026, 11, 14))

    def test_uses_actual_period_instead_of_filing_fiscal_year(self):
        report = self.extract(example_payload())
        self.assertEqual([r["fiscal_year"] for r in report["rows"]], [2023, 2024, 2025, 2026])
        self.assertAlmostEqual(report["revenue_cagr_pct"], 10)
        first = report["rows"][0]
        self.assertAlmostEqual(first["operating_margin_pct"], 20)
        self.assertEqual(first["free_cash_flow"], 18_000_000)
        self.assertIsNone(first["revenue_growth_pct"])

    def test_excludes_quarters_in_annual_filings_and_annual_facts_in_quarterly_filings(self):
        payload = example_payload()
        series = observations(payload, "revenue")
        series.append({**series[-1], "start": "2026-04-01", "val": 999, "filed": "2026-09-01"})
        series.append({**series[-2], "form": "10-Q", "val": 888, "filed": "2026-10-01"})
        self.assertAlmostEqual(self.extract(payload)["rows"][-1]["revenue"], 133_100_000)

    def test_chooses_latest_annual_fact_but_excludes_future_publications(self):
        payload = example_payload()
        series = observations(payload, "revenue")
        series.append({**series[-1], "val": 140_000_000, "form": "10-K/A", "filed": "2026-08-01"})
        series.append({**series[-1], "val": 150_000_000, "filed": "2026-12-01"})
        last = self.extract(payload)["rows"][-1]
        self.assertEqual(last["revenue"], 140_000_000)
        self.assertEqual(last["sources"]["revenue"]["filed"], "2026-08-01")

    def test_keeps_newest_year_with_missing_metric_and_does_not_substitute_old_value(self):
        payload = example_payload()
        observations(payload, "capex").pop()
        report = self.extract(payload)
        self.assertEqual(report["rows"][-1]["fiscal_year"], 2026)
        self.assertIsNone(report["rows"][-1]["capex"])
        self.assertIsNone(report["rows"][-1]["free_cash_flow"])
        self.assertEqual(len(report["warnings"]), 1)

    def test_negative_cash_flow_is_preserved_capex_sign_is_normalized(self):
        payload = example_payload()
        observations(payload, "operating_cash_flow")[-1]["val"] = -5_000_000
        observations(payload, "capex")[-1]["val"] = -12_000_000
        row = self.extract(payload)["rows"][-1]
        self.assertEqual(row["free_cash_flow"], -17_000_000)
        self.assertEqual(row["sources"]["capex"]["source_value"], -12_000_000)

    def test_zero_revenue_does_not_produce_infinite_margin(self):
        payload = example_payload()
        observations(payload, "revenue")[-1]["val"] = 0
        row = self.extract(payload)["rows"][-1]
        self.assertIsNone(row["operating_margin_pct"])

    def test_rejects_conflicting_values_in_the_same_report(self):
        payload = example_payload()
        series = observations(payload, "revenue")
        series.append({**series[-1], "val": 999})
        with self.assertRaises(SecDataError):
            self.extract(payload)

    def test_only_usd_and_microsoft_data_are_accepted(self):
        payload = example_payload()
        payload["cik"] = 123
        with self.assertRaises(SecDataError):
            self.extract(payload)
        payload = example_payload()
        fact = payload["facts"]["us-gaap"][METRICS["revenue"]["tags"][0]]
        fact["units"] = {"EUR": fact["units"]["USD"]}
        self.assertIsNone(self.extract(payload)["rows"][-1]["revenue"])

    def test_missing_intermediate_year_does_not_shorten_growth_horizon(self):
        payload = example_payload()
        for definition in METRICS.values():
            series = payload["facts"]["us-gaap"][definition["tags"][0]]["units"]["USD"]
            series.pop(1)
        report = self.extract(payload)
        self.assertEqual([r["fiscal_year"] for r in report["rows"]], [2023, 2024, 2025, 2026])
        self.assertIsNone(report["rows"][2]["revenue_growth_pct"])

    def test_growth_edge_cases(self):
        self.assertIsNone(percentage_change(10, 0))
        self.assertIsNone(percentage_change(10, -5))
        self.assertIsNone(percentage_change(None, 10))
        self.assertEqual(percentage_change(0, 10), -100)
        self.assertIsNone(revenue_cagr([{"fiscal_year": 2023, "revenue": 100}, {"fiscal_year": 2026, "revenue": 133.1}]))


class MicrosoftSnapshotTests(unittest.TestCase):
    def test_offline_reuses_raw_snapshot_without_reading_metadata_as_facts(self):
        raw = json.dumps(example_payload()).encode()
        with TemporaryDirectory() as directory:
            location = Path(directory)
            with patch("market_data.sec.download_companyfacts", return_value=raw) as download:
                first = build_report(data_dir=location, user_agent="test contact@example.com", as_of=date(2026, 11, 14))
                second = build_report(data_dir=location, offline=True, as_of=date(2026, 11, 14))
                self.assertEqual(first, second)
                download.assert_called_once()
            self.assertNotIn("contact@example.com", (location / "annual_metrics.json").read_text())
            self.assertEqual((location / first["raw_file"]).read_bytes(), raw)

    def test_tampering_with_cached_source_is_detected(self):
        raw = json.dumps(example_payload()).encode()
        with TemporaryDirectory() as directory:
            location = Path(directory)
            with patch("market_data.sec.download_companyfacts", return_value=raw):
                report = build_report(data_dir=location, as_of=date(2026, 11, 14))
            (location / report["raw_file"]).write_bytes(raw + b" ")
            with self.assertRaisesRegex(SecDataError, "tarkistussumma"):
                build_report(data_dir=location, offline=True, as_of=date(2026, 11, 14))

    def test_bad_download_does_not_replace_existing_report(self):
        raw = json.dumps(example_payload()).encode()
        with TemporaryDirectory() as directory:
            location = Path(directory)
            with patch("market_data.sec.download_companyfacts", return_value=raw):
                build_report(data_dir=location, as_of=date(2026, 11, 14))
            before = (location / "annual_metrics.json").read_bytes()
            with patch("market_data.sec.download_companyfacts", return_value=b"<html>error</html>"):
                with self.assertRaises(SecDataError):
                    build_report(data_dir=location, refresh=True)
            self.assertEqual((location / "annual_metrics.json").read_bytes(), before)
            self.assertEqual(len(list((location / "raw").glob("*.metadata.json"))), 1)

    def test_offline_without_snapshot_reports_missing_data(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(SecDataError, "Paikallista"):
                build_report(data_dir=Path(directory), offline=True)


if __name__ == "__main__":
    unittest.main()
