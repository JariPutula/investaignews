"""Pilot coverage and explicit first-download behavior."""

from unittest import TestCase
from unittest.mock import patch

from market_data.companies import COMPANIES
from market_data.pilot import coverage_rows, fetch_missing_reports, load_pilot
from market_data.sec import SecDataError
from tests.test_sec_companies import extract


class PilotTests(TestCase):
    def test_coverage_distinguishes_missing_data_from_missing_metrics(self):
        report = extract("NEE")
        report["rows"][-1]["capex"] = None
        table = {row["Tunnus"]: row for row in coverage_rows({"NEE": report}, {"AAPL": "Virheellinen aineisto"})}
        self.assertEqual(len(table), 20)
        self.assertIn("Käyttöomaisuusinvestoinnit", table["NEE"]["Puuttuvat lähtöluvut"])
        self.assertEqual(table["AAPL"]["Aineisto"], "Virheellinen aineisto")
        self.assertEqual(table["NEE"]["Liiketoiminta-alue"], "Sähkö ja energia")

    def test_first_download_reuses_existing_reports_and_continues_after_one_failure(self):
        existing = {ticker: extract(ticker) for ticker in COMPANIES if ticker not in {"LLY", "COST"}}
        with patch("market_data.pilot.load_pilot", return_value=(existing, {})), \
             patch("market_data.pilot.build_report", side_effect=[SecDataError("Tilapäinen yhteysvirhe"), extract("LLY")]) as build:
            progress = []
            loaded, failures = fetch_missing_reports("test-agent", on_progress=lambda *values: progress.append(values))
        self.assertEqual(len(loaded), 19)
        self.assertEqual(set(failures), {"COST"})
        self.assertEqual([call.kwargs["company"].ticker for call in build.call_args_list], ["COST", "LLY"])
        self.assertTrue(all(call.kwargs["user_agent"] == "test-agent" for call in build.call_args_list))
        self.assertEqual(progress[-1][:2], (2, 2))

    def test_bad_company_report_does_not_hide_the_remaining_pilot(self):
        def load(*, company):
            if company.ticker == "AAPL":
                raise SecDataError("Väärä yhteenvetoversio")
            return extract(company.ticker)
        with patch("market_data.pilot.load_report", side_effect=load):
            reports, issues = load_pilot()
        self.assertEqual(len(reports), 19)
        self.assertEqual(issues, {"AAPL": "Väärä yhteenvetoversio"})
