"""Exercise company switching and incomplete data without SEC or local files."""

from copy import deepcopy
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.test_sec_companies import extract
from market_data.companies import COMPANIES


PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "01_Yhtioanalyysi.py")


def reports():
    result = {ticker: extract(ticker) for ticker in COMPANIES}
    for report in result.values():
        report["retrieved_at"] = "2026-11-15T10:00:00+00:00"
        report["raw_sha256"] = "fixture-checksum"
    return result


def comparison_table(app):
    return next(table.value for table in app.dataframe if "Tilikausi alkaa" in table.value.index)


class LearningPageTests(TestCase):
    def setUp(self):
        index_patch = patch("learning.analysis_page.list_saved_analyses", return_value=[])
        self.saved_index = index_patch.start()
        self.addCleanup(index_patch.stop)
        quote_patch = patch("learning.investor_controls.load_valuation", return_value=None)
        quote_patch.start()
        self.addCleanup(quote_patch.stop)

    def test_company_brief_shows_four_topics_and_expandable_history_without_paid_call(self):
        from tests.test_company_brief import brief_fixture
        data = reports()
        row = data["AMZN"]["rows"][-1]
        annual = {"cik": data["AMZN"]["cik"], "accession": row["sources"]["revenue"]["accession"],
                  "url": "https://www.sec.gov/fixture.htm", "filed": "2026-08-01", "sha256": "fixture",
                  "excerpts": [{"id": "R0001", "text": "Operating cash flow increased."}]}
        saved = {"model": "fixture-model", "created_at": "2026-11-18T10:00:00+00:00", "input_tokens": 100,
                 "output_tokens": 50, "cost_estimate_usd": .001, "analysis": brief_fixture().model_dump()}
        with patch("learning.company_page.load_report", side_effect=lambda *, company: deepcopy(data[company.ticker])), \
             patch("learning.analysis_page.load_annual_report", side_effect=lambda company, row, **kwargs: annual if company.ticker == "AMZN" else None), \
             patch("learning.analysis_page.load_analysis", return_value=saved), \
             patch("learning.analysis_page.generate_analysis") as paid:
            app = AppTest.from_file(PAGE).run(timeout=30)
            app.selectbox(key="learning_company").set_value("AMZN").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertFalse(app.error)
            markdown = [item.value for item in app.markdown]
            for title in ("Kasvu", "Kannattavuus", "Kassavirta", "Maksuvalmius", "Vahvuus", "Huomio", "Seuraavaksi selvitettävä"):
                self.assertIn(f"**{title}**", markdown)
            history = next(table.value for table in app.dataframe if "Liikevaihto, mrd. USD" in table.value.columns)
            self.assertEqual(len(history), 4)
            self.assertEqual(history.iloc[0]["Liikevaihto, mrd. USD"], .1)
            details = next(expander for expander in app.expander if expander.label == "Tarkat luvut ja neljän tilikauden kehitys")
            self.assertFalse(details.proto.expanded)
            for text in ("**Tulevaisuus — oletuksiin perustuva arvio**", "**Sijoittajan kokonaisarvio — tulkinta**", "**Mikä voisi kumota tämän näkemyksen?**"):
                self.assertIn(text, markdown)
            app.run(timeout=30)
            paid.assert_not_called()

    def test_editing_and_saving_scenarios_does_not_trigger_paid_analysis(self):
        data = reports()
        row = data["AMZN"]["rows"][-1]
        annual = {"cik": data["AMZN"]["cik"], "accession": row["sources"]["revenue"]["accession"],
                  "url": "https://www.sec.gov/fixture.htm", "filed": "2026-08-01", "sha256": "fixture", "excerpts": []}
        with patch("learning.company_page.load_report", side_effect=lambda *, company: deepcopy(data[company.ticker])), \
             patch("learning.analysis_page.load_annual_report", side_effect=lambda company, row, **kwargs: annual if company.ticker == "AMZN" else None), \
             patch("learning.analysis_page.load_analysis", return_value=None), \
             patch("learning.analysis_page.legacy_analysis", return_value=None), \
             patch("learning.investor_controls.save_assumptions") as save, \
             patch("learning.analysis_page.generate_analysis") as paid:
            app = AppTest.from_file(PAGE).run(timeout=30)
            app.selectbox(key="learning_company").set_value("AMZN").run(timeout=30)
            app.number_input(key="scenario_AMZN_2025_continuity_growth_pct").set_value(25.)
            next(button for button in app.button if button.label == "Tallenna skenaario-oletukset").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(save.call_args.args[-1][1]["growth_pct"], 25.)
            paid.assert_not_called()

    def test_saved_shortcut_selects_company_year_and_analysis_tab_without_paid_call(self):
        from tests.test_company_analysis import analysis_fixture
        data = reports()
        self.saved_index.return_value = [{"ticker": "AMZN", "company": "Amazon", "fiscal_year": 2025}]
        row = data["AMZN"]["rows"][-1]
        annual = {"cik": data["AMZN"]["cik"], "accession": row["sources"]["revenue"]["accession"],
                  "url": "https://www.sec.gov/fixture.htm", "filed": "2026-08-01", "sha256": "fixture-annual",
                  "excerpts": [{"id": "R0001", "text": "Operating cash flow increased primarily due to higher net income."}]}
        saved = {"model": "fixture-model", "created_at": "2026-11-18T10:00:00+00:00", "input_tokens": 100,
                 "output_tokens": 50, "cost_estimate_usd": .001, "analysis": analysis_fixture().model_dump()}
        with patch("learning.company_page.load_report", side_effect=lambda *, company: deepcopy(data[company.ticker])), \
             patch("learning.analysis_page.load_annual_report", side_effect=lambda company, row, **kwargs: annual if company.ticker == "AMZN" else None), \
             patch("learning.analysis_page.load_analysis", return_value=saved), \
             patch("learning.analysis_page.generate_analysis") as paid_call:
            app = AppTest.from_file(PAGE).run(timeout=30)
            app.button(key="open_saved_AMZN_2025").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="learning_company").value, "AMZN")
            self.assertEqual(app.selectbox(key="year_AMZN").value, 2025)
            self.assertEqual(app.session_state["learning_tab"], "6 Tekoälyanalyysi")
            self.assertTrue(app.success)
            self.assertTrue(any("Harjoitusanalyysi" in item.value for item in app.markdown))
            paid_call.assert_not_called()

    def test_generation_failure_shows_specific_error_and_does_not_retry_on_rerun(self):
        from learning.analysis_errors import AnalysisGenerationError
        data = reports()
        row = data["MSFT"]["rows"][-1]
        annual = {"cik": data["MSFT"]["cik"], "accession": row["sources"]["revenue"]["accession"],
                  "url": "https://www.sec.gov/fixture.htm", "filed": "2026-08-01", "sha256": "fixture-annual",
                  "excerpts": [{"id": "R0001", "text": "Operating cash flow increased."}]}
        error = AnalysisGenerationError("Tekoälypalvelun vastausta ei saatu määräajassa.", {"error_type": "APITimeoutError"})
        with patch("learning.company_page.load_report", side_effect=lambda *, company: deepcopy(data[company.ticker])), \
             patch("learning.analysis_page.load_annual_report", return_value=annual), \
             patch("learning.analysis_page.load_analysis", return_value=None), \
             patch("learning.analysis_page.get_openai_api_key", return_value="test-key"), \
             patch("learning.analysis_page.generate_analysis", side_effect=error) as paid_call:
            app = AppTest.from_file(PAGE).run(timeout=30)
            app.button(key="analyze_MSFT_2026").click().run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any("määräajassa" in item.value for item in app.error))
            self.assertTrue(any(item.label == "Virheen tekniset tiedot" for item in app.expander))
            app.run(timeout=30)
            paid_call.assert_called_once()

    def test_first_run_without_data_does_not_make_network_requests(self):
        with patch("learning.company_page.load_report", return_value=None), patch("market_data.sec.urlopen") as network:
            app = AppTest.from_file(PAGE).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(app.info)
            network.assert_not_called()

    def test_switching_company_and_comparison_preserves_different_periods(self):
        data = reports()
        with patch("learning.company_page.load_report", side_effect=lambda *, company: deepcopy(data[company.ticker])), patch("market_data.sec.urlopen") as network:
            app = AppTest.from_file(PAGE).run(timeout=30)
            self.assertFalse(app.exception)
            app.selectbox(key="learning_company").set_value("KO").run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(app.selectbox(key="year_KO").value, 2025)
            comparison = comparison_table(app)
            self.assertEqual(comparison.loc["Tilikausi päättyy", "Microsoft"], "30.06.2025")
            self.assertEqual(comparison.loc["Tilikausi päättyy", "Coca-Cola"], "31.12.2025")
            self.assertEqual(len(comparison.columns), 5)
            app.multiselect(key="comparison_companies").set_value(["AMZN", "NVDA"]).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(list(comparison_table(app).columns), ["Amazon", "NVIDIA"])
            app.multiselect(key="comparison_companies").set_value(list(COMPANIES)).run(timeout=30)
            app.radio(key="comparison_mode").set_value("Viimeisin tilikausi kustakin yhtiöstä").run(timeout=30)
            self.assertFalse(app.exception)
            comparison = comparison_table(app)
            self.assertEqual(comparison.loc["Tilikausi päättyy", "Microsoft"], "30.06.2026")
            network.assert_not_called()

    def test_missing_optional_metrics_render_without_becoming_zero(self):
        data = reports()
        optional = ("cost_of_revenue", "net_income", "current_assets", "current_liabilities",
                    "gross_profit", "gross_margin_pct", "net_margin_pct", "current_ratio")
        for report in data.values():
            for row in report["rows"]:
                for key in optional:
                    row[key] = None
                row["sources"] = {key: source for key, source in row["sources"].items() if key not in optional}
        with patch("learning.company_page.load_report", side_effect=lambda *, company: deepcopy(data[company.ticker])), patch("market_data.sec.urlopen") as network:
            app = AppTest.from_file(PAGE).run(timeout=30)
            self.assertFalse(app.exception)
            values = {metric.label: metric.value for metric in app.metric}
            self.assertEqual(values["Bruttomarginaali"], "ei saatavilla")
            self.assertEqual(values["Lyhyen aikavälin maksuvalmiussuhde"], "ei saatavilla")
            network.assert_not_called()
