"""Exercise company switching and incomplete data without SEC or local files."""

from copy import deepcopy
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.test_sec_companies import extract


PAGE = str(Path(__file__).resolve().parents[1] / "pages" / "01_Yhtioanalyysi.py")


def reports():
    result = {ticker: extract(ticker) for ticker in ("MSFT", "KO")}
    for report in result.values():
        report["retrieved_at"] = "2026-11-15T10:00:00+00:00"
    return result


class LearningPageTests(TestCase):
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
            comparison = app.dataframe[0].value
            self.assertEqual(comparison.loc["Tilikausi päättyy", "Microsoft"], "30.06.2025")
            self.assertEqual(comparison.loc["Tilikausi päättyy", "Coca-Cola"], "31.12.2025")
            app.radio(key="comparison_mode").set_value("Kummankin viimeisin tilikausi").run(timeout=30)
            self.assertFalse(app.exception)
            comparison = app.dataframe[0].value
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
