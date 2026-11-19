"""The brief covers four topics, stays short, and preserves source/cost boundaries."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from learning.company_brief import BriefInterpretation, InvestorInterpretation, TOPICS, generate_analysis, make_packet, validate_brief
from market_data.annual_reports import excerpts_from_html
from market_data.companies import get_company
from market_data.sec import SecDataError
from tests.test_sec_companies import extract


def brief_packet():
    report = extract("AMZN")
    report["raw_sha256"] = "fixture"
    annual = {"accession": report["rows"][-1]["sources"]["revenue"]["accession"], "cik": report["cik"],
              "filed": "2026-08-01", "sha256": "annual", "url": "https://www.sec.gov/fixture.htm",
              "excerpts": [{"id": "R0001", "text": "Operating cash flow increased due to higher net income."}]}
    return make_packet(report, report["rows"][-1], annual)


def brief_fixture():
    def claim(text, source):
        return {"text": text, "evidence": [{"source_id": source, "quote": ""}]}
    return InvestorInterpretation.model_validate({
        "overview": claim("Myynti, kannattavuus ja rahavirta muodostavat tarkastelun lähtökohdan.", "M2025_revenue"),
        "topics": [
            {"topic": "growth", **claim("Myynnin kehitys kuvaa liiketoiminnan kokoa ja kasvun jatkuvuutta.", "M2025_revenue")},
            {"topic": "profitability", **claim("Marginaali kertoo, paljonko myynnistä jää liikevoittoa.", "M2025_operating_margin_pct")},
            {"topic": "cash_flow", **claim("Investointien jälkeen jäävä rahavirta auttaa arvioimaan rahoituksen tarvetta.", "M2025_free_cash_flow")},
            {"topic": "balance", **claim("Lyhytaikaiset varat suhteessa velkoihin kuvaavat maksuvalmiutta tilinpäätöshetkellä.", "M2025_current_ratio")},
        ],
        "strengths": [claim("Liiketoiminta tuottaa investointien jälkeistä rahavirtaa.", "M2025_free_cash_flow")],
        "counterpoints": [claim("Maksuvalmius ei yksin kuvaa koko velkariskiä.", "D_balance")],
        "questions": [claim("Kuinka suuri osa investoinneista ylläpitää nykyistä toimintaa?", "M2025_capex")],
        "outlook": {**claim("Kasvun jatkuminen voisi vahvistaa tulosta, jos marginaalit säilyvät.", "S_continuity"), "basis": "oletus"},
        "investment_case": {**claim("Liiketoiminta on tarkastelun arvoinen, mutta hinnan houkuttelevuutta ei voi arvioida ilman kurssiaineistoa.", "V_missing"), "basis": "tulkinta", "price_assessment": "unavailable"},
        "rebuttal": {**claim("Investointien kasvu voisi kumota myönteisen kassavirtanäkemyksen.", "M2025_capex"), "basis": "tulkinta"},
    })


class CompanyBriefTests(unittest.TestCase):
    def test_packet_contains_full_history_and_balance_sources(self):
        packet = brief_packet()
        self.assertEqual(len(packet["periods"]), 4)
        self.assertEqual(packet["periods"][0]["fiscal_year"], 2022)
        self.assertIn("M2025_current_ratio", packet["sources"])
        self.assertEqual(set(packet["sources"]["M2025_current_ratio"]["fact_sources"]), {"current_assets", "current_liabilities"})
        self.assertIn("Kokonaisvelat", packet["sources"]["D_balance"]["text"])
        self.assertEqual(len(validate_brief(brief_fixture(), packet).topics), 4)

    def test_selected_old_year_does_not_receive_later_numbers(self):
        report = extract("AMZN")
        report["raw_sha256"] = "fixture"
        row = report["rows"][1]
        annual = {"accession": row["sources"]["revenue"]["accession"], "cik": report["cik"], "filed": "2026-08-01",
                  "sha256": "annual", "url": "https://www.sec.gov/fixture.htm", "excerpts": []}
        packet = make_packet(report, row, annual)
        self.assertEqual([p["fiscal_year"] for p in packet["periods"]], [2022, 2023])
        self.assertFalse(any(key.startswith("M2024") or key.startswith("M2025") for key in packet["sources"]))

    def test_missing_balance_is_explicit_and_may_be_cited(self):
        packet = brief_packet()
        for key in ("current_assets", "current_liabilities", "current_ratio"):
            del packet["sources"][f"M2025_{key}"]
        analysis = brief_fixture()
        analysis.topics[-1].text = "Maksuvalmiuden tiedot puuttuvat; arviota ei muodosteta."
        analysis.topics[-1].evidence[0].source_id = "D_balance"
        validate_brief(analysis, packet)

    def test_duplicate_topics_wrong_metric_and_excess_text_are_rejected(self):
        for defect in ("duplicate", "source", "length"):
            analysis = brief_fixture()
            if defect == "duplicate":
                analysis.topics[1].topic = "growth"
            elif defect == "source":
                analysis.topics[1].evidence[0].source_id = "M2025_revenue"
            else:
                for claim in [analysis.overview, *analysis.topics, *analysis.strengths, *analysis.counterpoints, *analysis.questions]:
                    claim.text = "tulkinta " * 75
            with self.subTest(defect=defect), self.assertRaises(SecDataError):
                validate_brief(analysis, brief_packet())

    def test_one_paid_call_is_reused_and_bad_result_is_not_saved(self):
        packet = brief_packet()
        client = Mock()
        client.chat.completions.parse.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(parsed=brief_fixture(), refusal=None))],
            usage=SimpleNamespace(prompt_tokens=1000, completion_tokens=500))
        with TemporaryDirectory() as directory, patch("learning.company_brief.analysis_path", return_value=Path(directory)/"brief.json"):
            first = generate_analysis(get_company("AMZN"), packet, "secret-key", client=client)
            self.assertEqual(first, generate_analysis(get_company("AMZN"), packet, "", client=client))
            client.chat.completions.parse.assert_called_once()
            self.assertNotIn("secret-key", (Path(directory)/"brief.json").read_text())
        bad = brief_fixture()
        bad.topics[0].evidence[0].source_id = "invented-source"
        client.chat.completions.parse.return_value.choices[0].message.parsed = bad
        with TemporaryDirectory() as directory, patch("learning.company_brief.analysis_path", return_value=Path(directory)/"brief.json"):
            with self.assertRaises(SecDataError):
                generate_analysis(get_company("AMZN"), packet, "secret-key", client=client)
            self.assertFalse((Path(directory)/"brief.json").exists())

    def test_company_excerpts_cover_more_than_cash_flow(self):
        sections = ["Revenue and net sales drive growth. ", "Gross margin and operating income reflect profitability. ",
                    "Operating activities and cash flow support investment. ", "Current assets and current liabilities reflect liquidity. "]
        raw = ("<html><body>" + "".join("<p>" + text*100 + "</p>" for text in sections) + "</body></html>").encode()
        text = " ".join(e["text"].lower() for e in excerpts_from_html(raw, scope="company"))
        for term in ("revenue", "gross margin", "cash flow", "current liabilities"):
            self.assertIn(term, text)
