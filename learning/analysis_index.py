"""A local index for finding existing analyses without any API calls."""

import json

from learning.company_analysis import Analysis, MODEL, PROMPT_VERSION
from learning.company_brief import BRIEF_VERSION, BriefInterpretation, InvestorInterpretation
from market_data.companies import COMPANIES


def list_saved_analyses():
    newest = {}
    for company in COMPANIES.values():
        for path in (company.data_dir / "analyses").glob("*_*.json"):
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                year = saved["fiscal_year"]
                if type(year) is not int or saved["model"] != MODEL:
                    continue
                brief = saved.get("analysis_scope") == "company_brief"
                if brief:
                    if saved["brief_version"] not in {1, BRIEF_VERSION} or path.stem != f"brief_{year}_{saved['fingerprint'][:20]}":
                        continue
                    schema = InvestorInterpretation if saved["brief_version"] >= 2 else BriefInterpretation
                    schema.model_validate(saved["analysis"])
                else:
                    if saved["prompt_version"] != PROMPT_VERSION or path.stem != f"{year}_{saved['fingerprint'][:20]}":
                        continue
                    Analysis.model_validate(saved["analysis"])
                entry = {"ticker": company.ticker, "company": company.name, "fiscal_year": year,
                         "created_at": saved["created_at"], "analysis_scope": "company_brief" if brief else "cash_flow",
                         "brief_version": saved.get("brief_version", 0)}
                key = (company.ticker, year)
                if key not in newest or (entry["brief_version"], entry["created_at"]) > (newest[key]["brief_version"], newest[key]["created_at"]):
                    newest[key] = entry
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return sorted(newest.values(), key=lambda item: item["created_at"], reverse=True)
