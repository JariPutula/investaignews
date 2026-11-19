"""Visible assumptions and Python arithmetic, not probability-weighted forecasts."""

import json
import math

from market_data.financial_metrics import ratio
from market_data.sec import SecDataError

CASES = {"cautious": "Varovaisempi oletus", "continuity": "Jatkuvuusoletus", "strong": "Vahvempi oletus"}
FIELDS = {"growth_pct": "Liikevaihdon kasvu, %", "net_margin_pct": "Nettomarginaali, %",
          "ocf_margin_pct": "Liiketoiminnan rahavirta / myynti, %", "capex_pct": "Investoinnit / myynti, %"}


def _shift(value, delta, *, lower=-100., upper=100.):
    return min(upper, max(lower, value + delta)) if value is not None else None


def default_assumptions(row):
    """A transparent sensitivity around the latest year, not a central forecast."""
    growth = row.get("revenue_growth_pct")
    result = []
    for key, delta in (("cautious", -1), ("continuity", 0), ("strong", 1)):
        result.append({"case": key, "growth_pct": _shift(growth, delta*5, lower=-90),
                       "net_margin_pct": _shift(row.get("net_margin_pct"), delta*2),
                       "ocf_margin_pct": _shift(row.get("operating_cash_flow_margin_pct"), delta*2),
                       "capex_pct": _shift(row.get("capex_to_revenue_pct"), -delta*2, lower=0)})
    return result


def validate_assumptions(assumptions):
    if not isinstance(assumptions, list) or [a.get("case") for a in assumptions] != list(CASES):
        raise SecDataError("Skenaarioissa tarvitaan kolme nimettyä oletusjoukkoa oikeassa järjestyksessä.")
    for item in assumptions:
        for key in FIELDS:
            value = item.get(key)
            if value is None:
                continue
            lower = 0 if key == "capex_pct" else -90 if key == "growth_pct" else -100
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lower <= value <= 100:
                raise SecDataError(f"Skenaarion {FIELDS[key]} ei ole sallitulla välillä {lower}–100.")
    return assumptions


def project_scenarios(row, assumptions, valuation=None):
    validate_assumptions(assumptions)
    results = []
    for item in assumptions:
        revenue = row.get("revenue")
        sales = revenue * (1 + item["growth_pct"]/100) if revenue is not None and item.get("growth_pct") is not None else None
        amounts = {key: sales * item[field]/100 if sales is not None and item.get(field) is not None else None
                   for key, field in (("net_income", "net_margin_pct"), ("operating_cash_flow", "ocf_margin_pct"), ("capex", "capex_pct"))}
        fcf = amounts["operating_cash_flow"] - amounts["capex"] if amounts["operating_cash_flow"] is not None and amounts["capex"] is not None else None
        market_cap = valuation["quote"]["market_cap"] if valuation and valuation["available"] else None
        results.append({"case": item["case"], "fiscal_year": row["fiscal_year"]+1, "assumptions": dict(item),
                        "revenue": sales, **amounts, "free_cash_flow": fcf,
                        "market_cap_to_scenario_net_income": ratio(market_cap, amounts["net_income"]),
                        "scenario_fcf_yield_at_current_market_cap_pct": ratio(fcf, market_cap, percentage=True)})
    return results


def load_assumptions(company, report, row):
    path = company.data_dir / "scenarios" / f"{row['fiscal_year']}.json"
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["raw_sha256"] == report["raw_sha256"]:
                return validate_assumptions(saved["assumptions"])
        except (ValueError, KeyError, OSError) as exc:
            raise SecDataError("Tallennetut skenaario-oletukset eivät avaudu.") from exc
    return default_assumptions(row)


def save_assumptions(company, report, row, assumptions):
    validate_assumptions(assumptions)
    path = company.data_dir / "scenarios" / f"{row['fiscal_year']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"raw_sha256": report["raw_sha256"], "assumptions": assumptions}), encoding="utf-8")
    temporary.replace(path)
