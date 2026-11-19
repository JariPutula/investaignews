"""SEC Company Facts for the explicitly configured learning companies.

Only full fiscal years and year-end balances are used, including NVIDIA's
52/53-week calendar. Company-specific cash investment definitions stay visible.
"""

from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import NotRequired, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from market_data.companies import Company
from market_data.financial_metrics import calculate_metrics, percentage_change, revenue_cagr

class MetricDefinition(TypedDict):
    label: str
    tags: tuple[str, ...]
    required: bool
    instant: NotRequired[bool]


METRICS: dict[str, MetricDefinition] = {
    "revenue": {"label": "Liikevaihto", "tags": ("RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet", "Revenues"), "required": True},
    "operating_income": {"label": "Liikevoitto", "tags": ("OperatingIncomeLoss",), "required": True},
    "operating_cash_flow": {"label": "Liiketoiminnan rahavirta", "tags": ("NetCashProvidedByUsedInOperatingActivities",), "required": True},
    "capex": {"label": "Käyttöomaisuusinvestoinnit", "tags": ("PaymentsToAcquirePropertyPlantAndEquipment",), "required": True},
    "cost_of_revenue": {"label": "Myytyjen tuotteiden ja palvelujen kustannukset", "tags": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"), "required": False},
    "net_income": {"label": "Emoyhtiön omistajille kuuluva nettotulos", "tags": ("NetIncomeLoss",), "required": False},
    "current_assets": {"label": "Lyhytaikaiset varat", "tags": ("AssetsCurrent",), "instant": True, "required": False},
    "current_liabilities": {"label": "Lyhytaikaiset velat", "tags": ("LiabilitiesCurrent",), "instant": True, "required": False},
}
CORE_METRICS = {key: value for key, value in METRICS.items() if value["required"]}
SCHEMA_VERSION = 2


class SecDataError(ValueError):
    """The source is unavailable, inconsistent or unsuitable for this exercise."""


def _valid_fact(fact, company, definition, as_of):
    try:
        end = date.fromisoformat(fact["end"])
        filed = date.fromisoformat(fact["filed"])
        start, expected_end = company.period(end.year)
        value = fact["val"]
        return (
            fact.get("form") in {"10-K", "10-K/A"}
            and end == expected_end
            and (not fact.get("start") if definition.get("instant") else fact.get("start") == start.isoformat())
            and end <= filed <= as_of
            and isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value)
            and bool(re.fullmatch(r"\d{10}-\d{2}-\d{6}", fact.get("accn", "")))
        )
    except (KeyError, TypeError, ValueError):
        return False


def extract_annual_metrics(payload, *, company: Company, years=4, as_of=None):
    """Use actual periods rather than SEC's filing-year fy labels."""
    if not 2 <= years <= 10:
        raise SecDataError("Valitse 2–10 tilikautta.")
    if payload.get("cik") != company.cik:
        raise SecDataError(f"Aineiston yhtiötunniste ei vastaa yhtiötä {company.name}.")
    as_of = as_of or date.today()
    facts = payload.get("facts", {}).get("us-gaap", {})
    candidates, available_years = {}, set()
    for key, definition in METRICS.items():
        candidates[key] = {}
        tags = company.capex_tags if key == "capex" else definition["tags"]
        for preference, tag in enumerate(tags):
            for fact in facts.get(tag, {}).get("units", {}).get("USD", []):
                if not _valid_fact(fact, company, definition, as_of):
                    continue
                year = date.fromisoformat(fact["end"]).year
                if definition["required"]:
                    available_years.add(year)
                candidates[key].setdefault(year, []).append((fact["filed"], -preference, fact["accn"], tag, fact))
    if not available_years:
        raise SecDataError(f"Yhtiön {company.name} kokonaisia USD-tilikausia ei löytynyt.")
    latest = max(available_years)
    rows, warnings, notes = [], [], []
    for year in range(latest - years + 1, latest + 1):
        start, end = company.period(year)
        row = {"fiscal_year": year, "period_start": start.isoformat(), "period_end": end.isoformat(), "sources": {}}
        for key, definition in METRICS.items():
            entries = candidates[key].get(year, [])
            if not entries:
                row[key] = None
                (warnings if definition["required"] else notes).append(f"{year}: {definition['label']} puuttuu.")
                continue
            rank = max(entry[:3] for entry in entries)
            best = [entry for entry in entries if entry[:3] == rank]
            if len({entry[4]["val"] for entry in best}) > 1:
                raise SecDataError(f"{year}: ristiriitaiset {definition['label']}-luvut samassa lähteessä.")
            _, _, accession, tag, fact = best[0]
            row[key] = abs(fact["val"]) if key == "capex" else fact["val"]
            row["sources"][key] = {
                "taxonomy": "us-gaap", "tag": tag, "unit": "USD", "source_value": fact["val"],
                "filed": fact["filed"], "form": fact["form"], "accession": accession,
                "period_start": fact.get("start"), "period_end": fact["end"],
                "url": f"https://www.sec.gov/Archives/edgar/data/{company.cik}/{accession}-index.html",
            }
        row.update(calculate_metrics(row, rows[-1] if rows else None))
        rows.append(row)
    return {
        "schema_version": SCHEMA_VERSION, "company": company.name,
        "sec_entity_name": payload.get("entityName"), "ticker": company.ticker,
        "cik": company.cik, "unit": "USD", "as_of": as_of.isoformat(),
        "source_url": company.source_url,
        "selection_policy": "Latest filed full fiscal-year USD facts and fiscal-year-end balances from 10-K or 10-K/A, as of the cutoff date.",
        "free_cash_flow_definition": (
            "Operating cash flow minus cash payments for property, plant, equipment and intangible assets."
            if company.capex_includes_intangibles else
            "Operating cash flow minus cash payments to acquire property, plant and equipment; capex is a positive expense."
        ),
        "capex_includes_intangibles": company.capex_includes_intangibles,
        "rows": rows, "revenue_cagr_pct": revenue_cagr(rows), "warnings": warnings, "data_notes": notes,
    }


def download_companyfacts(user_agent: str, *, company: Company) -> bytes:
    user_agent = user_agent.strip()
    if not user_agent or "@" not in user_agent or "\n" in user_agent or "\r" in user_agent:
        raise SecDataError("Anna SEC_USER_AGENT-tunniste, jossa on sovelluksen nimi ja yhteyssähköposti.")
    request = Request(company.source_url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    for attempt in range(2):
        try:
            with urlopen(request, timeout=30) as response:
                return response.read()
        except HTTPError as exc:
            if attempt == 0 and exc.code in {429, 503}:
                time.sleep(1)
                continue
            raise SecDataError(f"SEC-haku epäonnistui (HTTP {exc.code}). Tarkista tunniste ja kokeile myöhemmin uudelleen.") from exc
        except (URLError, TimeoutError) as exc:
            raise SecDataError("SEC-yhteys ei onnistunut. Tarkista verkkoyhteys; aiempi aineisto säilyy.") from exc
    raise SecDataError("SEC-haku ei onnistunut.")


def build_report(*, company: Company, data_dir=None, user_agent="", refresh=False, offline=False, as_of=None):
    data_dir = Path(data_dir) if data_dir is not None else company.data_dir
    snapshots = sorted(path for path in (data_dir / "raw").glob("companyfacts_*.json") if not path.name.endswith(".metadata.json"))
    if refresh and offline:
        raise SecDataError("Offline-tilaa ja uutta hakua ei voi valita samanaikaisesti.")
    is_new = refresh or not snapshots
    if is_new:
        if offline:
            raise SecDataError("Paikallista SEC-aineistoa ei vielä ole. Tee ensin yksi verkkohaku.")
        raw = download_companyfacts(user_agent, company=company)
        retrieved = datetime.now(timezone.utc)
        snapshot = data_dir / "raw" / f"companyfacts_{retrieved.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    else:
        snapshot = snapshots[-1]
        raw = snapshot.read_bytes()
        metadata_path = snapshot.with_suffix(".metadata.json")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SecDataError("Aineiston lähdemetadata puuttuu tai ei avaudu; hae aineisto uudelleen.") from exc
        if metadata.get("sha256") != hashlib.sha256(raw).hexdigest():
            raise SecDataError("Paikallisen lähdeaineiston tarkistussumma ei täsmää; hae uudelleen.")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SecDataError("SEC-aineisto ei ole kelvollista JSON-dataa.") from exc
    report = extract_annual_metrics(payload, company=company, as_of=as_of)
    if is_new:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(raw)
        metadata = {"source_url": company.source_url, "retrieved_at": retrieved.isoformat(), "sha256": hashlib.sha256(raw).hexdigest()}
        snapshot.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report.update({"retrieved_at": metadata["retrieved_at"], "raw_sha256": metadata["sha256"], "raw_file": f"raw/{snapshot.name}"})
    output = data_dir / "annual_metrics.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    return report


def load_report(*, company: Company, data_dir=None):
    output = (Path(data_dir) if data_dir is not None else company.data_dir) / "annual_metrics.json"
    if not output.exists():
        return None
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SecDataError("Tallennettu yhteenveto ei avaudu. Muodosta se uudelleen hakukomennolla.") from exc
    if report.get("cik") != company.cik or report.get("schema_version") != SCHEMA_VERSION:
        raise SecDataError("Tallennettu yhteenveto tarvitsee päivityksen. Muodosta luvut uudelleen paikallisesta aineistosta.")
    for index, row in enumerate(report["rows"]):
        row.update(calculate_metrics(row, report["rows"][index - 1] if index else None))
    return report


def comparison_rows(reports, fiscal_year=None):
    """Keep actual periods visible even when the selected year labels match."""
    result = []
    for report in reports:
        row = report["rows"][-1] if fiscal_year is None else next((r for r in report["rows"] if r["fiscal_year"] == fiscal_year), None)
        if row is not None:
            result.append({"company": report["company"], "ticker": report["ticker"], "as_of": report["as_of"], **row})
    return result
