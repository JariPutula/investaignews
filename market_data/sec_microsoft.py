"""Fetch Microsoft's SEC Company Facts and extract full fiscal years.

Intentionally scoped to Microsoft: its fiscal years run July 1 to June 30.
No prices, estimates, quarterly data or inferred missing financial values.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


CIK = 789019
SOURCE_URL = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK:010d}.json"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "research" / "msft"
METRICS = {
    "revenue": {
        "label": "Liikevaihto",
        "tags": (
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "Revenues",
        ),
    },
    "operating_income": {"label": "Liikevoitto", "tags": ("OperatingIncomeLoss",)},
    "operating_cash_flow": {
        "label": "Liiketoiminnan rahavirta",
        "tags": ("NetCashProvidedByUsedInOperatingActivities",),
    },
    "capex": {
        "label": "Aineelliset käyttöomaisuusinvestoinnit",
        "tags": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    },
}


class SecDataError(ValueError):
    """The source is unavailable, inconsistent or unsuitable for this exercise."""


def _valid_annual_fact(fact: dict, as_of: date) -> bool:
    try:
        start = date.fromisoformat(fact["start"])
        end = date.fromisoformat(fact["end"])
        filed = date.fromisoformat(fact["filed"])
        value = fact["val"]
        return (
            fact.get("form") in {"10-K", "10-K/A"}
            and start == date(end.year - 1, 7, 1)
            and end.month == 6
            and end.day == 30
            and end <= filed <= as_of
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and bool(re.fullmatch(r"\d{10}-\d{2}-\d{6}", fact.get("accn", "")))
        )
    except (KeyError, TypeError, ValueError):
        return False


def percentage_change(new: float | None, old: float | None) -> float | None:
    """A conventional growth percentage needs a positive starting value."""
    if new is None or old is None or old <= 0:
        return None
    return (new / old - 1) * 100


def revenue_cagr(rows: list[dict]) -> float | None:
    """Annualized revenue growth over the selected, consecutive fiscal years."""
    if len(rows) < 2:
        return None
    years = [r["fiscal_year"] for r in rows]
    if years != list(range(years[0], years[-1] + 1)):
        return None
    first, last = rows[0]["revenue"], rows[-1]["revenue"]
    if first is None or last is None or first <= 0 or last < 0:
        return None
    return ((last / first) ** (1 / (years[-1] - years[0])) - 1) * 100


def extract_annual_metrics(
    payload: dict, *, years: int = 4, as_of: date | None = None
) -> dict:
    """Select each period's latest filed USD facts and preserve their provenance.

    SEC's ``fy`` describes the filing's fiscal year, including comparative
    columns. The actual start/end dates determine the year being measured.
    A missing metric stays None, including for the newest available year.
    """
    if not 2 <= years <= 10:
        raise SecDataError("Valitse 2–10 tilikautta.")
    if payload.get("cik") != CIK:
        raise SecDataError("Aineiston yhtiötunniste ei vastaa Microsoftia.")
    as_of = as_of or date.today()
    facts = payload.get("facts", {}).get("us-gaap", {})
    candidates: dict[str, dict[int, list[tuple]]] = {}
    available_years: set[int] = set()
    for key, definition in METRICS.items():
        candidates[key] = {}
        for preference, tag in enumerate(definition["tags"]):
            for fact in facts.get(tag, {}).get("units", {}).get("USD", []):
                if not _valid_annual_fact(fact, as_of):
                    continue
                year = date.fromisoformat(fact["end"]).year
                available_years.add(year)
                candidates[key].setdefault(year, []).append(
                    (fact["filed"], -preference, fact["accn"], tag, fact)
                )
    if not available_years:
        raise SecDataError("Microsoftin kokonaisia USD-tilikausia ei löytynyt.")
    latest_year = max(available_years)
    selected_years = range(latest_year - years + 1, latest_year + 1)
    rows, warnings = [], []
    for year in selected_years:
        row = {
            "fiscal_year": year,
            "period_start": date(year - 1, 7, 1).isoformat(),
            "period_end": date(year, 6, 30).isoformat(),
            "sources": {},
        }
        for key, definition in METRICS.items():
            entries = candidates[key].get(year, [])
            if not entries:
                row[key] = None
                warnings.append(f"{year}: {definition['label']} puuttuu.")
                continue
            rank = max(entry[:3] for entry in entries)
            best = [entry for entry in entries if entry[:3] == rank]
            if len({entry[4]["val"] for entry in best}) > 1:
                raise SecDataError(f"{year}: ristiriitaiset {definition['label']}-luvut samassa lähteessä.")
            _, _, accession, tag, fact = best[0]
            value = abs(fact["val"]) if key == "capex" else fact["val"]
            row[key] = value
            row["sources"][key] = {
                "taxonomy": "us-gaap", "tag": tag, "unit": "USD",
                "source_value": fact["val"], "filed": fact["filed"],
                "form": fact["form"], "accession": accession,
                "period_start": fact["start"], "period_end": fact["end"],
                "url": f"https://www.sec.gov/Archives/edgar/data/{CIK}/{accession}-index.html",
            }
        revenue, income = row["revenue"], row["operating_income"]
        cash_flow, capex = row["operating_cash_flow"], row["capex"]
        row["operating_margin_pct"] = (
            income / revenue * 100 if income is not None and revenue is not None and revenue > 0 else None
        )
        row["free_cash_flow"] = (
            cash_flow - capex if cash_flow is not None and capex is not None else None
        )
        row["revenue_growth_pct"] = (
            percentage_change(revenue, rows[-1]["revenue"]) if rows else None
        )
        rows.append(row)
    return {
        "schema_version": 1, "company": payload.get("entityName", "Microsoft"),
        "ticker": "MSFT", "cik": CIK, "unit": "USD", "as_of": as_of.isoformat(),
        "source_url": SOURCE_URL,
        "selection_policy": "Latest filed full July–June USD facts from 10-K or 10-K/A, as of the cutoff date.",
        "free_cash_flow_definition": "Operating cash flow minus cash payments to acquire property, plant and equipment; capex is a positive expense.",
        "rows": rows, "revenue_cagr_pct": revenue_cagr(rows), "warnings": warnings,
    }


def download_companyfacts(user_agent: str) -> bytes:
    """One request, with a single limited retry for a busy SEC server."""
    user_agent = user_agent.strip()
    if not user_agent or "@" not in user_agent or "\n" in user_agent or "\r" in user_agent:
        raise SecDataError("Anna SEC_USER_AGENT-tunniste, jossa on sovelluksen nimi ja yhteyssähköposti.")
    request = Request(SOURCE_URL, headers={"User-Agent": user_agent, "Accept": "application/json"})
    for attempt in range(2):
        try:
            with urlopen(request, timeout=30) as response:
                return response.read()
        except HTTPError as exc:
            if attempt == 0 and exc.code in {429, 503}:
                time.sleep(1)
                continue
            raise SecDataError(
                f"SEC-haku epäonnistui (HTTP {exc.code}). Tarkista tunniste ja kokeile myöhemmin uudelleen."
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise SecDataError("SEC-yhteys ei onnistunut. Tarkista verkkoyhteys; aiempi aineisto säilyy.") from exc
    raise SecDataError("SEC-haku ei onnistunut.")


def build_report(
    *, data_dir: Path = DEFAULT_DATA_DIR, user_agent: str = "", refresh: bool = False,
    offline: bool = False, as_of: date | None = None,
) -> dict:
    """Use a local immutable snapshot, downloading only on request or first use."""
    data_dir = Path(data_dir)
    snapshots = sorted(
        path for path in (data_dir / "raw").glob("companyfacts_*.json")
        if not path.name.endswith(".metadata.json")
    )
    if refresh and offline:
        raise SecDataError("Offline-tilaa ja uutta hakua ei voi valita samanaikaisesti.")
    is_new = refresh or not snapshots
    if is_new:
        if offline:
            raise SecDataError("Paikallista SEC-aineistoa ei vielä ole. Tee ensin yksi verkkohaku.")
        raw = download_companyfacts(user_agent)
        retrieved = datetime.now(timezone.utc)
        snapshot = data_dir / "raw" / f"companyfacts_{retrieved.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    else:
        snapshot = snapshots[-1]
        raw = snapshot.read_bytes()
        retrieved = None
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SecDataError("SEC-aineisto ei ole kelvollista JSON-dataa.") from exc
    report = extract_annual_metrics(payload, as_of=as_of)
    # Validate the response before committing a new snapshot or replacing output.
    if is_new:
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(raw)
        metadata = {
            "source_url": SOURCE_URL, "retrieved_at": retrieved.isoformat(),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        snapshot.with_suffix(".metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
    else:
        metadata_path = snapshot.with_suffix(".metadata.json")
        if not metadata_path.exists():
            raise SecDataError("Aineiston lähdemetadata puuttuu; hae aineisto uudelleen.")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("sha256") != hashlib.sha256(raw).hexdigest():
            raise SecDataError("Paikallisen lähdeaineiston tarkistussumma ei täsmää; hae uudelleen.")
    report.update({
        "retrieved_at": metadata["retrieved_at"],
        "raw_sha256": metadata["sha256"], "raw_file": f"raw/{snapshot.name}",
    })
    output = data_dir / "annual_metrics.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(output)
    return report


def load_report(data_dir: Path = DEFAULT_DATA_DIR) -> dict | None:
    output = Path(data_dir) / "annual_metrics.json"
    if not output.exists():
        return None
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise SecDataError("Tallennettu yhteenveto ei avaudu. Muodosta se uudelleen hakukomennolla.") from exc
    if report.get("schema_version") != 1 or report.get("cik") != CIK:
        raise SecDataError("Tallennettu yhteenveto ei vastaa Microsoftin oppimisharjoitusta.")
    return report
