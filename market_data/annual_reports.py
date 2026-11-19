"""Download and retain the exact official filing used by the selected numbers."""

from datetime import datetime, timezone
import hashlib
import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

from market_data.sec import SecDataError


def download(url, user_agent):
    if "@" not in user_agent or "\n" in user_agent or "\r" in user_agent:
        raise SecDataError("Anna SEC-haun yhteyssähköposti aineiston hakuosiossa.")
    try:
        with urlopen(Request(url, headers={"User-Agent": user_agent}), timeout=30) as response:
            raw = response.read(20_000_001)
        if len(raw) > 20_000_000:
            raise SecDataError("Vuosiraportti ylittää tämän harjoituksen kokorajan.")
        return raw
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SecDataError("Virallisen vuosiraportin haku epäonnistui. Aiempi aineisto säilyy.") from exc


def excerpts_from_html(raw, *, scope="cash_flow"):
    """Select bounded cash-flow/risk excerpts; never send hidden XBRL or scripts."""
    soup = BeautifulSoup(raw, "html.parser")
    for tag in list(soup.find_all(["script", "style", "ix:header", "ix:hidden", "head"])):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    if len(text) < 5_000:
        raise SecDataError("Ladattu sivu ei sisällä riittävästi vuosiraportin tekstiä.")
    chunks = [text[i:i + 1800] for i in range(0, len(text), 1800)]
    terms = ("cash flow", "operating activities", "capital expenditure", "property and equipment",
             "working capital", "finance lease", "liquidity", "purchase obligation", "risk", "data center")
    ranked = sorted(range(len(chunks)), key=lambda i: sum(chunks[i].lower().count(t) for t in terms), reverse=True)
    selected = sorted(i for i in ranked[:24] if any(t in chunks[i].lower() for t in terms))
    if scope == "company":
        groups = (
            ("net sales", "revenue", "demand", "growth", "segment"),
            ("gross margin", "operating income", "operating expenses", "profitability"),
            ("cash flow", "operating activities", "capital expenditure", "working capital"),
            ("liquidity", "current assets", "current liabilities", "balance sheet", "risk"),
        )
        chosen = set()
        for group in groups:
            candidates = sorted(range(len(chunks)), key=lambda i: sum(chunks[i].lower().count(t) for t in group), reverse=True)
            chosen.update(i for i in candidates[:6] if any(t in chunks[i].lower() for t in group))
        selected = sorted(chosen)
    elif scope != "cash_flow":
        raise ValueError("Tuntematon analyysin aineistorajaus.")
    if not selected:
        raise SecDataError("Vuosiraportista ei löytynyt kassavirran analyysiin tarvittavia kohtia.")
    return [{"id": f"R{i + 1:04}", "text": chunks[i]} for i in selected]


def load_annual_report(company, row, *, scope="cash_flow"):
    source = row.get("sources", {}).get("revenue")
    if not source:
        return None
    directory = company.data_dir / "annual_reports"
    metadata_path = directory / f"{source['accession']}.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        raw = (directory / f"{source['accession']}.html").read_bytes()
        if metadata["sha256"] != hashlib.sha256(raw).hexdigest():
            raise ValueError("checksum")
        if metadata["accession"] != source["accession"] or metadata["cik"] != company.cik:
            raise ValueError("identity")
        return {**metadata, "excerpts": excerpts_from_html(raw, scope=scope)}
    except (OSError, ValueError, KeyError) as exc:
        raise SecDataError("Tallennettu vuosiraportti ei läpäise eheystarkistusta. Hae se uudelleen.") from exc


def prepare_annual_report(company, row, user_agent, *, scope="cash_flow"):
    existing = load_annual_report(company, row, scope=scope)
    if existing:
        return existing
    source = row.get("sources", {}).get("revenue")
    if not source:
        raise SecDataError("Valitun tilikauden liikevaihdon lähde puuttuu.")
    accession = source["accession"]
    archive = f"https://www.sec.gov/Archives/edgar/data/{company.cik}/{accession.replace('-', '')}/"
    submissions = json.loads(download(f"https://data.sec.gov/submissions/CIK{company.cik:010d}.json", user_agent))
    recent = submissions["filings"]["recent"]
    if accession in recent["accessionNumber"]:
        index = recent["accessionNumber"].index(accession)
        url = urljoin(archive, recent["primaryDocument"][index])
    else:
        soup = BeautifulSoup(download(source["url"], user_agent), "html.parser")
        url = None
        for tr in soup.select("table.tableFile tr"):
            cells = tr.find_all("td")
            if len(cells) >= 4 and cells[3].get_text(strip=True) in {"10-K", "10-K/A"}:
                anchor = cells[2].find("a", href=True)
                if anchor:
                    url = urljoin(archive, str(anchor["href"]).replace("/ix?doc=", ""))
                    break
    if not url or not url.startswith(archive):
        raise SecDataError("Vuosiraportin virallista asiakirjaa ei löytynyt lähderaportista.")
    raw = download(url, user_agent)
    excerpts = excerpts_from_html(raw, scope=scope)
    metadata = {"cik": company.cik, "accession": accession, "filed": source["filed"],
                "url": url, "sha256": hashlib.sha256(raw).hexdigest(),
                "retrieved_at": datetime.now(timezone.utc).isoformat()}
    directory = company.data_dir / "annual_reports"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{accession}.html").write_bytes(raw)
    (directory / f"{accession}.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {**metadata, "excerpts": excerpts}
