"""Dated MSFT facts: exact fiscal periods, TTM bridge and official text sources."""

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo

from market_data.companies import get_company
from market_data.sec import SecDataError, build_report
from market_data.annual_reports import download, excerpts_from_html, prepare_annual_report
from market_data.financial_metrics import percentage_change, ratio

COMPANY = get_company("MSFT")
ROOT = COMPANY.data_dir / "exercise"
VERSION = 1
FLOW_TAGS = {
    "revenue": ("RevenueFromContractWithCustomerExcludingAssessedTax",),
    "operating_income": ("OperatingIncomeLoss",),
    "net_income": ("NetIncomeLoss",),
    "operating_cash_flow": ("NetCashProvidedByUsedInOperatingActivities",),
    "cash_capex": ("PaymentsToAcquirePropertyPlantAndEquipment",),
    "depreciation": ("Depreciation",),
    "amortization": ("AmortizationOfIntangibleAssets",),
    "finance_lease_additions": ("RightOfUseAssetObtainedInExchangeForFinanceLeaseLiability",),
    "receivables_increase": ("IncreaseDecreaseInAccountsReceivable",),
    "inventory_increase": ("IncreaseDecreaseInInventories",),
    "other_current_assets_increase": ("IncreaseDecreaseInOtherCurrentAssets",),
    "payables_increase": ("IncreaseDecreaseInAccountsPayable",),
    "other_current_liabilities_increase": ("IncreaseDecreaseInOtherCurrentLiabilities",),
    "customer_liabilities_increase": ("IncreaseDecreaseInContractWithCustomerLiability",),
}
BALANCE_TAGS = {
    "cash_and_short_term_investments": ("CashCashEquivalentsAndShortTermInvestments",),
    # LongTermDebt includes current maturities: never add LongTermDebtCurrent again.
    "borrowings": ("LongTermDebt",),
    "finance_lease_debt": ("FinanceLeaseLiability",),
    "shares": ("CommonStockSharesOutstanding",),
}


def today_rome():
    return datetime.now(ZoneInfo("Europe/Rome")).date()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def select_fact(payload, tags, *, end, start=None, cutoff, unit="USD", required=True):
    candidates = []
    for rank, tag in enumerate(tags):
        for fact in payload.get("facts", {}).get("us-gaap", {}).get(tag, {}).get("units", {}).get(unit, []):
            try:
                valid = (fact.get("start") == start and fact["end"] == end
                         and fact.get("form") in {"10-K", "10-K/A", "10-Q", "10-Q/A"}
                         and end <= fact["filed"] <= cutoff and bool(re.fullmatch(r"\d{10}-\d{2}-\d{6}", fact.get("accn", "")))
                         and isinstance(fact["val"], (float, int)) and not isinstance(fact["val"], bool) and math.isfinite(fact["val"]))
            except (KeyError, TypeError, ValueError):
                continue
            if valid:
                candidates.append((fact["filed"], -rank, fact["accn"], tag, fact))
    if not candidates:
        if required:
            raise SecDataError(f"Harjoituksen lähtötieto puuttuu: {tags[0]}, {start or 'tase'}–{end}.")
        return None
    best_rank = max(item[:3] for item in candidates)
    best = [item for item in candidates if item[:3] == best_rank]
    if len({item[-1]["val"] for item in best}) != 1:
        raise SecDataError("SEC-aineistossa on ristiriitaisia lukuja samalla kaudella.")
    filed, _, accession, tag, fact = best[0]
    return {"value": fact["val"], "tag": tag, "taxonomy": "us-gaap", "unit": unit,
            "start": start, "end": end, "filed": filed, "accession": accession,
            "url": f"https://www.sec.gov/Archives/edgar/data/{COMPANY.cik}/{accession}-index.html"}


def add_derived(row):
    row["free_cash_flow"] = row["operating_cash_flow"] - row["cash_capex"]
    row["total_capex_including_finance_leases"] = row["cash_capex"] + row["finance_lease_additions"]
    row["depreciation_and_amortization"] = row["depreciation"] + row["amortization"]
    row["operating_margin_pct"] = ratio(row["operating_income"], row["revenue"], percentage=True)
    row["working_capital_investment"] = (
        row["receivables_increase"] + row["inventory_increase"] + row["other_current_assets_increase"]
        - row["payables_increase"] - row["other_current_liabilities_increase"] - row["customer_liabilities_increase"])
    return row


def extract_financials(payload, annual_report, *, cutoff):
    if payload.get("cik") != COMPANY.cik:
        raise SecDataError("Harjoitus hyväksyy vain Microsoftin aineiston.")
    sources = {}
    def period(start, end, label):
        row = {"period_start": start, "period_end": end}
        for key, tags in FLOW_TAGS.items():
            fact = select_fact(payload, tags, start=start, end=end, cutoff=cutoff)
            sources[f"{label}_{key}"] = fact
            row[key] = fact["value"]
        return add_derived(row)
    annuals = []
    for original in annual_report["rows"]:
        row = period(original["period_start"], original["period_end"], f"A{original['fiscal_year']}")
        annuals.append({"fiscal_year": original["fiscal_year"], **row})
    annual = annuals[-1]
    candidates = []
    for fact in payload["facts"]["us-gaap"][FLOW_TAGS["revenue"][0]]["units"]["USD"]:
        try:
            end = date.fromisoformat(fact["end"])
            start = date.fromisoformat(fact["start"])
            fiscal_start = date(end.year if end.month >= 7 else end.year-1, 7, 1)
            if (start == fiscal_start and end.month in {9, 12, 3} and end.day in {30, 31}
                    and end.isoformat() > annual["period_end"] and fact.get("form") in {"10-Q", "10-Q/A"}
                    and end.isoformat() <= fact["filed"] <= cutoff):
                candidates.append(end)
        except (KeyError, ValueError, TypeError):
            continue
    if not candidates:
        raise SecDataError("Uusinta tilikautta seuraavaa osavuosiraporttia ei löytynyt. Harjoitus tarvitsee päivitetyn osavuosiaineiston.")
    end = max(candidates)
    start = date(end.year if end.month >= 7 else end.year-1, 7, 1)
    prior_start, prior_end = start.replace(year=start.year-1), end.replace(year=end.year-1)
    ytd = period(start.isoformat(), end.isoformat(), "YTD")
    prior_ytd = period(prior_start.isoformat(), prior_end.isoformat(), "PYTD")
    ttm = {"period_start": (prior_end+timedelta(days=1)).isoformat(), "period_end": end.isoformat()}
    for key in FLOW_TAGS:
        ttm[key] = annual[key] - prior_ytd[key] + ytd[key]
    add_derived(ttm)
    if end.month == 9:
        quarter, prior_quarter = ytd.copy(), prior_ytd.copy()
    else:
        previous_end = (date(end.year, end.month-2, 1) - timedelta(days=1))
        before = period(start.isoformat(), previous_end.isoformat(), "YTD_PREV")
        prior_before = period(prior_start.isoformat(), previous_end.replace(year=previous_end.year-1).isoformat(), "PYTD_PREV")
        quarter, prior_quarter = {}, {}
        for target, current, old in ((quarter, ytd, before), (prior_quarter, prior_ytd, prior_before)):
            target.update({"period_start": (date.fromisoformat(old["period_end"])+timedelta(days=1)).isoformat(), "period_end": current["period_end"]})
            target.update({key: current[key]-old[key] for key in FLOW_TAGS})
            add_derived(target)
    balances = {}
    for key, tags in BALANCE_TAGS.items():
        fact = select_fact(payload, tags, end=end.isoformat(), cutoff=cutoff, unit="shares" if key == "shares" else "USD")
        balances[key] = fact["value"]
        sources[f"BS_{key}"] = fact
    balances["period_end"] = end.isoformat()
    return {"annuals": annuals, "ytd": ytd, "prior_ytd": prior_ytd, "quarter": quarter,
            "prior_quarter": prior_quarter, "ttm": ttm, "balances": balances, "sources": sources,
            "quarter_growth_pct": {key: percentage_change(quarter[key], prior_quarter[key]) for key in ("revenue", "operating_income", "cash_capex", "free_cash_flow")},
            "ttm_formula": "Viimeisin koko tilikausi − edellisvuoden vastaava kumulatiivinen osavuosijakso + uusin kumulatiivinen osavuosijakso."}


def _prices(history, cutoff):
    if history.empty:
        raise SecDataError("Kurssihistoriaa ei saatu.")
    rows = []
    for timestamp, row in history.iterrows():
        day = timestamp.date()
        close = float(row["Close"])
        if day <= cutoff and math.isfinite(close) and close > 0:
            rows.append({"date": day.isoformat(), "close": close})
    if len(rows) < 200 or len({r["date"] for r in rows}) != len(rows):
        raise SecDataError("Kurssihistoria on liian lyhyt tai sisältää päällekkäisiä päiviä.")
    return sorted(rows, key=lambda r: r["date"])


def price_summary(rows):
    from calendar import monthrange
    current = rows[-1]
    end = date.fromisoformat(current["date"])
    result = {"price": current["close"], "price_date": current["date"]}
    for months, key in ((1, "month"), (3, "three_months")):
        month_index = end.year*12+end.month-1-months
        year, month = divmod(month_index, 12)
        month += 1
        anchor = date(year, month, min(end.day, monthrange(year, month)[1]))
        earlier = [r for r in rows if r["date"] <= anchor.isoformat()]
        if not earlier or (anchor-date.fromisoformat(earlier[-1]["date"])).days > 7:
            raise SecDataError("Vertailupäivän kurssi puuttuu.")
        result[key] = {**earlier[-1], "change_pct": percentage_change(current["close"], earlier[-1]["close"])}
    peak = max((r for r in rows if r["date"] >= (end-timedelta(weeks=52)).isoformat()), key=lambda r: r["close"])
    result["peak_52w"] = {**peak, "change_pct": percentage_change(current["close"], peak["close"])}
    return result


def fetch_market(*, ticker_factory=None):
    if ticker_factory is None:
        import yfinance as yf
        yf.set_tz_cache_location(str(ROOT / "quote_cache"))
        ticker_factory = yf.Ticker
    now = datetime.now(ZoneInfo("America/New_York"))
    cutoff = now.date() if (now.hour, now.minute) >= (16, 15) else now.date()-timedelta(days=1)
    start = cutoff-timedelta(days=400)
    histories = {}
    try:
        for ticker in ("MSFT", "SPY"):
            history = ticker_factory(ticker).history(start=start.isoformat(), end=(cutoff+timedelta(days=1)).isoformat(), auto_adjust=False, actions=False, raise_errors=True)
            histories[ticker] = _prices(history, cutoff)
    except Exception as exc:
        raise SecDataError("Kurssihistorian haku epäonnistui. Aiempi harjoitus säilyy.") from exc
    end = min(histories[ticker][-1]["date"] for ticker in histories)
    for ticker in histories:
        histories[ticker] = [r for r in histories[ticker] if r["date"] <= end]
        if histories[ticker][-1]["date"] != end:
            raise SecDataError("Osakkeen ja vertailurahaston yhteistä kaupankäyntipäivää ei löytynyt.")
    if (today_rome()-date.fromisoformat(end)).days > 7:
        raise SecDataError("Saatu kurssihistoria on yli seitsemän päivää vanha.")
    return {"histories": histories, "summaries": {t: price_summary(r) for t, r in histories.items()},
            "source_urls": {t: f"https://finance.yahoo.com/quote/{t}/history/" for t in histories},
            "definition": "Päätöskurssin muutos, osakesplitit huomioitu; osinkoja ei sijoiteta uudelleen. SPY on S&P 500 -indeksiä seuraava ETF.",
            "retrieved_at": datetime.now(timezone.utc).isoformat()}


def make_source_text(raw, *, url, label, prefix, filed):
    excerpts = excerpts_from_html(raw, scope="company")
    return {f"{prefix}{i:02}": {"text": item["text"], "url": url, "label": label, "filed": filed}
            for i, item in enumerate(excerpts, 1)}


def refresh_exercise(user_agent, *, root=None):
    root = Path(root) if root else ROOT
    annual = build_report(company=COMPANY, data_dir=root, user_agent=user_agent, refresh=True, as_of=today_rome())
    raw = (root / annual["raw_file"]).read_bytes()
    financials = extract_financials(json.loads(raw), annual, cutoff=annual["as_of"])
    market = fetch_market()
    source = financials["sources"]["YTD_revenue"]
    quarterly = prepare_annual_report(COMPANY, {"sources": {"revenue": {"accession": source["accession"], "url": source["url"], "filed": source["filed"]}}}, user_agent, scope="company")
    quarter_raw = (COMPANY.data_dir / "annual_reports" / f"{source['accession']}.html").read_bytes()
    # Verify that Microsoft's LongTermDebt tag is the reported TOTAL debt row.
    # Its current portion must not be added twice, nor missing short debt set to zero.
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(quarter_raw, "html.parser")
    debt_rows = [tr for tr in soup.find_all("tr") if re.match(r"^Total debt\s+(?:\$\s*)?[\d,]+", tr.get_text(" ", strip=True), re.I)
                 and tr.find(None, attrs={"name": "us-gaap:LongTermDebt"})]
    if len(debt_rows) != 1:
        raise SecDataError("Raportin kokonaisvelkariviä ei voitu varmistaa. Velkoja ei oleteta nollaksi.")
    financials["balances"]["borrowings_are_reported_total"] = True
    financials["sources"]["BS_borrowings"]["text"] = debt_rows[0].get_text(" ", strip=True)
    fiscal_year = date.fromisoformat(source["end"]).year + (date.fromisoformat(source["end"]).month >= 7)
    quarter_number = ((date.fromisoformat(source["end"]).month-7) % 12)//3+1
    guidance_url = f"https://www.microsoft.com/en-us/investor/events/fy-{fiscal_year}/earnings-fy-{fiscal_year}-q{quarter_number}"
    guidance_raw = download(guidance_url, user_agent)
    guidance_text = BeautifulSoup(guidance_raw, "html.parser").get_text(" ", strip=True)
    if "Transcript" not in guidance_text or f"FY{str(fiscal_year)[-2:]}" not in guidance_text:
        raise SecDataError("Johdon tulospuhelun oikeaa raportointikautta ei voitu varmistaa.")
    (root / "guidance.html").write_bytes(guidance_raw)
    sources = make_source_text(quarter_raw, url=quarterly["url"], label="Uusin SEC-osavuosiraportti", prefix="Q", filed=source["filed"])
    sources.update(make_source_text(guidance_raw, url=guidance_url, label="Johdon tulospuhelu: tulevaisuus on johdon arvio", prefix="G", filed=source["filed"]))
    snapshot = {"version": VERSION, "ticker": "MSFT", "fiscal_year": fiscal_year, "as_of": annual["as_of"],
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "raw_sha256": annual["raw_sha256"],
                "financials": financials, "market": market, "texts": sources,
                "quarter_report": quarterly["url"], "guidance_url": guidance_url}
    snapshot["fingerprint"] = digest(snapshot)
    validate_snapshot(snapshot)
    atomic_json(root / "snapshots" / f"{snapshot['as_of']}_{snapshot['fingerprint'][:16]}.json", snapshot)
    atomic_json(root / "latest.json", snapshot)
    return snapshot


def validate_snapshot(snapshot, *, require_current=False):
    try:
        if snapshot["version"] != VERSION or snapshot["ticker"] != "MSFT":
            raise ValueError("identity")
        if snapshot["fingerprint"] != digest({k: v for k, v in snapshot.items() if k != "fingerprint"}):
            raise ValueError("checksum")
        end = snapshot["market"]["summaries"]["MSFT"]["price_date"]
        if snapshot["financials"]["ttm"]["period_end"] > end or any(s["filed"] > end for s in snapshot["financials"]["sources"].values()):
            raise ValueError("lookahead")
        if date.fromisoformat(end) > today_rome() or require_current and (today_rome()-date.fromisoformat(end)).days > 7:
            raise ValueError("stale price")
        if date.fromisoformat(snapshot["as_of"]) > today_rome():
            raise ValueError("future snapshot")
        if require_current and (today_rome()-date.fromisoformat(snapshot["financials"]["ttm"]["period_end"])).days > 160:
            raise ValueError("stale report")
    except (KeyError, TypeError, ValueError) as exc:
        raise SecDataError("Harjoituksen aineisto ei läpäise tunniste-, päiväys- tai eheystarkistusta. Päivitä aineisto; aiempi tallennettu harjoitus säilyy luettavana.") from exc
    return snapshot


def load_exercise(*, root=None):
    path = (Path(root) if root else ROOT) / "latest.json"
    if not path.exists():
        return None
    try:
        return validate_snapshot(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise SecDataError("Tallennettu Microsoft-harjoituksen aineisto ei avaudu.") from exc
