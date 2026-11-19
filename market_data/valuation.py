"""Explicitly fetched, dated equity-price context; never fetch during a rerun."""

from datetime import datetime, timezone
import json
import math
from zoneinfo import ZoneInfo
from typing import TypeGuard

from market_data.companies import DATA_ROOT
from market_data.financial_metrics import ratio
from market_data.sec import SecDataError


def _positive(value) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def validate_quote(quote, company):
    if quote.get("ticker") != company.ticker or quote.get("currency") != "USD":
        raise SecDataError("Kurssiaineiston yhtiö tai valuutta ei vastaa harjoituksen USD-lukuja.")
    if not all(_positive(quote.get(key)) for key in ("price", "shares", "market_cap")):
        raise SecDataError("Kurssi, osakemäärä tai markkina-arvo puuttuu tai ei ole kelvollinen.")
    if not math.isclose(quote["price"] * quote["shares"], quote["market_cap"], rel_tol=1e-8):
        raise SecDataError("Kurssin, osakemäärän ja lasketun markkina-arvon täsmäytys epäonnistui.")
    try:
        for key in ("quoted_at", "retrieved_at"):
            parsed = datetime.fromisoformat(quote[key])
            if parsed.tzinfo is None:
                raise ValueError("timezone")
        if not quote["source_url"].startswith("https://finance.yahoo.com/quote/" + company.ticker + "/"):
            raise ValueError("source")
    except (KeyError, TypeError, ValueError) as exc:
        raise SecDataError("Kurssiaineiston päiväys tai lähde puuttuu.") from exc
    return quote


def fetch_valuation(company, *, ticker_factory=None):
    if ticker_factory is None:
        import yfinance as yf
        yf.set_tz_cache_location(str(DATA_ROOT / "_quote_cache"))
        ticker_factory = yf.Ticker
    try:
        info = ticker_factory(company.ticker).get_info()
    except Exception as exc:
        raise SecDataError("Kurssihaku ei onnistunut. Aiempi kurssiaineisto säilyy; voit yrittää myöhemmin uudelleen.") from exc
    if info.get("symbol") != company.ticker or info.get("currency") != "USD":
        raise SecDataError("Kurssilähde palautti väärän yhtiön tai valuutan.")
    price, shares, timestamp = info.get("regularMarketPrice"), info.get("sharesOutstanding"), info.get("regularMarketTime")
    if not _positive(price) or not _positive(shares) or not _positive(timestamp):
        raise SecDataError("Kurssilähteestä puuttuu kurssi, osakemäärä tai kaupankäynnin aikaleima.")
    quote = validate_quote({"ticker": company.ticker, "currency": "USD", "price": price, "shares": shares,
                            "market_cap": price * shares,
                            "quoted_at": datetime.fromtimestamp(timestamp, timezone.utc).isoformat(),
                            "retrieved_at": datetime.now(timezone.utc).isoformat(),
                            "source_url": f"https://finance.yahoo.com/quote/{company.ticker}/",
                            "shares_date": None, "provider": "Yahoo Finance / yfinance"}, company)
    path = company.data_dir / "valuation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(quote, indent=2), encoding="utf-8")
    temporary.replace(path)
    return quote


def load_valuation(company):
    path = company.data_dir / "valuation.json"
    if not path.exists():
        return None
    try:
        return validate_quote(json.loads(path.read_text(encoding="utf-8")), company)
    except (ValueError, OSError) as exc:
        raise SecDataError("Tallennettu kurssiaineisto ei avaudu. Hae kurssitiedot uudelleen.") from exc


def valuation_context(report, row, quote, *, today=None):
    today = today or datetime.now(ZoneInfo("Europe/Rome")).date()
    result = {"available": False, "reason": "Kurssiaineisto puuttuu. Osakkeen hinnan houkuttelevuutta ei voida arvioida.", "quote": quote}
    if row["fiscal_year"] != report["rows"][-1]["fiscal_year"]:
        result["reason"] = "Valittuna on vanha tilikausi. Nykyhintaa ei yhdistetä tämän historiallisen harjoituksen arvostusjohtopäätökseen."
        return result
    if quote is None:
        return result
    quoted_date = datetime.fromisoformat(quote["quoted_at"]).date()
    if not 0 <= (today - quoted_date).days <= 7:
        result["reason"] = "Kurssin päiväys on yli seitsemän päivää vanha tai tulevaisuudessa. Päivitä aineisto ennen nykyhinnan arviointia."
        return result
    period_end = datetime.fromisoformat(row["period_end"]).date()
    if quoted_date < period_end:
        result["reason"] = "Kurssi on tilikauden päättymistä vanhempi; sitä ei yhdistetä myöhemmin valmistuneisiin tilikausilukuihin."
        return result
    return {"available": True, "reason": "Päivätty hinta suhteessa viimeisimpään tallennettuun kokonaiseen tilikauteen; ei rullaava 12 kk eikä analyytikkoennuste.",
            "quote": quote, "market_cap_to_annual_net_income": ratio(quote["market_cap"], row.get("net_income")),
            "annual_fcf_yield_pct": ratio(row.get("free_cash_flow"), quote["market_cap"], percentage=True),
            "financial_period_end": row["period_end"], "financial_data_age_days_at_quote": (quoted_date-period_end).days,
            "limits": "Osakemäärän tarkkaa päiväystä lähde ei toimita. Markkina-arvo lasketaan kurssi × lähteen osakemäärä. Kokonaisvelat, osavuosiluvut ja kaikki viimeaikaiset tapahtumat eivät sisälly arvioon."}
