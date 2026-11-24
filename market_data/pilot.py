"""Local coverage of the pilot and an explicit, sequential first download."""

from market_data.companies import COMPANIES
from market_data.sec import METRICS, SecDataError, build_report, load_report


def load_pilot():
    reports, issues = {}, {}
    for ticker, company in COMPANIES.items():
        try:
            report = load_report(company=company)
            if report is not None:
                reports[ticker] = report
            else:
                issues[ticker] = "Aineistoa ei ole vielä haettu."
        except (SecDataError, OSError) as exc:
            issues[ticker] = str(exc)
    return reports, issues


def coverage_rows(reports, issues):
    rows = []
    for ticker, company in COMPANIES.items():
        report = reports.get(ticker)
        missing = []
        if report:
            for key, definition in METRICS.items():
                years = [str(row["fiscal_year"]) for row in report["rows"] if row.get(key) is None]
                if years:
                    missing.append(f"{definition['label']} ({', '.join(years)})")
        rows.append({
            "Yhtiö": company.name, "Tunnus": ticker, "Liiketoiminta-alue": company.sector,
            "Tilikaudet": ", ".join(str(row["fiscal_year"]) for row in report["rows"]) if report else "—",
            "Aineisto": ("Lähtölukuja puuttuu" if missing else "Lähtöluvut saatavilla") if report else issues.get(ticker, "Ei aineistoa"),
            "Puuttuvat lähtöluvut": "; ".join(missing) or "—",
        })
    return rows


def fetch_missing_reports(user_agent, *, on_progress=None):
    """Reuse valid reports; rebuild from raw data before making new requests.

    One company's failure never prevents the remaining companies from loading.
    No annual-report downloads, quotes or paid analysis calls happen here.
    """
    reports, _ = load_pilot()
    pending = [company for ticker, company in COMPANIES.items() if ticker not in reports]
    failures = {}
    for index, company in enumerate(pending):
        try:
            reports[company.ticker] = build_report(company=company, user_agent=user_agent)
        except (SecDataError, OSError) as exc:
            failures[company.ticker] = str(exc)
        if on_progress:
            on_progress(index + 1, len(pending), company.name)
    return reports, failures
