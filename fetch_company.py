"""Fetch one learning company's annual data, or rebuild a local snapshot."""

import argparse
from datetime import date
import os
from pathlib import Path
import sys

from market_data.companies import COMPANIES, get_company
from market_data.sec import SecDataError, build_report


def main():
    parser = argparse.ArgumentParser(description="Hae harjoitusyhtiön vuosiluvut SEC:stä tai paikallisesta aineistosta.")
    parser.add_argument("--company", choices=COMPANIES, required=True)
    parser.add_argument("--refresh", action="store_true", help="Hae uusi alkuperäisaineisto SEC:stä.")
    parser.add_argument("--offline", action="store_true", help="Käytä vain aiemmin tallennettua aineistoa.")
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    company = get_company(args.company)
    try:
        report = build_report(company=company, data_dir=args.data_dir, user_agent=os.getenv("SEC_USER_AGENT", ""), refresh=args.refresh, offline=args.offline, as_of=args.as_of)
    except (SecDataError, OSError) as exc:
        print(f"Haku ei onnistunut: {exc}", file=sys.stderr)
        return 1
    print(f"{company.name} / {company.ticker} | miljoonaa USD")
    print("Vuosi  Liikevaihto  Liikevoitto  Vapaa kassavirta  Tilikausi päättyi")
    for row in report["rows"]:
        values = ["puuttuu" if row[key] is None else f"{row[key]/1e6:.0f}" for key in ("revenue", "operating_income", "free_cash_flow")]
        print(row["fiscal_year"], *values, row["period_end"])
    for warning in report["warnings"] + report["data_notes"]:
        print("Huomio:", warning)
    print("Tallennettu:", (args.data_dir or company.data_dir) / "annual_metrics.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
