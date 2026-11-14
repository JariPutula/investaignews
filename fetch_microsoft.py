"""Fetch the four Microsoft fiscal years used by the learning exercise."""

import argparse
from datetime import date
import os
from pathlib import Path
import sys

from market_data.sec_microsoft import DEFAULT_DATA_DIR, SecDataError, build_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Hae Microsoftin neljä tilikautta SEC:stä tai paikallisesta aineistosta.")
    parser.add_argument("--refresh", action="store_true", help="Hae uusi alkuperäisaineisto SEC:stä.")
    parser.add_argument("--offline", action="store_true", help="Käytä vain aiemmin tallennettua aineistoa.")
    parser.add_argument("--as-of", type=date.fromisoformat, help="Huomioi vain tähän päivään mennessä julkaistut luvut (YYYY-MM-DD).")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    try:
        report = build_report(
            data_dir=args.data_dir, user_agent=os.environ.get("SEC_USER_AGENT", ""),
            refresh=args.refresh, offline=args.offline, as_of=args.as_of,
        )
    except (SecDataError, OSError) as exc:
        print(f"Haku ei onnistunut: {exc}", file=sys.stderr)
        return 1

    def number(value):
        return f"{value / 1_000_000:,.0f}".replace(",", " ") if value is not None else "puuttuu"

    print(f"Microsoft / MSFT | luvut miljoonaa USD | aineisto haettu {report['retrieved_at']}")
    print(f"{'Tilikausi':<11}{'Liikevaihto':>15}{'Liikevoitto':>15}{'Rahavirta':>15}{'Investoinnit':>15}{'Vapaa rahavirta':>18}")
    for row in report["rows"]:
        print(f"{row['fiscal_year']:<11}" + "".join(
            f"{number(row[key]):>{width}}" for key, width in [
                ("revenue", 15), ("operating_income", 15), ("operating_cash_flow", 15),
                ("capex", 15), ("free_cash_flow", 18),
            ]
        ))
    for warning in report["warnings"]:
        print(f"Huomio: {warning}")
    print(f"Yhteenveto ja lähdeviitteet: {args.data_dir / 'annual_metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
