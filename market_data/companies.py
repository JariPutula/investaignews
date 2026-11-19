"""The deliberately small company registry for the C learning exercises."""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "research"


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    cik: int
    fiscal_end_month: int
    fiscal_end_day: int
    last_sunday: bool = False
    capex_tags: tuple[str, ...] = ("PaymentsToAcquirePropertyPlantAndEquipment",)
    capex_includes_intangibles: bool = False

    @property
    def source_url(self) -> str:
        return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{self.cik:010d}.json"

    @property
    def data_dir(self) -> Path:
        return DATA_ROOT / self.ticker.lower()

    def period(self, year: int) -> tuple[date, date]:
        end = date(year, self.fiscal_end_month, self.fiscal_end_day)
        previous_end = date(year - 1, self.fiscal_end_month, self.fiscal_end_day)
        if self.last_sunday:
            end -= timedelta(days=(end.weekday() + 1) % 7)
            previous_end -= timedelta(days=(previous_end.weekday() + 1) % 7)
        return previous_end + timedelta(days=1), end


COMPANIES = {
    "MSFT": Company("MSFT", "Microsoft", 789019, 6, 30),
    "KO": Company("KO", "Coca-Cola", 21344, 12, 31),
    "AMZN": Company("AMZN", "Amazon", 1018724, 12, 31,
                    capex_tags=("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")),
    "NVDA": Company("NVDA", "NVIDIA", 1045810, 1, 31, last_sunday=True,
                    capex_tags=("PaymentsToAcquireProductiveAssets",),
                    capex_includes_intangibles=True),
    "PG": Company("PG", "Procter & Gamble", 80424, 6, 30),
}


def get_company(ticker: str) -> Company:
    try:
        return COMPANIES[ticker.upper()]
    except KeyError as exc:
        raise ValueError(f"Tuntematon harjoitusyhtiö: {ticker}") from exc
