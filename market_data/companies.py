"""The explicit, dated 20-company pilot for the C learning exercises."""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "research"
UNIVERSE_DATE = "2026-11-20"


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
    sector: str = ""
    fiscal_weekday: int | None = None
    nearest_weekday: bool = False
    revenue_tags: tuple[str, ...] | None = None
    net_income_tags: tuple[str, ...] | None = None
    cost_of_revenue_tags: tuple[str, ...] | None = None
    capex_definition: str | None = None
    notes: tuple[str, ...] = ()

    def year_end(self, year: int) -> date:
        end = date(year, self.fiscal_end_month, self.fiscal_end_day)
        weekday = 6 if self.last_sunday else self.fiscal_weekday
        if weekday is not None:
            distance = (weekday - end.weekday()) % 7
            if self.nearest_weekday:
                end += timedelta(days=distance if distance <= 3 else distance - 7)
            else:
                end -= timedelta(days=(-distance) % 7)
        return end

    @property
    def source_url(self) -> str:
        return f"https://data.sec.gov/api/xbrl/companyfacts/CIK{self.cik:010d}.json"

    @property
    def data_dir(self) -> Path:
        return DATA_ROOT / self.ticker.lower()

    def period(self, year: int) -> tuple[date, date]:
        return self.year_end(year - 1) + timedelta(days=1), self.year_end(year)


COMPANIES = {
    "MSFT": Company("MSFT", "Microsoft", 789019, 6, 30, sector="Teknologia"),
    "KO": Company("KO", "Coca-Cola", 21344, 12, 31, sector="Päivittäistavarat"),
    "AMZN": Company("AMZN", "Amazon", 1018724, 12, 31,
                    capex_tags=("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"),
                    sector="Kauppa"),
    "NVDA": Company("NVDA", "NVIDIA", 1045810, 1, 31, last_sunday=True,
                    capex_tags=("PaymentsToAcquireProductiveAssets",),
                    capex_includes_intangibles=True, sector="Teknologia"),
    "PG": Company("PG", "Procter & Gamble", 80424, 6, 30, sector="Päivittäistavarat"),
    "AAPL": Company("AAPL", "Apple", 320193, 9, 30, fiscal_weekday=5, sector="Teknologia"),
    "GOOGL": Company("GOOGL", "Alphabet", 1652044, 12, 31, sector="Viestintäpalvelut"),
    "META": Company("META", "Meta Platforms", 1326801, 12, 31, sector="Viestintäpalvelut"),
    "TSLA": Company("TSLA", "Tesla", 1318605, 12, 31, sector="Autot"),
    "V": Company("V", "Visa", 1403161, 9, 30, sector="Maksupalvelut",
                 capex_tags=("PaymentsToAcquireProductiveAssets",), capex_includes_intangibles=True,
                 notes=("Visan investointirivi sisältää käyttöomaisuuden lisäksi teknologiaa. Vapaan kassavirran rajaus on siksi laajempi kuin pelkät aineelliset investoinnit.",)),
    "MA": Company("MA", "Mastercard", 1141391, 12, 31, sector="Maksupalvelut",
                  notes=("Mastercardin konsernin nettotulosta ei käytetä emoyhtiön nettotuloksen korvikkeena. Nettomarginaali ja tuloskerroin jäävät puuttuviksi, jos emoyhtiön lukua ei saada. Erillisiä ohjelmistoinvestointeja ei vähennetä tämän aineellisiin investointeihin perustuvan FCF:n laskennassa.",)),
    "WMT": Company("WMT", "Walmart", 104169, 1, 31, sector="Kauppa", revenue_tags=("Revenues",),
                   notes=("Walmartin liikevaihtona käytetään kokonaisliikevaihtoa, joka sisältää myös jäsenmaksut ja muut tuotot. Tilikauden vuosiluku on tammikuun päättymisvuosi.",)),
    "COST": Company("COST", "Costco", 909832, 8, 31, fiscal_weekday=6,
                    nearest_weekday=True, sector="Kauppa"),
    "LLY": Company("LLY", "Eli Lilly", 59478, 12, 31, sector="Lääkkeet",
                   capex_tags=("PaymentsToAcquireOtherPropertyPlantAndEquipment",),
                   notes=("Liikevoitto jää puuttuvaksi, jos sitä ei ole tuetussa SEC-tietokentässä. Sitä ei korvata tuloksella ennen veroja. Tutkimusmenot, patentit ja lääkeaihioiden riskit eivät sisälly pelkkään investointiasteeseen.",)),
    "ABBV": Company("ABBV", "AbbVie", 1551152, 12, 31, sector="Lääkkeet"),
    "COP": Company("COP", "ConocoPhillips", 1163165, 12, 31, sector="Energia",
                   revenue_tags=("Revenues",), cost_of_revenue_tags=(),
                   notes=("Liikevaihto sisältää myynnin ja muut liiketoiminnan tuotot. Liikevoitto ja investoinnit voivat puuttua tuetuista tietokentistä. Pelkkiä ostettujen hyödykkeiden kustannuksia ei tulkita kaikiksi myynnin kustannuksiksi, joten bruttomarginaalia ei lasketa.",)),
    "CVX": Company("CVX", "Chevron", 93410, 12, 31, sector="Energia",
                   capex_tags=("PaymentsToAcquireProductiveAssets",), cost_of_revenue_tags=(),
                   capex_definition="Operating cash flow minus reported cash capital expenditures (PaymentsToAcquireProductiveAssets); not total investing cash flow.",
                   notes=("Investointeina käytetään rahavirtalaskelman Capital expenditures -riviä. Liikevoittoa ei korvata tuloksella ennen veroja. Ostetun öljyn ja tuotteiden kulut eivät kata kaikkia myynnin kustannuksia, joten bruttomarginaalia ei lasketa.",)),
    "CAT": Company("CAT", "Caterpillar", 18230, 12, 31, sector="Teollisuus",
                   net_income_tags=("NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"), cost_of_revenue_tags=(),
                   notes=("Luvut sisältävät myös rahoitusliiketoiminnan. Pelkkiä tavaramyynnin kuluja ei suhteuteta koko konsernin liikevaihtoon bruttomarginaaliksi. Nettotuloksena voidaan käyttää raportin emoyhtiön kantaosakkeille kuuluvaa Profit-lukua.",)),
    "MCD": Company("MCD", "McDonald's", 63908, 12, 31, sector="Ravintolat",
                   notes=("Franchising-ravintoloista kirjataan pääosin maksuja ja vuokria, omista ravintoloista myyntiä. Marginaaleja ei siksi voi verrata suoraan pelkästään omia ravintoloita pyörittävään yhtiöön.",)),
    "NEE": Company("NEE", "NextEra Energy", 753308, 12, 31, sector="Sähkö ja energia",
                   revenue_tags=("RegulatedAndUnregulatedOperatingRevenue",),
                   notes=("Liikevaihtona käytetään säännellyn ja sääntelemättömän toiminnan yhteenlaskettuja tuottoja. Investoinnit raportoidaan useina yhtiökohtaisina riveinä, joita ei vielä yhdistetä. Vapaa kassavirta ja investointiaste jäävät tällöin puuttuviksi.",)),
}


def get_company(ticker: str) -> Company:
    try:
        return COMPANIES[ticker.upper()]
    except KeyError as exc:
        raise ValueError(f"Tuntematon harjoitusyhtiö: {ticker}") from exc
