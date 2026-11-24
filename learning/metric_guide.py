"""Definitions and interpretation guidance shown next to the calculations."""

from typing import TypedDict


class MetricGuide(TypedDict):
    label: str
    unit: str
    inputs: tuple[str, ...]
    meaning: str
    formula: str
    assessment: str
    limit: str

GUIDE: dict[str, MetricGuide] = {
    "revenue_growth_pct": {
        "label": "Liikevaihdon vuosikasvu", "unit": "pct", "inputs": ("revenue",),
        "meaning": "Liikevaihto tarkoittaa myyntituottoja. Vuosikasvu kertoo myynnin muutoksen edellisestä tilikaudesta.",
        "formula": "(uusi liikevaihto / edellisen vuoden liikevaihto − 1) × 100",
        "assessment": "Katso kasvun jatkuvuutta ja sen suhdetta kannattavuuteen. Selvitä, tuliko kasvu myyntimääristä, hinnoista, yritysostoista vai valuuttakursseista.",
        "limit": "Kasvu ei yksin osoita kannattavuuden paranemista tai osakkeen edullisuutta. Nolla tai negatiivinen vertailuvuoden liikevaihto estää tämän tavallisen prosenttilaskennan.",
    },
    "operating_margin_pct": {
        "label": "Liikevoittomarginaali", "unit": "pct", "inputs": ("operating_income", "revenue"),
        "meaning": "Kuinka suuri osa myynnistä jää liiketoiminnan tulokseksi ennen rahoituseriä ja veroja. Esimerkiksi 20 % tarkoittaa 20 dollarin liikevoittoa sadan dollarin myynnistä.",
        "formula": "liikevoitto / liikevaihto × 100",
        "assessment": "Seuraa kehitystä saman yhtiön historiassa. Paraneva marginaali voi liittyä hinnoitteluun, kustannuksiin tai liiketoimintajakaumaan; syy tarkistetaan raportista.",
        "limit": "Eri toimialojen marginaalit eivät ole suoraan paremmuusjärjestys. Kertaluonteinen erä voi vaikuttaa raportoituihin lukuihin. Muutos 20 prosentista 22 prosenttiin on kaksi prosenttiyksikköä.",
    },
    "free_cash_flow": {
        "label": "Vapaa kassavirta", "unit": "usd_m", "inputs": ("operating_cash_flow", "capex"),
        "meaning": "Liiketoiminnan rahavirrasta jäävä määrä käyttöomaisuusinvestointien jälkeen. Laskemme tämän itse raportoiduista luvuista. Joidenkin yhtiöiden investointirivi sisältää myös teknologiaa tai aineettomia hyödykkeitä; rajaus kerrotaan erikseen.",
        "formula": "liiketoiminnan rahavirta − käyttöomaisuusinvestointien rahamenot (yhtiökohtainen rajaus näkyy näkymässä)",
        "assessment": "Tutki useita vuosia ja muutoksen syitä. Suuri kasvua rakentava investointi voi pienentää tämän vuoden vapaata kassavirtaa.",
        "limit": "Tämä määritelmä ei automaattisesti vähennä yritysostoja tai rahoitusleasingin lyhennyksiä. Yritysostoon liittyvä maksu voi myös sisältyä jo liiketoiminnan rahavirtaan. Luku ei ole sama kuin voitto tai jaettavissa oleva osinko.",
    },
    "gross_margin_pct": {
        "label": "Bruttomarginaali", "unit": "pct", "inputs": ("revenue", "cost_of_revenue"),
        "meaning": "Kuinka suuri osa myynnistä jää jäljelle myytyjen tuotteiden ja palvelujen kustannusten jälkeen, ennen muita liiketoiminnan kuluja.",
        "formula": "(liikevaihto − myytyjen tuotteiden ja palvelujen kustannukset) / liikevaihto × 100",
        "assessment": "Muutos voi liittyä hinnoitteluun, tuotannon kustannuksiin tai siihen, mitä tuotteita ja palveluja myydään. Vertaa sitä liikevoittomarginaalin kehitykseen.",
        "limit": "Kustannusten luokittelu ja liiketoimintamalli vaikuttavat vertailuun. Bruttomarginaali voi laskea samalla, kun liikevoittomarginaali paranee muiden kulujen kasvaessa hitaammin.",
    },
    "net_margin_pct": {
        "label": "Nettomarginaali", "unit": "pct", "inputs": ("net_income", "revenue"),
        "meaning": "Kuinka suuri osa liikevaihdosta jää emoyhtiön omistajille kuuluvaksi tilikauden voitoksi myös rahoituserien ja verojen jälkeen.",
        "formula": "emoyhtiön omistajille kuuluva nettotulos / liikevaihto × 100",
        "assessment": "Vertaa nettomarginaalia liikevoittomarginaaliin. Ero auttaa huomaamaan rahoituserien, verojen ja muiden tuloserien vaikutuksen.",
        "limit": "Sijoitusten myyntivoitot, verot ja muut kertaluonteiset erät voivat nostaa lukua ilman varsinaisen liiketoiminnan paranemista. Nettomarginaali voi olla liikevoittomarginaalia suurempi.",
    },
    "fcf_margin_pct": {
        "label": "Vapaan kassavirran marginaali", "unit": "pct", "inputs": ("operating_cash_flow", "capex", "revenue"),
        "meaning": "Paljonko sadasta myyntidollarista jää vapaata kassavirtaa. Esimerkiksi 15 % tarkoittaa 15 dollaria sadan dollarin myynnistä.",
        "formula": "(liiketoiminnan rahavirta − investoinnit) / liikevaihto × 100",
        "assessment": "Seuraa, muuttuuko myynnin kasvu myös investointien jälkeiseksi rahavirraksi. Suhdeluku helpottaa erikokoisten yhtiöiden tarkastelua.",
        "limit": "Investointien ajoitus, veromaksut ja käyttöpääoma voivat heiluttaa yhtä vuotta. Luku ei ole FCF-tuotto: FCF-tuoton nimittäjä olisi osakemarkkina-arvo, joka ei ole mukana tässä harjoituksessa.",
    },
    "capex_to_revenue_pct": {
        "label": "Investointiaste", "unit": "pct", "inputs": ("capex", "revenue"),
        "meaning": "Käyttöomaisuusinvestointeihin käytetty rahamäärä suhteessa myyntiin. Nämä voivat olla esimerkiksi datakeskuksia, koneita tai rakennuksia. Raportin rivi voi sisältää myös teknologiaa tai aineettomia hyödykkeitä; yhtiökohtainen rajaus kerrotaan erikseen.",
        "formula": "käyttöomaisuusinvestointien rahamenot / liikevaihto × 100",
        "assessment": "Nouseva osuus kertoo kasvavasta investointipanoksesta suhteessa myyntiin. Selvitä, rakennetaanko kasvua vai ylläpidetäänkö nykyistä toimintaa.",
        "limit": "Korkea arvo ei yksin ole huono eikä matala arvo hyvä. Liiketoimintamallit eroavat. Tämä ei kata kaikkia investointeja, kuten tutkimuskuluja, yritysostoja tai leasingillä hankittua kapasiteettia.",
    },
    "current_ratio": {
        "label": "Lyhyen aikavälin maksuvalmiussuhde", "unit": "ratio", "inputs": ("current_assets", "current_liabilities"),
        "meaning": "Current ratio vertaa tilinpäätöspäivän lyhytaikaisia varoja lyhytaikaisiin velkoihin. Arvo 1,2 tarkoittaa 1,20 dollarin varoja yhtä velkadollaria kohti.",
        "formula": "lyhytaikaiset varat / lyhytaikaiset velat",
        "assessment": "Alle yksi tarkoittaa, että taseen lyhytaikaiset velat ylittävät lyhytaikaiset varat. Tarkista varojen laatu, erääntymiset ja liiketoiminnan rahavirta.",
        "limit": "Arvo ei yksin ratkaise maksukykyä. Kaikki varat eivät ole käteistä, ja veloissa voi olla esimerkiksi asiakkailta saatuja ennakoita. Korkea luku ei takaa tehokasta pääoman käyttöä eikä yksi ole yleinen osto- tai hylkäysraja.",
    },
    "operating_cash_flow_margin_pct": {
        "label": "Liiketoiminnan rahavirran marginaali", "unit": "pct", "inputs": ("operating_cash_flow", "revenue"),
        "meaning": "Kuinka paljon liiketoiminta tuottaa rahavirtaa sataa myyntidollaria kohti ennen käyttöomaisuusinvestointeja.",
        "formula": "liiketoiminnan rahavirta / liikevaihto × 100",
        "assessment": "Vertaa kehitystä FCF-marginaaliin. Niiden ero on tässä laskennassa investointiaste.",
        "limit": "Maksujen ajoitus, käyttöpääoma ja kertaluonteiset maksut vaikuttavat. Tämä ei ole kassatilin saldo eikä osakkeen tuotto.",
    },
    "cash_conversion_ratio": {
        "label": "Liiketoiminnan rahavirta / nettotulos", "unit": "ratio", "inputs": ("operating_cash_flow", "net_income"),
        "meaning": "Vertaa rahavirtaa emoyhtiön omistajille kuuluvaan kirjanpidolliseen voittoon. Kerroin 1,2 tarkoittaa 1,20 dollaria rahavirtaa yhtä tulosdollaria kohti.",
        "formula": "liiketoiminnan rahavirta / emoyhtiön omistajille kuuluva nettotulos",
        "assessment": "Tutki useita vuosia. Poistot, osakeperusteiset palkkiot ja käyttöpääoman muutokset voivat selittää eroa.",
        "limit": "Korkea kerroin ei takaa hyvää tuloksen laatua. Nettotuloksen ollessa nolla tai negatiivinen kerrointa ei esitetä. Rahavirta voi sisältää myös tytäryhtiöiden vähemmistöosuuksia.",
    },
}


COMPANY_CASH_NOTES = {
    "AMZN": "Yhteinen laskelmamme vähentää käyttöomaisuusostot bruttomääräisinä. Amazonin itse raportoima vapaa kassavirta vähentää investoinneista myös omaisuuden myyntituloja ja kannustimia, joten yhtiön FCF voi poiketa tästä. Rahoitusleasingin lyhennyksiä ei vähennetä tässä erikseen.",
    "NVDA": "NVIDIAn raportoima investointirivi sisältää aineellista käyttöomaisuutta ja aineettomia hyödykkeitä. Sitä käytetään sellaisenaan ja rajaus eroaa muiden harjoitusyhtiöiden aineellisista investoinneista. Tilikausi päättyy tammikuun viimeisenä sunnuntaina ja voi kestää 52 tai 53 viikkoa.",
    "PG": "Näytämme raportoidun rahavirran vähennettynä käyttöomaisuusinvestoinneilla. P&G:n oma adjusted free cash flow ja sen productivity-tunnusluku voivat sisältää oikaisuja, joita tähän laskelmaan ei tehdä.",
}

KO_REPORT = "https://investors.coca-colacompany.com/filings-reports/all-sec-filings/content/0001628280-26-010047/ko-20251231.htm"
CONTEXT = {
    ("KO", 2024): "Vuoden 2024 liiketoiminnan rahavirtaan sisältyi 6,0 miljardin dollarin IRS-veroriitaan liittyvä talletus. Se vaikuttaa myös tässä laskettuun vapaaseen kassavirtaan. Raportoitu luku kannattaa erottaa arviosta jatkuvasta rahavirrasta.",
    ("KO", 2025): "Vuoden 2025 liiketoiminnan rahavirtaan sisältyi 6,1 miljardia dollaria fairlife-yrityskaupan lisäkauppahinnan loppumaksusta. Maksu pienentää myös tässä laskettua vapaata kassavirtaa, vaikka se ei ole normaali vuotuinen käyttöomaisuusinvestointi.",
}
