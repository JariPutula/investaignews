"""The first C learning exercise; also runs as a standalone Streamlit page."""

import os
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from market_data.sec_microsoft import METRICS, SecDataError, build_report, load_report


st.set_page_config(page_title="Microsoft – opi talousluvut", page_icon="📖", layout="wide")
st.title("Microsoft – opi talousluvut")
st.write("Tutkitaan kasvua, kannattavuutta ja rahavirtaa yksi laskelma kerrallaan. Luvut haetaan Microsoftin SEC-raporteista.")

with st.expander("Hae tai päivitä aineisto", expanded=False):
    st.write("SEC-haku on maksuton eikä tarvitse API-avainta. Yhteyssähköposti lähetetään SEC:lle sovelluksen tunnistetietona. Sitä ei tallenneta aineistoon.")
    contact = st.text_input("SEC-haun yhteyssähköposti", value="", placeholder="oma.osoite@example.com")
    if st.button("Hae uusimmat luvut SEC:stä", type="primary"):
        agent = f"InvestaiLearning {contact.strip()}" if contact.strip() else os.environ.get("SEC_USER_AGENT", "")
        try:
            with st.spinner("Haetaan Microsoftin vuosiluvut…"):
                build_report(user_agent=agent, refresh=True)
            st.success("Vuosiluvut ja lähteet tallennettu.")
        except (SecDataError, OSError) as exc:
            st.error(str(exc))

try:
    report = load_report()
except SecDataError as exc:
    st.error(str(exc))
    st.stop()
if report is None:
    st.info("Aineistoa ei vielä ole. Avaa yllä oleva haku tai suorita projektikansiossa fetch_microsoft.py.")
    st.stop()

rows = report["rows"]
st.caption(f"Tilikaudet {rows[0]['fiscal_year']}–{rows[-1]['fiscal_year']}. Microsoftin tilikausi päättyy 30.6. Julkaistut tiedot huomioitu päivään {report['as_of']} asti. Lähdeaineisto haettu {report['retrieved_at']}.")
for warning in report["warnings"]:
    st.warning(warning)

frame = pd.DataFrame(rows).set_index("fiscal_year")
table = frame[["revenue", "operating_income", "operating_cash_flow", "capex", "free_cash_flow"]].copy() / 1_000_000
table.columns = ["Liikevaihto", "Liikevoitto", "Liiketoiminnan rahavirta", "Investoinnit", "Vapaa kassavirta"]
table.index.name = "Tilikausi"
st.subheader("Lähtöluvut ja vapaa kassavirta")
st.caption("Miljoonaa Yhdysvaltain dollaria. Investoinnit esitetään positiivisena vähennettävänä menona. Puuttuva luku ei tarkoita nollaa.")
st.dataframe(table, width="stretch")

year = st.selectbox("Minkä tilikauden laskelmaa katsotaan?", [row["fiscal_year"] for row in rows], index=len(rows) - 1)
row = next(item for item in rows if item["fiscal_year"] == year)
position = rows.index(row)


def amount(value):
    return f"{value / 1_000_000:,.0f}".replace(",", " ") if value is not None else "puuttuu"


def percent(value):
    return f"{value:.2f} %".replace(".", ",") if value is not None else "ei laskettavissa"


def show_sources(keys):
    for key in keys:
        source = row["sources"].get(key)
        if source:
            st.markdown(f"[{METRICS[key]['label']}: alkuperäinen SEC-raportti]({source['url']}) — tilikausi {source['period_start']}–{source['period_end']}, julkaistu {source['filed']}.")
        else:
            st.warning(f"{METRICS[key]['label']}: lähtötieto puuttuu.")


growth_tab, margin_tab, cash_tab = st.tabs(["1 Liikevaihdon kasvu", "2 Liikevoittomarginaali", "3 Vapaa kassavirta"])
with growth_tab:
    st.write("Liikevaihto tarkoittaa yrityksen myyntituottoja. Vuosikasvu kertoo, kuinka paljon myynti muuttui edellisestä tilikaudesta.")
    st.metric(f"Liikevaihdon vuosikasvu {year}", percent(row["revenue_growth_pct"]))
    st.line_chart(frame[["revenue"]].rename(columns={"revenue": "Liikevaihto, miljoonaa USD"}) / 1_000_000)
    with st.expander("Näytä laskenta", expanded=True):
        st.write("Vuosikasvu = (uusi liikevaihto / edellisen vuoden liikevaihto − 1) × 100.")
        if position and row["revenue_growth_pct"] is not None:
            previous = rows[position - 1]
            st.write(f"({amount(row['revenue'])} / {amount(previous['revenue'])} − 1) × 100 = {percent(row['revenue_growth_pct'])}")
            previous_source = previous["sources"].get("revenue")
            if previous_source:
                st.markdown(f"[Vertailuvuoden {previous['fiscal_year']} lähde]({previous_source['url']}) — julkaistu {previous_source['filed']}.")
        else:
            st.info("Vuosikasvu tarvitsee kaksi liikevaihtolukua ja positiivisen vertailuluvun. Valitun aineiston ensimmäisen vuoden vertailuluku ei ole mukana.")
        show_sources(["revenue"])
        st.write("Kasvu ei yksin kerro kannattavuudesta tai siitä, onko osake edullinen.")
    with st.expander("Seuraavaksi kolmen vuoden keskikasvu"):
        st.write("Neljä tilikautta muodostaa kolme vuosiväliä. CAGR kertoo tasaisen vuosikasvun, jolla lähtövuodesta päädyttäisiin viimeiseen vuoteen.")
        st.write(f"(({amount(rows[-1]['revenue'])} / {amount(rows[0]['revenue'])}) potenssiin 1/3 − 1) × 100 = {percent(report['revenue_cagr_pct'])}")
        for endpoint in (rows[0], rows[-1]):
            source = endpoint["sources"].get("revenue")
            if source:
                st.markdown(f"[Vuoden {endpoint['fiscal_year']} lähtöluku]({source['url']}) — julkaistu {source['filed']}.")
        st.write("Keskikasvu ei kerro, oliko kehitys välivuosina tasaista.")
with margin_tab:
    st.write("Liikevoitto kuvaa liiketoiminnan tulosta ennen rahoituseriä ja veroja. Marginaali kertoo, kuinka suuri osa myynnistä jää liikevoitoksi.")
    st.metric(f"Liikevoittomarginaali {year}", percent(row["operating_margin_pct"]))
    st.line_chart(frame[["operating_margin_pct"]].rename(columns={"operating_margin_pct": "Liikevoittomarginaali, %"}))
    with st.expander("Näytä laskenta", expanded=True):
        st.write("Liikevoittomarginaali = liikevoitto / liikevaihto × 100.")
        if row["operating_margin_pct"] is not None:
            st.write(f"{amount(row['operating_income'])} / {amount(row['revenue'])} × 100 = {percent(row['operating_margin_pct'])}")
        show_sources(["operating_income", "revenue"])
        st.write("Liikevaihdon kasvu ei automaattisesti paranna marginaalia. Eri toimialojen marginaalit eivät ole suoraan vertailukelpoisia.")
        st.info("Pohdittavaa: jos myynti kasvaa mutta liikevoitto pysyy ennallaan, mitä marginaalille tapahtuu?")
with cash_tab:
    st.write("Liiketoiminnan rahavirta kertoo toiminnan tuottamasta tai käyttämästä rahasta. Tässä harjoituksessa siitä vähennetään aineelliset käyttöomaisuusinvestoinnit.")
    st.metric(f"Vapaa kassavirta {year}, miljoonaa USD", amount(row["free_cash_flow"]))
    st.bar_chart(table[["Liiketoiminnan rahavirta", "Investoinnit", "Vapaa kassavirta"]], stack=False)
    with st.expander("Näytä laskenta", expanded=True):
        st.write("Vapaa kassavirta = liiketoiminnan rahavirta − aineelliset käyttöomaisuusinvestoinnit.")
        if row["free_cash_flow"] is not None:
            st.write(f"{amount(row['operating_cash_flow'])} − {amount(row['capex'])} = {amount(row['free_cash_flow'])} miljoonaa USD")
        show_sources(["operating_cash_flow", "capex"])
        st.write("Määritelmä ei automaattisesti sisällä yritysostoja tai rahoitusleasingin maksuja. Vapaa kassavirta ei ole sama kuin voitto tai jaettavissa oleva osinko.")
        st.info("Pohdittavaa: voiko tulevaa kasvua rakentava investointi pienentää tämän vuoden vapaata kassavirtaa?")

with st.expander("Tarkat lähdetiedot"):
    details = []
    for item in rows:
        for key, source in item["sources"].items():
            details.append({"Tilikausi": item["fiscal_year"], "Luku": METRICS[key]["label"], "SEC-kenttä": source["tag"], "Julkaistu": source["filed"], "Raportti": source["accession"], "Yksikkö": source["unit"]})
    st.dataframe(pd.DataFrame(details), width="stretch", hide_index=True)
    st.caption("Samalle tilikaudelle valitaan viimeisin määräpäivään mennessä julkaistu vuosiraportin havainto. Vertailuvuosien lähde voi siksi olla uudempi raportti. Osavuosilukuja ei käytetä.")
