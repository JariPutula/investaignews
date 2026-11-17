"""One shared learning page for the configured companies."""

from datetime import datetime
import math
import os
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st

from learning.metric_guide import CONTEXT, GUIDE, KO_REPORT
from market_data.companies import COMPANIES, get_company
from market_data.sec import METRICS, SecDataError, build_report, comparison_rows, load_report


def display_date(value):
    return datetime.fromisoformat(value).strftime("%d.%m.%Y")


def display_time(value):
    return datetime.fromisoformat(value).astimezone(ZoneInfo("Europe/Rome")).strftime("%d.%m.%Y klo %H.%M %Z")


def format_value(value, unit="usd_m"):
    if value is None or not math.isfinite(value):
        return "ei saatavilla"
    if unit == "usd_m":
        return f"{value/1e6:,.0f}".replace(",", " ")
    return f"{value:.2f}".replace(".", ",") + (" %" if unit == "pct" else " ×")


def explain(key, *, identity):
    guide = GUIDE[key]
    with st.popover(f"Tietoa: {guide['label']}", key=f"info_{identity}_{key}"):
        st.markdown(f"**{guide['label']}**")
        st.write(guide["meaning"])
        st.markdown("**Kaava**")
        st.write(guide["formula"])
        st.markdown("**Miten arviota kannattaa tehdä?**")
        st.write(guide["assessment"])
        st.markdown("**Tulkinnan rajat**")
        st.write(guide["limit"])


def metric_card(container, key, row, identity):
    guide = GUIDE[key]
    with container:
        label = guide["label"] + (", milj. USD" if guide["unit"] == "usd_m" else "")
        st.metric(label, format_value(row.get(key), guide["unit"]), help=guide["meaning"])
        explain(key, identity=identity)


def show_sources(row, keys):
    for key in dict.fromkeys(keys):
        source = row["sources"].get(key)
        if not source:
            st.caption(f"{METRICS[key]['label']}: lähtötieto puuttuu.")
            continue
        period = (
            f"{display_date(source['period_start'])}–{display_date(source['period_end'])}"
            if source["period_start"] else f"tilanne {display_date(source['period_end'])}"
        )
        st.markdown(f"[{METRICS[key]['label']}: lähderaportti]({source['url']}) — {period}, julkaistu {display_date(source['filed'])}.")


def calculation(key, row, previous=None):
    guide = GUIDE[key]
    with st.expander(f"Näytä laskenta: {guide['label']}"):
        st.write(guide["formula"])
        value = row.get(key)
        if value is None:
            st.info("Laskentaan tarvittava tieto puuttuu tai jakaja ei ole positiivinen. Aineiston ensimmäisen vuoden vuosikasvu tarvitsee myös sitä edeltävän vuoden.")
        elif key == "revenue_growth_pct":
            st.write(f"({format_value(row['revenue'])} / {format_value(previous['revenue'])} − 1) × 100 = {format_value(value, 'pct')}")
            show_sources(previous, ["revenue"])
        elif key == "free_cash_flow":
            st.write(f"{format_value(row['operating_cash_flow'])} − {format_value(row['capex'])} = {format_value(value)} milj. USD")
        elif key == "gross_margin_pct":
            st.write(f"({format_value(row['revenue'])} − {format_value(row['cost_of_revenue'])}) / {format_value(row['revenue'])} × 100 = {format_value(value, 'pct')}")
        elif key == "fcf_margin_pct":
            st.write(f"({format_value(row['operating_cash_flow'])} − {format_value(row['capex'])}) / {format_value(row['revenue'])} × 100 = {format_value(value, 'pct')}")
        else:
            numerator, denominator = guide["inputs"]
            multiplier = " × 100" if guide["unit"] == "pct" else ""
            st.write(f"{format_value(row[numerator])} / {format_value(row[denominator])}{multiplier} = {format_value(value, guide['unit'])}")
        st.caption("Rahamäärät laskelman esityksessä ovat miljoonia USD; varsinainen laskenta käyttää pyöristämättömiä dollarimääriä.")
        show_sources(row, guide["inputs"])
        st.write(guide["limit"])


def show_comparison():
    st.subheader("Microsoft ja Coca-Cola rinnakkain")
    st.write("Vertaa ensisijaisesti kehitystä ja liiketoimintamallia. Yksi korkeampi tunnusluku ei tee yrityksestä parempaa sijoitusta.")
    reports = []
    for company in COMPANIES.values():
        try:
            report = load_report(company=company)
            if report is not None:
                reports.append(report)
            else:
                st.info(f"{company.name}: hae aineisto yhtiön omasta näkymästä.")
        except SecDataError as exc:
            st.warning(f"{company.name}: {exc}")
    if len(reports) < 2:
        st.info("Rinnakkainen vertailu tulee näkyviin, kun molempien yhtiöiden aineisto on tallennettu.")
        return
    same_year = st.radio("Vertailun ajanjakso", ["Sama tilikauden päättymisvuosi", "Kummankin viimeisin tilikausi"], horizontal=True, key="comparison_mode")
    selected_year = None
    if same_year == "Sama tilikauden päättymisvuosi":
        common_years = sorted(set.intersection(*(set(row["fiscal_year"] for row in report["rows"]) for report in reports)))
        if not common_years:
            st.info("Yhteisiä tilikausien päättymisvuosia ei ole.")
            return
        selected_year = st.selectbox("Vertailuvuosi", common_years, index=len(common_years)-1, key="comparison_year")
    compared = comparison_rows(reports, selected_year)
    st.info("Microsoftin tilikausi päättyy kesäkuussa ja Coca-Colan joulukuussa. Sama vuosiluku ei tarkoita samaa ajanjaksoa. Tarkat jaksot näkyvät taulukossa.")
    with st.popover("Miksi tilikausien ero on tärkeä?"):
        st.write("Eri tilikaudet voivat sisältää erilaisen taloussuhdanteen, valuuttakurssit ja tapahtumat. Viimeisimpien tilikausien vertailussa myös vuosiluvut voivat erota. Tässä ei muodosteta yhteisen kalenterijakson osavuosilaskelmaa.")
    columns = {}
    comparison_keys = ("revenue_growth_pct", "operating_margin_pct", "gross_margin_pct", "net_margin_pct", "fcf_margin_pct", "capex_to_revenue_pct", "current_ratio")
    for item in compared:
        column = {
            "Tilikausi alkaa": display_date(item["period_start"]),
            "Tilikausi päättyy": display_date(item["period_end"]),
            "Liikevaihto, milj. USD": format_value(item["revenue"]),
            "Vapaa kassavirta, milj. USD": format_value(item["free_cash_flow"]),
        }
        column.update({GUIDE[key]["label"]: format_value(item.get(key), GUIDE[key]["unit"]) for key in comparison_keys})
        columns[item["company"]] = column
    st.dataframe(pd.DataFrame(columns), width="stretch")
    for item in compared:
        st.caption(f"{item['company']}: julkaistut tiedot huomioitu päivään {display_date(item['as_of'])} asti.")
        note = CONTEXT.get((item["ticker"], item["fiscal_year"]))
        if note:
            st.caption(f"{item['company']}: {note} [Vuosiraportti]({KO_REPORT})")
    trend_key = st.selectbox("Vertaa tunnusluvun kehitystä", ["operating_margin_pct", "fcf_margin_pct", "capex_to_revenue_pct"], format_func=lambda key: GUIDE[key]["label"], key="comparison_trend")
    points = [{"Yhtiö": report["company"], "Tilikauden päättyminen": pd.to_datetime(row["period_end"]), "Prosenttia": row.get(trend_key)} for report in reports for row in report["rows"]]
    st.plotly_chart(px.line(pd.DataFrame(points), x="Tilikauden päättyminen", y="Prosenttia", color="Yhtiö", markers=True), width="stretch")
    explain(trend_key, identity="comparison")


def render():
    st.set_page_config(page_title="Yhtiöanalyysi – opi talousluvut", page_icon="📖", layout="wide")
    st.title("Yhtiöanalyysi")
    st.write("Opettele kasvua, kannattavuutta ja rahavirtaa. Aloita kolmesta perusluvusta; tarkemmat selitykset avautuvat Tietoa-painikkeista.")
    ticker = st.selectbox("Yhtiö", list(COMPANIES), format_func=lambda key: f"{COMPANIES[key].name} ({key})", key="learning_company")
    company = get_company(ticker)
    with st.expander("Hae tai päivitä aineisto"):
        st.write("SEC-haku on maksuton. Yhteyssähköposti lähetetään SEC:lle hakijan tunnisteena; sitä ei tallenneta aineistoon. Päivitys koskee valittua yhtiötä.")
        contact = st.text_input("SEC-haun yhteyssähköposti", placeholder="oma.osoite@example.com", key="sec_contact")
        if st.button(f"Hae {company.name}: uusimmat vuosiluvut", type="primary", key="refresh_company"):
            agent = f"InvestaiLearning {contact.strip()}" if contact.strip() else os.getenv("SEC_USER_AGENT", "")
            try:
                with st.spinner("Haetaan ja tallennetaan vuosiluvut…"):
                    build_report(company=company, user_agent=agent, refresh=True)
                st.success("Vuosiluvut ja lähteet tallennettu.")
            except (SecDataError, OSError) as exc:
                st.error(str(exc))
    try:
        report = load_report(company=company)
    except SecDataError as exc:
        st.error(str(exc))
        return
    if report is None:
        st.info("Aineistoa ei vielä ole. Hae se yllä olevasta osiosta.")
        return
    rows = report["rows"]
    year = st.selectbox("Tilikausi", [row["fiscal_year"] for row in rows], index=len(rows)-1, key=f"year_{ticker}")
    row = next(item for item in rows if item["fiscal_year"] == year)
    index = rows.index(row)
    previous = rows[index-1] if index else None
    st.caption(f"{company.name} · tilikausi {display_date(row['period_start'])}–{display_date(row['period_end'])} · aineisto haettu {display_time(report['retrieved_at'])} · julkaistut tiedot päivään {display_date(report['as_of'])} asti.")
    for warning in report["warnings"]:
        st.warning(warning)
    columns = st.columns(3)
    for container, key in zip(columns, ("revenue_growth_pct", "operating_margin_pct", "free_cash_flow")):
        metric_card(container, key, row, f"top_{ticker}_{year}")
    st.caption("Puuttuva lähtötieto ei tarkoita nollaa. Yksittäinen mittari ei ole osto- tai myyntisuositus.")
    frame = pd.DataFrame(rows).set_index("fiscal_year")
    frame.index.name = "Tilikausi"
    growth, profit, cash, liquidity, comparison = st.tabs(["1 Kasvu", "2 Kannattavuus", "3 Kassavirta", "4 Maksuvalmius", "5 Yhtiövertailu"])
    with growth:
        st.subheader("Miten myynti on kehittynyt?")
        st.line_chart(frame[["revenue"]].rename(columns={"revenue": "Liikevaihto, milj. USD"}) / 1e6)
        calculation("revenue_growth_pct", row, previous)
        with st.expander("Kolmen vuoden keskikasvu eli CAGR"):
            span = rows[-1]["fiscal_year"]-rows[0]["fiscal_year"]
            st.metric(f"Liikevaihdon keskikasvu {rows[0]['fiscal_year']}–{rows[-1]['fiscal_year']}", format_value(report["revenue_cagr_pct"], "pct"))
            st.write(f"Neljä tilikautta muodostaa {span} vuosiväliä. CAGR kuvaa tasaista vuosikasvua lähtövuodesta viimeiseen vuoteen.")
            st.write(f"(({format_value(rows[-1]['revenue'])} / {format_value(rows[0]['revenue'])}) potenssiin 1/{span} − 1) × 100")
            st.write("Keskikasvu ei ole vuosikasvujen aritmeettinen keskiarvo eikä kerro välivuosien tasaisuudesta.")
            show_sources(rows[0], ["revenue"])
            show_sources(rows[-1], ["revenue"])
    with profit:
        st.subheader("Mitä myynnistä jää tulokseksi?")
        for container, key in zip(st.columns(2), ("gross_margin_pct", "net_margin_pct")):
            metric_card(container, key, row, f"profit_{ticker}_{year}")
        st.line_chart(frame[["gross_margin_pct", "operating_margin_pct", "net_margin_pct"]].rename(columns={key: GUIDE[key]["label"] for key in ("gross_margin_pct", "operating_margin_pct", "net_margin_pct")}))
        st.caption("Prosenttia liikevaihdosta. Nettotulokseen vaikuttavat myös liiketoiminnan ulkopuoliset erät ja verot.")
        for key in ("operating_margin_pct", "gross_margin_pct", "net_margin_pct"):
            calculation(key, row)
        with st.popover("Prosentti vai prosenttiyksikkö?"):
            st.write("Marginaalin nousu 20 prosentista 22 prosenttiin on +2 prosenttiyksikköä. Suhteellisesti marginaali nousi 10 %. Käytä vertailussa oikeaa yksikköä.")
    with cash:
        st.subheader("Mihin liiketoiminnan tuottama raha menee?")
        for container, key in zip(st.columns(2), ("fcf_margin_pct", "capex_to_revenue_pct")):
            metric_card(container, key, row, f"cash_{ticker}_{year}")
        cash_frame = frame[["operating_cash_flow", "capex", "free_cash_flow"]].rename(columns={"operating_cash_flow": "Liiketoiminnan rahavirta", "capex": "Investoinnit", "free_cash_flow": "Vapaa kassavirta"}) / 1e6
        st.bar_chart(cash_frame, stack=False)
        st.caption("Miljoonaa USD. Investoinnit ovat aineellisiin käyttöomaisuushyödykkeisiin maksettuja menoja.")
        note = CONTEXT.get((ticker, year))
        if note:
            st.info(note)
            st.markdown(f"[Tarkista selitys Coca-Colan vuosiraportin rahavirtaosiosta]({KO_REPORT})")
        for key in ("free_cash_flow", "fcf_margin_pct", "capex_to_revenue_pct"):
            calculation(key, row)
        with st.popover("Miksi voitto ja rahavirta eroavat?"):
            st.write("Myynti voidaan kirjata tulokseen ennen maksun saamista. Investointi kuluttaa rahaa nyt, mutta sen hankintameno voi näkyä tuloksessa vähitellen poistoina. Myös varastot, asiakkaiden ja toimittajien maksuajat, veromaksut ja osakeperusteiset palkkiot vaikuttavat eroon.")
    with liquidity:
        st.subheader("Lyhyen aikavälin maksuvalmius")
        metric_card(st.container(), "current_ratio", row, f"liquidity_{ticker}_{year}")
        st.line_chart(frame[["current_ratio"]].rename(columns={"current_ratio": "Lyhytaikaiset varat / lyhytaikaiset velat"}))
        st.write("Tämä on tilinpäätöspäivän taseeseen perustuva näkökulma. Se ei mittaa yrityksen koko velkariskiä tai ennusta maksukyvyttömyyttä.")
        calculation("current_ratio", row)
    with comparison:
        show_comparison()
    with st.expander("Kaikki lähtöluvut ja lähdetiedot"):
        labels = {key: value["label"] for key, value in METRICS.items()}
        table = frame[list(METRICS)].rename(columns=labels) / 1e6
        st.caption("Rahamäärät miljoonaa USD. Taseen lyhytaikaiset varat ja velat ovat tilikauden päättymispäivältä; muut luvut kattavat koko tilikauden.")
        st.dataframe(table, width="stretch")
        details = [{"Tilikausi": item["fiscal_year"], "Luku": METRICS[key]["label"], "SEC-kenttä": source["tag"], "Julkaistu": source["filed"], "Raportti": source["accession"], "Yksikkö": source["unit"]} for item in rows for key, source in item["sources"].items()]
        st.dataframe(pd.DataFrame(details), width="stretch", hide_index=True)
        for note in report["data_notes"]:
            st.caption(note)
        st.caption("Samalle kaudelle käytetään viimeisintä määräpäivään mennessä julkaistua vuosiraportin havaintoa. Vertailuvuosien lähde voi siksi olla uudempi raportti. Osavuosilukuja ei käytetä.")
