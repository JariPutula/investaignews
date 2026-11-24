"""The guided MSFT exercise; opening and reading never calls external services."""

from datetime import datetime
import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st

from learning.microsoft_exercise import (
    MODEL, PRICING_URL, ask_question, estimate_exercise_cost, estimate_question_cost, generate_exercise,
    list_runs, load_conversation, load_run, source_catalog,
)
from learning.analysis_errors import AnalysisGenerationError
from market_data.microsoft_dcf import CASE_LABELS
from market_data.microsoft_exercise import load_exercise, refresh_exercise, validate_snapshot
from market_data.sec import SecDataError
from news_sentinel.config import get_openai_api_key


def _contact():
    path = Path(__file__).resolve().parents[1]/".streamlit"/"local_preferences.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("sec_contact", "")
    except (OSError, ValueError):
        return ""


def _money(value):
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def show_claim(claim, sources, title):
    st.write(claim["text"])
    with st.expander(f"Perustelun lähteet: {title}"):
        for evidence in claim["evidence"]:
            source = sources[evidence["source_id"]]
            if source["url"]:
                st.markdown(f"[{source.get('label', evidence['source_id'])}]({source['url']})")
            else:
                st.caption(source["label"])
            st.text(evidence["quote"] or source["text"])
        st.caption("Lähdeviite ja raporttilainaus tarkistetaan automaattisesti. Perustelun sisältö on tekoälyn tulkinta.")


def show_facts(snapshot):
    price = snapshot["market"]["summaries"]["MSFT"]
    benchmark = snapshot["market"]["summaries"]["SPY"]
    financials = snapshot["financials"]
    st.caption(f"Aineisto {snapshot['as_of']} · päätöskurssi {price['price_date']} · talouslukujen viimeinen päivä {financials['ttm']['period_end']}.")
    for column, label, value in zip(st.columns(3), ("Microsoftin päätöskurssi, USD", "Kurssimuutos kuukaudessa", "Muutos 52 viikon huipusta"),
                                   (_money(price["price"]), f"{price['month']['change_pct']:.2f} %", f"{price['peak_52w']['change_pct']:.2f} %")):
        column.metric(label, value)
    st.caption(f"Kuukauden vertailupäivä {price['month']['date']} · huipun päivä {price['peak_52w']['date']} · SPY:n kuukauden muutos {benchmark['month']['change_pct']:.2f} %. Kyse on päätöskurssien muutoksesta ilman osinkojen uudelleensijoitusta.")
    details = st.expander("Kurssikuva ja raportoidut luvut", key="exercise_facts", on_change="rerun")
    if details.open:
        with details:
            histories = snapshot["market"]["histories"]
            frame = pd.concat({ticker: pd.DataFrame(rows).set_index("date")["close"] for ticker, rows in histories.items()}, axis=1).dropna()
            frame.index = pd.to_datetime(frame.index)
            st.line_chart(frame/frame.iloc[0]*100, alt="Microsoftin ja SPY:n päätöskurssit yhteisestä alkutasosta sata")
            st.caption("Sama lähtötaso helpottaa kurssien kehityksen vertailua. SPY seuraa S&P 500 -indeksiä.")
            for ticker, url in snapshot["market"]["source_urls"].items():
                st.markdown(f"[{ticker}: kurssihistorian lähde]({url})")
            quarter = financials["quarter"]
            previous = financials["prior_quarter"]
            rows = []
            for key, label in (("revenue", "Liikevaihto"), ("operating_income", "Liikevoitto"), ("cash_capex", "Käteisinvestoinnit"), ("free_cash_flow", "Raportoituun rahavirtaan perustuva FCF")):
                rows.append({"Erä, mrd. USD": label, previous["period_end"]: previous[key]/1e9, quarter["period_end"]: quarter[key]/1e9,
                             "Vuosimuutos, %": financials["quarter_growth_pct"][key]})
            st.dataframe(pd.DataFrame(rows), hide_index=True, alt="Viimeisin vuosineljännes ja edellisvuoden sama neljännes")
            st.caption("Negatiivisen lähtöluvun päälle ei lasketa tavallista kasvuprosenttia. Viimeisimmän neljänneksen lukua ei kerrota neljällä.")
            ttm = financials["ttm"]
            st.write(f"Viimeiset 12 kk {ttm['period_start']}–{ttm['period_end']}: liikevaihto {_money(ttm['revenue']/1e9)} mrd. USD, liikevoitto {_money(ttm['operating_income']/1e9)} mrd. USD ja vapaa kassavirta {_money(ttm['free_cash_flow']/1e9)} mrd. USD.")
            st.caption(financials["ttm_formula"])
            st.dataframe(pd.DataFrame([{"Tilikausi": row["fiscal_year"], "Liikevaihto, mrd. USD": row["revenue"]/1e9,
                "Liikevoitto, mrd. USD": row["operating_income"]/1e9, "Käteisinvestoinnit, mrd. USD": row["cash_capex"]/1e9,
                "Uudet rahoitusleasing-hyödykkeet, mrd. USD": row["finance_lease_additions"]/1e9} for row in financials["annuals"]]),
                hide_index=True, alt="Neljän tilikauden liikevaihto, liikevoitto ja investoinnit")
            st.markdown(f"[Uusin osavuosiraportti]({snapshot['quarter_report']}) · [Johdon tulospuhelu ja näkymät]({snapshot['guidance_url']})")


def show_run(run, *, archived=False):
    snapshot, plan, results = run["snapshot"], run["plan"], run["results"]
    sources = source_catalog(snapshot, results)
    if archived:
        st.info(f"Arkistoitu harjoitus: laadittu {run['created_at'][:10]}, kurssi {snapshot['market']['summaries']['MSFT']['price_date']}. Kaikki alla olevat tulokset käyttävät tämän harjoituksen omaa aineistoa.")
    st.subheader("Mitä hinnalle ja liiketoiminnalle tapahtui?")
    show_claim(plan["price_view"], sources, "hinta")
    show_claim(plan["current_view"], sources, "liiketoiminta")
    st.subheader("Kolme tekoälyn valmistelemaa tulevaisuutta")
    st.caption("Perus on työoletus. Skenaariot eivät ole analyytikkoennusteita, todennäköisyyksiä tai ostosuosituksia.")
    table = []
    for key, result in results["values"].items():
        table.append({"Skenaario": CASE_LABELS[key], "Arvioitu arvo/osake, USD": result["value_per_share"],
                      "Ero päätöskurssiin, %": result["difference_to_price_pct"],
                      "Seuraavan 12 kk skenaarion P/E": result["first_year_scenario_pe"],
                      "Jatkuvan vaiheen osuus yritysarvosta, %": result["terminal_share_pct"]})
    st.dataframe(pd.DataFrame(table), hide_index=True, key=f"exercise_values_{snapshot['fingerprint'][:8]}", alt="Kolmen skenaarion arvot, tuloskertoimet ja jatkuvan vaiheen osuus")
    st.caption(f"Historiallinen 12 kk P/E {_money(results['ttm_price_to_earnings']) if results['ttm_price_to_earnings'] is not None else 'ei saatavilla'} × · saman jakson vapaan kassavirran tuotto {_money(results['ttm_fcf_yield_pct'])} %. Skenaarion P/E käyttää tekoälyn nettomarginaalioletusta ja muuttumatonta päivättyä markkina-arvoa.")
    for scenario in plan["scenarios"]:
        key = scenario["case"]
        with st.expander(f"{CASE_LABELS[key]}: miksi tekoäly valitsi nämä oletukset?"):
            for driver in scenario["drivers"]:
                show_claim(driver, sources, {"growth": "kasvu", "profitability": "kannattavuus", "investment": "investoinnit"}[driver["topic"]])
            st.markdown("**Seurattava havainto**")
            show_claim(scenario["watch"], sources, "seuraava raportti")
            fields = {"growth_pct": "Myynnin kasvu, %", "operating_margin_pct": "Liikevoittomarginaali, %", "net_margin_pct": "Nettomarginaali, %",
                      "cash_capex_pct": "Käteisinvestoinnit/myynti, %", "lease_capex_pct": "Uudet rahoitusleasing-hyödykkeet/myynti, %", "da_pct": "Poistot/myynti, %"}
            assumptions = scenario["assumptions"]
            st.dataframe(pd.DataFrame({"Vuosi tästä eteenpäin": list(range(1, 6)), **{label: assumptions[field] for field, label in fields.items()}}),
                         hide_index=True, alt="Tekoälyn viiden vuoden oletukset prosentteina")
            st.caption(f"Käyttöpääoman lisäraha / myynnin muutos: {assumptions['working_capital_pct_of_sales_change']:.1f} %. Kaikki luvut ovat mallioletuksia.")
    st.subheader("Millaista kasvua nykyinen hinta edellyttäisi?")
    reversed_rows = []
    for key, result in results["reverse"].items():
        if result["available"]:
            reversed_rows.append({"Muut oletukset": CASE_LABELS[key], "Hinnan edellyttämä tasainen vuosikasvu, %": result["growth_pct"]})
        else:
            st.info(f"{CASE_LABELS[key]}: {result['reason']}")
    if reversed_rows:
        st.dataframe(pd.DataFrame(reversed_rows), hide_index=True, alt="Nykyhinnan edellyttämä tasainen viiden vuoden liikevaihdon kasvu eri muilla oletuksilla")
    st.caption("Ratkaisussa vain liikevaihdon kasvupolku vaihtuu tasaiseksi. Marginaalien ja investointien oletukset pysyvät valitun tapauksen mukaisina. Siksi kolme tapausta voivat antaa eri kasvuluvun samalle hinnalle.")
    st.subheader("Miten tätä kannattaa tulkita?")
    for key, title in (("conclusion", "Hinta ja arvo"), ("implied_growth", "Hinnan edellyttämä kasvu"), ("counterargument", "Vahvin vastaperustelu"), ("next_question", "Seuraavaksi seurattava")):
        st.markdown(f"**{title}**")
        show_claim(run["interpretation"][key], sources, title)
    with st.expander("Tuottovaatimus, herkkyys ja laskennan opastus"):
        common = plan["common"]
        st.write(f"Yrityksen tuottovaatimus (WACC) {common['wacc_pct']:.1f} %, veroaste {common['tax_pct']:.1f} %, jatkuva kasvu {common['terminal_growth_pct']:.1f} % ja uuden pääoman tuotto jatkuvassa vaiheessa {common['terminal_roic_pct']:.1f} %.")
        st.caption("Tuottovaatimus on tässä tekoälyn harkinnanvarainen herkkyysoletus. Sitä ei vielä lasketa markkinoiden korko- ja riskitiedoista.")
        show_claim(plan["valuation_reason"], sources, "yhteiset arvostusoletukset")
        st.dataframe(pd.DataFrame(results["sensitivity"]).rename(columns={"wacc_pct": "Tuottovaatimus, %", "terminal_growth_pct": "Jatkuva kasvu, %",
            "value_per_share": "Arvo/osake, USD", "implied_growth_pct": "Hinnan edellyttämä vuosikasvu, %"}), hide_index=True, alt="Perusskenaarion herkkyys tuottovaatimukselle ja jatkuvalle kasvulle")
        st.write("Kasvu kertoo myynnin muutoksesta. Marginaali kertoo, paljonko myynnistä jää voitoksi. Investoinnit käyttävät rahaa nyt ja voivat tuottaa sitä myöhemmin. Tuottovaatimus pienentää kaukana tulevaisuudessa saatavan rahan nykyarvoa. Jatkuva arvo arvioi rahavirrat viiden ennustevuoden jälkeen.")
        st.write(results["definition"])
        st.caption(results["limits"])
        balance = snapshot["financials"]["balances"]
        st.dataframe(pd.DataFrame({"Arvosta omistajille, mrd. USD": ["Raportoidut lainat yhteensä", "Rahoitusleasingvelat", "Käteinen ja lyhytaikaiset sijoitukset", "Nettovelka vähennettäväksi yritysarvosta"],
            "Määrä": [balance["borrowings"]/1e9, balance["finance_lease_debt"]/1e9, balance["cash_and_short_term_investments"]/1e9, results["values"]["base"]["net_debt"]/1e9]}),
            hide_index=True, alt="Yritysarvon ja omistajille kuuluvan arvon välissä huomioitavat velat ja varat")
        st.caption(f"Osakemäärä {balance['shares']/1e9:.3f} miljardia · tasepäivä {balance['period_end']}. Lainojen lyhytaikaista osuutta ei lisätä uudelleen kokonaisvelkaan.")
        st.dataframe(pd.DataFrame([{"Skenaario": CASE_LABELS[key], "Sama arvo kuukauden aikaisella hinnalla: ero, %": value["difference_to_month_price_pct"],
                                   "Sama arvo nykyhinnalla: ero, %": value["difference_to_price_pct"]} for key, value in results["values"].items()]),
                     hide_index=True, alt="Saman kassavirtamallin vertailu kuukauden aikaisempaan ja nykyiseen hintaan")
        st.caption("Tässä pidetään arvolaskelma samana molemmille hinnoille. Se ei väitä yhtiön todellisen arvon pysyneen samana eikä rekonstruoi aiempia markkinaodotuksia.")
        for key, value in results["values"].items():
            st.markdown(f"**{CASE_LABELS[key]}: Pythonin rahavirtalaskelma**")
            frame = pd.DataFrame(value["forecast"]).set_index("year")
            frame.index.name = "Vuosi tästä eteenpäin"
            labels = {"revenue": "Liikevaihto", "operating_income": "Liikevoitto", "net_income": "Nettotulos",
                      "nopat": "Liikevoitto verojen jälkeen", "da": "Poistot", "cash_capex": "Käteisinvestoinnit", "lease_capex": "Uudet rahoitusleasing-hyödykkeet",
                      "working_capital_investment": "Käyttöpääoman lisäraha", "fcff": "Yrityksen vapaa kassavirta", "present_value": "Rahavirran nykyarvo"}
            for field in labels:
                frame[field] = frame[field]/1e9
            frame = frame.rename(columns={**{key: label+", mrd. USD" for key, label in labels.items()}, "growth_pct": "Myynnin kasvu, %"})
            st.dataframe(frame, alt="Arvostuksen vuosikohtaiset laskelmat miljardeina dollareina")
            st.caption(f"Rahavirran muutos viimeisestä ennustevuodesta jatkuvan vaiheen ensimmäiseen vuoteen: {_money(value['year5_to_terminal_fcff_change']/1e9)} mrd. USD. Suuri muutos on tärkeä malliriski.")
    st.caption(f"Laadittu {run['created_at']} · malli {run['model']} · tallennetun version vaiheiden tokenkäytön kustannusarvio {run['cost_estimate_usd']:.4f} USD. Mahdolliset hylätyt kutsut ovat erillisiä kuluja. Tallennetun harjoituksen lukeminen on maksutonta.")
    if run.get("review_notes"):
        with st.expander("Harjoituksen tarkistusmerkinnät"):
            for note in run["review_notes"]:
                st.write(note)
    st.download_button("Lataa harjoitus oletuksineen ja lähteineen", json.dumps(run, ensure_ascii=False, indent=2),
                       file_name=f"Microsoft_harjoitus_{snapshot['as_of']}.json", mime="application/json", key=f"download_exercise_{snapshot['fingerprint'][:8]}")
    st.subheader("Kysy tämän harjoituksen luvuista")
    conversation = load_conversation(run)
    for item in conversation:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            show_claim(item["answer"], sources, "harjoitusvastaus")
    key = get_openai_api_key()
    st.caption(f"Uusi vastaus käyttää yhtä maksullista tekoälykutsua, varovainen kustannusarvio enintään noin {estimate_question_cost(run):.2f} USD. Sama tallennettu kysymys palautetaan ilman uutta kutsua. Vastaus käyttää tämän päivätyn harjoituksen aineistoa.")
    st.caption("Palveluun lähetetään kysymyksesi, enintään viisi edellistä harjoituskysymystä vastauksineen sekä tämän harjoituksen julkinen lähdepaketti. Keskustelu tallentuu paikallisesti harjoituksen yhteyteen.")
    with st.form(f"exercise_question_{snapshot['fingerprint'][:8]}", clear_on_submit=True):
        question = st.text_area("Mikä jäi epäselväksi?", placeholder="Esimerkiksi: miksi suuret investoinnit pienentävät tämänhetkistä arviota?", max_chars=1500, key="exercise_question_text")
        submitted = st.form_submit_button("Kysy tekoälyltä", disabled=not bool(key))
    if submitted:
        with st.spinner("Selitetään tämän harjoituksen avulla…"):
            ask_question(run, question, key)
        st.rerun()


def show_exercise():
    st.title("Microsoft: mitä nykyinen hinta edellyttää?")
    st.write("Tekoäly valmistelee skenaariot ja opastaa. Python laskee arvostuksen. Aloita valmiista harjoituksesta; sinun ei tarvitse antaa kasvu- tai marginaalioletuksia.")
    contact = _contact()
    with st.expander("Aineiston haun yhteystieto"):
        contact = st.text_input("SEC-haun yhteyssähköposti", value=contact, key="exercise_sec_contact")
        st.caption("Paikallisiin asetuksiin tallennettua yhteystietoa käytetään vain SEC-haun tunnisteena. Se ei kuulu tekoälyn aineistoon.")
    if st.button("Päivitä aineisto", key="refresh_msft_exercise"):
        agent = f"InvestaiLearning {contact.strip()}" if contact.strip() else os.getenv("SEC_USER_AGENT", "")
        try:
            with st.spinner("Haetaan raportoidut luvut, kurssihistoria ja johdon näkymät…"):
                refresh_exercise(agent)
            st.success("Uusi päivätty aineisto tallennettu. Tekoälykutsuja ei tehty.")
        except (SecDataError, OSError) as error:
            st.error(str(error))
    try:
        snapshot = load_exercise()
        if snapshot is None:
            st.info("Päivitä ensin aineisto. Haku käyttää maksuttomia julkisia lähteitä.")
            return
        current = True
        try:
            validate_snapshot(snapshot, require_current=True)
        except SecDataError:
            current = False
            st.warning("Näytetään aiempi päivätty aineisto. Päivitä aineisto ennen uutta nykyhinnan arviointia; tallennettua harjoitusta voi edelleen lukea.")
        show_facts(snapshot)
        run = load_run(snapshot)
        if run is None:
            estimate = estimate_exercise_cost(snapshot)
            st.write(f"Laadi tekoälyharjoitus: kaksi maksullista kutsua, varovainen kustannusarvio enintään noin {estimate['upper_estimate_usd']:.2f} USD.")
            st.caption(f"Malli {MODEL} · arvio tarkistettu {estimate['pricing_checked']}. [Hinnoittelu]({PRICING_URL}). Vain julkiset yhtiötiedot ja raporttiotteet lähetetään palveluun; API-tallennus on pois käytöstä. Kesken jäänyt valmis skenaariovaihe käytetään uudelleen.")
            key = get_openai_api_key()
            if not key:
                st.info("OpenAI-avain puuttuu palvelimen asetuksista.")
            if st.button("Laadi tekoälyharjoitus", type="primary", key="generate_msft_exercise", disabled=not bool(key) or not current):
                with st.spinner("Laaditaan perusteltua harjoitusta…"):
                    progress = st.empty()
                    run = generate_exercise(snapshot, key, on_stage=progress.caption)
                    progress.empty()
        if run:
            show_run(run)
        archives = list_runs()
        if archives:
            archive = st.expander("Aiemmat päivätyt harjoitukset", key="exercise_archive", on_change="rerun")
            if archive.open:
                with archive:
                    selected = st.selectbox("Tallennettu harjoitus", list(range(len(archives))), format_func=lambda i: f"{archives[i]['created_at']} · kurssi {archives[i]['price_date']}", key="exercise_archive_selection")
                    saved = json.loads(Path(archives[selected]["path"]).read_text(encoding="utf-8"))
                    historical = load_run(saved["snapshot"])
                    st.json({"Laadittu": historical["created_at"], "Oletukset": historical["plan"], "Tulokset": historical["results"], "Tulkinta": historical["interpretation"]}, expanded=False)
        with st.expander("Lähtölukujen tarkat lähteet"):
            st.dataframe(pd.DataFrame([{"Tunnus": key, **source} for key, source in snapshot["financials"]["sources"].items()]),
                         hide_index=True, alt="Harjoituksen talouslukujen lähteet, kaudet ja julkaisupäivät")
    except (SecDataError, OSError) as error:
        st.error(str(error))
        if isinstance(error, AnalysisGenerationError):
            with st.expander("Virheen tekniset tiedot"):
                st.json(error.details)
