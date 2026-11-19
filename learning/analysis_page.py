"""Short company interpretations, expandable evidence and explicit paid actions."""

import json
import os

import pandas as pd
import streamlit as st

from learning import company_analysis as cash_analysis
from learning.analysis_errors import AnalysisGenerationError
from learning.analysis_index import list_saved_analyses
from learning.company_brief import TOPICS, MODEL, PRICING_URL, estimate_cost, generate_analysis, load_analysis, make_packet, make_company_packet
from learning.investor_controls import show_investor_controls
from market_data.annual_reports import load_annual_report, prepare_annual_report
from market_data.sec import SecDataError
from news_sentinel.config import get_openai_api_key


def open_saved_analysis(ticker, year):
    st.session_state['learning_company'] = ticker
    st.session_state[f'year_{ticker}'] = year
    st.session_state['learning_tab'] = '6 Tekoälyanalyysi'


def show_saved_analysis_shortcuts():
    saved = list_saved_analyses()
    if not saved:
        return
    with st.container(border=True):
        st.markdown('**Valmiit tekoälyanalyysit**')
        st.caption('Avaa tallennettu analyysi suoraan. Avaaminen ei tee uutta maksullista kutsua.')
        with st.container(horizontal=True):
            for item in saved:
                st.button(f"Avaa {item['company']} · {item['fiscal_year']}",
                          key=f"open_saved_{item['ticker']}_{item['fiscal_year']}",
                          on_click=open_saved_analysis, args=(item['ticker'], item['fiscal_year']))


def show_claim(claim, packet, label):
    st.write(claim['text'])
    with st.expander(f'Lähteet: {label}'):
        for evidence in claim['evidence']:
            source = packet['sources'][evidence['source_id']]
            if source.get('kind') == 'assumption':
                st.markdown(f"**{evidence['source_id']}: oletus ja Pythonin laskelma** · [lähtölukujen lähde]({source['url']})")
            else:
                st.markdown(f"[{evidence['source_id']}: lähde]({source['url']})")
            st.text(evidence['quote'] or source['text'])
            for key, fact in source.get('fact_sources', {}).items():
                st.markdown(f"[{key}: {fact['tag']}]({fact['url']}) · julkaistu {fact['filed']}")
        st.caption('Lähdeviitteet ja annetut lainaukset on tarkistettu automaattisesti. Johtopäätös on tekoälyn tulkintaa.')


def show_saved(saved, packet, annual):
    st.success('Tallennettu analyysi: lukeminen ja näkymän vaihtaminen eivät tee uusia API-kutsuja.')
    analysis = saved['analysis']
    if 'overview' in analysis:
        if 'outlook' in analysis:
            st.caption('Luvut ovat Pythonin laskemia havaintoja. Nykytilan johtopäätökset ovat tekoälyn tulkintaa; tulevaisuusarvio perustuu näkyviin oletuksiin.')
        st.subheader('Yhtiön tilanne lyhyesti')
        show_claim(analysis['overview'], packet, 'kokonaiskuva')
        with st.expander('Tunnusluvut selkokielellä: mitä luvut ja kehitys kertovat?', expanded='outlook' not in analysis):
            for topic in analysis['topics']:
                label = TOPICS[topic['topic']]
                st.markdown(f'**{label}**')
                show_claim(topic, packet, label.lower())
                lesson = packet.get('lessons', {}).get(topic['topic'])
                if lesson:
                    st.markdown('**Mitä luku tarkoittaa?**')
                    st.write(lesson['meaning'])
                    st.write(lesson['example'])
                    if lesson.get('change'):
                        st.write(lesson['change'])
                    st.caption(lesson['limit'])
        if 'outlook' in analysis:
            for key, label in (('outlook', 'Tulevaisuus — oletuksiin perustuva arvio'),
                               ('investment_case', 'Sijoittajan kokonaisarvio — tulkinta'),
                               ('rebuttal', 'Mikä voisi kumota tämän näkemyksen?')):
                st.markdown(f'**{label}**')
                show_claim(analysis[key], packet, label.lower())
            if not packet['valuation']['available']:
                st.info(packet['valuation']['reason'])
            with st.expander('Vahvuus ja muu huomio'):
                for key, label in (('strengths', 'Vahvuus'), ('counterpoints', 'Huomio')):
                    for claim in analysis[key]:
                        st.markdown(f'**{label}**')
                        show_claim(claim, packet, label.lower())
            groups = (('questions', 'Seuraavaksi selvitettävä'),)
        else:
            groups = (('strengths', 'Vahvuus'), ('counterpoints', 'Huomio'), ('questions', 'Seuraavaksi selvitettävä'))
        for key, label in groups:
            st.markdown(f'**{label}**')
            for claim in analysis[key]:
                show_claim(claim, packet, label.lower())
        with st.expander('Tarkat luvut ja neljän tilikauden kehitys'):
            fields = {'revenue': ('Liikevaihto, mrd. USD', 1e9), 'operating_margin_pct': ('Liikevoittomarginaali, %', 1),
                      'net_margin_pct': ('Nettomarginaali, %', 1), 'operating_cash_flow': ('Liiketoiminnan rahavirta, mrd. USD', 1e9),
                      'capex': ('Investoinnit, mrd. USD', 1e9), 'free_cash_flow': ('Vapaa kassavirta, mrd. USD', 1e9),
                      'current_ratio': ('Maksuvalmiussuhde, kerroin', 1)}
            rows = [{'Tilikausi': item['fiscal_year'], **{label: item.get(key)/scale if item.get(key) is not None else None
                     for key, (label, scale) in fields.items()}} for item in packet['periods']]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', key='analysis_history_table')
            st.caption('Puuttuva tieto säilyy puuttuvana. Maksuvalmius kuvaa vain lyhytaikaisia varoja ja velkoja, ei koko velkariskiä.')
    else:
        st.write(analysis['summary'])
        for group, label in (('explanations', 'Raportista löytyvät selitykset'), ('counterpoints', 'Tulkinnan rajat'), ('questions', 'Avoimet kysymykset')):
            st.markdown(f'**{label}**')
            for index, claim in enumerate(analysis[group], 1):
                show_claim(claim, packet, f'{label.lower()} {index}')
        with st.expander('Pythonin laskemat havainnot'):
            for claim in analysis['observations']:
                show_claim(claim, packet, 'laskelma')
    with st.expander('Analyysin tiedot ja tarkistusmerkinnät'):
        st.write(f"Malli {saved['model']} · laadittu {saved['created_at']} · arvioitu API-kulu {saved['cost_estimate_usd']:.4f} USD.")
        st.caption(f"{saved['input_tokens']} syöte- ja {saved['output_tokens']} vastaustokenia.")
        if saved.get('citation_repairs'):
            st.caption('Osa lähdeviitteistä kohdistettiin automaattisesti siihen annettuun raporttiotteeseen, josta lainaus löytyi sanatarkasti. Laskelmiin viitataan niiden omilla tunnisteilla.')
            st.json(saved['citation_repairs'])
        if saved.get('reviewed_by'):
            st.caption(f"Sisältö tarkistettu: {saved['reviewed_by']} · {saved['reviewed_at']}.")
            for note in saved.get('review_notes', []):
                st.write(note)
    with st.expander('Mallille annetut raporttiotteet'):
        for excerpt in annual['excerpts']:
            st.markdown(f"**{excerpt['id']}**")
            st.text(excerpt['text'])
    st.download_button('Lataa analyysi lähteineen (JSON)', data=json.dumps({**saved, 'evidence_packet': packet}, ensure_ascii=False, indent=2),
                       file_name=f"{packet['ticker']}_{packet['fiscal_year']}_analyysi.json", mime='application/json',
                       key=f"download_analysis_{packet['ticker']}_{packet['fiscal_year']}")


def legacy_analysis(company, report, row):
    if any((company.data_dir / 'analyses').glob(f"brief_{row['fiscal_year']}_*.json")):
        annual = load_annual_report(company, row, scope='company')
        if annual:
            packet = make_company_packet(report, row, annual)
            saved = load_analysis(company, packet, version=1)
            if saved:
                return saved, packet, annual
    if not any((company.data_dir / 'analyses').glob(f"{row['fiscal_year']}_*.json")):
        return None
    annual = load_annual_report(company, row)
    if annual is None:
        return None
    packet = cash_analysis.make_packet(report, row, annual)
    saved = cash_analysis.load_analysis(company, packet)
    return (saved, packet, annual) if saved else None


def show_analysis(company, report, row, contact):
    st.subheader(f'{company.name}: tekoälyn yhtiöanalyysi')
    st.write('Selkokielinen nykytilan tulkinta, tulevaisuusarvio oletuksineen ja sijoittajan kokonaisarvio vastaperusteluineen. Tarkat laskelmat ja opastukset avautuvat erikseen.')
    st.caption('Luvut kuvaavat julkaistuja tilikausia. Päivätty kurssi on erillinen aineisto; kokonaisvelat, osavuosiluvut ja kaikki uusimmat tapahtumat eivät sisälly arvioon.')
    try:
        annual = load_annual_report(company, row, scope='company')
        if annual is None:
            st.info('Lataa ensin virallinen vuosiraportti. Maksuton SEC-haku käyttää yläosan aineistohaussa annettua yhteyssähköpostia.')
            if st.button('Lataa analyysin vuosiraportti', key=f"annual_{company.ticker}_{row['fiscal_year']}"):
                agent = f'InvestaiLearning {contact.strip()}' if contact.strip() else os.getenv('SEC_USER_AGENT', '')
                with st.spinner('Ladataan virallista vuosiraporttia…'):
                    annual = prepare_annual_report(company, row, agent, scope='company')
            if annual is None:
                return
        st.markdown(f"[Avaa analyysin vuosiraportti]({annual['url']}) · julkaistu {annual['filed']}")
        assumptions, quote = show_investor_controls(company, report, row)
        packet = make_packet(report, row, annual, assumptions=assumptions, quote=quote)
        saved = load_analysis(company, packet)
        if saved is None:
            estimate = estimate_cost(packet)
            st.write(f"Malli {MODEL} · yhden analyysin varovainen kustannusarvio enintään noin {estimate['upper_estimate_usd']:.3f} USD.")
            st.caption(f'Vain valitun yhtiön julkiset luvut ja raporttiotteet lähetetään OpenAI:lle. Uusi maksullinen kutsu tehdään vain painikkeesta. [Hinnoittelu, tarkistettu 17.11.2026]({PRICING_URL})')
            key = get_openai_api_key()
            if not key:
                st.info('OpenAI-avain puuttuu palvelimen asetuksista. Aseta OPENAI_API_KEY ympäristömuuttujaan tai Streamlitin salaisuuksiin.')
            if st.button('Laadi tekoälyanalyysi', disabled=not bool(key), type='primary', key=f"analyze_{company.ticker}_{row['fiscal_year']}"):
                with st.spinner('Tulkitaan yhtiön kehitystä ja tarkistetaan lähteitä…'):
                    saved = generate_analysis(company, packet, key)
            if saved is None:
                previous = legacy_analysis(company, report, row)
                if previous:
                    st.info('Alla on aiempi analyysiversio. Yllä olevasta painikkeesta saat opastukset, tulevaisuusarvion ja sijoittajan johtopäätöksen nykyisellä aineistolla ja oletuksilla.')
                    show_saved(*previous)
                return
        show_saved(saved, packet, annual)
    except (SecDataError, OSError) as exc:
        st.error(str(exc))
        if isinstance(exc, AnalysisGenerationError):
            with st.expander('Virheen tekniset tiedot'):
                st.json(exc.details)
