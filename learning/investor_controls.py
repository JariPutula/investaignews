"""Local scenario edits and one explicit public quote refresh."""

import pandas as pd
import streamlit as st

from market_data.scenarios import CASES, FIELDS, load_assumptions, save_assumptions, project_scenarios
from market_data.valuation import load_valuation, fetch_valuation, valuation_context
from market_data.sec import SecDataError


def show_investor_controls(company, report, row):
    assumptions = load_assumptions(company, report, row)
    try:
        quote = load_valuation(company)
    except SecDataError as error:
        st.warning(str(error))
        quote = None
    with st.expander("Hinta ja tulevaisuusskenaariot"):
        st.markdown("**Päivätty hinta**")
        if st.button("Hae tai päivitä kurssitiedot", key=f"quote_refresh_{company.ticker}"):
            try:
                with st.spinner("Haetaan julkisia kurssitietoja…"):
                    quote = fetch_valuation(company)
            except SecDataError as error:
                st.warning(str(error))
        context = valuation_context(report, row, quote)
        if quote:
            st.write(f"Kurssi {quote['price']:.2f} USD · markkina-arvo {quote['market_cap']/1e9:,.1f} mrd. USD.")
            st.markdown(f"[Yahoo Finance: {company.ticker}]({quote['source_url']}) · kurssin aikaleima {quote['quoted_at']} · haettu {quote['retrieved_at']}.")
        if not context["available"]:
            st.info(context["reason"])
        else:
            st.write(f"Markkina-arvo / valitun tilikauden nettotulos: {context['market_cap_to_annual_net_income']:.1f} ×" if context["market_cap_to_annual_net_income"] is not None else "Tuloskerrointa ei esitetä, koska vuositulos puuttuu tai ei ole positiivinen.")
            if context["annual_fcf_yield_pct"] is not None:
                st.write(f"Valitun tilikauden vapaa kassavirta / markkina-arvo: {context['annual_fcf_yield_pct']:.2f} %.")
            st.caption(f"Tilikausi päättyy {context['financial_period_end']}; kyseessä ei ole rullaava 12 kuukauden tulos. {context['limits']}")
        st.markdown("**Jos tällainen kehitys toteutuisi**")
        st.write(f"Skenaariot koskevat tilikautta {row['fiscal_year']+1} vuoden {row['fiscal_year']} lukujen pohjalta. Ne ovat herkkyyslaskelmia, eivät toteutumisennusteita.")
        st.caption("Alkuoletukset jatkavat viimeisintä kasvua ja marginaaleja. Muissa tapauksissa kasvu muuttuu ±5 ja marginaalit ±2 prosenttiyksikköä. Rajaus on laskentaharjoitus, ei arvio todennäköisimmästä kehityksestä. Muuta oletuksia tarvittaessa.")
        with st.form(f"scenario_form_{company.ticker}_{row['fiscal_year']}"):
            edited = []
            for column, item in zip(st.columns(3), assumptions):
                with column:
                    st.markdown(f"**{CASES[item['case']]}**")
                    result = {"case": item["case"]}
                    for field, label in FIELDS.items():
                        lower = 0. if field == "capex_pct" else -90. if field == "growth_pct" else -100.
                        result[field] = st.number_input(label, min_value=lower, max_value=100.,
                            value=float(item[field]) if item.get(field) is not None else None, step=1.,
                            key=f"scenario_{company.ticker}_{row['fiscal_year']}_{item['case']}_{field}")
                    edited.append(result)
            if st.form_submit_button("Tallenna skenaario-oletukset"):
                save_assumptions(company, report, row, edited)
                assumptions = edited
                st.success("Oletukset tallennettu. Tekoälyn päivitys tehdään erikseen analyysipainikkeesta.")
        cases = project_scenarios(row, assumptions, context)
        table = [{"Oletus": CASES[item["case"]], "Myynti, mrd. USD": item["revenue"]/1e9 if item["revenue"] is not None else None,
                  "Nettotulos, mrd. USD": item["net_income"]/1e9 if item["net_income"] is not None else None,
                  "Vapaa kassavirta, mrd. USD": item["free_cash_flow"]/1e9 if item["free_cash_flow"] is not None else None,
                  "Nykyarvo / skenaarion tulos": item["market_cap_to_scenario_net_income"]} for item in cases]
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch", key="investor_scenarios_table")
        st.caption("Myynti = viimeisin myynti × (1 + kasvu). Tulos ja rahavirrat = skenaarion myynti × niiden oletettu marginaali. FCF = liiketoiminnan rahavirta − investoinnit. Tuloskerroin käyttää muuttumatonta päivättyä markkina-arvoa: se ei ole tuleva osakekurssi tai tuottoennuste.")
    return assumptions, quote
