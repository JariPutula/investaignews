"""Plain-language definitions and examples calculated from the selected year."""


def _pct(value):
    return f"{value:.1f}".replace(".", ",") if value is not None else "ei saatavilla"


def metric_lessons(row, previous=None):
    growth, margin, fcf_margin, current = (row.get(key) for key in ("revenue_growth_pct", "operating_margin_pct", "fcf_margin_pct", "current_ratio"))
    change = None if previous is None or previous.get("operating_margin_pct") is None or margin is None else margin - previous["operating_margin_pct"]
    return {
        "growth": {
            "meaning": "Liikevaihto on yrityksen myynti. Kasvuprosentti vertaa myyntiä edelliseen tilikauteen.",
            "example": f"Jos edellisen vuoden myynti oli 100 dollaria, sama kasvuvauhti tekisi siitä {_pct(100+growth)} dollaria." if growth is not None else "Vertailuvuoden lukua ei ole tai kasvuprosenttia ei voi mielekkäästi laskea.",
            "limit": "Myynti voi muuttua myyntimäärien, hintojen, valuuttojen tai yritysostojen vuoksi. Kasvu ei yksin tarkoita parempaa tulosta.",
        },
        "profitability": {
            "meaning": "Liikevoittomarginaali kertoo, paljonko myynnistä jää liiketoiminnan tulokseksi ennen rahoituseriä ja veroja.",
            "example": f"Sadasta myyntidollarista jäi {_pct(margin)} dollaria liikevoittoa." if margin is not None else "Liikevoittomarginaali puuttuu.",
            "change": f"Muutos edellisestä vuodesta on {_pct(change)} prosenttiyksikköä. Prosenttiyksikkö kuvaa kahden prosenttiluvun erotusta." if change is not None else "Edellisvuoden marginaalimuutosta ei voi laskea.",
            "limit": "Marginaalia kannattaa verrata yhtiön omaan historiaan. Eri toimialojen luvut eivät ole suora paremmuusjärjestys.",
        },
        "cash_flow": {
            "meaning": "Vapaa kassavirta on liiketoiminnan rahavirta vähennettynä tässä käytetyillä käyttöomaisuusinvestoinneilla. Se ei ole sama asia kuin tilikauden voitto.",
            "example": f"Sadasta myyntidollarista jäi {_pct(fcf_margin)} dollaria investointien jälkeistä rahavirtaa." if fcf_margin is not None and fcf_margin >= 0 else
                       f"Investointien jälkeinen rahavirta oli negatiivinen: sataa myyntidollaria kohti käytettiin {_pct(-fcf_margin)} dollaria enemmän kuin liiketoiminta tuotti." if fcf_margin is not None else "Vapaan kassavirran marginaali puuttuu.",
            "limit": "Kasvuinvestoinnit voivat pienentää tämän vuoden lukua. Yritysostot, leasing ja yhtiökohtainen investointirajaus pitää tutkia erikseen.",
        },
        "balance": {
            "meaning": "Maksuvalmiussuhde vertaa lyhytaikaisia varoja lyhytaikaisiin velkoihin tilinpäätöspäivänä.",
            "example": f"Yhtä lyhytaikaista velkadollaria kohti oli {_pct(current)} dollaria lyhytaikaisia varoja." if current is not None else "Maksuvalmiuden suhdeluku puuttuu.",
            "limit": "Kaikki varat eivät ole käteistä. Tämä ei kata kokonaisvelkoja eikä yksin osoita, pystyykö yritys maksamaan velkansa.",
        },
    }
