---
title: Investaiappver2
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Investaiappver2
Tämä sovellus on julkaistu Hugging Face Spaceen Dockerin avulla.

## Yhtiöanalyysin oppimisnäkymä

**19.11.2026:** tekoälyarviossa erotetaan nykytilan havainnot, tulevaisuusoletukset ja sijoittajan tulkinta vastaperusteluineen. Pääarvio on lyhyt, tunnuslukujen selkokieliset opastukset ja laskelmat avautuvat erikseen. Mukana ovat muokattavat tilikausiskenaariot sekä käsin päivitettävä, päivätty kurssi- ja markkina-arvohaku. [Käyttöohje ja laskelmien rajaukset](instructions/SIJOITUSARVIO_JA_SKENAARIOT.md).

Yhteinen näkymä sisältää Microsoftin, Coca-Colan, Amazonin, NVIDIAn ja Procter & Gamblen neljän tilikauden luvut sekä valittavien yhtiöiden rinnakkaisen vertailun. Kassavirta-välilehti näyttää rahavirran, investointien ja vapaan kassavirran muutokset sekä rahavirran suhteen myyntiin ja nettotulokseen. Tietoa-painikkeet avaavat kaavat, tulkinnan ja rajoitteet. NVIDIAn 52/53 viikon tilikausi ja myös aineettomia hyödykkeitä sisältävä investointirivi näkyvät erillisinä rajauksina. SEC-haku on maksuton eikä tarvitse API-avainta.

**6 Tekoälyanalyysi** avaa yhtiön kasvua, kannattavuutta, kassavirtaa ja lyhyen aikavälin maksuvalmiutta. Oletusnäkymässä on tiivis kokonaiskuva, neljä lyhyttä tulkintaa, tärkein vahvuus, huomio ja jatkokysymys (enintään 260 sanaa). Neljän vuoden tarkat luvut ja lähteet avautuvat erikseen. Python laskee luvut; OpenAI tulkitsee niiden yhteyksiä ja valikoituja vuosiraporttiotteita. Aiemmat kassavirtaharjoitukset säilyvät. Raportti ladataan ja maksullinen analyysi käynnistetään erillisistä painikkeista, ja tallennetut tulokset avautuvat ilman uutta API-kutsua. Avaimena käytetään aiemman News Sentinel -toiminnon `OPENAI_API_KEY`-asetusta; yhtiöanalyysi ei muuta uutisanalyysin mallia.

Käynnistä projektikansiossa PowerShellissä:

```powershell
& .\.venv-run\Scripts\python.exe -m streamlit run pages/01_Yhtioanalyysi.py
```

Sivu löytyy myös alkuperäisen `app.py`-sovelluksen Streamlit-sivuvalikosta nimellä Yhtioanalyysi. Aiempi `pages/01_Microsoft.py` on korvattu tällä yhteisellä sivulla; päivitä suora käynnistyskomento. Tallennettuja lukuja voi käyttää ilman uutta verkkohakua. Päivittäminen tapahtuu valitulle yhtiölle osiossa **Hae tai päivitä aineisto**.

Komentorivillä paikallisen aineiston saa näkyviin näin:

```powershell
& .\.venv-run\Scripts\python.exe fetch_company.py --company MSFT --offline
& .\.venv-run\Scripts\python.exe fetch_company.py --company KO --offline
```

Uuteen komentorivihakuun aseta `SEC_USER_AGENT` muotoon `InvestaiLearning oma.yhteys@example.com` ja suorita `fetch_company.py --company KO --refresh` (tai MSFT). Yhteystietoa ei tallenneta aineistoon. Aineisto tallentuu yhtiöittäin `data/research/msft/`- ja `data/research/ko/`-kansioihin. Alkuperäinen `fetch_microsoft.py` toimii edelleen Microsoftin komentorivityökaluna.

Ajantasaiset käyttöohjeet: [yhtiövertailu](instructions/C2_YHTIOVERTAILU.md) ja [kassavirta ja tekoälyanalyysi](instructions/C3_KASSAVIRTA_JA_TEKOALY.md). Ensimmäisen harjoituksen perusteet: [instructions/C1_DATAKOE.md](instructions/C1_DATAKOE.md).
