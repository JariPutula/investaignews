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

## Microsoftin talouslukujen oppimisnäkymä

Ensimmäinen C-harjoitus hakee Microsoftin vuosiluvut ohjelmallisesti SEC:stä ja näyttää liikevaihdon kasvun, liikevoittomarginaalin sekä vapaan kassavirran laskelmineen. SEC-haku on maksuton eikä tarvitse API-avainta.

Käynnistä projektikansiossa PowerShellissä:

```powershell
& .\.venv-run\Scripts\python.exe -m streamlit run pages/01_Microsoft.py
```

Sivu löytyy myös alkuperäisen `app.py`-sovelluksen Streamlit-sivuvalikosta. Tallennettuja lukuja voi käyttää ilman uutta verkkohakua. Jos aineistoa ei ole tai haluat päivittää sen, avaa näkymän **Hae tai päivitä aineisto** ja anna SEC-haun yhteyssähköposti.

Komentorivillä paikallisen aineiston saa näkyviin näin:

```powershell
& .\.venv-run\Scripts\python.exe fetch_microsoft.py --offline
```

Uuteen komentorivihakuun aseta `SEC_USER_AGENT` muotoon `InvestaiLearning oma.yhteys@example.com` ja suorita `fetch_microsoft.py --refresh`. Yhteystietoa ei tallenneta aineistoon. Raakavastaukset ja lähteistetty yhteenveto tallentuvat `data/research/msft/`-kansioon.

Oppimispolku ja ensimmäisen harjoituksen ohjeet: [instructions/C1_DATAKOE.md](instructions/C1_DATAKOE.md).
