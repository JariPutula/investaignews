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

Yhteinen näkymä sisältää Microsoftin ja Coca-Colan, neljän tilikauden kehityksen sekä rinnakkaisen vertailun. Peruslaskelmien lisäksi mukana ovat brutto- ja nettomarginaali, vapaan kassavirran marginaali, investointiaste ja maksuvalmiussuhde. Tietoa-painikkeet avaavat kaavat, tulkinnan ja rajoitteet. SEC-haku on maksuton eikä tarvitse API-avainta.

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

Ajantasainen käyttöohje: [instructions/C2_YHTIOVERTAILU.md](instructions/C2_YHTIOVERTAILU.md). Ensimmäisen harjoituksen perusteet: [instructions/C1_DATAKOE.md](instructions/C1_DATAKOE.md).
