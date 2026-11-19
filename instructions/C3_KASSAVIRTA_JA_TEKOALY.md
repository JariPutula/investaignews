# C3 Kassavirta, viisi yhtiötä ja tekoälyanalyysi

17.11.2026 | Käyttöohje

**Uusi versio 19.11.2026:** mukana ovat nyt avattavat selkokieliset opastukset, nykytilan tulkinta, tulevaisuusarvio, muokattavat skenaariot, päivätty hinnan tarkastelu ja sijoittajan arvio vastaperusteluineen. Ajantasainen ohje: [SIJOITUSARVIO_JA_SKENAARIOT.md](SIJOITUSARVIO_JA_SKENAARIOT.md). Alla kuvataan aiemmat harjoitukset ja niiden laskentapohja.

**Tiivis yhtiöanalyysi 18.11.2026:** tekoäly avaa nyt kasvua, kannattavuutta, kassavirtaa ja maksuvalmiutta koko valittuun vuoteen päättyvän lukuhistorian avulla. Näkymässä on kokonaiskuva, neljä lyhyttä tulkintaa, tärkein vahvuus, huomio ja jatkokysymys. Koko pääteksti rajataan enintään 260 sanaan. Tarkat luvut, aineiston rajaukset ja lähteet avataan erikseen. Uudet Amazonin ja NVIDIAn esimerkit ovat tallennettuina; vanhat kassavirtaharjoitukset säilyvät erillisinä tiedostoina ja näkyvät, jos uutta yhtiöanalyysiä ei vielä ole.

**Päivitys 18.11.2026:** sivun alussa on Valmiit tekoälyanalyysit -osio. Sen Avaa-painike valitsee oikean yhtiön, tilikauden ja Tekoälyanalyysi-välilehden ilman maksullista kutsua. Tallennettuna ovat Amazonin tilikausi 2025 ja NVIDIAn tilikausi 2026. NVIDIAn uudelleen tarkistettu analyysi valmistui; aiemman yleisvirheen tarkkaa syytä ei voitu jälkikäteen varmistaa.

API-virheet erotellaan nyt aikakatkaisuihin, yhteysvirheisiin, käyttövaraan/kutsurajaan, avaimen tai mallin käyttöoikeuteen ja katkenneeseen tai virheelliseen vastaukseen. Virheen tekniset tiedot näkyvät avattavassa osiossa. Sama peitetty diagnostiikka tallentuu yhtiön analyses/errors-kansioon; siihen ei kirjoiteta API-avainta, palvelun raakavirheviestiä tai pyyntö-/vastausrunkoja. Automaattista uutta maksullista kutsua ei tehdä.

## Käynnistä

```powershell
& .\.venv-run\Scripts\python.exe -m streamlit run pages/01_Yhtioanalyysi.py
```

Tavallinen paikallinen osoite on http://127.0.0.1:8501. Jos sovellus on jo auki, päivitä sivu. Alkuperäinen app.py toimii edelleen erillisenä salkkusovelluksena, jonka sivuvalikosta sama yhtiöanalyysi löytyy.

## Viiden yhtiön aineisto

| Yhtiö | Tunnus | Tallennetut tilikaudet | Tilikauden päättyminen |
| --- | --- | --- | --- |
| Microsoft | MSFT | 2023–2026 | 30.6. |
| Coca-Cola | KO | 2022–2025 | 31.12. |
| Amazon | AMZN | 2022–2025 | 31.12. |
| NVIDIA | NVDA | 2023–2026 | Tammikuun viimeinen sunnuntai |
| Procter & Gamble | PG | 2023–2026 | 30.6. |

Uusien kolmen yhtiön aineisto haettiin 17.11.2026. Niiden kaikkien neljän tilikauden 96 lähtölukua täsmäytettiin yhdeksän virallisen vuosiraportin näkyviin inline XBRL -lukuihin. Alkuperäinen SEC-JSON, tarkistussumma, lähteet, tilikaudet ja hakuaika säilyvät yhtiön data/research-kansiossa. Ladattua aineistoa ei versionhallita.

NVIDIAn tilikauden 2025 jakso on 29.1.2024–26.1.2025. Sen tilikausi 2027 kestää 53 viikkoa, mutta julkaisemattomia tai osavuosilukuja ei oteta vuosivertailuun. Yhtiövertailun todelliset ajanjaksot näkyvät aina taulukossa.

## Kassavirran harjoitus

Valitse **Amazon → 2025 → 3 Kassavirta**. Sovelluksen yhteisellä määritelmällä:

| Erä, mrd. USD | 2024 | 2025 | Muutos |
| --- | ---: | ---: | ---: |
| Liiketoiminnan rahavirta | 115,877 | 139,514 | +23,637 |
| Käyttöomaisuusinvestoinnit | 82,999 | 131,819 | +48,820 |
| Vapaa kassavirta | 32,878 | 7,695 | −25,183 |

Vapaan kassavirran muutos = rahavirran muutos − investointien muutos = 23,637 − 48,820 = −25,183 miljardia USD. Myös negatiivinen vapaa kassavirta ja negatiiviset muutokset säilyvät näkyvissä. Puuttuvia havaintoja ei korvata nollalla, eikä negatiivisen lähtöarvon päälle lasketa harhaanjohtavaa kasvuprosenttia.

Sivulla näkyvät myös liiketoiminnan rahavirta / liikevaihto sekä liiketoiminnan rahavirta / emoyhtiön nettotulos. Jälkimmäinen jätetään puuttuvaksi nettotuloksen ollessa nolla tai negatiivinen. Korkea kerroin ei yksin osoita tuloksen laatua: poistot, käyttöpääoma ja osakeperusteiset palkkiot vaikuttavat.

Muutoslaskelma kertoo laskennallisen erittelyn. Se ei vielä todista investointien liiketaloudellista syytä tai tulevaa tuottoa.

## Määritelmien erot

- Amazonin oma vapaa kassavirta vähentää investoinnit **myyntitulojen ja investointikannustimien jälkeen**. Vuoden 2025 oma raportoitu FCF on 11,194 mrd. USD, kun tämän sovelluksen bruttoinvestointeihin perustuva FCF on 7,695 mrd. USD. Ero 3,499 mrd. USD vastaa kyseisen vuoden käyttöomaisuuden myyntituloja ja kannustimia. Molemmat pitää nimetä oikein.
- NVIDIAn raportoitu investointirivi sisältää käyttöomaisuuden lisäksi aineettomia hyödykkeitä. Tätä laajempaa riviä ei esitetä pelkkinä aineellisina investointeina.
- P&G:n adjusted free cash flow voi sisältää yhtiön omia oikaisuja. Sovellus näyttää raportoidun liiketoiminnan rahavirran vähennettynä investointien rahamenoilla.
- Coca-Colan aiemmat IRS- ja fairlife-huomiot säilyvät. Maksuja ei oikaista automaattisesti pois.

## Tekoälyanalyysi

Valitse yhtiö ja tilikausi, sitten **6 Tekoälyanalyysi**.

1. Jos vuosiraporttia ei ole tallennettu, anna yhteyssähköposti yläosan Hae tai päivitä aineisto -osiossa ja paina **Lataa analyysin vuosiraportti**. Raportti haetaan juuri valittujen lukujen liikevaihdon lähderaportin tunnisteella. Myöhemmin julkaistu raportti voi sisältää vanhan vuoden vertailuluvut; julkaisupäivä näytetään.
2. Tarkista raporttilinkki, malli, aineiston rajaus ja kustannusarvio.
3. Paina **Laadi tekoälyanalyysi**. Tämä on yksi maksullinen OpenAI-kutsu. Automaattista uusintaa ei tehdä.
4. Lue kokonaiskuva ja neljä aihetta. Lopussa ovat yksi vahvuus, huomio ja jatkokysymys. Avaa tarvittaessa kohdan lähteet tai Tarkat luvut ja neljän tilikauden kehitys -osio.
5. Tallennettu tulos avautuu seuraavilla kerroilla ilman uutta kutsua. JSON-lataus sisältää analyysin ja sen lähdepaketin.

Numeeriset havainnot, summat, suhdeluvut ja kasvulaskelmat muodostetaan Pythonilla. Malli tulkitsee valittuun vuoteen päättyviä lukuja ja enintään 24 otetta yhdestä virallisesta vuosiraportista. Otteet valitaan tasapainoisesti myynnin, kannattavuuden, rahavirtojen ja maksuvalmiuden aiheista. Raportti voi olla myöhemmältä vuodelta, jos valitut luvut on julkaistu siinä uudelleen; myöhempi suunnitelma ei selitä vanhan vuoden toteutunutta muutosta. Maksuvalmius perustuu vain lyhytaikaisiin varoihin ja velkoihin. Kokonaisvelat ja omavaraisuusaste eivät ole aineistossa. Osakkeen hintaa, arvostusta, tavoitehintaa tai uutisia ei käsitellä.

Uudet tiiviit yhtiöanalyysit tallentuvat brief_-alkuisiin tiedostoihin. Tallennettujen analyysien valikko suosii saman yhtiön ja tilikauden uutta yhtiöanalyysiä; vanhaa kassavirta-analyysiä ei poisteta. Valitun vanhan tilikauden analyysiin ei anneta myöhempien vuosien lukuja. Puuttuvat tiedot näytetään puuttuvina.

Malli on **gpt-5.4-mini**. Käytetään olemassa olevaa OPENAI_API_KEY-asetusta ympäristömuuttujasta tai Streamlitin salaisuuksista. News Sentinelin malliasetusta ei muuteta. Mallille lähetetään vain tämän yhtiön julkiset luvut ja raporttiotteet. Salkku ja SEC-yhteyssähköposti eivät kuulu pakettiin. Vastausten API-tallennus on pois käytöstä (`store=False`); tämä ei itsessään tarkoita palveluntarjoajan kaiken lokituksen poistamista.

Hinnoittelu tarkistettu 17.11.2026: 0,75 USD / miljoona syötetokenia ja 4,50 USD / miljoona vastaustokenia. Ennen kutsua näytetään varovainen arvio syötteen UTF-8-tavumäärästä ja enintään 4 000 vastaustokenista, myös mallin päättely mukaan lukien. Todellinen tokenmäärä ja kustannusarvio tallennetaan vastauksen mukana. Tämä on arvio, ei laskutustakuu.

Tulosta ei tallenneta valmiina, jos lähde-ID:t ovat tuntemattomia tai annetut raporttilainaukset eivät löydy annetusta otteesta. Tulkinnan oikeellisuutta ei voi todistaa pelkällä lainauksen tarkistuksella: tärkeät johtopäätökset pitää lukea lähdettä vasten. Hylättyjen vastausten diagnostiikka säilyy paikallisessa analyses/rejected-kansiossa eikä näy valmiina analyysinä.

## Tarkistukset

```powershell
& .\.venv-run\Scripts\python.exe -m unittest tests.test_sec_microsoft tests.test_sec_companies tests.test_company_analysis tests.test_company_brief tests.test_analysis_errors tests.test_learning_page -v
```

Testit tarkistavat muun muassa NVIDIAn 52/53 viikon jaksot, yhtiökohtaiset investointikentät, puuttuvat lähtöluvut, muutoslaskelman, viiden yhtiön valinnan, keksittyjen lähteiden ja lainausten hylkäyksen sekä sen, ettei tallennetun analyysin lukeminen tee uutta maksullista kutsua. Kehityksen ajonaikainen tarkistus käyttää vain localhostiin sidottua Streamlitin agent API:a; sitä ei kytketä julkaisun asetuksiin.

## Alkuperäislähteet

- [Amazonin vuosiraportti 2025: rahavirrat ja FCF-täsmäytys](https://www.sec.gov/Archives/edgar/data/1018724/000101872426000004/amzn-20251231.htm)
- [NVIDIAn vuosiraportti 2026: tilikausikalenteri ja investointien rajaus](https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm)
- [P&G:n vuosiraportti 2026](https://www.sec.gov/Archives/edgar/data/80424/000008042426000103/pg-20260630.htm)
- [OpenAI GPT-5.4 mini: ominaisuudet ja hinnoittelu](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
