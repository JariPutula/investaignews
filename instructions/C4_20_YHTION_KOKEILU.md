# C4: 20 yhtiön käyttökokeilu

20.11.2026 | Paikallinen yhtiöanalyysin oppimisnäkymä

## Mitä seuraavaksi tehdään

Yhtiöjoukko on laajennettu viidestä kahteenkymmeneen. Tarkoitus on kokeilla nykyisiä kasvu-, kannattavuus-, kassavirta- ja arvostusnäkymiä erilaisilla liiketoiminnoilla ennen uusien tunnuslukujen lisäämistä. Tämä ei ole kattava markkinaseula tai ostolista. Pankit, vakuutusyhtiöt ja Eurooppa eivät kuulu tähän vaiheeseen.

1. Käynnistä sovellus alla olevalla komennolla tai päivitä jo käynnissä oleva näkymä.
2. Valitse **Yhtiö**-valikosta tuttu yritys. Kaikille 20 yhtiölle on tallennettu neljän tilikauden aineisto paikallisesti.
3. Avaa **5 Yhtiövertailu** ja rajaa liiketoiminta-alue. Kokeile esimerkiksi **Kauppa → Walmart ja Costco** tai **Viestintäpalvelut → Alphabet ja Meta Platforms**. Vertailu näyttää yhtiöt riveinä, ja numerosarakkeet voi järjestää otsikosta.
4. Katso myynnin kehitystä, liikevoittomarginaalia ja investointien jälkeistä kassavirtaa. Tarkista todelliset tilikaudet ja avaa tarvittaessa selitys. Liiketoiminta-alue on karkea ryhmä, ei lupaus samanlaisesta liiketoimintamallista.
5. Kerro, missä yhtiössä tulkinta auttoi ja mikä kysymys jäi avoimeksi. Sen perusteella päätetään, tarvitsemmeko esimerkiksi tarkempaa velkatietoa, toimialakohtaisia lukuja tai uusia raporttiotteita.

```powershell
& .\.venv-run\Scripts\python.exe -m streamlit run pages/01_Yhtioanalyysi.py
```

Sivun alun **Yhtiöluettelo ja aineiston kattavuus** kertoo tilikaudet ja puuttuvat lähtöluvut. Jos käytät uutta asennusta, avaa **Hae tai päivitä aineisto**, anna SEC-haun yhteyssähköposti ja paina **Hae puuttuvien yhtiöiden vuosiluvut**. Toiminto käyttää aiempaa paikallista aineistoa ja hakee vain puuttuvat yhteenvedot. Yhden yhtiön epäonnistuminen ei keskeytä muita. Puuttuvia tunnuslukuja sisältävän, muuten kelvollisen yhteenvedon uusiminen tehdään yhtiön omalla päivityspainikkeella.

Talouslukujen lukeminen, vertailu ja yhtiön vaihtaminen eivät tee verkkohakuja tai maksullisia tekoälykutsuja. Uusien yhtiöiden vuosiraportit ovat myös paikallisesti saatavilla. Uusi tulkinta laaditaan erikseen **6 Tekoälyanalyysi → Laadi tekoälyanalyysi** -painikkeesta. Kurssit haetaan edelleen erillisestä painikkeesta. Tässä laajennuksessa ei laadittu uusia maksullisia analyysejä.

## Päivätty yhtiöjoukko

| Liiketoiminta-alue | Yhtiöt ja tunnukset |
| --- | --- |
| Teknologia | Microsoft MSFT, NVIDIA NVDA, Apple AAPL |
| Viestintäpalvelut | Alphabet GOOGL, Meta Platforms META |
| Kauppa | Amazon AMZN, Walmart WMT, Costco COST |
| Päivittäistavarat | Coca-Cola KO, Procter & Gamble PG |
| Autot | Tesla TSLA |
| Maksupalvelut | Visa V, Mastercard MA |
| Lääkkeet | Eli Lilly LLY, AbbVie ABBV |
| Energia | ConocoPhillips COP, Chevron CVX |
| Teollisuus | Caterpillar CAT |
| Ravintolat | McDonald's MCD |
| Sähkö ja energia | NextEra Energy NEE |

Yhtiötunnisteet tarkistettiin [SEC:n julkisesta tunnisteluettelosta](https://www.sec.gov/files/company_tickers.json). Luettelo on käsin rajattu kokeilujoukko päivälle 20.11.2026; tunnisteet ja yhtiömuutokset tarkistetaan uudelleen seuraavassa laajennuksessa. Alphabetista mukana on vain yksi osakesarja.

## Aineiston rajaukset

Microsoftin, NVIDIAn, P&G:n, Applen, Visan, Walmartin ja Costcon aineisto sisältää tilikausien päättymisvuodet 2023–2026. Muilla mukana ovat 2022–2025. Neljän tilikauden jaksot eivät tarkoita samoja kalenterikuukausia. Apple käyttää syyskuun viimeistä lauantaita, Costco elokuun loppua lähintä sunnuntaita ja NVIDIA tammikuun viimeistä sunnuntaita. Myös 53 viikon tilikaudet käsitellään todellisilla alku- ja loppupäivillä. Osavuosia ei sekoiteta vuosilukuihin.

Yhtiökohtaiset valinnat ja puutteet ovat näkyvissä:

- **Walmart:** liikevaihto sisältää myös jäsenmaksut ja muut tuotot. Pelkkä Net sales -rivi ei ole koko liikevaihto.
- **ConocoPhillips:** liikevaihtona käytetään myyntiä ja muita liiketoiminnan tuottoja. Pelkkä asiakassopimusten myynti olisi tätä suppeampi. Tuettu investointirivi ja liikevoitto puuttuvat, joten FCF:ää ja liikevoittomarginaalia ei lasketa.
- **NextEra:** käytetään säännellyn ja sääntelemättömän toiminnan kokonaisliikevaihtoa. Useaan yhtiökohtaiseen investointiriviin jakautuvaa menoa ei vielä yhdistetä, joten FCF ja investointiaste puuttuvat.
- **Visa:** investointirivi sisältää käyttöomaisuuden lisäksi teknologiaa. **Chevron:** käytetään rahavirtalaskelman Capital expenditures -riviä. Koko investointien rahavirtaa ei käytetä kummankaan korvikkeena.
- **Eli Lilly ja Chevron:** liikevoitto puuttuu tuetuista kentistä. Tulosta ennen veroja ei nimetä liikevoitoksi.
- **Mastercard:** tuettu emoyhtiön nettotulos puuttuu. Konsernin vähemmistöosuudet sisältävää nettotulosta ei käytetä automaattisena korvikkeena, joten myös siihen perustuva arvostuskerroin puuttuu. Aineellisiin investointeihin perustuva FCF ei vähennä erillisiä ohjelmistoinvestointeja.
- **ConocoPhillips, Chevron ja Caterpillar:** osittaista kustannusriviä ei käytetä koko konsernin bruttomarginaaliin. Myös eräiden palveluyhtiöiden bruttomarginaali jää puuttuvaksi.
- **Caterpillar:** luvut sisältävät rahoitusliiketoiminnan. Raportin emoyhtiön kantaosakkeille kuuluvaa Profit-lukua tuetaan nettotuloksen lähteenä.
- **McDonald's:** franchising-maksut ja vuokrat sekä omien ravintoloiden myynti vaikuttavat marginaalien vertailuun.

Puuttuva tieto ei ole nolla eikä automaattisesti huonon liiketoiminnan merkki. Koko velkariskiä, lääkekehitystä tai energiahintojen herkkyyttä ei voi arvioida näillä yhteisillä luvuilla yksin. Näistä mahdollisesti tarvittavat lisätiedot valitaan käyttökokeilun perusteella.

Hintojen, skenaarioiden ja tekoälytulkinnan ohje: [SIJOITUSARVIO_JA_SKENAARIOT.md](SIJOITUSARVIO_JA_SKENAARIOT.md).

## Tarkistus

Kaikkien 20 yhtiön neljälle tilikaudelle valitut **588 saatavilla olevaa lähtölukua** täsmäytettiin **60 alkuperäisen vuosiraportin** näkyviin inline XBRL -havaintoihin: tietokenttä, ajanjakso, konsernitason konteksti ja arvo. Täsmäytys ei tarkoita kaikkien liiketoiminnan tulkintojen tai yhtiöiden välisen vertailukelpoisuuden todistamista.

Yhtiökohtaiset lähde- ja tilikausivalinnat testataan erikseen. Käynnissä olevassa sovelluksessa tarkistettiin jokaisen 20 yhtiön avaaminen, kattavuusluettelo sekä kaupan, maksupalvelujen, energian ja yhden yhtiön ryhmän rajaukset. Pelkkä lukeminen ei tee maksullista kutsua. Tyyppitarkistus meni läpi.

```powershell
& .\.venv-run\Scripts\python.exe -m unittest tests.test_sec_microsoft tests.test_sec_companies tests.test_company_analysis tests.test_company_brief tests.test_investor_inputs tests.test_analysis_errors tests.test_pilot tests.test_learning_page -v
```

Alkuperäisaineisto, raportit ja yhteenvedot ovat paikallisessa `data/research/`-kansiossa, jota ei versionhallita. Yhteyssähköpostia ei tallenneta koodiin tai aineistoon.

Kalenterien alkuperäislähteet: [Applen sijoittaja-FAQ](https://investor.apple.com/faq/default.aspx), [Costcon vuosiraportti 2026](https://www.sec.gov/Archives/edgar/data/909832/000090983226000093/cost-20260830.htm).
