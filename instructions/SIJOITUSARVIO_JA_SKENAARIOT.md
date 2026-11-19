# Selkokielinen sijoitusarvio ja skenaariot

19.11.2026 | Yhtiöanalyysin käyttöohje

## Avaa analyysi

Päivitä paikallinen sovellus ja valitse alusta **Valmiit tekoälyanalyysit → Avaa Amazon** tai **Avaa NVIDIA**. Uusi versio sisältää nykytilan tulkinnan, tulevaisuusarvion oletuksineen, sijoittajan kokonaisarvion, vastaperustelun ja seurattavan kysymyksen. Näkyvä pääteksti on enintään 260 sanaa. Lähteet, opastukset ja lisähuomiot avautuvat erikseen.

Sama toiminto on kaikkien viiden yhtiön **6 Tekoälyanalyysi** -välilehdellä. Tallennetun tuloksen lukeminen on maksutonta. Uusi OpenAI-kutsu tehdään vain Laadi tekoälyanalyysi -painikkeesta; hinta tai oletusten muuttaminen ei itsessään tee kutsua.

## Tunnusluvut selkokielellä

Avattava osio selittää kasvun, kannattavuuden, kassavirran ja maksuvalmiuden. Opastukset sisältävät määritelmän, valitun tilikauden luvuista lasketun esimerkin ja tulkinnan rajoitteen. Esimerkiksi marginaali suhteutetaan sadan dollarin myyntiin. Marginaalin muutos näytetään prosenttiyksikköinä. Negatiivinen vapaa kassavirta tarkoittaa rahankäyttöä, eikä puuttuvaa lukua korvata nollalla.

Python laskee luvut. Tekoäly yhdistää havainnot ja raporttiotteet tulkinnaksi. Tulevaisuusarvio merkitään oletuksiin perustuvaksi, sijoitusjohtopäätös tulkinnaksi. Tekoäly ei käsittele yhtiöiden tekoälyosaamista omana teemanaan tässä versiossa.

## Päivätty hinta

Avaa **Hinta ja tulevaisuusskenaariot → Hae tai päivitä kurssitiedot**. Haku käyttää Yahoo Financea yfinance-kirjaston kautta. Kurssi, kaupankäynnin aikaleima, hakuaika ja lähdelinkki tallentuvat paikallisesti. Haku ei tarvitse uutta API-avainta.

Markkina-arvo lasketaan kurssi × lähteen osakemäärä. Osakemäärän tarkkaa päiväystä lähde ei toimita. Arvostuskonteksti kelpaa vain uusimmalle tallennetulle tilikaudelle ja enintään seitsemän päivää vanhalle kurssille. Tulevaisuuteen päivättyä tai tilikauden päättymistä vanhempaa kurssia ei käytetä. Epäonnistunut haku ei korvaa aiempaa aineistoa.

Arvostus näyttää:

- Markkina-arvo / valitun täyden tilikauden emoyhtiön nettotulos. Kerroin jätetään puuttuvaksi, jos nettotulos puuttuu, on nolla tai negatiivinen.
- Valitun täyden tilikauden vapaa kassavirta / markkina-arvo. Negatiivinen FCF-tuotto säilytetään negatiivisena.

Nämä eivät ole rullaavan 12 kuukauden tunnuslukuja tai analyytikkojen ennusteita. Nykyinen hinta voi sisältää tietoa, jota vanha vuositilinpäätös ei kata. Kokonaisvelat, osavuositiedot ja kaikki uusimmat tapahtumat eivät sisälly analyysiin. Taseen arvio rajoittuu lyhytaikaisiin varoihin ja velkoihin.

Jos kelvollinen hinta puuttuu, analyysi voi tulkita liiketoimintaa ja tulevaisuuden ehtoja mutta ilmoittaa, ettei hinnan houkuttelevuutta voi arvioida. Se ei saa merkitä hintatietoa saatavilla olevaksi.

## Muokattavat skenaariot

Kolme tapausta ovat **Varovaisempi oletus**, **Jatkuvuusoletus** ja **Vahvempi oletus**. Ne kuvaavat valittua tilikautta seuraavaa tilikautta. Historiallisen vuoden valinnalla ne ovat kyseiseen vanhaan vuoteen perustuvia laskentaharjoituksia; myöhempien vuosien toteutuneita lukuja ei syötetä mukaan.

Alkuarvot ovat herkkyyslaskelma viimeisimmän vuoden kasvun ja marginaalien ympärillä: kasvu ±5 ja marginaalit ±2 prosenttiyksikköä, investointiosuus vastakkaiseen suuntaan. Käytetyt prosentit näkyvät lomakkeessa. Ne eivät ole todennäköisyyksiä tai yhtiökohtaisia analyytikkoennusteita. Rajat näkyvät syöttökentissä. Puuttuva lähtömarginaali säilyy puuttuvana, kunnes käyttäjä antaa oletuksen.

Voit muuttaa liikevaihdon kasvua, nettomarginaalia, liiketoiminnan rahavirran osuutta myynnistä ja investointien osuutta myynnistä. **Tallenna skenaario-oletukset** tallentaa ne paikallisesti. Oletukset palautuvat myöhemmin samalle yhtiölle ja tilikaudelle. Jos lähdeaineiston tarkistussumma muuttuu, vanhoja oletuksia ei siirretä siihen hiljaisesti.

Python laskee:

1. Skenaarion liikevaihto = lähtövuoden liikevaihto × (1 + kasvu / 100).
2. Nettotulos = skenaarion liikevaihto × nettomarginaali / 100.
3. Liiketoiminnan rahavirta ja investoinnit = skenaarion liikevaihto × niiden oma oletusprosentti / 100.
4. Vapaa kassavirta = liiketoiminnan rahavirta − investoinnit.
5. Päivätty markkina-arvo / skenaarion positiivinen nettotulos.

Viimeinen kerroin käyttää **muuttumatonta päivättyä markkina-arvoa**. Se auttaa pohtimaan hinnan vaatimuksia, mutta ei ennusta osakkeen hintaa, tavoitehintaa tai tuottoa. Jatkuvuusoletusta ei nimetä todennäköisimmäksi kehitykseksi. Tekoälyn tulevaisuuspäättely on ehdollista ja lähteistettyä, ei lupaus toteutumisesta.

## Lähteet ja vanhat analyysit

Uusi analyysiversio sisältää oman aineisto- ja oletustunnisteensa. Samat tiedot ja oletukset käyttävät tallennettua tulosta; muutetulla hinnalla tai oletuksilla uuden analyysin voi pyytää erikseen. Valmiiden analyysien lista suosii saman yhtiön ja vuoden uudempaa versiota. Vanhat kassavirta- ja yhtiöanalyysit säilyvät tiedostoina ja ovat varalla, jos uutta versiota ei vielä ole.

Raporttilainauksen pitää löytyä sanatarkasti annetusta aineistosta. Jos malli sijoittaa oikean lainauksen väärään raporttiotteeseen, viite voidaan kohdistaa automaattisesti siihen annettuun otteeseen, josta lainaus löytyy. Korjaukset näkyvät tarkistusmerkinnöissä. Keksittyjä lainauksia ei korjata keksimällä lähdettä. Skenaarioihin viitataan oletuksina ja Python-laskelmina, ei raportin toteutuneina lukuina.

## Tarkistukset

```powershell
& .\.venv-run\Scripts\python.exe -m unittest tests.test_sec_microsoft tests.test_sec_companies tests.test_company_analysis tests.test_company_brief tests.test_investor_inputs tests.test_analysis_errors tests.test_learning_page -v
```

Tarkistukset kattavat muun muassa negatiivisen kassavirran ja nettotuloksen, puuttuvat ja virheelliset oletukset, kurssin päiväyksen ja yhtiön, oletusten tallennuksen, hinnan tilan lähdevaatimukset, virheelliset raporttiviitteet sekä sen, ettei oletusten muokkaaminen tai analyysin lukeminen tee maksullista kutsua.
