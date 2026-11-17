# C2 Microsoftin ja Coca Colan vertailu

15.11.2026 | Yhtiöanalyysin käyttöohje

## Käynnistä ja valitse yhtiö

Yksi yhteinen sivu palvelee Microsoftia ja Coca-Colaa. Molempien alkuperäiset SEC-aineistot ja lasketut vuosiluvut on tallennettu paikallisesti. Käynnistä projektikansiossa:

```powershell
& .\.venv-run\Scripts\python.exe -m streamlit run pages/01_Yhtioanalyysi.py
```

Alkuperäisen app.py-sovelluksen sivuvalikossa sivu näkyy nimellä Yhtioanalyysi. Vanha sivutiedosto pages/01_Microsoft.py on nimetty uudelleen; se ei ole enää käynnistystiedosto. Komentorivityökalu fetch_microsoft.py säilyy yhteensopivana.

Valitse yhtiö ja tilikausi. Kolme peruslukua näkyvät ylhäällä. Muut luvut ja vertailu on jaettu viidelle välilehdelle. Tietoa-painike avaa selitysikkunan, ja Näytä laskenta avaa todelliset lähtöluvut ja lähteet. Aloita tutuista perusluvuista ja avaa lisäluvut yksi kerrallaan.

## Tunnusluvut

| Luku | Kysymys ja laskenta | Tulkinnan tärkeä rajoite |
| --- | --- | --- |
| Liikevaihdon vuosikasvu | Kuinka paljon myynti muuttui? Uusi / edellinen − 1. | Kasvu voi tulla myös hinnoista, valuutoista ja yritysostoista. |
| Liikevoittomarginaali | Paljonko myynnistä jää liiketoiminnan tulokseksi? Liikevoitto / liikevaihto. | Vertaa kehitystä ja samankaltaisia liiketoimintoja. |
| Vapaa kassavirta | Mitä rahavirrasta jää investointien jälkeen? Rahavirta − aineelliset käyttöomaisuusinvestoinnit. | Yritysostot ja leasingin kaikki rahavirrat eivät automaattisesti sisälly vähennykseen. |
| Bruttomarginaali | Mitä myynnistä jää myytyjen suoritteiden kustannusten jälkeen? (Liikevaihto − myytyjen tuotteiden ja palvelujen kustannukset) / liikevaihto. | Kulujen luokittelu ja myyntijakauma vaikuttavat vertailuun. |
| Nettomarginaali | Mitä jää emoyhtiön omistajille myös rahoituserien ja verojen jälkeen? Nettotulos / liikevaihto. | Sijoitustuotot, verot ja kertaluonteiset erät voivat muuttaa lukua. |
| Vapaan kassavirran marginaali | Kuinka paljon myyntidollarista jää vapaata kassavirtaa? Vapaa kassavirta / liikevaihto. | Tämä ei ole osakekurssiin perustuva FCF-tuotto. |
| Investointiaste | Kuinka paljon aineellisia käyttöomaisuusinvestointeja tehdään suhteessa myyntiin? Investoinnit / liikevaihto. | Tämä ei kata esimerkiksi kaikkia leasinghankintoja tai tutkimuskuluja. |
| Maksuvalmiussuhde | Kuinka suuret ovat lyhytaikaiset varat suhteessa lyhytaikaisiin velkoihin? Varat / velat. | Tilinpäätöspäivän tasekuva ei yksin ratkaise maksukykyä. |

Prosenttiluvut kerrotaan sadalla. Maksuvalmiussuhde näytetään kertoimena. Rahamäärät näytetään miljoonina dollareina; laskenta käyttää pyöristämättömiä dollarimääriä. Nollan suuruinen tai negatiivinen jakaja ei tuota harhaanjohtavaa suhdelukua. Negatiiviset tulokset ja rahavirrat säilytetään.

Kasvu-välilehdellä on myös neljän tilikauden väliseen kolmeen vuosiväliin perustuva CAGR. Kannattavuus-välilehti selittää prosenttien ja prosenttiyksikköjen eron. Kassavirta-välilehti avaa voiton ja rahavirran eroa.

## Mitä hyvä arvo tarkoittaa

Sovellus ei anna yleistä vihreä–punainen-luokitusta tai osto- ja myyntipisteitä. Korkea marginaali voi olla myönteinen havainto, mutta sen syy ja pysyvyys pitää tutkia. Korkea investointiaste voi kuvata kasvupanostusta tai raskasta ylläpitotarvetta. Current ratio alle yksi tarkoittaa taseessa lyhytaikaisten velkojen ylittävän lyhytaikaiset varat, mutta se ei yksin osoita maksukyvyttömyyttä. Varojen laatu, erääntymiset, ennakkomaksut ja rahavirta vaikuttavat tulkintaan.

Lukujen yhteydessä olevat Tietoa-ikkunat sisältävät määritelmän, kaavan, arviointiohjeen ja rajoitteen. Laskelmat perustuvat raportoituihin lukuihin. Niitä ei automaattisesti oikaista poikkeuksellisista eristä.

## Tilikausien vertailu

Microsoftin tallennetut tilikaudet ovat 2023–2026 ja Coca-Colan 2022–2025. Microsoftin tilikausi päättyy kesäkuussa ja Coca-Colan joulukuussa. Yhtiövertailussa voi valita:

- Saman tilikauden päättymisvuoden, esimerkiksi 2025. Ajanjaksot ovat silti eri: Microsoft 1.7.2024–30.6.2025 ja Coca-Cola 1.1.2025–31.12.2025.
- Kummankin viimeisimmän julkaistun tilikauden. Vuosiluvutkin voivat silloin erota.

Taulukko näyttää aina jaksojen alut ja loput. Kehityskuvaajan vaaka-akseli käyttää todellisia tilinpäätöspäiviä. Taseen varat ja velat ovat tilikauden päättymispäivän havaintoja; muut luvut kattavat koko vuoden. Uusinta osavuosikatsausta tai yhteistä rullaavaa 12 kuukauden jaksoa ei vielä muodosteta.

## Coca Colan rahavirtojen oppimisesimerkit

Vuoden 2024 liiketoiminnan rahavirtaan sisältyi 6,0 miljardin dollarin IRS-veroriitaan liittyvä talletus. Vuonna 2025 mukana oli 6,1 miljardia dollaria fairlife-yrityskaupan lisäkauppahinnan loppumaksusta. Nämä vaikuttavat myös sovelluksen laskemaan vapaaseen kassavirtaan. Näkymä kertoo asiasta kyseisten vuosien kohdalla ja vertailussa sekä linkittää alkuperäisraporttiin. Lukuja ei muuteta arvioksi normaalista kassavirrasta.

Lähde: [Coca-Colan vuoden 2025 raportti, Cash Flows from Operating Activities](https://investors.coca-colacompany.com/filings-reports/all-sec-filings/content/0001628280-26-010047/ko-20251231.htm).

## Haku ja ohjelman rakenne

Uusi haku tehdään vain käyttäjän pyynnöstä valitulle yhtiölle. Yhtiön tai tilikauden vaihtaminen ja vertailun avaaminen käyttävät paikallisia tiedostoja. Sähköpostia käytetään SEC-pyynnön tunnistetietona eikä sitä tallenneta aineistoon.

```powershell
& .\.venv-run\Scripts\python.exe fetch_company.py --company MSFT --offline
& .\.venv-run\Scripts\python.exe fetch_company.py --company KO --offline
```

--refresh hakee uuden aineiston ja tarvitsee SEC_USER_AGENT-ympäristömuuttujan, jossa on sovelluksen nimi ja yhteyssähköposti. Aineisto tallentuu yhtiöittäin data/research/msft- ja data/research/ko-kansioihin. Microsoftin aiempi alkuperäisaineisto on käytetty uudelleen; Coca-Colan aineisto on haettu SEC:stä 15.11.2026.

| Tiedosto | Tehtävä |
| --- | --- |
| market_data/companies.py | Yhtiöiden tunnisteet ja tilikausien asetukset. |
| market_data/sec.py | Yhteinen SEC-haku, vuosilukujen ja tilinpäätöspäivän tase-erien valinta sekä tallennus. |
| market_data/financial_metrics.py | Yhteiset laskukaavat. |
| learning/metric_guide.py | Selitykset, arviointiohjeet ja lähteistetyt yhtiöhuomiot. |
| learning/company_page.py | Yhteinen käyttöliittymä ja rinnakkainen vertailu. |
| pages/01_Yhtioanalyysi.py | Streamlit-sivun käynnistystiedosto. |
| market_data/sec_microsoft.py | Alkuperäisten Microsoft-kutsujen yhteensopivuus. |

Nykyinen kausivalinta on testattu näille kahdelle kiinteään päivämäärään päättyvälle tilikaudelle. Muiden yhtiöiden lisääminen vaatii tunnisteiden, tietokenttien ja tilikausikalenterin tarkistuksen. Esimerkiksi 52/53-viikkoisia tilikausia ei vielä tueta yleisesti.

## Tarkistukset

24 laskennan ja aineiston yksikkötestiä tarkistavat muun muassa kausivalinnan, vertailuvuosien lähteet, puuttuvat arvot, lisälaskelmien nimittäjät, taseen päivämäärät ja yhtiöiden erillisen tallennuksen. Lisäksi kolme Streamlit AppTest -testiä tarkistaa tyhjän aineiston, puuttuvat lisätunnusluvut sekä yhtiön ja vertailutilan vaihtamisen ilman verkkohakuja. Kaikki 27 testiä menevät läpi. Erillinen AppTest-tarkistus kattaa myös Coca-Colan poikkeuksellisen maksun huomion eri tilikausilla.

```powershell
& .\.venv-run\Scripts\python.exe -m unittest tests.test_sec_microsoft tests.test_sec_companies tests.test_learning_page -v
```

## Seuraava harjoitus

Valitse vertailuvuodeksi 2025. Katso ensin liikevoittomarginaalia ja investointiastetta. Avaa molempien selitykset ja tarkista ajanjaksot. Tutki sitten vapaata kassavirtaa ja Coca-Colan poikkeuksellista maksua. Tarkoitus on oppia, miksi yksi taulukon luku ei yksin ratkaise sijoituskohteen laatua.
