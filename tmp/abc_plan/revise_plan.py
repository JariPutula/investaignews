from pathlib import Path

root = Path(__file__).resolve().parents[2]
main_path = root / 'instructions/KEHITYSSUUNNITELMA_ABC.md'
parts = main_path.read_text(encoding='utf-8').split('<!-- pagebreak -->')
assert len(parts) == 8, 'Unexpected plan structure; review before updating'

parts[0] = '''# Sijoitusanalyysin oppimisympäristön kehityssuunnitelma

## C Yhtiöanalyysin opettelu A Salkun riskit B Omistusten seuranta

14.11.2026 | Projektiversio investaignews | Suunnitelma 2

**Tavoite:** opetellaan sijoitusanalyysin laskelmia vähitellen ja rakennetaan samalla ymmärrettävä sovellus. Aloitetaan Microsoftin neljästä tilikaudesta ja kolmesta laskelmasta. Jokaisesta näytetään lähtöluvut, kaava, selitys ja lähde. USA:n yhtiöihin rajattu seula tulee mukaan vasta oppimispolun myöhemmässä vaiheessa.

**Ensimmäinen rajaus:** yksi USA-yhtiö, raportoidut vuosiluvut ja maksuttomat lähteet. Erillisten talousdatatilausten tavoitekulu on 0 euroa kuukaudessa. Tekoälyn ja mahdollisten hakupalvelujen kulut käsitellään erikseen, kun ne otetaan käyttöön. Eurooppa, automaattiset seulontarajat ja ajastus jäävät myöhemmiksi.

## Mitä nykyisestä sovelluksesta voi käyttää

| Nykyinen osa | Hyödynnettävä pohja | Ensimmäinen lisäys |
| --- | --- | --- |
| Streamlit ja Plotly | Käyttöliittymä ja kuvaajat | Yhden yhtiön oppimisnäkymä |
| Salkun CSV ja historiakuvat | Myöhempi yhteys omistuksiin | C:n harjoitus toimii itsenäisesti |
| News Sentinel | Myöhemmin uutishaku | Ensin luvut ja virallinen vuosiraportti |
| Portfolio-moduulit | Nykyinen salkkuanalyysi | Erilliset selitettävät yhtiölaskelmat |

Nykyinen uutisagentti käsittelee otsikoita ja hakukatkelmia. Yhtiöiden talouslukujen oppimisnäkymää, numeerista yhtiöseulaa tai rahastojen läpivalaisua ei vielä ole. Useat riskivälilehden tunnusluvut käyttävät omistusten tuottoprosentteja varsinaisen tuottoaikasarjan korvikkeena. C:n uudet laskelmat pidetään erillään tästä logiikasta.

**Tietojen tuoreus:** nykyinen data/latest_assets_jari.csv sisältää 46 omistusriviä ja vastaa 10.9.2026 päivättyä salkkukuvaa. Sen käyttäminen ei ole ensimmäisen C-harjoituksen edellytys. Latest-tiedoston päiväyskäsittely korjataan ennen historiatietojen käyttöä sijoituspäätösten tukena.

## Etenemisjärjestys

1. Microsoftin neljä tilikautta ja kolme selitettävää laskelmaa.
2. Viiden USA-yhtiön vertailu samoilla määritelmillä.
3. Yksi vuosiraporttiin ja tarkistettuihin lukuihin perustuva tekoälyanalyysi.
4. Laajennus 20 yhtiöön, sitten 50–100 yhtiöön ja lopulta noin 200 yhtiön seulaan.
5. A:n salkkuriskit ja B:n omistusten seuranta erillisinä myöhempinä kehitysvaiheina.

**Ensimmäinen onnistuminen:** käyttäjä ymmärtää, mistä kolme laskelmaa muodostuvat, osaa tarkistaa ne vuosiraportista ja tunnistaa vähintään yhden tulkinnan rajoitteen. Seuraava vaihe aloitetaan tämän pohjalta.
'''

parts[1] = parts[1].replace('Tavoite on nähdä', '**Myöhempi kehitysvaihe.** Tavoite on nähdä', 1)
parts[2] = parts[2].replace('Jokaiselle seurattavalle yhtiölle tallennetaan', '**Myöhempi kehitysvaihe.** Jokaiselle seurattavalle yhtiölle tallennetaan', 1)

parts[3] = '''# C Sijoitusanalyysin opettelu yhtiö kerrallaan

Ensimmäisen version tarkoitus on auttaa ymmärtämään yrityksen kasvua, kannattavuutta ja kassavirtaa. Yhtiöitä ei vielä luokitella ostettaviksi tai hylättäviksi. Ymmärrettävä laskelma ja sen tarkistaminen ovat ensimmäinen käytännön hyöty.

## Ensimmäiset kolme laskelmaa

| Laskelma | Mitä sillä opitaan |
| --- | --- |
| Liikevaihdon kasvu | Kuinka nopeasti yrityksen myynti kasvaa? Aluksi vuosimuutos, sitten kolmen vuoden keskikasvu. |
| Liikevoittomarginaali | Kuinka paljon liikevaihdosta jää liikevoitoksi ja miten osuus muuttuu? |
| Vapaa kassavirta | Paljonko liiketoiminnan rahavirrasta jää investointimenojen jälkeen? |

Liikevaihto tarkoittaa yrityksen myyntituottoja, ei voittoa. Liikevoitto kuvaa liiketoiminnan tulosta ennen rahoituseriä ja veroja. Rahavirta kuvaa rahojen liikettä; se voi poiketa tilikauden voitosta. Selitykset näytetään laskelmien yhteydessä tavallisella kielellä.

Microsoft toimii ensimmäisenä harjoitusyhtiönä. Sen jälkeen lisätään Amazon, Coca-Cola, NVIDIA ja Procter & Gamble. Yhtiöt ovat opetuskohteita, eivät ostosuosituksia. Viiden yhtiön valikoima ei vielä edusta koko markkinaa; laajennuksessa lisätään myös muita toimialoja.

## Näytä laskenta

Jokaisesta tunnusluvusta voi avata käytetyt tilikaudet, lähtöluvut, laskukaavan ja alkuperäisen lähteen. Näkymä kertoo, miten tulosta voi tulkita ja mitä siitä ei voi päätellä yksinään. Puuttuva tieto näytetään puuttuvana.

Esimerkiksi suuret investoinnit voivat pienentää vapaata kassavirtaa kasvua rakennettaessa. Kasvava liikevaihto ei puolestaan yksin osoita kannattavuuden paranemista. Sovellus näyttää kehityksen rinnakkain ja auttaa muodostamaan kysymyksiä.

## Arvostus opetellaan myöhemmin

Nettovelka/EBITDA, EV/EBIT, FCF-tuotto ja osakekohtaiset tunnusluvut lisätään vasta peruslaskelmien jälkeen. Niiden yhteydessä opetellaan velan, osakemäärän, hintatiedon ja käytettävän raportointikauden merkitys. Ensimmäinen harjoitus ei vaadi osakekurssia tai osakekohtaisten lukujen laskentaa.

## Tekoäly mukaan yhden yhtiön analyysiin

Mallille annetaan tarkistetut laskelmat ja yksi virallinen vuosiraportti. Sen tehtävä on erottaa aineiston osoittama havainto, raportista löytyvä selitys ja avoin kysymys. Se etsii myös perusteluja omaa tulkintaa vastaan. Olennaiset väitteet yhdistetään lähdekohtiin.

Esimerkiksi kassavirran ja liikevoiton kasvun eroa tutkitaan vasta, jos laskelmat osoittavat sellaisen. Tekoäly ei saa olettaa eron olemassaoloa kysymyksen sanamuodon perusteella. Ensimmäinen tekoälyversio käsittelee yhtä yhtiötä kerrallaan. Uutishaku lisätään tarvittaessa myöhemmin.
'''

parts[4] = '''# Kevyt toteutus ja maksuton aineisto

Streamlit säilyy käyttöliittymänä. Ensimmäiseen versioon riittää yksi selkeä yhtiönäkymä, pieni SEC-aineiston hakutoiminto ja kolme erillistä laskentafunktiota. Datan haku käynnistetään käsin ja aineisto tallennetaan paikallisesti uudelleenkäyttöä varten.

## Lähteet ja kustannukset

| Lähde | Käyttö ensimmäisissä vaiheissa |
| --- | --- |
| SEC EDGAR Company Facts | Microsoftin historialliset talousluvut ilman API-avainta [1] |
| Virallinen vuosiraportti | Lukujen, tilikausien ja tulkintojen tarkistus |
| yfinance myöhemmin | Hintadata arvostusharjoituksiin; epävirallinen tutkimus- ja opetustyökalu [3] |
| News Sentinel myöhemmin | Tarvittaessa täydentävä uutishaku |

**Talousdatatilaukset alussa 0 euroa kuukaudessa.** Maksullista palvelua harkitaan vasta, kun sen ratkaisema ongelma on tunnistettu käytännössä. Aikaisempi 20–50 euron budjetti jää mahdolliseksi myöhemmäksi harkinnaksi. Se ei ole aloitusvaiheen kustannustavoite tai tilauspäätös. Tekoälykulut arvioidaan erikseen ennen sen käyttöönottoa.

SEC:n aineisto vaatii silti huolellista käsittelyä. Valitaan oikeat vuosijaksot, mittayksiköt ja yhtiön käyttämät tietokentät. Sama vuosi voi esiintyä useissa raporteissa. Valittu lähdeversio tallennetaan niin, että laskelma voidaan toistaa. Aluksi tuetaan Microsoftin tarvitsemia tietoja ja laajennetaan vasta havaittujen tarpeiden mukaan.

## Ensimmäiseen harjoitukseen tarvittava data

| Tieto | Miksi se tallennetaan |
| --- | --- |
| Yhtiön nimi ja SEC-tunniste | Oikean yhtiön tunnistaminen |
| Neljän tilikauden alku ja loppu | Vuosimuutokset ja kolmen vuoden kasvujakso |
| Liikevaihto ja liikevoitto | Kasvu ja liikevoittomarginaali |
| Liiketoiminnan rahavirta ja investointimenot | Vapaa kassavirta |
| Valuutta ja mittayksikkö | Dollareita, tuhansia ja miljoonia ei sekoiteta |
| Lähde, julkaisuaika ja hakuaika | Tarkistettavuus ja tietojen tuoreus |

Investointimeno esitetään laskennassa positiivisena vähennettävänä määränä. Sovittu ensimmäinen vapaan kassavirran määritelmä on liiketoiminnan rahavirta vähennettynä aineellisilla käyttöomaisuusinvestoinneilla. Yritysostot ja rahoitusleasingiin liittyvät maksut eivät sisälly tähän automaattisesti; rajaus kerrotaan käyttäjälle.

## Mitä rakenteita tarvitaan nyt

Paikalliset alkuperäiset JSON-tiedostot, selkeästi tallennetut poimitut luvut ja lähdeviitteet riittävät. Uutta tietokantaa, ajastusta, monen datapalvelun yleisrajapintaa tai laajaa agenttijärjestelmää ei rakenneta ensimmäisen harjoituksen edellytykseksi. A:n ja B:n tietorakenteet suunnitellaan niiden omissa vaiheissa.
'''

parts[5] = '''# Toteutusjärjestys ja oppimisen tavoitteet

Edetään yksi tarkistettava tulos kerrallaan. Vaiheiden välissä katsotaan yhdessä laskelmia ja niiden tulkintaa. Oppimisen eteneminen ratkaisee laajentamisen ajankohdan.

| Vaihe | Valmis tulos | Alustava työarvio |
| --- | --- | --- |
| C1 Microsoft | Neljä tilikautta, kolme laskelmaa ja avattavat selitykset | 2–4 pv |
| C2 Viisi yhtiötä | Samat laskelmat ja liiketoimintojen erojen vertailu | lisäksi 1–3 pv |
| C3 Yksi tekoälyanalyysi | Lähteisiin perustuvat havainnot, selitykset ja vastaväitteet | lisäksi 1–2 pv |
| C4 Laajentaminen | 20 yhtiön tarkistus, sitten 50–100 ja noin 200 yhtiön seula | arvioidaan C2:n jälkeen |
| A Salkkuriskit | Rahastojen läpivalaisu ja skenaariot | myöhemmin |
| B Omistusten seuranta | Sijoitusmuistiot ja niiden muutosten seuranta | myöhemmin |

Työarviot ovat omia kehitysarvioita yhden käyttäjän paikalliselle kokeilulle. Ne olettavat nykyisen käyttöliittymän hyödyntämisen ja tarvittavien lähdetietojen saatavuuden. Microsoftin oppimisnäkymä on ensimmäinen toimitettava kokonaisuus. Oppimiskeskusteluille ei aseteta työpäiväarvioon sidottua aikataulua.

Viiden yhtiön vertailu ja yhden yhtiön tekoälyanalyysi olisivat näillä oletuksilla noin 4–9 kehitystyöpäivän kokonaisuus. Suurimman epävarmuuden muodostavat yhtiökohtaiset tietokentät ja lähteiden käsittely. Aikaisemman laajan USA–Eurooppa-pilotin työarvio ei koske tätä kevyttä aloitusta.

## Miten oppimista tuetaan

Jokaisessa vaiheessa otetaan esiin uusi käsite, näytetään laskenta ja käydään läpi yksi tulkintakysymys. Esimerkiksi marginaalin kohdalla tarkastellaan, mitä tapahtuu, jos myynti kasvaa mutta liikevoitto pysyy ennallaan. Käyttäjän ei tarvitse tuntea tunnuslukuja etukäteen.

Automaattisia vähimmäisrajoja, kokonaispisteitä tai BUY/HOLD/SELL-luokituksia ei käytetä aloituksessa. Kun käsitteet ovat tuttuja, kokeillaan yksittäisiä suodattimia ja katsotaan, miten ne muuttavat tulosjoukkoa. Mahdolliset seulontasäännöt nimetään kokeiluoletuksiksi.

## Laajentamisen ehdot

Viiden yhtiön luvut tarkistetaan ennen 20 yhtiön koetta. Laajempi yhtiöjoukko sisältää erilaisia toimialoja ja myös laskennan poikkeustapauksia. Viiden tutun yhtiön onnistuminen ei todista koko markkinan datakattavuutta.

Noin 200 yhtiön vaiheessa sovitaan päivätty yhtiölista, tiedon tuoreuden ehdot ja näkyvä käsittely puuttuville tiedoille. Ajastus, laaja uutishaku, Eurooppa ja maksullinen data ovat erillisiä jatkopäätöksiä. A:n avulla arvioidaan myöhemmin, toisiko löydetty ehdokas omaan salkkuun uutta hajautusta.
'''

parts[6] = '''# C Ensimmäinen harjoitus Microsoft

**Valmistuva näkymä:** Microsoftin neljä viimeisintä julkaistua, vertailukelpoista tilikautta. Yhteenvetotaulukko, kehityskuvaajat ja jokaisen laskelman Näytä laskenta -osio. Todelliset tilikaudet ja niiden päättymispäivät valitaan aineiston haun yhteydessä.

Ensimmäinen harjoitus käyttää raportoituja vuosilukuja. Osavuositietoja, oikaistuja tuloksia tai osakekurssia ei tarvita. Sama mittayksikkö säilytetään kaikkien laskelman lähtölukujen välillä.

## Yksi käsite kerrallaan

**Liikevaihdon kasvu** kertoo myynnin muutoksesta. Vuosikasvu lasketaan jakamalla uusi liikevaihto edellisvuoden liikevaihdolla, vähentämällä yksi ja kertomalla sadalla. Harjoitellaan ensin tätä muutosta, sitten kolmen vuoden vuotuista keskikasvua eli CAGR-lukua.

**Kolmen vuoden keskikasvu prosentteina = ((uusin liikevaihto / kolme vuotta aiempi liikevaihto) potenssiin 1/3 − 1) × 100.** Se kertoo tasaisen vuosikasvun, jolla lähtötasosta päädyttäisiin lopputasoon. Neljä tilikautta muodostaa kolme vuosiväliä. Keskikasvu ei kerro, oliko kehitys välivuosina tasaista. Nolla tai negatiivinen lähtöarvo estää tämän tavallisen prosenttilaskennan.

**Liikevoittomarginaali prosentteina = liikevoitto / liikevaihto × 100.** Se kertoo, kuinka suuri osuus myynnistä jää liikevoitoksi. Korkeampi myynti ei yksin tarkoita parempaa marginaalia. Muutos 20 prosentista 22 prosenttiin on kaksi prosenttiyksikköä.

**Vapaa kassavirta = liiketoiminnan rahavirta − investointimenot.** Tässä harjoituksessa käytetään aineellisia käyttöomaisuusinvestointeja. Luku kuvaa investointien jälkeen jäävää rahavirtaa valitulla määritelmällä. Se ei ole sama asia kuin tilikauden voitto tai automaattisesti osinkoihin jaettavissa oleva raha.

## Havainnollistavat harjoitusluvut

Seuraavat luvut ovat keksittyjä esimerkkejä, eivät Microsoftin talouslukuja.

| Lähtötiedot | Laskettu tulos |
| --- | --- |
| Liikevaihto 100 → 133,1 miljoonaa kolmessa vuodessa | Vuotuinen keskikasvu 10 % |
| Liikevoitto 20 ja liikevaihto 100 miljoonaa | Liikevoittomarginaali 20 % |
| Liiketoiminnan rahavirta 30 ja investoinnit 12 miljoonaa | Vapaa kassavirta 18 miljoonaa |

## Milloin harjoitus on valmis

Käyttäjä löytää kunkin luvun lähteestä, pystyy seuraamaan kaavaa ja osaa kuvata tuloksen merkityksen omin sanoin. Sovellus näyttää samat tulokset uudelleen samalla aineistolla. Jokaisen laskelman yhteydessä on vähintään yksi tulkinnan rajoite. Puuttuvaa tai epäselvää lukua ei arvata.

Tarkka toteutustyölista on tiedostossa instructions/C1_DATAKOE.md. Yhtiönäkymää ja aineiston hakua ei ole vielä toteutettu tämän suunnitelman päivityksen yhteydessä.
'''

parts[7] = '''# Laadun varmistus ja lähteet

## Ensimmäisen harjoituksen tarkistukset

Varmistetaan Microsoftin tunniste, neljä täyttä tilikautta, mittayksiköt ja investointimenojen etumerkit. Liikevaihto, liikevoitto, liiketoiminnan rahavirta ja investointimenot täsmäytetään vuosiraportteihin. Samalle tilikaudelle löytyvien raporttiversioiden valinta dokumentoidaan.

Kaavoja tarkistetaan pienillä käsin laskettavilla esimerkeillä. Puuttuva tieto ja jakaminen nollalla käsitellään näkyvästi. Liikevaihdon vuosimuutos, kolmen vuoden keskikasvu ja marginaalin prosenttiyksikkömuutos pidetään erillään. Kuvaajien yksiköt ja tilikaudet näkyvät käyttäjälle.

## Myöhempien vaiheiden tarkistukset

**Viiden yhtiön vertailu:** käytetään samoja määritelmiä ja näytetään yhtiöiden erilaiset tilikaudet. Liiketoimintojen eroja käsitellään ennen yhteisiä seulontarajoja. Osakemäärien ja kurssitiedon tarkistus lisätään arvostusvaiheessa.

**Tekoäly:** keskeiset väitteet pitää voida tarkistaa annetusta raportista. Laskelmat tuottaa Python. Malli erottaa havainnon, selityksen ja avoimen kysymyksen. Epävarma sanamuoto ei korvaa lähdettä. Ulkoisten raporttien sisältämät toimintaohjeet eivät ohjaa sovellusta.

**A ja B:** A:ssa tarkistetaan rahastojen omistusten kattavuus, päiväykset ja kaksinkertaisen laskennan estäminen. B:ssä alkuperäinen sijoitusperustelu säilyy ja uudet tulkinnat kirjataan erikseen. Nämä tarkistukset toteutetaan kyseisten osioiden kehitysvaiheessa.

## Onnistumisen arviointi

Aluksi arvioidaan laskelmien oikeellisuutta ja käyttäjän ymmärrystä. Käyttäjä osaa näyttää lähtöluvut, selittää kaavan ja tunnistaa tulkinnan rajoitteen. Myöhemmin arvioidaan yhtiövertailun ja tekoälyn havaintojen hyödyllisyyttä. Lyhyen jakson kurssituotto ei ole ensimmäisen kokeilun onnistumismittari.

## Lähteet ja tarkistuspäivä

Verkkolähteet tarkistettu keskustelun yhteydessä 14.11.2026. Yhtiökohtaiset alkuperäisraportit ja tarkat tilikaudet kirjataan ensimmäisen aineistohaun yhteydessä. Työmäärät ja etenemisvaiheet ovat tämän suunnitelman arvioita.

[1] SEC EDGAR Application Programming Interfaces. Raportointihistoria ja rakenteinen talousdata ilman API-avainta. https://www.sec.gov/search-filings/edgar-application-programming-interfaces

[2] iShares Core MSCI World UCITS ETF. Esimerkki myöhemmän A-osion rahaston omistusdatasta. https://www.ishares.com/uk/individual/en/products/251882/ISHARES_MSCI_WORLD_UCITS_ETF

[3] yfinance projektin kuvaus. Yahoosta riippumaton tutkimus- ja opetustyökalu myöhempiä hintatietoharjoituksia varten. https://github.com/ranaroussi/yfinance

**Projektihavainnot:** app.py; config.py; news_sentinel/agent.py; portfolio/classification.py ja risk_metrics.py; portfolio/historical/snapshot_loader.py sekä data/latest_assets_jari.csv. A-osion Amazon-esimerkki perustuu 10.9.2026 salkkukuvaan. Se ei ole tämänhetkinen riskiarvio.

**Seuraava toteutettava tulos:** Microsoftin neljä tilikautta, kolme tarkistettavaa laskelmaa ja selitykset. Päivitetty C1-työohje ohjaa tätä harjoitusta. Suunnitelman päivitys ei vielä muuta sovelluksen toimintaa.
'''

main_path.write_text('\n\n<!-- pagebreak -->\n\n'.join(p.strip() for p in parts) + '\n', encoding='utf-8')

c1 = '''# C1 Microsoftin talouslukujen ensimmäinen harjoitus

14.11.2026 | Suunnitelma 2 | Käytännön työohje

## Tavoite ja rajaus

Opetellaan sijoitusanalyysin kolme peruslaskelmaa Microsoftin neljän viimeisimmän julkaistun, vertailukelpoisen tilikauden avulla. Käyttäjän ei tarvitse tuntea laskelmia etukäteen. Näkymä näyttää lähtötiedot, kaavan, tuloksen, selityksen ja lähteen jokaiselle laskelmalle.

Tämä työohje vastaa päivitettyä kevyttä C-suunnitelmaa. Aloitus käyttää USA:n yhtiötä ja maksuttomia lähteitä. Erilliset talousdatatilaukset ovat aluksi 0 €/kk. Ensimmäisessä vaiheessa ei tarvita hintadataa, tekoälyä, osavuosiraportteja, seulontarajoja, ajastusta tai uutta tietokantaa.

## Sovittu oppimispolku

1. Microsoft: neljä tilikautta ja kolme selitettävää laskelmaa.
2. Viiden USA-yhtiön vertailu: Microsoft, Amazon, Coca-Cola, NVIDIA ja Procter & Gamble.
3. Yksi tekoälyavusteinen yhtiöanalyysi tarkistetuista luvuista ja virallisesta vuosiraportista.
4. Laajennus 20 yhtiöön, sitten 50–100 yhtiöön ja lopulta noin 200 yhtiön seulaan.
5. A:n salkkuriskit ja B:n omistusten seuranta myöhemmin.

Viiden yhtiön ryhmä on opetuksen lähtöjoukko, ei ostosuositus tai markkinaa edustava otos. Yhtiöjoukkoa laajennettaessa otetaan mukaan muita toimialoja ja poikkeustapauksia. Eurooppa ja maksullinen data arvioidaan vasta myöhemmän tarpeen perusteella.

## Tilanne nyt

- [x] Sovittu kevyt aloitus USA-yhtiöillä ja maksuttomalla datalla.
- [x] Ensimmäinen harjoitusyhtiö on Microsoft.
- [x] Ensimmäiset laskelmat on rajattu kasvuun, marginaaliin ja kassavirtaan.
- [x] Oppimispolku ja käytännön valmistumiskriteerit on päivitetty.
- [ ] Microsoftin SEC-tunniste ja alkuperäisraportit on varmistettu.
- [ ] Neljän tilikauden alkuperäisaineisto on haettu ja tallennettu.
- [ ] Lähtöluvut on täsmäytetty vuosiraportteihin.
- [ ] Kolme laskelmaa ja Näytä laskenta -osiot on toteutettu.
- [ ] Käyttäjä on käynyt ensimmäisen harjoituksen läpi.

Dokumentin päivitys ei tarkoita, että aineistohaku tai sovelluksen uusi näkymä olisi jo toteutettu.

## Ensimmäisen harjoituksen laskelmat

### Liikevaihdon kasvu

Liikevaihto tarkoittaa yrityksen myyntituottoja. Harjoitellaan ensin yhden vuoden muutosta:

**Vuosikasvu prosentteina = (uusi liikevaihto / edellisvuoden liikevaihto − 1) × 100.**

Seuraavaksi katsotaan kolmen vuoden vuotuista keskikasvua eli CAGR-lukua:

**CAGR prosentteina = ((uusin liikevaihto / kolme vuotta aiempi liikevaihto) potenssiin 1/3 − 1) × 100.**

Neljä tilikautta muodostaa kolme vuosiväliä. Luku kertoo tasaisen vuosikasvun, joka veisi lähtötasosta lopputasoon. Se ei ole vuosikasvujen tavallinen aritmeettinen keskiarvo. Nolla tai negatiivinen lähtöarvo estää tämän tavallisen prosenttilaskennan ja näytetään erillisenä tapauksena.

Harjoituskysymys: voiko kolmen vuoden keskikasvu olla positiivinen, vaikka viimeisin vuosi olisi edellistä heikompi?

### Liikevoittomarginaali

Liikevoitto kuvaa liiketoiminnan tulosta ennen rahoituseriä ja veroja. Marginaali suhteuttaa sen yrityksen myyntiin:

**Liikevoittomarginaali prosentteina = liikevoitto / liikevaihto × 100.**

Harjoituskysymys: jos liikevaihto kasvaa mutta liikevoitto pysyy ennallaan, mitä marginaalille tapahtuu? Muutos 20 prosentista 22 prosenttiin on kaksi prosenttiyksikköä. Muutos esitetään tällä nimellä eikä kahden prosentin kasvuna.

### Vapaa kassavirta

Liiketoiminnan rahavirta kertoo toiminnan tuottamasta tai käyttämästä rahasta. Tässä harjoituksessa vapaa kassavirta määritellään näin:

**Vapaa kassavirta = liiketoiminnan rahavirta − aineelliset käyttöomaisuusinvestoinnit.**

Investointimenot yhtenäistetään positiivisiksi vähennettäviksi määriksi. Yritysostoja, kaikkia aineettomia investointeja tai rahoitusleasingiin liittyviä maksuja ei lisätä määritelmään huomaamatta. Käytetyn rajauksen vaikutus kerrotaan.

Harjoituskysymys: voiko kasvua rakentava yritys näyttää heikompaa vapaata kassavirtaa investointien lisääntyessä? Luku ei ole sama kuin tilikauden voitto eikä sellaisenaan kerro jaettavasta osingosta.

## Käsin tarkistettavat harjoitusluvut

Seuraavat ovat keksittyjä opetusesimerkkejä, eivät Microsoftin talouslukuja.

| Laskelma | Lähtöluvut | Oikea tulos |
| --- | --- | --- |
| Vuosikasvu | Liikevaihto 100 → 110 miljoonaa vuodessa | 10 % |
| Kolmen vuoden keskikasvu | Liikevaihto 100 → 133,1 miljoonaa kolmessa vuodessa | 10 % vuodessa |
| Liikevoittomarginaali | Liikevoitto 20 ja liikevaihto 100 miljoonaa | 20 % |
| Vapaa kassavirta | Liiketoiminnan rahavirta 30 ja investoinnit 12 miljoonaa | 18 miljoonaa |

## Kerättävä aineisto

Käytetään SEC:n Company Facts -aineistoa sekä Microsoftin virallisia vuosiraportteja lukujen tarkistamiseen. API ei vaadi avainta, mutta automaattisen haun SEC-ohjeita noudatetaan. Aluksi tuetaan vain tämän harjoituksen tarvitsemia tietokenttiä.

| Tieto | Tallennettava sisältö |
| --- | --- |
| Yhtiö | Nimi ja varmistettu SEC CIK -tunniste |
| Raportointikausi | Tilikauden alku ja loppu; neljä täyttä vertailukelpoista tilikautta |
| Liikevaihto ja liikevoitto | Raportoidut vuosiluvut sekä niiden tietokentät |
| Rahavirta ja investoinnit | Liiketoiminnan rahavirta sekä sovitun määritelmän mukainen investointimeno |
| Valuutta ja yksikkö | USD sekä käytetty suuruusluokka; esityksessä esimerkiksi miljoonaa USD |
| Lähde | Raportti, SEC-tietokenttä ja aineistoversio tai raportin tunniste |
| Ajat | Raportin julkaisuaika ja tiedon hakuaika |

Osavuosilukuja tai kuluvan vuoden kumulatiivisia rahavirtoja ei sekoiteta koko tilikauden lukuihin. Jos sama tilikausi on useassa raportissa, valitun version peruste kirjataan. Puuttuva havainto säilyy puuttuvana. Tarkka neljän tilikauden joukko ratkaistaan haun yhteydessä julkaistujen raporttien perusteella.

Alkuperäinen aineisto säilytetään paikallisena JSON-tiedostona. Poimitut tiedot ja niiden lähteet tallennetaan erikseen, jotta laskelma voidaan toistaa ilman uutta hakua. Tietokanta ei ole tämän harjoituksen edellytys.

## Oppimisnäkymä

Yhtiönäkymässä on neljän tilikauden taulukko ja selkeä kehityskuva kullekin laskelmalle. Prosentit ja rahamäärät esitetään erillään. Jokaisen laskelman Näytä laskenta -osio sisältää:

- Mihin kysymykseen laskelma vastaa.
- Käytetyt vuodet, lähtöluvut ja mittayksiköt.
- Kaavan, välivaiheet ja lasketun tuloksen.
- Lyhyen tulkinnan ja vähintään yhden rajoitteen.
- Linkin tai tarkan viitteen alkuperäiseen lähteeseen.

Käyttöliittymän ei tarvitse selittää SEC:n teknisiä tietokenttiä päätekstissä. Ne löytyvät tarkemmista lähdetiedoista. Keskeistä on, että laskenta on ymmärrettävissä ilman talousalan ennakkotietoja.

## Toteutuksen työjärjestys

1. Varmista Microsoftin tunniste ja valitse neljä viimeisintä julkaistua, vertailukelpoista vuosiraporttia.
2. Hae ja tallenna tarvittava alkuperäinen SEC-aineisto. Tuo liikevaihto, liikevoitto, liiketoiminnan rahavirta ja investoinnit tilikausittain.
3. Täsmäytä lähtöluvut vuosiraportteihin. Kirjaa tietokentät, yksiköt, ajanjaksot ja mahdolliset epäselvyydet.
4. Toteuta kolme erillistä laskentaa. Tarkista kaavat yllä olevilla pienillä esimerkeillä sekä puuttuvilla arvoilla ja nollajakajalla.
5. Lisää nykyiseen Streamlit-sovellukseen rajattu oppimisnäkymä ja Näytä laskenta -osiot. Pidä laskenta erillään nykyisistä salkun riskimittareista.
6. Käy käyttäjän kanssa yksi laskelma kerrallaan läpi. Laajenna vasta tämän jälkeen muihin neljään harjoitusyhtiöön.

## Valmistumiskriteerit

- Microsoftin neljä tilikautta ja niiden päiväykset ovat näkyvissä.
- Kolme laskelmaa perustuvat tarkistettuihin lähtölukuihin ja ovat toistettavia.
- Jokaiselle luvulle löytyvät määritelmä, lähtötiedot, laskenta ja lähde.
- Puuttuva tieto tai nollan suuruinen nimittäjä näkyy ymmärrettävänä selityksenä.
- Käyttäjä pystyy kuvaamaan kunkin laskelman merkityksen ja vähintään yhden rajoitteen omin sanoin. Tämä on keskustelun tavoite, ei koe tai sovelluksen pakollinen portti.
- Sovellus ei vielä käytä automaattisia hyväksytty/hylätty-rajoja tai sijoitussuosituksia.

## Myöhempiin vaiheisiin siirretyt asiat

Nettovelka/EBITDA, EV/EBIT, FCF-tuotto, osakekohtaiset luvut, osakekurssit, osavuosianalyysi, tekoälyn raporttitulkinta, uutishaku, maksulliset datapalvelut ja automaattinen seula otetaan mukaan erikseen. Tekoälyn ensimmäinen tehtävä on yksi lähteistetty yhtiöanalyysi; sen kulut arvioidaan ennen käyttöönottoa. yfinance voi toimia myöhemmän kurssiharjoituksen lähteenä.

## Lähteet

- [SEC EDGAR API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [yfinance projektin kuvaus myöhempää hintadataa varten](https://github.com/ranaroussi/yfinance)
- Projektin yhteinen suunnitelma: instructions/KEHITYSSUUNNITELMA_ABC.md ja output/pdf/Sijoitusanalyysisovellus_kehityssuunnitelma_ABC.pdf.
'''
(root / 'instructions/C1_DATAKOE.md').write_text(c1, encoding='utf-8')
print('Updated main plan and C1 learning instructions')
