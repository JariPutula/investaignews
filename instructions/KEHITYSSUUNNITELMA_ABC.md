# Sijoitusanalyysin oppimisympäristön kehityssuunnitelma

## C Yhtiöanalyysin opettelu A Salkun riskit B Omistusten seuranta

15.11.2026 | Projektiversio investaignews | Suunnitelma 3

**Tavoite:** opetellaan sijoitusanalyysin laskelmia vähitellen ja rakennetaan samalla ymmärrettävä sovellus. Aloitetaan Microsoftin neljästä tilikaudesta ja kolmesta laskelmasta. Jokaisesta näytetään lähtöluvut, kaava, selitys ja lähde. USA:n yhtiöihin rajattu seula tulee mukaan vasta oppimispolun myöhemmässä vaiheessa.

**Nykyinen rajaus:** Microsoft ja Coca-Cola, raportoidut vuosiluvut ja maksuttomat lähteet. Aloitus säilyy kolmessa peruslaskelmassa, joita täydentävät avattavat lisätunnusluvut. Talousdatatilaukset maksavat alussa 0 euroa kuukaudessa. Tekoäly, Eurooppa, automaattinen seula ja ajastus jäävät myöhemmiksi.

## Mitä nykyisestä sovelluksesta voi käyttää

| Nykyinen osa | Hyödynnettävä pohja | Ensimmäinen lisäys |
| --- | --- | --- |
| Streamlit ja Plotly | Käyttöliittymä ja kuvaajat | Yhtiövalinta ja rinnakkainen vertailu |
| Salkun CSV ja historiakuvat | Myöhempi yhteys omistuksiin | C:n harjoitus toimii itsenäisesti |
| News Sentinel | Myöhemmin uutishaku | Ensin luvut ja virallinen vuosiraportti |
| Portfolio-moduulit | Nykyinen salkkuanalyysi | Erilliset selitettävät yhtiölaskelmat |

Microsoftin C1-harjoitus ja C2:n kahden yhtiön vertailu on toteutettu. Microsoftilta ovat mukana vuodet 2023–2026 ja Coca-Colalta 2022–2025. Näkymässä on avattavat Tietoa-selitykset sekä laskelmien lähteet. Seula ja rahastojen läpivalaisu ovat myöhempiä vaiheita. C:n laskenta on erillään vanhoista salkun riskimittareista.

**Tietojen tuoreus:** nykyinen data/latest_assets_jari.csv sisältää 46 omistusriviä ja vastaa 10.9.2026 päivättyä salkkukuvaa. Sen käyttäminen ei ole ensimmäisen C-harjoituksen edellytys. Latest-tiedoston päiväyskäsittely korjataan ennen historiatietojen käyttöä sijoituspäätösten tukena.

## Etenemisjärjestys

1. Microsoftin neljä tilikautta ja kolme selitettävää laskelmaa.
2. Microsoftin ja Coca-Colan vertailu; sen jälkeen laajennus viiteen USA-yhtiöön.
3. Yksi vuosiraporttiin ja tarkistettuihin lukuihin perustuva tekoälyanalyysi.
4. Laajennus 20 yhtiöön, sitten 50–100 yhtiöön ja lopulta noin 200 yhtiön seulaan.
5. A:n salkkuriskit ja B:n omistusten seuranta erillisinä myöhempinä kehitysvaiheina.

**Ensimmäinen onnistuminen:** käyttäjä ymmärtää, mistä kolme laskelmaa muodostuvat, osaa tarkistaa ne vuosiraportista ja tunnistaa vähintään yhden tulkinnan rajoitteen. Seuraava vaihe aloitetaan tämän pohjalta.

<!-- pagebreak -->

# A Salkun todelliset riskit ja päällekkäisyydet

**Myöhempi kehitysvaihe.** Tavoite on nähdä suorat ja rahastojen kautta syntyvät yhtiöaltistukset yhtenä kokonaisuutena. Instrumenttien lukumäärä, rahaston nimi tai listausmaa ei yksin osoita hajautusta. Omistusten päällekkäisyys on laskettavissa; riippuvuus samasta taloudellisesta ilmiöstä vaatii lisäksi harkintaa.

## Läpivalaisu ja skenaariot

Jokainen rahasto tunnistetaan oikean osuussarjan ISIN-tunnuksella. Sen omistuslista haetaan ensisijaisesti liikkeeseenlaskijalta tai tuodaan aluksi tiedostona. Listan päiväys ja tunnettu osuus tallennetaan. Esimerkiksi iShares julkaisee omistustietoja ladattavina tiedostoina [2]. Esimerkkilähde ei yksin tunnista salkun omaa rahasto-osuutta.

**Yhtiön kokonaispaino = suora salkkupaino + rahastokohtaisten välillisten painojen summa. Välillinen paino = rahaston salkkupaino × yhtiön paino rahastossa.** Rahastorivit korvataan niiden läpivalaistuilla osuuksilla; rahastopainoa ei lisätä niiden päälle. Samat yhtiöt, eri osakesarjat ja ADR-todistukset yhdistetään yhteiseen yhtiötunnisteeseen.

**Tuntematon osuus säilyy tuntemattomana.** Jos saatavilla ovat vain suurimmat omistukset, loppuosaa ei oleteta hajautetuksi eikä tunnettuja rivejä skaalata sataan prosenttiin. Rahastojen rahastot avataan vaiheittain; käteinen ja johdannaiset käsitellään erikseen. Synteettinen tai vivutettu rahasto tarvitsee oman altistusmallin.

| Kysymys | Toteutettava tarkastelu |
| --- | --- |
| Suuret teknologiayhtiöt | Päivätty yhtiöjoukko ja suorat sekä välilliset painot |
| AI-investointien riippuvuus | Perustellut teemaluokitukset ja useita skenaarioita |
| Puolijohteet −30 % | Kohdeyhtiöiden läpivalaistu paino × hintashokki |
| Terveydenhuolto +20 % | Vastaava erillinen toimialaskenaario |
| Uusi hajautus | Keskittymien ja skenaariotappioiden muutos samalla rahoitustavalla |

AI-riippuvuutta ei pidä esittää täsmällisenä mitattuna prosenttina ilman sovittua määritelmää. Sama yhtiö voi hyötyä investoinneista, rahoittaa niitä ja kantaa niihin liittyvää riskiä. Päällekkäisiä teemaluokkia ei lasketa yhteen erillisinä tappioina.

## Amazon esimerkin tarkistus

10.9.2026 aineistossa salkun arvo on 193 706,92 euroa ja Amazonin suora omistus 19 062,12 euroa eli 9,84 %. Suoran omistuksen puolittuminen laskisi salkkua **9 531,06 euroa eli 4,92 %**, jos muut sijoitukset ja valuuttakurssit pysyisivät ennallaan. Käyttäjän esimerkki on näin tälle aineistolle oikeansuuntainen. Rahastojen Amazon-omistusten samanaikainen lasku lisäisi vaikutusta; sitä ei ole vielä laskettu.

Skenaario kertoo valittujen oletusten seurauksen, ei tapahtuman todennäköisyyttä. Yhdistelmäskenaariossa jokaiselle yhtiölle annetaan yksi yhteinen hintamuutos. Euroissa laskettu rahastokurssi ei poista taustalla olevia valuutta-altistuksia. Kotipaikka, liikevaihdon maantiede ja valuutta esitetään erillään.

**Ensimmäinen A-versio:** ajantasainen salkku, 3–5 suurimman rahaston täydet omistuslistat, kattavuusprosentti, yhtiökohtaiset kokonaispainot ja neljä skenaariota. Nykyinen korrelaatioiden avainsanakorvike ei kelpaa laskelman perustaksi.

<!-- pagebreak -->

# B Tekoäly sijoitusperustelujen seuraajana

**Myöhempi kehitysvaihe.** Jokaiselle seurattavalle yhtiölle tallennetaan sijoitusmuistio. Siinä kuvataan omistamisen syy, aikajänne, kasvu- ja kannattavuusoletukset, arvostus, keskeiset riskit ja havainnot, jotka kumoaisivat perustelun. Alkuperäinen versio säilytetään. Muutoksille kirjataan päiväys ja syy.

Jos alkuperäistä ostoperustelua ei ole tallessa, käyttäjä laatii tekoälyn avulla uuden lähtömuistion. Se merkitään jälkikäteen laadituksi. Malli ei saa keksiä sijoittajan aikaisempaa ajattelua tai korjata vanhoja perusteluja jälkiviisaasti.

## Viikoittaisen raportin sisältö

1. **Muutos:** mitä olennaista tapahtui edellisen tarkastelun jälkeen ja mikä on alkuperäinen lähde?
2. **Oletus:** mitä sijoitusmuistion oletusta havainto vahvistaa tai heikentää?
3. **Laskelma:** miten vaikutus näkyy kasvussa, marginaaleissa, kassavirrassa tai arvostusskenaariossa?
4. **Vastaväite:** mikä on vahvin näyttö omaa tulkintaa vastaan ja mikä havainto ratkaisisi asian?
5. **Seuraus:** ei olennaista muutosta, seurattava muutos tai sijoitusperustelun uudelleenarviointi.

Raportin tulee erottaa lähteestä todettu fakta, käyttäjän oletus, laskettu tulos ja tekoälyn tulkinta. Jokainen olennainen väite yhdistetään lähdekohtaan ja päiväykseen. Pelkkä uskottavan näköinen lähdelinkki ei riitä, jos lähde ei tue väitettä.

## Arvostus ja markkinoiden odotukset

Python laskee heikon, perus- ja vahvan kehityksen skenaariot näkyvillä oletuksilla. Käänteinen arvostuslaskelma tutkii, millainen kasvu tai kannattavuus perustelisi nykyisen hinnan valitulla tuottovaatimuksella ja muilla oletuksilla. Yhtä oikeaa markkinan kasvuoletusta ei voida päätellä hinnasta, koska useat oletusyhdistelmät tuottavat saman arvon.

Analyytikkoennusteita voidaan käyttää vertailuna, jos niitä saadaan budjetissa. Väite siitä, ettei markkina ole huomioinut jotakin, merkitään tutkittavaksi hypoteesiksi. Uutisen myönteisyys ei sellaisenaan osoita osakkeen houkuttelevuutta.

## Yhtiökohtaisuus ja salkkupaino

Pilvipalveluyhtiön seuranta voi painottaa segmenttikasvua, investointeja ja pääoman tuottoa. Lääkeyhtiön seuranta tarvitsee lisäksi esimerkiksi patenttien, tutkimustulosten ja lääkeaihioiden tarkastelua. Yhteinen talouslukupohja yhdistetään yhtiökohtaisiin seurantakysymyksiin.

Viikoittain tarkistetaan muutokset. Täysi muistio ja arvostus päivitetään tulosraportin, merkittävän tapahtuman tai sovitun tarkastelupäivän yhteydessä. Myös ”ei olennaista muutosta” on hyödyllinen tulos.

Salkkupainon uudelleenarviointi käyttää A:n tietoja kokonaisaltistuksista sekä käyttäjän omia rajoja. B:n ensimmäinen versio antaa perustellun tarkastelutarpeen. Tarkan tavoitepainon laskeminen ei kuulu ensimmäiseen julkaisuun.

**Toteutus:** versioidut muistiot, alkuperäisten raporttien käsittely, lähteisiin sidottu väiterekisteri, deterministinen arvostus ja vertailu edelliseen raporttiin. C:n jatkotutkimuksen lähde- ja laskentakerros palvelee tätä myöhemmin.

<!-- pagebreak -->

# C Sijoitusanalyysin opettelu yhtiö kerrallaan

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

## Täydentävät tunnusluvut ja arvostus

C2 lisää brutto- ja nettomarginaalin, vapaan kassavirran marginaalin, investointiasteen sekä lyhyen aikavälin maksuvalmiussuhteen. Niiden määritelmä, laskenta ja tulkinnan rajat avautuvat Tietoa-painikkeista. Yleisiä osto- tai hylkäysrajoja ei anneta.

Nettovelka/EBITDA, EV/EBIT, FCF-tuotto ja osakekohtaiset tunnusluvut lisätään vasta peruslaskelmien jälkeen. Niiden yhteydessä opetellaan velan, osakemäärän, hintatiedon ja käytettävän raportointikauden merkitys. Ensimmäinen harjoitus ei vaadi osakekurssia tai osakekohtaisten lukujen laskentaa.

## Tekoäly mukaan yhden yhtiön analyysiin

Mallille annetaan tarkistetut laskelmat ja yksi virallinen vuosiraportti. Sen tehtävä on erottaa aineiston osoittama havainto, raportista löytyvä selitys ja avoin kysymys. Se etsii myös perusteluja omaa tulkintaa vastaan. Olennaiset väitteet yhdistetään lähdekohtiin.

Esimerkiksi kassavirran ja liikevoiton kasvun eroa tutkitaan vasta, jos laskelmat osoittavat sellaisen. Tekoäly ei saa olettaa eron olemassaoloa kysymyksen sanamuodon perusteella. Ensimmäinen tekoälyversio käsittelee yhtä yhtiötä kerrallaan. Uutishaku lisätään tarvittaessa myöhemmin.

<!-- pagebreak -->

# Kevyt toteutus ja maksuton aineisto

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

<!-- pagebreak -->

# Toteutusjärjestys ja oppimisen tavoitteet

Edetään yksi tarkistettava tulos kerrallaan. Vaiheiden välissä katsotaan yhdessä laskelmia ja niiden tulkintaa. Oppimisen eteneminen ratkaisee laajentamisen ajankohdan.

| Vaihe | Valmis tulos | Alustava työarvio |
| --- | --- | --- |
| C1 Microsoft | Neljä tilikautta, kolme laskelmaa ja avattavat selitykset | 2–4 pv |
| C2 Yhtiövertailu | Microsoft ja Coca-Cola valmiina; laajennus viiteen yhtiöön seuraavaksi | lisäksi 1–3 pv |
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

<!-- pagebreak -->

# C Ensimmäinen harjoitus Microsoft

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

**Toteutettu 15.11.2026:** yhteinen SEC-haku, Microsoftin ja Coca-Colan luvut, yhtiövertailu ja Tietoa-ikkunat. Käynnistä: .venv-run/Scripts/python.exe -m streamlit run pages/01_Yhtioanalyysi.py. Ohjeet ovat tiedostoissa instructions/C1_DATAKOE.md ja instructions/C2_YHTIOVERTAILU.md.

<!-- pagebreak -->

# Laadun varmistus ja lähteet

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

Verkkolähteet tarkistettu keskustelun yhteydessä 14.–15.11.2026. Yhtiöiden lähderaportit, tilikaudet ja hakupäivät säilyvät aineistossa. Työmäärät ja etenemisvaiheet ovat tämän suunnitelman arvioita.

[1] SEC EDGAR Application Programming Interfaces. Raportointihistoria ja rakenteinen talousdata ilman API-avainta. https://www.sec.gov/search-filings/edgar-application-programming-interfaces

[2] iShares Core MSCI World UCITS ETF. Esimerkki myöhemmän A-osion rahaston omistusdatasta. https://www.ishares.com/uk/individual/en/products/251882/ISHARES_MSCI_WORLD_UCITS_ETF

[3] yfinance projektin kuvaus. Yahoosta riippumaton tutkimus- ja opetustyökalu myöhempiä hintatietoharjoituksia varten. https://github.com/ranaroussi/yfinance

**Projektihavainnot:** app.py; config.py; news_sentinel/agent.py; portfolio/classification.py ja risk_metrics.py; portfolio/historical/snapshot_loader.py sekä data/latest_assets_jari.csv. A-osion Amazon-esimerkki perustuu 10.9.2026 salkkukuvaan. Se ei ole tämänhetkinen riskiarvio.

**Seuraava askel:** vertaa yhtiöiden vuoden 2025 lukuja, tarkista tilikaudet ja avaa tunnuslukujen selitykset. Coca-Colan vuosien 2024–2025 rahavirtojen poikkeukselliset maksut on merkitty lähteistettyinä huomioina [6]. Yksittäinen tunnusluku ei ratkaise sijoituksen laatua.

[4] Microsoftin vuosiraportti 2026. https://www.microsoft.com/investor/reports/ar26/index.html

[5] Microsoftin vuosiraportti 2025. https://www.microsoft.com/investor/reports/ar25/index.html

[6] Coca-Colan vuosiraportti 2025. https://investors.coca-colacompany.com/filings-reports/all-sec-filings/content/0001628280-26-010047/ko-20251231.htm
