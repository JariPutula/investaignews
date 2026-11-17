# C1 Microsoftin talouslukujen ensimmäinen harjoitus

14.11.2026 | Suunnitelma 2 | Käytännön työohje

**Päivitys 15.11.2026:** Microsoftin harjoitus on nyt osa yhteistä Microsoftin ja Coca-Colan oppimisnäkymää. Tämä ohje säilyy kolmen peruslaskelman harjoituksena. Yhtiövertailun ja lisätunnuslukujen ajantasainen ohje on instructions/C2_YHTIOVERTAILU.md.

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

## Käynnistä oppimisnäkymä

Microsoftin vuosien 2023–2026 luvut on haettu ja tallennettu paikallisesti 14.11.2026. Ensimmäinen näkymä ei tarvitse uutta verkkohakua tai API-avainta. Aja projektikansiossa PowerShellissä:

```powershell
& .\.venv-run\Scripts\python.exe -m streamlit run pages/01_Yhtioanalyysi.py
```

Avaa selaimessa Streamlitin ilmoittama paikallinen osoite, yleensä http://localhost:8501. Valitse yhtiöksi Microsoft ja avaa Kasvu-välilehdeltä Näytä laskenta. Sama Yhtioanalyysi-sivu löytyy alkuperäisen app.py-sovelluksen Streamlit-sivuvalikosta. Jos sovellus oli avoinna sivua nimettäessä uudelleen, käynnistä se tarvittaessa uudelleen.

Tallennetun aineiston saa tulostettua myös ilman selainta ja verkkoyhteyttä:

```powershell
& .\.venv-run\Scripts\python.exe fetch_microsoft.py --offline
```

Uuden aineiston voi hakea oppimisnäkymän Hae tai päivitä aineisto -osiosta antamalla SEC-haun yhteyssähköpostin. Komentorivihaussa tunniste annetaan SEC_USER_AGENT-ympäristömuuttujana ja uusi haku pyydetään --refresh-valinnalla. Sähköpostia ei kirjoiteta lähdekoodiin tai aineistotiedostoihin.

Alkuperäiset SEC-vastaukset ja lähdemetadata tallentuvat data/research/msft/raw-kansioon. Poimitut luvut, laskelmat ja lähdeviitteet ovat data/research/msft/annual_metrics.json-tiedostossa. Alkuperäisen aineiston tarkistussumma tarkistetaan sitä uudelleenkäytettäessä. Ladattua aineistoa ei lisätä Git-versionhallintaan automaattisesti.

## Tilanne nyt

- [x] Sovittu kevyt aloitus USA-yhtiöillä ja maksuttomalla datalla.
- [x] Ensimmäinen harjoitusyhtiö on Microsoft.
- [x] Ensimmäiset laskelmat on rajattu kasvuun, marginaaliin ja kassavirtaan.
- [x] Oppimispolku ja käytännön valmistumiskriteerit on päivitetty.
- [x] Microsoftin SEC-tunniste ja alkuperäisraportit on varmistettu.
- [x] Neljän tilikauden alkuperäisaineisto on haettu ja tallennettu.
- [x] Lähtöluvut on täsmäytetty vuosiraportteihin.
- [x] Kolme laskelmaa ja Näytä laskenta -osiot on toteutettu.
- [ ] Käyttäjä on käynyt ensimmäisen harjoituksen läpi.

Toteutus sisältää yhteisen SEC-haun, paikallisen aineiston ja kahden yhtiön oppimisnäkymän. Viiden yhtiön vertailua tai automaattista seulaa ei vielä ole toteutettu. Käyttäjä voi jatkaa kolmen perusluvun harjoitusta ja avata lisäluvut omaan tahtiin.

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

## Toteutettu työjärjestys ja seuraava harjoitus

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

## Toteutuksen tarkistus

14 laskennan ja aineistonkäsittelyn testiä tarkistavat muun muassa tilikausien valinnan, osavuosien poissulkemisen, vertailuvuosien raportointipäivät, puuttuvat arvot ja paikallisen aineiston eheyden. Oppimisnäkymä on tarkistettu Streamlitin AppTest-työkalulla sekä viimeisen että ensimmäisen tilikauden valinnalla. Aineiston 16 lähtölukua on verrattu Microsoftin vuoden 2025 ja 2026 raportteihin.

```powershell
& .\.venv-run\Scripts\python.exe -m unittest tests.test_sec_microsoft -v
```

## Lähteet

- [SEC EDGAR API](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [Microsoftin vuosiraportti 2026](https://www.microsoft.com/investor/reports/ar26/index.html)
- [Microsoftin vuosiraportti 2025](https://www.microsoft.com/investor/reports/ar25/index.html)
- [yfinance projektin kuvaus myöhempää hintadataa varten](https://github.com/ranaroussi/yfinance)
- Projektin yhteinen suunnitelma: instructions/KEHITYSSUUNNITELMA_ABC.md ja output/pdf/Sijoitusanalyysisovellus_kehityssuunnitelma_ABC.pdf.
