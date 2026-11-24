# Microsoft: mitä nykyinen hinta edellyttää?

21.11.2026 | Ensimmäinen tekoälyn ohjaama arvostusharjoitus

## Avaa valmis harjoitus

Käynnistä tavallinen yhtiöanalyysi ja paina sivun alusta **Microsoft: hinta ja odotukset — avaa harjoitus**. Takaisin-painike palauttaa yhtiövalintaan. Alkuperäisen salkkusovelluksen sivuvalikossa harjoitus näkyy myös omana sivunaan.

Harjoituksen voi käynnistää suoraan projektikansiossa:

```powershell
& .\.venv-run\Scripts\python.exe -m streamlit run pages/02_Microsoft_harjoitus.py
```

Ensimmäinen valmis harjoitus on tallennettu paikallisesti. Se käyttää 20.11.2026 päätöskursseja ja Microsoftin 30.9.2026 päättyneen vuosineljänneksen lukuja. Raportti julkaistiin 28.10.2026. Neljän tilikauden historia ja johdon virallinen tulospuhelu ovat mukana. Tallennetun harjoituksen avaaminen ei tee uusia hakuja tai tekoälykutsuja.

Lue ensin hinnan ja liiketoiminnan kuvaus, sitten skenaariotaulukko ja **Millaista kasvua nykyinen hinta edellyttäisi?**. Sinun ei tarvitse antaa kasvuprosentteja tai muita oletuksia. Niiden perustelut löytyvät skenaarioiden avattavista osioista.

Kokeile lopuksi kysymystä: *Miksi suuret investoinnit voivat pienentää nykyarvoa, vaikka niiden tarkoitus on kasvattaa liiketoimintaa?* Kysymysosio käyttää juuri tämän päivätyn harjoituksen tietoja. Vastaus ei hae uusia uutisia tai kurssia. Uusi vastaus on erillinen maksullinen kutsu; kustannusarvio näkyy lomakkeen yhteydessä. Sama tallennettu kysymys palautetaan maksutta.

## Päivittäminen ja tallennus

- **Päivitä aineisto** hakee SEC:n talousluvut, Microsoftin tulospuhelun sekä MSFT:n ja SPY:n päätöskurssihistorian. Haku ei käytä maksullisia tekoälypalveluja.
- **Laadi tekoälyharjoitus** näkyy, kun nykyiselle aineistolle ei ole valmista harjoitusta. Se näyttää ensin kustannusarvion. Ensimmäinen kutsu valmistelee ja perustelee skenaariot, Python laskee tulokset ja toinen kutsu selittää ne.
- Valmis ensimmäinen skenaariovaihe tallennetaan välituloksena. Jos selitys epäonnistuu, seuraava erikseen pyydetty yritys käyttää tätä vaihetta uudelleen. Automaattisia maksullisia uusintoja ei ole.
- Hylätty vastaus ei korvaa valmista harjoitusta. Hylätyistä kutsuista voi silti syntyä API-kuluja; tallennetun version kustannusarvio kattaa siihen liitetyt vaiheet.
- Jokainen uusi aineistoversio ja valmis harjoitus tallentuvat erikseen. Vanhojen harjoitusten oletuksia ja tuloksia voi katsoa arkistosta tai ladata JSON-tiedostona. Kysymykset säilyvät oman harjoituksensa yhteydessä.

Sovittu SEC-yhteyssähköposti on vain paikallisessa, versionhallinnasta pois jätetyssä asetustiedostossa. Se ei kuulu talousaineistoon tai tekoälysyötteeseen. OpenAI käyttää sovelluksen olemassa olevaa avainta. Malli on `gpt-5.4-mini`; API-tallennus on pois käytöstä (`store=False`). Tämä asetus ei itsessään poista palveluntarjoajan kaikkea lokitusta.

## Mitä lasketaan

**Hinta:** kuukauden ja kolmen kuukauden muutos sekä muutos viimeisten 52 viikon korkeimmasta päätöskurssista. Kuukausivertailu käyttää kalenterikuukauden takaista päivää tai sitä edeltävää viimeistä kaupankäyntipäivää. SPY on S&P 500 -indeksiä seuraava ETF. Kyse on päätöskurssien muutoksesta osakesplitit huomioiden, ilman osinkojen uudelleensijoitusta. Osake ja vertailurahasto päättyvät samaan kaupankäyntipäivään.

**Raportoidut luvut:** neljän vuoden historia, viimeisin vuosineljännes verrattuna edellisvuoden samaan neljännekseen sekä viimeiset 12 kuukautta. Jälkimmäinen lasketaan kaavalla viimeinen koko tilikausi − edellisen vuoden vastaava kumulatiivinen osavuosijakso + uusi kumulatiivinen osavuosijakso. Vuosineljännestä ei kerrota neljällä. Kassavirtalaskelman kumulatiivisista luvuista erotetaan tarvittaessa viimeisin neljännes.

**Historiallinen FCF:** liiketoiminnan rahavirta − käteisenä maksetut käyttöomaisuusinvestoinnit. Sen tuotto lasketaan päivättyä markkina-arvoa vasten. Historiallinen P/E käyttää viimeisten 12 kuukauden emoyhtiön nettotulosta. Poikkeukselliset sijoitustulokset voivat vaikuttaa GAAP-nettotulokseen.

**Tekoälyn skenaariot:** viiden seuraavan liukuvan 12 kuukauden jakson kasvu, liikevoitto- ja nettomarginaali, käteisinvestoinnit, uudet rahoitusleasing-hyödykkeet, poistot ja käyttöpääoman lisätarve. Heikko, perus ja vahva ovat vaihtoehtoisia oletusjoukkoja. Perus ei tarkoita todennäköisintä kehitystä. Skenaarion P/E käyttää omaa nettomarginaalioletusta ja nykyistä markkina-arvoa; se ei ole analyytikkojen konsensusennuste.

**Arvostus:** käytetään koko liiketoiminnan vapaata kassavirtaa (FCFF):

> Liikevoitto × (1 − veroaste) + poistot − käteisinvestoinnit − uudet rahoitusleasing-hyödykkeet − käyttöpääoman lisäraha.

Käyttöpääoman historiallinen vertailu rajataan myyntisaamisten, varastojen, muiden lyhytaikaisten operatiivisten varojen, ostovelkojen, muiden lyhytaikaisten operatiivisten velkojen ja asiakasvelvoitteiden rahavirtamuutoksiin. Se ei ole kaikkien tase-erien kattava käyttöpääomamalli. Ennusteessa lisäraha suhteutetaan myynnin muutokseen.

Rahavirrat diskontataan yrityksen pääomakustannusta kuvaavalla WACC-oletuksella. Tämän ensimmäisen harjoituksen WACC on tekoälyn harkinnanvarainen herkkyysoletus, jota ei estimoida riskittömän koron, betan ja riskipreemioiden markkina-aineistosta. Korkeampi WACC pienentää saman rahavirran nykyarvoa. Se ei ole henkilökohtainen tuottolupaus.

Yritysarvosta vähennetään raportoidut lainat ja rahoitusleasingvelat sekä lisätään käteinen ja lyhytaikaiset sijoitukset. Microsoftin LongTermDebt-tietokentän merkitys varmistetaan raportin **Total debt** -riviltä. Lainojen lyhytaikaista osuutta ei lisätä kokonaisvelkaan toistamiseen eikä puuttuvaa velkariviä oleteta nollaksi. Operating lease -kulut säilyvät liiketoiminnan kuluina, joten näitä leasingvelkoja ei vähennetä erikseen. Osakepalkkiokulua ei lisätä takaisin liikevoittoon. Tulevia osakkeiden takaisinostoja tai laimennusta ei ennusteta; osakekohtainen arvo käyttää tasepäivän osakemäärää.

**Jatkuva vaihe:** viiden ennustevuoden jälkeen verojen jälkeisestä liikevoitosta vähennetään kasvun vaatima jälleensijoitus. Sen osuus on jatkuva kasvu / uuden pääoman tuotto. Tämä on erillinen vahva mallioletus. Ensimmäisessä harjoituksessa siirtymä nostaa rahavirtaa voimakkaasti, ja jatkuva vaihe muodostaa suuren osan arvosta. Molemmat rajaukset näkyvät näkymässä ja tulkinnassa. Arvoa ei pidä käsitellä täsmällisenä tavoitehintana.

**Käänteinen arvostus:** ratkaistaan tasainen viiden vuoden liikevaihdon vuosikasvu, joka vastaa nykyhintaa muiden oletusten pysyessä ennallaan. Ratkaisu ei paljasta yhtä markkinoiden todellista kasvuodotusta. Herkkyystaulukossa vaihdetaan myös WACC:ia ja jatkuvaa kasvua. Jos kasvun ja arvon suhde ei ole yksikäsitteisesti nouseva tai ratkaisua ei löydy sallitulta alueelta, kasvulukua ei näytetä.

## Tarkistus ja käytännön rajat

88 lähtöhavaintoa täsmäytettiin kolmen alkuperäisen SEC-raportin inline XBRL -lukuihin, niiden kausiin ja konsernitason konteksteihin. Tekoäly valitsee raporttilainauksen annetusta alkuperäisestä otteesta indeksillä; Python liittää todellisen lainauksen. Tuntemattomat lähteet ja keksityt lainaukset hylätään. Viitteen oikeellisuus ei yksin todista johtopäätöksen oikeellisuutta.

Ensimmäisen harjoituksen tekstit tarkistettiin kehityksessä Pythonin laskelmia vasten. Tuottovaatimuksen selitystä ja ehdollisen kasvun tulkintaa täsmennettiin. Skenaarioiden prosenttioletuksia ei muutettu. Tarkistusmerkinnät ovat myös harjoituksessa. Automaattinen lähde- ja laskentatarkistus ei ole kaikkien tulevien tekoälypäätelmien sisällöllinen tarkastus.

Yli seitsemän päivää vanha kurssi tai yli 160 päivää vanha talouskauden päättymispäivä estää uuden nykyhintaharjoituksen. Vanha päivätty harjoitus säilyy luettavana. Myöhemmin julkaistuja talouslukuja ei yhdistetä niitä vanhempaan hintaan. Laskelma ei kata kaikkia viimeaikaisia uutisia tai sijoittajan henkilökohtaista tilannetta.

```powershell
& .\.venv-run\Scripts\python.exe -m unittest tests.test_microsoft_exercise -v
```

Koko yhtiöanalyysin testisarja sisältää lisäksi aiemmat SEC-, analyysi-, skenaario- ja sivutestit. Muutettujen tiedostojen tyyppitarkistus on ajettu. Käynnissä olevassa Streamlitissä tarkistetaan valmis harjoitus, raporttilukujen avaaminen ja arkisto ilman maksullisia painalluksia.

## Lähteet

- [Microsoftin osavuosiraportti 30.9.2026](https://www.sec.gov/Archives/edgar/data/789019/000119312526436133/msft-20260930.htm)
- [Microsoftin FY2027 Q1 -tulospuhelu](https://www.microsoft.com/en-us/investor/events/fy-2027/earnings-fy-2027-q1)
- [Microsoftin kurssihistoria](https://finance.yahoo.com/quote/MSFT/history/)
- [SPY:n kurssihistoria](https://finance.yahoo.com/quote/SPY/history/)
- [Damodaran: kassavirtojen ja arvostuksen määritelmät](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/definitions.html)
- [OpenAI: GPT-5.4 mini, hinnoittelu tarkistettu 21.11.2026](https://developers.openai.com/api/docs/models/gpt-5.4-mini)
