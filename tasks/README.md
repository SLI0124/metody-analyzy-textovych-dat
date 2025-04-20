# Cvičení 1: Automatické doplňování slov (N-gram model)

V této úloze byste si měli vyzkoušet vytvořit jednoduchý model jazyka. Model jazyka je statistickým modelem, který se
snaží odhadnout pravděpodobnost výskytu slova na základě jeho kontextu. V této úloze se zaměříme na n-gram model, který
je založen na pravděpodobnosti výskytu n po sobě jdoucích slov. Při řešení této úlohy můžete používat umělou inteligenci
v libovolném rozsahu.

## Základní seznámení s n-gramy (jednodušší) - 1 bod

### Úkol:

- Načtěte zvolený český textový soubor např. [opus.nlpl.eu](https://opus.nlpl.eu/results/en&cs/corpus-result-table).
- Rozdělte text na tokeny (slova). Můžete si vybrat, zda odstraníte interpunkci (pokud čárky a tečky v textu zachováte,
  pak je berte jako jedno slovo) a zda převedete text na malá písmena.
- Vytvořte unigramový (jednoslovný), bigramový (dvoslovný) a trigramový (tříslovný) model frekvencí slov.
- Zobrazte nejčastější n-gramy.

### Cíl:

Naučit se základní práci s textem a pochopit, co jsou n-gramy.

## Pravděpodobnost n-gramů (střední) - 1 bod

### Úkol:

- Na základě Vašich dat vytvořte bigramový a trigramový model přepočítaný na pravděpodobnosti.
- Vypočítejte pravděpodobnost výskytu slova následujícího po určitém n-gramu.
- Zjistěte, co to je Laplaceovo vyhlazování a upravte výpočet pravděpodobnosti.

### Cíl:

Seznámit se s pravděpodobnostním modelem n-gramů a základním vyhlazováním.

## Predikce slova pomocí bigramového modelu (střední) - 1 bod

### Úkol:

- Implementujte jednoduchý autokomplet, který na základě zadaného slova nabídne nejpravděpodobnější následující slovo.
- Testujte na různých vstupech a porovnejte výsledky.

### Cíl:

Pochopit princip predikce slov a být schopen vytvořit základní model automatického doplňování.

## Vytvoření generátoru textu (obtížnější) - 2 body

### Úkol:

- Pomocí trigramového modelu vytvořte jednoduchý generátor textu.
- Model by měl na základě vstupního slova vygenerovat další slova a vytvořit několik vět.

### Cíl:

Aplikovat n-gram model pro tvorbu textu.

## Evaluace modelu pomocí perplexity (náročnější) - 2 body

## Úkol:

- Nastudujte si pojem perplexity.
- Implementujte metodu pro výpočet perplexity trénovaného n-gram modelu.
- Porovnejte perplexity různých n-gram modelů (unigramy, bigramy, trigramy).
- Zhodnoťte kvalitu modelu na testovacím datasetu.

### Cíl:

Naučit se měřit kvalitu jazykového modelu a pochopit význam perplexity.

## Kontrolní otázky k úloze: Automatické doplňování slov (N-gram model)

- Základní seznámení s n-gramy
    - Co je to n-gram?
    - Jak se počítají frekvence unigramů, bigramů a trigramů?
- Pravděpodobnost n-gramů
    - Jak se počítá pravděpodobnost výskytu slova v rámci n-gramového modelu?
    - Co je Laplaceovo vyhlazování a proč se používá?
- Predikce slova pomocí bigramového modelu
    - Jak lze použít bigramový model pro automatické doplňování slov?
    - Jak se určuje nejpravděpodobnější následující slovo?
- Vytvoření generátoru textu
    - Jak lze trigramový model využít pro generování textu?
    - Jak lze vylepšit kvalitu generovaného textu?
- Evaluace modelu pomocí perplexity
    - Co je to perplexity a jak se vypočítává?
    - Jaká je interpretace hodnoty perplexity pro jazykový model?
    - Jak se perplexity mění s rostoucí velikostí n-gramu?

# Cvičení 2: Přímé vyhledávání v textových datech - 7 bodů

V této úloze budete analyzovat různé algoritmy pro vyhledávání vzorů v textu. Zaměříte se na porovnání tří algoritmů:
hrubé síly, Knuth-Morris-Pratt (KMP) a Boyer-Moore-Horspool (BMH). Cílem je pochopit, kdy je který algoritmus výhodnější
a jak se chovají při různých typech textů a vzorů. Při řešení této úlohy můžete používat umělou inteligenci v libovolném
rozsahu.

## Příprava implementací (jednodušší) - 2 body

### Úkol:

- Připravte implementace tří algoritmů: hrubá síla, KMP a BMH (můžete využít AI).
- Upravte algoritmy tak, aby vracely nejen nalezené pozice vzoru, ale i počet porovnání znaků.

### Cíl:

Získat implementace algoritmů a zajistit, aby poskytovaly statistiky o porovnání znaků.

## Testování na různých datech (střední) - 2 body

### Úkol:

Otestujte implementace na třech různých typech textů:

- Krátký text (~100 znaků)
- Dlouhý text (~1000 znaků)
- Náhodně generovaný text z malé abecedy (cca do čtyř znaků, např. sekvence DNA „AGCTAGCT…“)

Pro každý typ textu proveďte testy s alespoň třemi různými vzory.

### Cíl:

Ověřit, jak algoritmy fungují na různých datech.

## Porovnání počtu porovnání znaků (střední) - 1 bod

### Úkol:

- Zaznamenejte počet porovnání znaků pro každý algoritmus a každou testovací sadu.
- Vytvořte tabulku s výsledky.

### Cíl:

Kvantifikovat efektivitu algoritmů.

## Vizualizace výkonu algoritmů (střední) - 1 bod

### Úkol:

- Vytvořte graf, který ukáže efektivitu algoritmů v závislosti na délce textu a vzoru.

### Cíl:

Graficky zobrazit výkonnost algoritmů.

## Analýza a rozhodování o vhodnosti algoritmů (náročnější) - 1 bod

### Úkol:

Odpovězte na následující otázky:

- Kdy se KMP chová lépe než BMH?
- Kdy je BMH rychlejší než Brute Force?
- Kdy je KMP nevýhodné používat?
- Jak algoritmy fungují na textech s opakujícími se vzory?

### Cíl:

Pochopit silné a slabé stránky jednotlivých algoritmů.

## 🎯 Bonusová úloha (+2 body navíc)

### Úkol:

Navrhněte hybridní přístup:

- Vytvořte heuristiku, která na základě délky a vlastností vzoru a textu vybere nejvhodnější algoritmus.
- Porovnejte výkonnost této strategie oproti jednotlivým algoritmům.

# Cvičení 3: Automatická oprava slov a vyhledávání s chybou - 7 bodů

V této úloze budete implementovat algoritmus pro automatickou opravu slov a analyzovat efektivitu různých přístupů k
vyhledávání slov s chybou. Základní inspirací pro implementaci je známý algoritmus Petera Norwiga. Vaším cílem bude
implementovat výpočet editační vzdálenosti a následně vytvořit systém pro automatickou opravu slov na základě
pravděpodobnosti výskytu slov ve slovníku. Při řešení této úlohy můžete používat umělou inteligenci v libovolném
rozsahu.

## Výpočet editační vzdálenosti (jednodušší) - 3 body

### Úkol:

- Implementujte algoritmus pro výpočet Levenshteinovy vzdálenosti mezi dvěma slovy.
- Otestujte vaši implementaci na vlastních datech – vyberte několik dvojic slov a ověřte, zda vzdálenost odpovídá
  očekávání.

### Cíl:

Porozumět principu editační vzdálenosti a zajistit správnou implementaci.

## Implementace automatické opravy slov (střední) - 4 body

### Příprava slovníku - 1 bod

- Vytvořte slovník slov na základě zvoleného datasetu.
- Uložte frekvenci jednotlivých slov, aby bylo možné určit jejich pravděpodobnost výskytu.

### Generování variant slov - 1 bod

- Pro vstupní slovo vygenerujte všechny možné varianty slov s editační vzdáleností maximálně 2.
- Uvažujte operace vložení, smazání, nahrazení, prohození sousedů.
- Zjistěte počet variant.

### Výběr nejpravděpodobnějšího slova - 1 bod

- Z vygenerovaných variant vyberte nejpravděpodobnější slovo podle jeho četnosti ve slovníku.
- Opravte následující větu:  
  _Dneska si dám oběť v restauarci a pak půjdu zpěť domů, kde se podívám na televezí._

### Alternativní přístup a porovnání efektivity - 1 bod

- Místo generování variant vypočítejte editační vzdálenost ke všem slovům ve slovníku a vyberte nejbližší kandidáty.
- Porovnejte výpočetní složitost a kvalitu výsledků obou přístupů.
- Pro délku _n_ nějakého slova určete počet vygenerovaných variant.

## 🎯 Bonusová úloha (+2 body navíc)

- Vylepšení výpočtu pravděpodobnosti pomocí n-gram modelu:
    - Navrhněte a implementujte vylepšený systém, který využívá n-gramy a podmíněné pravděpodobnosti pro určení
      nejpravděpodobnějšího opraveného slova.
    - Porovnejte výsledky s původním přístupem a analyzujte zlepšení.

# Cvičení 4: Boolean Information Retrieval – Invertovaný index a dotazy – 7 bodů

V této úloze si vyzkoušíte základní principy booleovského vyhledávání v textových datech. Vytvoříte invertovaný index s
normalizací tokenů, naparsujete a vyhodnotíte dotazy se závorkami a různými logickými operátory. Následně rozšíříte svůj
systém o kompaktní reprezentaci indexu a analyzujete jeho efektivitu. Úloha je určena pro hlubší pochopení principů
klasického IR modelu. Při řešení této úlohy můžete používat umělou inteligenci v libovolném rozsahu.

### Ke studiu

- [Introduction to Information
  Retrieval – Chapter 1, pages 7-9.](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf)

## Úkoly

### Invertovaný index s normalizací – 1 bod

**Úkol:**

- Vytvořte textový korpus (alespoň 50 dokumentů) a vytvořte invertovaný index.
- Při tvorbě indexu odstraňte stop slova.
- Index by měl uchovávat také četnost výskytu každého slova v jednotlivých dokumentech.

**Cíl:**  
Seznámit se s reprezentací textového korpusu pomocí invertovaného indexu a předzpracováním textu.

### Parsování a vyhodnocení složitých boolean dotazů – 2 body

**Úkol:**

- Implementujte zpracování boolean dotazů včetně operátorů `AND`, `OR`, `NOT`.
- Vyhodnocení proveďte jako množinové operace nad invertovaným indexem.

**Cíl:**  
Naučit se správně parsovat složitější logické dotazy a efektivně je vyhodnocovat.

### Efektivita a velikost indexu – 1 bod

**Úkol:**

- Analyzujte velikost vytvořeného indexu: kolik obsahuje tokenů, průměrná délka seznamů, celkový počet záznamů.

**Cíl:**  
Uvědomit si nároky invertovaného indexu.

### Rozhraní pro dotazování a srovnání dotazů – 1 bod

**Úkol:**

- Vytvořte jednoduché rozhraní (např. konzolové nebo skriptové), které umožní zadání libovolného dotazu a zobrazení
  výsledků.
- Zobrazte alespoň ID dokumentů nebo první větu z každého nalezeného dokumentu.

**Cíl:**  
Usnadnit práci s vyhledávačem a demonstrovat dopad různých dotazů na výstup.

### Rozšířený boolean model s váhováním – 2 body

**Úkol:**

- Navrhněte a implementujte jednoduchý rozšířený boolean model, který umožňuje přiřazení skóre dokumentům.
- Na vstupu je dotaz, na výstupu seřazený seznam dokumentů podle skóre relevance.
- Porovnejte kvalitu výsledků oproti čistému boolean přístupu.

**Cíl:**  
Seznámit se s principem váženého boolean modelu a úvodem do relevance ranking.

# Cvičení 5: Vektorový model a výpočet tf-idf – 7 bodů

V tomto cvičení si vyzkoušíte praktickou práci s vektorovým modelem reprezentace dokumentů. Ručně spočítáte tf-idf váhy,
porovnáte dokumenty pomocí kosinové podobnosti a zamyslíte se nad limity této metody. Úloha je navržena tak, abyste
porozuměli principům vážení slov a podobnosti dokumentů, nikoli jen použili hotové funkce. Při řešení této úlohy můžete
používat umělou inteligenci pro implementaci i konzultaci návrhu, ale výstupy musí být vaším vlastním zpracováním a
interpretací a očekává se vaše schopnost problematiku vysvětlit, nikoli pouze předložit výstup nástroje.

## Předzpracování textu – 1 bod

### Úkol:

- Vyberte si dataset s alespoň 20 dokumenty, můžete použít např. [NLTK](https://www.nltk.org/nltk_data/), Gutenberg,
  Twitter, recenze atd.
- Pro zadané dokumenty proveďte:
    - převod na malá písmena,
    - odstranění interpunkce,
    - tokenizaci,
    - odstranění stopslov.
- Vytvořte si vlastní seznam stopslov (alespoň 5 výrazů).
- Výstupem by měl být seznam termů pro každý dokument.

**Cíl:** Seznámit se s manuálním předzpracováním textu a připravit jej pro vektorovou reprezentaci.

## Výpočet tf a idf – 2 body

### Úkol:

- Spočítejte term frequency (tf) každého slova *t* ve všech dokumentech.
    - Použijte nějakou formu normování četnosti (relativní četnost) nebo zdůvodněte použití nenormované verze.
- Spočítejte inverse document frequency (idf) s využitím vzorce:

  $$  idf(t) = \log \left(\frac{N}{df(t)}\right)  $$

- Spočítejte tf-idf váhy:

  $$  tf\text{-}idf(t,d) = tf(t,d) \times idf(t)  $$

- Spočítejte skóre pro termy v dotazu *q*:

  $$  Score(q,d) = \sum_{t \in q} tf\text{-}idf(t,d)  $$

- Vraťte dokumenty setříděné podle skóre.

**Cíl:** Porozumět výpočtu jednotlivých komponent tf-idf a jejich významu v kontextu textového korpusu.

## Výpočet tf-idf a kosinová podobnost – 2 body

### Úkol:

- Spočítejte kosinovou podobnost mezi všemi dvojicemi dokumentů.
- Určete, které dva dokumenty jsou si nejpodobnější, a interpretujte proč.
- Jak by se výsledky změnily, kdyby se použilo jen tf bez idf?

**Cíl:** Prakticky aplikovat vektorový model a pochopit princip výpočtu podobnosti dokumentů.

## Význam idf v různých doménách – 1 bod (úvaha)

### Úkol:

- Uveďte příklad oblasti nebo tématu, kde by častá slova mohla být navzdory vysoké frekvenci velmi důležitá.
- Vysvětlete, proč v takovém případě může být použití klasického idf nevhodné.
- Navrhněte úpravu výpočtu, která by tento problém zmírnila.

**Cíl:** Kriticky zhodnotit omezení vektorového modelu a navrhnout jeho úpravy pro konkrétní situace.

## Návrh alternativního váhovacího schématu – 1 bod (úvaha)

### Úkol:

- Navrhněte váhovací schéma pro krátké texty (např. tweety), které by lépe zachytilo význam slov než klasické tf-idf.
- Popište, jak by vaše schéma vážilo slova:
    - velmi častá napříč korpusem,
    - vyskytující se pouze jednou,
    - vyskytující se v části dokumentů.

**Cíl:** Podpořit kreativní přístup k návrhu vlastních modelů a pochopení významu jednotlivých komponent vážení.

## Cvičení 6: Komprese invertovaného indexu – 5 bodů

(Tohle se nepovedlo, nedostal jsem body za druhou polovinu, tak na to tu upozorňuji, nepochopil jsem to, špatně jsem si
to obhájil a šel jsem na to úplně špatně, implementaci kompresních algoritmů jsem udělal správně, ale pak jsem to špatně
otestoval, místo genereování ASCII znaků jsem měl generovat čísla, což mi dává menší smysl, ale budiž, v zdaání jsou
slova, ale nebudu se hádat, tím pádem ty velikosti a časy nedávají smysl, čas jsem taky testoval na jednom dokumentu,
což je moje chyba, protože to takhle není statisticky správně, měl jsem to testovat na více dokumentech a počítat
průměr, tím že jsem to nepochopil a už jsem (ne)dostal body, tak to tu jen tak napíšu místot toho, abych to opravoval)

V tomto cvičení si vyzkoušíte různé metody bezztrátové komprese seznamu dokumentových identifikátorů (docIDs) v
invertovaném indexu. Zaměříte se na jejich implementaci, experimentální vyhodnocení kompresního poměru i vlivu na
rychlost vyhledávání. Úloha vás provede základními technikami komprese pomocí kódování rozdílů a univerzálních kódů.

### Implementace kompresních algoritmů – 3 body

**Úkol:**

- Implementujte kompresi i dekompresi tří univerzálních kódů:
    - **Unární kódování/dekódování** (1 bod)  
      Vysvětlete a naimplementujte.
    - **Eliasovo gamma kódování/dekódování** (1 bod)  
      Vysvětlete a naimplementujte.
    - **Fibonacciho kódování/dekódování** (1 bod)  
      Vysvětlete a naimplementujte.
- Kódová slova reprezentujte v textové podobě, není nutné je ukládat binárně.

**Cíl:**  
Pochopit princip univerzálního kódování a vytvořit funkční implementaci pro experimenty.

### Simulace dat a kódování – 1 bod

**Úkol:**

- Vygenerujte slovník s 1000 náhodnými slovy a předpokládejte kolekci s 10 000 dokumenty.
- Vytvořte milion náhodných unikátních dvojic *(slovo, docID)* a sestavte invertovaný seznam docIDs pro každé slovo.
- Seznam docIDs pro každé slovo seřaďte a zakódujte jako sekvenci rozdílů mezi po sobě jdoucími hodnotami, zvlášť pro
  každý ze tří kódovacích algoritmů.

**Cíl:**  
Ověřit funkčnost komprese na synteticky vytvořených datech a připravit podklady pro srovnání velikostí.

### Srovnání velikostí a rychlosti – 1 bod

**Úkol:**

- Porovnejte velikost zakódovaného seznamu se seznamem nezakódovaným (např. jako seznam čísel v textové podobě).
- Otestujte vyhledávání konkrétního docID ve všech variantách a určete rozdíl v době běhu (např. pomocí časové funkce).

**Cíl:**  
Kvantitativně zhodnotit přínos i cenu komprese z pohledu velikosti a výkonu.

### Doporučené zdroje:

- [Prezentace ACS – slide kódování s proměnlivým počtem bytů](https://homel.vsb.cz/~vas218/pdf/acs/lecture3-ext.pdf)
- [Introduction to Information Retrieval – Kapitola 5, str. 96–98](https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf)
- [Článek: Porovnání různých prefixových kódů](https://pdfs.semanticscholar.org/add5/81f36e848c47c4a1d7a0d1b72acc0ced7420.pdf)
