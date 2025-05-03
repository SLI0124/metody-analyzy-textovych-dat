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

# Cvičení 7: Překlad slov pomocí vektorových reprezentací – 9 bodů

V tomto cvičení si vyzkoušíte přenos významu mezi jazyky pomocí word embeddingů a lineární transformace. Nejprve získáte
dvojjazyčná data (vektorové reprezentace a překladové dvojice), následně implementujete metodu pro učení transformační
matice pomocí gradient descent a nakonec ověříte kvalitu překladu pomocí přesnosti. Cvičení spojuje praktické
programování s pochopením matematického základu.

## Příprava dat – 2 body

**Vstupní otázka:** Co to je embedding?

**Úkol:**

- Stáhněte si dva embeddingy z webu [FastText](https://fasttext.cc/docs/en/crawl-vectors.html) (např. čeština a
  angličtina), v textovém formátu `.vec`.
- Stáhněte si překladové dvojice z
  projektu [MUSE – bilingual dictionaries](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries),
  konkrétně soubory `cs-en train` a   `cs-en test`.
- Načtěte embeddingy a vytvořte matice **X** (zdrojový jazyk) a **Y** (cílový jazyk) pro trénovací a testovací část.

**Cíl:** Získat konzistentní embeddingy a překladové páry pro experimenty bez nutnosti vytvářet vlastní slovník.

## Implementace trénovacího algoritmu – 5 bodů

**Vstupní otázka:** Co dělá gradient descent?

**Úkol:**

- Implementujte funkci pro výpočet Frobeniovy normy: $\|XW^\top - Y\|_F^2$ *(1 bod)*
- Vypočítejte výraz $XW^\top - Y$ *(0.5 bodu)*
- Odvoďte gradient ztrátové funkce vůči matici $W^\top$ *(1 bod)*
- Implementujte gradient ztrátové funkce vůči matici $W^\top$ *(1 bod)*
- Implementujte gradient descent s parametrem `alpha` (učící koeficient) *(1 bod)*
- Vytvořte učení ve smyčce: pevný počet kroků a zároveň konvergence (např. musí dojít k poklesu loss funkce v posledních
  deseti iteracích) *(0.5 bodu)*

**Cíl:** Pochopit a implementovat trénovací proces pro učení lineární transformace mezi jazyky.

## Překlad a vyhodnocení – 2 body

**Vstupní otázka:** Jak bychom museli změnit matice X a Y, pokud bychom použili matici W místo $W^\top$?

**Úkol:**

- Na základě naučené matice **W** implementujte funkci, která přeloží zadané slovo – výstupem bude 5 nejpodobnějších
  slov v cílovém jazyce dle kosinové podobnosti *(1 bod)*
- Otestujte přesnost na testovacích slovech a spočítejte přesnost překladu (accuracy - top 1, případně accuracy - top 5)
  *(1 bod)*

**Cíl:** Ověřit, že naučená transformace umožňuje smysluplný překlad mezi jazyky.

## Doporučené zdroje

- [Prezentace - překlady](https://homel.vsb.cz/~vas218/docs/matd/translation.pdf)
- [FastText – předtrénované modely](https://fasttext.cc/docs/en/crawl-vectors.html)
- [MUSE – dvojjazyčné slovníky](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries)

# Cvičení 8: CBOW model pro češtinu – 20 nebo 30 bodů (cvičení a projekt)

V tomto cvičení si vytvoříte vlastní model typu Continuous Bag of Words (CBOW) pro češtinu. Cílem je naučit model
předpovídat slovo na základě jeho kontextu a tím získat kvalitní distribuované vektorové reprezentace slov (embeddingy).
Úloha má dvě varianty: použití existujících knihoven (20 bodů) nebo plná implementace od nuly (30 bodů).

## Příprava a zpracování dat – 5 bodů

**Vstupní otázka:** Co je CBOW model a jak funguje?

**Úkol:**

- Stáhněte nebo použijte připravený český korpus (např. z hugginface.co nebo výřez Wikipedie).
- Proveďte tokenizaci a vytvořte slovník (např. 10 000 nejčastějších slov).
- Vytvořte trénovací dvojice: kontextová slova (např. 2 vlevo + 2 vpravo) → cílové slovo.
- Slova mimo slovník označte jako <UNK>.

**Cíl:** Připravit data pro trénink CBOW modelu ve formátu vhodném pro učení.

## Trénování modelu – 10 bodů (varianta A) nebo 20 bodů (varianta B)

**Vstupní otázka:** Co se učí CBOW model a jak se optimalizují jeho váhy?

**Varianta A – použití knihovny (max. 20 bodů celkem):**

- Využijte PyTorch, Keras, TensorFlow nebo jinou knihovnu s Embedding vrstvou.
- Implementujte architekturu CBOW a trénujte model pomocí CrossEntropyLoss a optimalizátoru (SGD/Adam).

**Varianta B – plná implementace (max. 30 bodů celkem):**

- Inicializujte matice E, W, b pomocí např. normálního rozdělení.
- Vypočítejte průměr embeddingů pro kontext.
- Implementujte softmax, ztrátu (křížová entropie), výpočet gradientů a SGD update ručně (např. v NumPy).
- Proveďte trénink přes více epoch s výpisem průběhu lossu.

**Cíl:** Porozumět principu CBOW a naučit se jej implementovat a trénovat.

## Vyhodnocení modelu – 5 bodů

**Úkol:**

- Pro vybraná slova (např. „pes", „škola", „krásný") najděte 5 nejbližších sousedů podle kosinové podobnosti.
- Vizualizujte embedding prostor pomocí t-SNE nebo PCA.
- Zhodnoťte, zda embeddingy zachycují významovou podobnost.

**Cíl:** Ověřit kvalitu naučených embeddingů a jejich interpretaci.

## Doporučené zdroje:

- [Jurafsky & Martin – Speech and Language Processing (kap. 6)](https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf)
- [Datasety](https://huggingface.co/datasets?task_categories=task_categories:text-classification&language=language:cs&p=1&sort=trending)
- [Prezentace – CBOW model (ke cvičení)](https://homel.vsb.cz/~vas218/docs/matd/cbow-prezentace.pdf)

# Cvičení 9: Transformer pro sumarizaci dialogů – 21 bodů (cvičení a projekt)

V tomto cvičení si vyzkoušíte implementaci modelu typu encoder-decoder založeného na architektuře Transformer pro úlohu
sumarizace krátkých dialogů ze Samsum datasetu. Cílem je vytvořit systém, který dokáže automaticky vygenerovat souhrn
daného dialogu.

## Příprava a zpracování dat – 2 body

**Odpovězte:** Jaká je struktura Samsum datasetu a proč je vhodný pro úlohu sumarizace?

**Úkol:**

- Stáhněte Samsum dataset (např. pomocí knihovny datasets z Hugging
  Face). [Odkaz na HuggingFace](https://huggingface.co/datasets/Samsung/samsum)
- Rozdělte data na trénovací (příp. také validační) a testovací množinu.

**Cíl:** Připravit trénovací a testovací data vhodná pro modelování sumarizace.

## Předzpracování a tokenizace – 2 body

**Odpovězte:** Jak funguje tokenizace a proč je důležitá při práci s modely pro textová data?

**Úkol:**

- Vyberte libovolný předtrénovaný tokenizer.
- Tokenizujte vstupní texty (dialogy) i výstupy (shrnutí).

**Cíl:** Převést textová data do tokenizovaného formátu vhodného pro trénink modelu.

## Implementace Transformer modelu – 6 bodů

**Odpovězte:** Jak funguje encoder-decoder architektura založená na Transformeru?

**Odpovězte:** Proč potřebujeme padding v encoderu a decoderu?

**Úkol:**

- Naimplementujte encoder-decoder model pro generování shrnutí pomocí Transformeru.
- Můžete využít:
    - nn.Transformer z PyTorch,
    - nebo TransformerEncoder a TransformerDecoder vrstvy v Kerasu.
- Přidejte token embedding a poziční encoding pro vstupní i cílové sekvence.
- Implementujte správné maskování:
    - Padding mask – ignorování padding tokenů,
    - Causal mask – zákaz nahlížení na budoucí tokeny v dekodéru.

**Cíl:** Postavit funkční Transformer model připravený k tréninku.

## Trénování modelu – 6 body

**Odpovězte:** Jak se trénuje sekvenční model na úlohu sekvence-sekvence?

**Úkol:**

- Definujte loss funkci (například CrossEntropyLoss pro predikci tokenů).
- Vyberte vhodný optimalizátor (například Adam/AdamW).
- Natrénujte model na trénovacích datech (validujte během trénování).

**Cíl:** Vytrénovat model schopný generovat shrnutí.

## Inference – generování shrnutí – 4 body

**Odpovězte:** Jakým způsobem lze generovat výstupní sekvenci v dekodérovém Transformeru?

**Nápověda:** K čemu nám jsou speciální tokeny <SOS> a <EOS>?

**Úkol:**

- Implementujte funkci, která na základě vstupního dialogu vygeneruje shrnutí.
- Nastudujte si a využijte greedy decoding nebo beam search.

**Cíl:** Umět z trénovaného modelu získat výstupní shrnutí.

## Vyhodnocení modelu – 1 bod

**Úkol:**

- Vyhodnoťte kvalitu generovaných shrnutí pomocí metrik ROUGE-1 a ROUGE-2.
- Porovnejte generovaná shrnutí s referenčními shrnutími v testovací množině.

**Cíl:** Kvantitativně ověřit kvalitu modelu pro sumarizaci.

## Doporučené zdroje:

- [Transformer - Jurafsky, Stanford](https://web.stanford.edu/~jurafsky/slp3/10.pdf)

