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
