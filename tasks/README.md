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
