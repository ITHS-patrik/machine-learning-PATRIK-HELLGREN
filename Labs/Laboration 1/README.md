# 🎬 Movie Recommender

En interaktiv rekommendationsmotor byggd med bland annat **Python**, **scikit‑learn**, **pandas** och **Streamlit**.
Projektet låter användaren ladda upp MovieLens‑data, ställa in hyperparametrar och generera personliga filmrekommendationer baserat på både **content‑** och **collaborative filtering**.
<br><br>

## 🧬 Funktionalitet


### ✔️ 1. Ladda upp data
Användaren laddar upp följande CSV‑filer:

- `movies_enriched.csv`* (kan laddas ner via applikationen vid behov)
- `media.csv` (kan laddas ner via applikationen vid behov)
- `ratings.csv`
- `tags.csv`
- `links.csv`

*: `movies_enriched.csv` är berikad med ytterligare metadata för alla ~86 500 filmerna i form av genres (kompletterat), plot/overview, keywords, cast och director(s) i syfte att förbättra content-delen av hybridmodellen. All data hämtades via TMDb-API:n.

CSV-filerna ovan används för att bygga:

- metadata‑profiler för filmerna
- en TF‑IDF‑matris
- en LSA‑matris
- en user‑item‑matris

När alla filer är uppladdade aktiveras knappen `Train model` (se steg 2).


### ✔️ 2. Ställ in hyperparametrar och träna modellen

I appen kan användaren justera/välja:

- **Diversifiera**: ja/nej
- **Antal kluster**: en film från varje kluster väljs (förutsatt att diversifiera är satt till ja)
- **Top‑n**: antal rekommendationer som önskas
- **LSA-komponenter**: antal dimensioner i LSA-matrisen
- **Alpha**: balansen mellan content‑ och collaborative-filtrering

Applikationen har defaultvärden på samtliga hyperparametrar som är satta från början om användaren vill träna modellen direkt istället. Dessa värden kan återställas genom att användaren trycker på knappen "Reset values".

När användaren klickar på `Train model`-knappen händer följande:

- datan läses in
- rekommendationsmodellen byggs (processen visas stegvis jämte knappen)
- textfältet för sökning låses upp


### ✔️ 3. Sök efter en film

Användaren skriver in en filmtitel.

- **Titeln matchas genom:**
    - prefix-matchning *(.startswith() på normaliserade titlar, e.g. "Matrix, The" -> "The Matrix")*
    - substring-matchning *(sökningen finns någonstans i titeln)*
    - fuzzy‑matchning *(bäst träff genereras även vid stavfel, o.d.)*
- **Rekommendationer genereras automatiskt baserat på den bästa träffen.**
    - användaren kan välja en annan film från en drop-down med andra filmtitlar som matchade sökningen.
- **Rekommendationerna presenteras i en tabell med värdena "Movie ID", "Title", "Content score", "Collaborative score" & "Hybrid Score".**
    - om diversify är satt till "Ja" inkluderas även en kolumn för "Cluster no." för att visa vilket kluster respektive film är tagen ifrån.


### ✔️ 4. Posters och trailers

När rekommendationerna har genererats så hämtas även trailers, posters & metadata för varje rekommenderad film (visas under steg 1-3).

- **Här kan användaren:**
    - expandera fältet *"Movie info"* för att få information om genrar, handling, regissör och skådespelare.
    - zooma in på postern för att se den i fullstorlek.
    - spela upp trailern.

<br>

## 🚀 Så här kör du projektet

#### Klona repot:
```bash
    git clone https://github.com/ITHS-patrik/machine-learning-PATRIK-HELLGREN.git
    cd <projektmapp>"\Labs\Laboration 1"
```

#### Skapa och aktivera ett venv:
```bash
    python -m venv venv
    source venv/bin/activate    # macOS/Linux
    venv\Scripts\activate       # Windows
```

#### Installera beroenden:
```bash
    pip install -r requirements.txt
```

#### Starta Streamlit‑appen:
```bash
    streamlit run app.py        # se till att terminalens path står i projektets root (Laboration 1)
```
