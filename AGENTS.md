# AGENTS.md

# Progetto

Questo repository riguarda la generazione di immagini satellitari multispettrali tramite modelli a diffusione (Diffusion Models).

L'obiettivo non è solamente ottenere immagini visivamente realistiche, ma soprattutto garantire la coerenza fisica delle bande spettrali generate.

Le metriche e i criteri di valutazione evolveranno nel tempo. Attualmente sono presenti:

- curva RMC per classe (firma spettrale media);
- confronto aggregato delle RMC reali e generate per tutte le classi;
- // aggiungere nuove metriche man mano che vengono introdotte.

---

# Dataset

Il progetto utilizza il dataset EuroSAT in due varianti:

- EuroSAT RGB
- EuroSAT Multispectral (MS)

---

# Struttura del progetto

La working directory principale è:

```
/main
```

Il filesystem è organizzato come segue.

## data/

Contiene esclusivamente i dataset.

Non salvare output o artefatti all'interno di questa directory.

---

## experiments/

Ogni esperimento deve avere una propria sottocartella.

Un esperimento contiene esclusivamente il codice necessario al suo funzionamento, ad esempio:

- training
- sampling
- evaluation
- definizione del modello
- utility strettamente legate all'esperimento
- config.json
- cartella `log/` per i log incrementali del training

Le funzioni di valutazione specifiche di un esperimento devono rimanere all'interno della relativa directory.

### experiments/common/

Contiene esclusivamente componenti riutilizzabili da più esperimenti, come:

- runtime di training;
- modello e processo di diffusione condivisi;
- sampling ed evaluation generici;
- gestione di dataset, checkpoint, logging e output.

Il codice in `common/` non deve importare moduli appartenenti a uno specifico esperimento. Ogni esperimento deve avere un entry point sottile che seleziona il proprio `config.json` e le sole componenti specifiche, riutilizzando il runtime comune.

Non duplicare all'interno degli esperimenti codice già disponibile in `common/`. Spostare in `common/` una componente soltanto quando è realmente condivisa e non contiene logica specifica di un esperimento.

---

## outputs/

Contiene tutti gli output visivi e i risultati prodotti dagli esperimenti.

I log testuali del training rappresentano l'unica eccezione e devono essere salvati nella cartella `log/` del relativo esperimento.

Nessuna immagine deve essere salvata al di fuori di questa directory.

La struttura è la seguente.

### outputs/previews/

Contiene le immagini generate durante il training.

Per ogni esperimento viene creata una cartella

```
outputs/previews/<nome_esperimento>/
```

Ogni training genera una sottodirectory

```
<nome_esperimento>-run_i
```

dove `i` identifica la run.

---

### outputs/results/

Contiene i risultati finali ottenuti tramite lo script di sampling.

Per ogni esperimento viene creata una cartella

```
outputs/results/<nome_esperimento>/
```

Ogni sampling genera una sottodirectory


```
<nome_esperimento>-sampling_i-run_j
```

dove `j` identifica la run del checkpoint utilizzato per il sampling.

contenente le immagini generate in:

```
samples/<classe>/MS/
samples/<classe>/RGB/
```

Il sampling deve completare e salvare una classe alla volta. Per ogni TIFF multispettrale deve essere salvata anche la corrispondente visualizzazione RGB.

I TIFF multispettrali generati devono essere salvati in `float32` e contenere direttamente i valori di riflettanza, senza fattore di scala intero.

Ogni evaluation genera invece:

```
<nome_esperimento>-eval_i-run_j/
```

dove `j` identifica la run del checkpoint utilizzato per l'evaluation.

contenente:

- `samples/` con le immagini utilizzate per la valutazione e suffisso `_eval`;
- `plots/` con tutte le figure ottenute dall'analisi.

---

# Evoluzione del progetto

Il filesystem definito sopra rappresenta la struttura ufficiale del progetto.

Non creare nuove directory nella root della repository.

Qualunque modifica alla struttura del progetto richiede esplicita approvazione.

---

# Configurazione degli esperimenti

Ogni esperimento deve possedere un file

```
config.json
```

che contiene:

- iperparametri
- configurazioni globali
- flag
- percorsi
- impostazioni del training

I file di configurazione devono essere il più possibile omogenei tra esperimenti.

Le stesse opzioni devono mantenere gli stessi nomi quando possibile.

All'inizio del file inserire un blocco commentato che descriva le principali sezioni configurabili.

## Documentazione degli esperimenti

Ogni nuovo esperimento deve aggiungere una scheda sintetica in `README.md` con:

- approccio;
- architettura;
- condizionamento;
- loss;
- focus dell'esperimento.

La scheda deve spiegare soltanto l'idea fondamentale e le differenze rispetto agli altri esperimenti. Non deve duplicare iperparametri, percorsi o dettagli operativi già presenti nel relativo `config.json`.

---

# Checkpoint

Ogni esperimento deve possedere una propria directory dei checkpoint.

Un checkpoint completo deve essere salvato alla fine di ogni epoca.

Mantenere due checkpoint distinti:

- `latest`, aggiornato a ogni epoca e utilizzato per il resume;
- `best`, aggiornato soltanto quando migliora la validation loss e utilizzato per sampling ed evaluation.

Quando l'esperimento utilizza EMA, validation, preview, checkpoint `best`, sampling ed evaluation devono usare i pesi EMA. Il checkpoint `latest` deve conservare sia i pesi di training sia lo stato EMA per il resume.

Il checkpoint deve contenere tutto il necessario per un resume fedele, inclusi:

- pesi del modello;
- stato dell'optimizer;
- stato del mixed-precision scaler;
- epoca raggiunta;
- split del dataset;
- stati dei generatori casuali;
- configurazione della run.

Il training deve poter:

- partire da zero;
- riprendere automaticamente dall'ultimo checkpoint;
- permettere all'utente di scegliere facilmente tra resume e nuovo training.

Il meccanismo di resume deve essere semplice, affidabile e trasparente.

---

# Split e bilanciamento del dataset

Ogni esperimento deve utilizzare uno split stratificato e riproducibile:

- 90% training;
- 5% validation;
- 5% test.

Il test set non deve essere utilizzato durante il training o per scegliere il checkpoint.

Il training deve utilizzare batch bilanciati per classe. Quando possibile, il batch size deve essere un multiplo del numero di classi, in modo che ogni batch contenga lo stesso numero di campioni per ciascuna classe.

---

# Logging del training

Ogni training deve mostrare una progress bar aggiornata durante ogni epoca.

I log devono essere salvati incrementalmente in:

```
experiments/<nome_esperimento>/log/
```

Ogni run deve possedere un file di log separato. Preferire il formato JSONL, in modo che ogni record venga scritto immediatamente senza dover mantenere l'intero log in memoria.

Registrare almeno:

- loss corrente e loss media;
- valore individuale e media di ogni componente o metrica restituita dalla loss;
- validation loss a ogni epoca;
- learning rate;
- norma del gradiente;
- epoca, batch e step globale;
- throughput;
- utilizzo massimo della VRAM, quando disponibile;
- metriche sintetiche delle preview utili a una valutazione preliminare.

Le immagini di preview devono continuare a essere salvate esclusivamente in `outputs/previews/`.

Una preview deve essere generata e salvata alla fine di ogni epoca.

---

# Interfaccia modulare delle loss

Il runtime di training condiviso non deve conoscere i nomi o la semantica delle loss specifiche degli esperimenti.

Ogni esperimento che introduce una loss personalizzata deve fornire una factory con interfaccia semplice:

```python
def create_loss(config, scheduler, band_stats):
    return loss_function
```

La funzione restituita riceve immagini pulite, immagini rumorose, rumore reale, rumore predetto, timestep ed etichette reali. Una loss che deve distinguere i campioni condizionati da quelli sottoposti a CFG dropout imposta `requires_conditioning_context = True` e riceve `model_labels` come argomento keyword. Una loss che necessita di ulteriori forward imposta `requires_model_context = True` e riceve anche `model` e `model_labels`. Il risultato deve essere un dizionario di tensori scalari:

```python
return {
    "loss": total_loss,
    "mse_loss": mse_loss,
    "physical_metric": physical_metric,
}
```

La chiave `loss` è obbligatoria ed è l'unico valore utilizzato per il backward e per determinare il checkpoint `best`. Tutte le altre chiavi rappresentano componenti o metriche valutate individualmente.

Il runtime deve automaticamente, senza condizioni specifiche per un esperimento:

- calcolare la media progressiva di ogni valore restituito;
- mostrare ogni media nella progress bar di training e validation;
- salvare valori correnti e medi nei log JSONL;
- includere le medie di validation nel log di epoca.

Utilizzare nomi brevi ma espliciti e stabili. Quando una componente viene pesata nella loss totale, esporre separatamente il valore originale e il contributo pesato se entrambi sono utili per interpretare il training. Le metriche destinate soltanto al monitoraggio devono essere scollegate dal grafo quando non devono contribuire al gradiente.

Una loss può esporre un dizionario opzionale `metric_labels` per assegnare etichette compatte alle metriche nella progress bar. Le chiavi dei log devono restare quelle originali e descrittive. Il runtime deve mostrare le metriche prima delle informazioni secondarie come throughput e VRAM, affinché rimangano visibili anche nei terminali stretti.

---

# Best practices

Il codice deve essere:

- modulare;
- leggibile;
- autoesplicativo;
- ben commentato quando necessario;
- semplice da seguire.

Il flusso di esecuzione deve risultare facilmente comprensibile leggendo il codice.

Preferire sempre soluzioni semplici rispetto a soluzioni sofisticate.

---

## Evitare

- duplicazione del codice;
- verbosità inutile;
- overengineering;
- meccanismi eccessivamente astratti;
- controlli difensivi non necessari (assumere dataset ben formati);
- fallback inutili;
- utilizzo eccessivo di parametri da CLI;
- meccanismi di lock;
- tecniche poco convenzionali se non strettamente motivate;
- dipendenze inutili.

---

## Preferire

- modularità;
- efficienza;
- correttezza teorica;
- semplicità;
- riutilizzabilità del codice;
- configurazioni centralizzate;
- componenti indipendenti e facilmente testabili.

# Regole per le modifiche

Quando modifichi il codice:

- modifica il minor numero possibile di file;
- mantieni lo stile già presente nel progetto;
- non cambiare API pubbliche senza necessità;
- non introdurre nuove dipendenze se non strettamente necessarie;
- non rinominare file o directory senza approvazione;
- mantieni la retrocompatibilità quando possibile;
- evita refactoring estesi se non richiesti esplicitamente.

## Firme e type hint

Le firme di funzioni e metodi devono essere brevi e immediatamente leggibili.

Preferire firme semplici, ad esempio:

```python
def __init__(self, root, transform=None):
```

Non utilizzare nelle firme:

- `Callable` completi;
- tipi annidati;
- `Union` o union con `|`;
- annotazioni lunghe o decorative;
- import da `typing` necessari soltanto a costruire firme complesse.

Se un type hint rende la firma più difficile da leggere, ometterlo. Utilizzare annotazioni soltanto quando sono semplici e chiariscono realmente il codice.
