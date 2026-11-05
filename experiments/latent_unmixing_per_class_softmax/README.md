# Latent unmixing per class softmax

Esperimento indipendente derivato da `latent_unmixing_per_class`. Mantiene
encoder e basi spettrali distinti per classe, ma addestra il DDPM sui **K logits
degli encoder**, senza trasformazione ILR. Dopo il denoising, la softmax produce
le abundance usate dal decoder lineare.

Il modello, i training, le utility e gli entry point sono locali a questa
directory: non importano gli altri esperimenti e non riutilizzano i loro
checkpoint. Il dataset EuroSAT MS resta quello condiviso in `dataset.py`.
Dimensioni delle reti, basi e iperparametri iniziali seguono la variante base;
il riferimento completo è [config.json](config.json).

## Architettura e flussi

Lo stadio encoder ricostruisce l'immagine standardizzata:

```text
immagine MS x + label y
  → encoder della classe y
  → K logits l
  → softmax(l / temperatura)
  → abundance a
  → miscela lineare con le basi della classe y
  → ricostruzione x_hat
```

Si ottimizzano encoder e basi con `MSE(x_hat, x)`. La softmax è la stessa
funzione in training e nella decodifica dei campioni generati.

Per il secondo stadio si carica il `best` encoder di questo esperimento e si
congelano sia gli encoder sia le basi. Le statistiche dei logits sono stimate
sui soli pixel del training set:

```text
l = encoder_y(x).logits                       [B,K,H,W]
z = (l - media_y) / std_y                     [B,K,H,W]
z_t = sqrt(alpha_bar_t) z + sqrt(1-alpha_bar_t) noise
prediction = U-Net(z_t, t, model_labels)      [B,K,H,W]
loss = MSE(prediction, noise)
```

La label reale seleziona sempre l'encoder e le statistiche. Soltanto
`model_labels`, passata alla U-Net, può essere sostituita dalla label nulla
per la classifier-free guidance. Non si applica la softmax ai tensori rumorosi
o alla predizione del rumore: il processo gaussiano lavora su valori liberi.

La generazione percorre invece:

```text
rumore gaussiano [B,K,H,W] + classe richiesta
  → denoising DDIM con classifier-free guidance
  → logits standardizzati z
  → l = z * std_y + media_y
  → a = softmax(l / temperatura)
  → miscela con le basi della classe y
  → immagine standardizzata
  → denormalizzazione delle bande → riflettanza
```

La softmax non è una proiezione euclidea sul simplesso: è una mappa da logits
reali a pesi positivi con somma uno. Per evitare zeri da underflow numerico,
il codice applica un limite inferiore minimo in `float32` e rinormalizza.
Non si effettuano proiezioni sui passi intermedi del denoising.

## Differenze rispetto alla ILR e motivazioni

| Aspetto | Variante ILR | Questa variante |
| --- | --- | --- |
| Target pulito della diffusione | Coordinate ILR delle abundance | Logits direttamente prodotti dall'encoder |
| Canali della U-Net | `K-1` | `K` |
| Conversione iniziale | Logaritmo, centraggio e base Helmert | Standardizzazione affine dei logits |
| Inversa prima del decoder | Inversa affine, base Helmert, softmax | Inversa affine e softmax con la temperatura dell'encoder |
| Statistiche latenti | Per classe e coordinata ILR | Per classe e canale dei logits |

La standardizzazione dei logits conserva scale comparabili per le classi e
per i canali su cui agisce la stessa schedule di rumore. Non rende la
distribuzione gaussiana e non allinea i materiali fra classi. Le std hanno un
limite inferiore configurabile per evitare divisioni per zero o amplificazioni
eccessive nelle coordinate quasi costanti.

È necessario invertire la standardizzazione **prima** della softmax: applicare
la softmax direttamente a `z` cambierebbe i rapporti delle abundance. Anche
la temperatura deve essere coerente con quella dell'encoder; viene pertanto
salvata in `logit_stats` nel checkpoint DDPM.

La normalizzazione delle **bande** resta distinta: divisione per la scala del
dataset e standardizzazione con statistiche globali del training. Il decoder
opera in questo spazio e l'inversa restituisce riflettanza. La MSE di
ricostruzione pesa implicitamente l'errore di ogni banda con l'inverso della
sua varianza. La scala RGB è usata soltanto per visualizzare i risultati.

### Limiti del confronto

La softmax soddisfa `softmax(l + costante) = softmax(l)`: i K logits hanno una
direzione ridondante, non determinata dalla sola ricostruzione. Questa variante
mantiene i logits grezzi, senza centraggio; la U-Net impara anche tale componente.
Il confronto con ILR cambia quindi sia le coordinate sia il numero di canali.
Non è un confronto a identico numero di parametri della U-Net.

Positività e somma uno riguardano le **abundance**. Le basi spettrali rimangono
parametri liberi e non sono necessariamente materiali reali. La softmax non
garantisce da sola la plausibilità fisica dell'immagine: valgono ancora il
limite dell'inviluppo convesso delle basi e la necessità di valutare i risultati
multispettrali. Un miglioramento rispetto alla ILR va misurato.

## Mappa del codice

| File / simbolo | Responsabilità |
| --- | --- |
| [model.py](model.py): `AbundanceEncoder`, `ClassSpecificLatentUnmixingModel` | Routing per classe, logits e decoder lineare |
| [model.py](model.py): `create_ddpm_model`, `generate_logits` | U-Net a K canali, schedule e generazione con CFG |
| [utils.py](utils.py): `compute_logit_statistics`, `to_latents` | Stima delle statistiche sul training e standardizzazione |
| [utils.py](utils.py): `from_latents`, `softmax_abundances` | Inversa della standardizzazione e vincolo sulle abundance |
| [train_encoder.py](train_encoder.py), [train_ddpm.py](train_ddpm.py) | Training separati, validation e checkpoint |
| [sample.py](sample.py), [evaluation.py](evaluation.py) | Sampling e confronto RMC sul test |
| [test_experiment.py](test_experiment.py) | Controlli numerici e prova completa su TIFF sintetici |

Non sono presenti trasformazioni Helmert, logaritmi delle abundance o
statistiche ILR. I checkpoint DDPM contengono `logit_stats`, oltre a modello
di unmixing, statistiche delle bande, classi e split.

## Utilizzo e resume

Dalla root, usando l'ambiente Python del progetto:

```bash
python -m experiments.latent_unmixing_per_class_softmax.train
python -m experiments.latent_unmixing_per_class_softmax.train --stage encoder
python -m experiments.latent_unmixing_per_class_softmax.train --stage ddpm
python -m experiments.latent_unmixing_per_class_softmax.sample --n 100
python -m experiments.latent_unmixing_per_class_softmax.sample --class River --n 100
python -m experiments.latent_unmixing_per_class_softmax.evaluation
```

Il comando senza `--stage` esegue i due stadi in sequenza. I flag `resume`
sono inizialmente attivi in entrambe le sezioni: in assenza di checkpoint si
parte da zero; altrimenti si riprende il rispettivo `latest` dall'epoca
successiva. Impostare `resume: false` nello stadio desiderato per una nuova run.

Dopo aver cambiato la rappresentazione o riaddestrato gli encoder, iniziare
anche una nuova run DDPM. Un DDPM in resume continua a usare l'unmixing
incorporato nel proprio checkpoint. La ricerca dei checkpoint sceglie il file
del tipo richiesto modificato più recentemente, non il migliore fra tutte le run.

`latest` conserva optimizer, scheduler LR, scaler, epoca, step globale, stati
RNG, split e config. Nel DDPM conserva separatamente pesi di training e stato
EMA. Validation, preview e `best` DDPM usano EMA; `best` è un checkpoint per
inferenza. La scelta del best è sulla validation MSE, non sul test o sulla RMC.

Per riprendere fedelmente mantenere invariati dati, architettura,
normalizzazioni, temperatura, schedule e impostazioni di training: gli script
leggono il config corrente, senza sostituirlo automaticamente con quello
salvato nel checkpoint. I vecchi checkpoint ILR non sono compatibili con questa
diffusione a K canali.

## Preview, file e analisi

I training mostrano progress bar e scrivono JSONL a ogni batch nelle directory
`log/` dell'esperimento: loss corrente e media, LR, grad norm quando viene
calcolata, epoca, batch, step globale, throughput e picco VRAM quando disponibile.
I record di epoca includono validation loss e metriche delle preview.

Gli encoder salvano basi e RMC reali/ricostruite a ogni epoca; il DDPM salva
una preview RGB con EMA e seed fisso. Questi file restano sotto `outputs/previews/`.

Il sampling completa una classe alla volta e salva, sotto `outputs/results/`,
coppie `samples/<classe>/MS/*.tif` e `samples/<classe>/RGB/*.png`. I TIFF sono
`float32` in riflettanza, senza fattore di scala intero. I valori negativi
vengono portati a zero nell'esportazione; non viene imposto un massimo al TIFF.
Il PNG usa B4/B3/B2 e la scala di visualizzazione del config.

L'evaluation genera campioni con suffisso `_eval`, confronta le RMC con il test
set e salva grafico e riepilogo `evaluation.json`. Si può valutare una directory
già generata con `--path <directory_samples>`, purché contenga tutte le classi.
Le metriche confrontano firme medie, non coppie di immagini corrispondenti.

## Verifica locale

```bash
python -m unittest experiments.latent_unmixing_per_class_softmax.test_experiment
```

Il test non richiede EuroSAT: usa piccole immagini sintetiche e reti ridotte
per verificare trasformazioni, statistiche per classe, training, resume, EMA,
sampling ed evaluation. I TIFF e le preview temporanei restano sotto `outputs/`
e vengono rimossi alla fine; log e checkpoint di test usano directory temporanee
nelle rispettive cartelle dell'esperimento. Non misura la qualità generativa.
