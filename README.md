# Generazione di immagini satellitari multispettrali

## LLUM (Latent Linear Unmixing Model)

- **Approccio:** apprendimento di abundance e basi spettrali specifiche per classe,
  con generazione delle abundance nello spazio ILR tramite DDPM condizionale.
- **Architettura:** encoder convoluzionale, decoder lineare per classe e U-Net DDPM.
- **Condizionamento:** la classe seleziona le statistiche ILR e le basi del decoder,
  e condiziona il DDPM con classifier-free guidance.
- **Loss:** MSE di ricostruzione per l'encoder e MSE sul rumore per il DDPM.
- **Focus:** coerenza fisica delle bande spettrali generate.

Configurazione, flussi e comandi in [`experiments/LLUM/`](experiments/LLUM/).

## DDPM

- **Approccio:** generazione diretta delle immagini multispettrali con DDPM.
- **Architettura:** U-Net condizionale operante sulle 13 bande.
- **Condizionamento:** label di classe e classifier-free guidance.
- **Loss:** MSE sul rumore predetto.
- **Focus:** riferimento per confrontare la generazione diretta con LLUM.

Configurazione e codice in [`experiments/DDPM/`](experiments/DDPM/).

## SD1.5 (Kaggle)

- **Approccio:** fine-tuning del VAE di SD 1.5 su EuroSAT MS, seguito da adattamento
  LoRA della U-Net nello spazio latente del VAE best congelato.
- **Architettura:** AutoencoderKL con ingresso e uscita a 13 bande inizializzati
  casualmente con la stessa regola per tutti i canali; pesi interni e quattro
  canali latenti conservati dal modello preaddestrato. La U-Net usa adapter
  sull'attenzione e sulle convoluzioni di ingresso e uscita; validation e preview
  generative usano i pesi EMA degli adapter.
- **Condizionamento:** VAE non condizionato e diffusione condizionata tramite
  embedding CLIP di prompt fissi per classe, con classifier-free guidance.
- **Loss:** L1 e KL media per coordinata latente per il VAE; MSE sul rumore per
  la diffusione.
- **Focus:** valutare ricostruzione e generazione multispettrale nello scenario
  multiclasse, distinguendo fedelta spaziale e spettrale.

Notebook autonomo in [`kaggle/stable-diffusion-msi.ipynb`](kaggle/stable-diffusion-msi.ipynb).
Include sampling per classe in riflettanza ed evaluation sul test, con confronto
delle ricostruzioni, statistiche spettrali, figure e manifest esportati.

La [documentazione tecnica](kaggle/stable-diffusion-msi.md) collega architettura,
normalizzazioni, flussi dei tensori e funzioni del notebook alle specifiche del
progetto, con una guida ai punti da modificare e alle analisi dei risultati.

Per proseguire solo lo stadio SD di altre dieci epoche, importa su Kaggle
[`stable-diffusion-msi-continue-sd.ipynb`](kaggle/stable-diffusion-msi-continue-sd.ipynb),
collega l'output completo della run precedente e il dataset EuroSAT MS, poi
esegui tutte le celle. Il notebook ripristina lo stato di training da `latest.pt`
e mantiene congelato lo stesso VAE. Al termine esegue sampling dal best EMA
ed evaluation sul test, salvando campioni MS/RGB, metriche, grafici e report.

Per generare **2500 immagini River** dal best EMA della continuazione, usa
[`stable-diffusion-msi-sample-river.ipynb`](kaggle/stable-diffusion-msi-sample-river.ipynb).
Collega l'output completo della continuazione, abilita GPU e Internet ed esegui
tutte le celle: il notebook salva TIFF a 13 bande in riflettanza float32 e PNG RGB
corrispondenti sotto `outputs/results/`, senza richiedere EuroSAT. La continuazione
ora esporta anche il VAE associato; per output precedenti collega anche la run
originale che contiene il suo checkpoint VAE `best.pt`.
