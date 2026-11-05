# Generazione di immagini satellitari multispettrali

## Latent unmixing per class softmax

- **Approccio:** generazione dei logits degli encoder per classe, seguita da
  softmax per ottenere abundance positive con somma unitaria.
- **Architettura:** encoder e basi indipendenti per classe, decoder lineare e
  unica U-Net DDPM sui K logits standardizzati; nessuna trasformazione ILR.
- **Condizionamento:** la classe seleziona encoder, basi e statistiche dei
  logits e condiziona la U-Net tramite embedding e classifier-free guidance.
- **Loss:** MSE di ricostruzione per gli encoder e MSE sul rumore per il DDPM.
- **Focus:** confrontare la rappresentazione a K logits con le K-1 coordinate
  ILR, applicando il vincolo sul simplesso soltanto prima della decodifica.

Configurazione, flussi e comandi in
[`experiments/latent_unmixing_per_class_softmax/`](experiments/latent_unmixing_per_class_softmax/README.md).

## Latent unmixing per class large

- **Approccio:** variante indipendente di latent unmixing per classe, con maggiore
  capacita sia nella ricostruzione sia nella generazione delle abundance.
- **Architettura:** encoder piu ampio e banco di basi spettrali piu numeroso per
  ciascuna classe; decoder lineare e unica U-Net DDPM piu ampia nello spazio ILR.
- **Condizionamento:** la classe seleziona encoder, basi e statistiche ILR e
  condiziona la U-Net tramite embedding e classifier-free guidance.
- **Loss:** MSE di ricostruzione per gli encoder e MSE sul rumore per il DDPM.
- **Focus:** misurare l'effetto della maggiore capacita su dettagli spaziali e
  coerenza spettrale, con checkpoint, log e output separati dall'esperimento base.

Configurazione e comandi in
[`experiments/latent_unmixing_per_class_large/`](experiments/latent_unmixing_per_class_large/README.md).

## Stable Diffusion multispettrale (Kaggle)

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
