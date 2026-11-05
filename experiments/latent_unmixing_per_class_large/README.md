# Latent unmixing per class large

Esperimento indipendente derivato da `latent_unmixing_per_class`. Aumenta la
larghezza dei dieci encoder, della U-Net condizionale e il numero di basi
spettrali per classe. Il decoder resta una miscela lineare delle basi e il DDPM
genera le coordinate ILR delle abundance, standardizzate per classe.

Il confronto riguarda l'aumento complessivo di capacita: le modifiche simultanee
non permettono di attribuire un eventuale miglioramento a una singola componente.
Loss, profondita delle reti, split, seed e durata del training seguono la variante
base. Tutti i valori e i percorsi sono definiti in `config.json`.

## Utilizzo

Dalla root del repository:

```bash
python -m experiments.latent_unmixing_per_class_large.train
python -m experiments.latent_unmixing_per_class_large.train --stage encoder
python -m experiments.latent_unmixing_per_class_large.train --stage ddpm
python -m experiments.latent_unmixing_per_class_large.sample --n 100
python -m experiments.latent_unmixing_per_class_large.sample --class River --n 100
python -m experiments.latent_unmixing_per_class_large.evaluation
```

Il primo comando esegue i due stadi in sequenza. Ogni stadio riprende dal proprio
checkpoint `latest`, se presente. Per una nuova run impostare `resume: false`
nella relativa sezione di training; per ripartire con entrambi gli stadi,
impostarlo in entrambe. Dopo un nuovo training degli encoder avviare anche una
nuova run DDPM, in modo da usare le nuove abundance e le nuove basi.

Il DDPM carica il checkpoint `best` degli encoder di questo esperimento;
validation, preview e checkpoint `best` DDPM usano i pesi EMA. Il `latest`
conserva pesi di training, EMA, optimizer, scheduler, scaler, split e stati RNG.

Il sampling completa una classe alla volta e salva coppie TIFF MS in riflettanza
`float32` e visualizzazioni RGB. L'evaluation genera campioni con suffisso `_eval`
e confronta le RMC con il test set. Le immagini restano sotto `outputs/` e i log
JSONL sono scritti a ogni batch nelle directory dedicate del nuovo esperimento.

Il training richiede piu memoria e tempo rispetto alla variante base. Il batch
di training resta bilanciato su tutte le classi; il gradient checkpointing della
U-Net e gia abilitato nella configurazione.
