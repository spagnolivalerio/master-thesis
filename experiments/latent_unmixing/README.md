# Latent unmixing

Pipeline autosufficiente a due stadi:

1. `train_encoder.py` apprende l'encoder delle abundance e le basi spettrali
   globali con ricostruzione lineare `A @ E`;
2. `train_ddpm.py` congela il primo stadio e apprende un DDPM condizionale sulle
   coordinate ILR standardizzate delle abundance.

`utils.py` espone esplicitamente `to_ILR` e `from_ILR`. La prima applica floor,
chiusura sul simplesso, ILR di Helmert e standardizzazione; la seconda applica
de-standardizzazione, inversa ILR e nuova chiusura sul simplesso. Le statistiche
ILR sono calcolate esclusivamente sul training set.

Il monitor dell'encoder salva soltanto l'evoluzione delle basi latenti e le RMC
medie reali/ricostruite. Non produce grafici delle abundance.

```bash
python -m experiments.latent_unmixing.train
python -m experiments.latent_unmixing.train --stage encoder
python -m experiments.latent_unmixing.train --stage ddpm
python -m experiments.latent_unmixing.evaluation
```

