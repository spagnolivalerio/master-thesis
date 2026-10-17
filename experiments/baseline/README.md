# Baseline

DDPM condizionale per le 13 bande EuroSAT MS con classifier-free guidance.

L'esperimento mantiene il flusso autosufficiente di `mybaseline` (training,
resume, EMA, preview e sampling locali), ma modello e iperparametri sono letti
dalla configurazione della baseline originale. Non dipende da `common`.

Esecuzione dalla root di `newrepo`:

```bash
python -m experiments.baseline.train
python -m experiments.baseline.sample
```

Il sampling senza argomenti, oppure con `--multi`, genera tutte le classi con
quantità e impostazioni definite in `config.json`.

Per generare soltanto una classe, usare `--single` con il nome esatto della
classe EuroSAT e il numero di campioni desiderato:

```bash
python -m experiments.baseline.sample --single --class Forest --n 100
python -m experiments.baseline.sample --multi
```

`--class` e `--n` sono obbligatori soltanto in modalità `--single`; `--n` deve
essere positivo. Per ogni campione vengono salvati il TIFF multispettrale e
la corrispondente visualizzazione RGB nelle directory di output abituali.
