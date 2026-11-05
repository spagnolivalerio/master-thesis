# Latent unmixing per class

La [guida tecnica dettagliata](DOCUMENTAZIONE_TECNICA.md) collega obiettivi,
architettura e codice: include motivazioni delle normalizzazioni, conversioni
ILR, diagrammi dei flussi, checkpoint e punti da modificare. Preview e analisi
dei risultati sono trattate separatamente dalla pipeline del modello.

La variante possiede dieci istanze indipendenti dello stesso encoder e dieci
banchi di basi apprendibili, uno per classe EuroSAT. Ogni banco contiene `K`
spettri, configurati tramite `model.num_bases` in `config.json`.

Un campione attraversa esclusivamente l'encoder associato alla sua classe e
viene ricostruito esclusivamente con il banco della stessa classe. Il DDPM e`
unico e condizionale, ma opera su coordinate ILR standardizzate con media e
deviazione standard distinte per classe. Durante la generazione la label
seleziona sia le statistiche ILR inverse sia il banco usato dal decoder.

Il monitor dell'encoder mostra soltanto i dieci banchi spettrali e le RMC
reali/ricostruite. Non vengono prodotti grafici delle abundance.

```bash
python experiments/latent_unmixing_per_class/train.py
python experiments/latent_unmixing_per_class/train.py --stage encoder
python experiments/latent_unmixing_per_class/train.py --stage ddpm
python experiments/latent_unmixing_per_class/evaluation.py
```

Per calcolare la FID RGB di ogni classe presente in `eval/all_samples/<classe>`:

```bash
pip install pytorch-fid==0.3.0
python experiments/latent_unmixing_per_class/eval/eval_fid.py
```

Lo script confronta tutte le immagini generate con quelle della stessa classe
in EuroSAT RGB. Usa [pytorch-fid](https://github.com/mseitzer/pytorch-fid) con
feature Inception-v3 a 2048 dimensioni e salva tabelle TXT e CSV con FID e
conteggi. Percorsi e batch sono nel config, nelle voci `fid_*`. La FID riguarda
le immagini RGB e non misura la coerenza delle tredici bande multispettrali.
Le tabelle includono anche la FID reali–reali sullo stesso identico insieme:
è un controllo numerico con risultato atteso vicino a zero.

