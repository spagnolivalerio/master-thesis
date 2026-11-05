# Evaluation latent unmixing per class large

Checkpoint best della run 1, epoca 100, pesi EMA. Generati 100 campioni per ciascuna delle 10 classi, per un totale di 1.000 TIFF multispettrali float32 e 1.000 PNG RGB.

- Sampling: 50 passi DDIM, CFG 3.0, batch 8, seed base 42.
- Dispositivo: NVIDIA GeForce RTX 5070 Ti; precisione del modello float32.
- Tempo totale: 9.34 minuti; picco VRAM allocata da PyTorch: 1866 MiB.
- Ogni classe viene completata e salvata prima della successiva. I TIFF contengono riflettanza diretta, senza scala intera, e hanno forma 64×64×13.
- Riferimento: Prime 100 immagini per classe degli indici test salvati nel checkpoint, come nello script evaluation.

## Metriche RMC

MAE macro: **0.003827**. RMSE macro: **0.004941**. Ogni classe pesa allo stesso modo nella media macro.

| Classe | MAE RMC | RMSE RMC |
|---|---:|---:|
| AnnualCrop | 0.006124 | 0.008046 |
| Forest | 0.002151 | 0.003221 |
| HerbaceousVegetation | 0.002691 | 0.003566 |
| Highway | 0.003478 | 0.004550 |
| Industrial | 0.002601 | 0.003438 |
| Pasture | 0.005609 | 0.006685 |
| PermanentCrop | 0.002793 | 0.004193 |
| Residential | 0.006807 | 0.008075 |
| River | 0.004580 | 0.005709 |
| SeaLake | 0.001438 | 0.001931 |

## Risultati

- [Curve RMC reali e generate](plots/rmc_all_classes.png)
- [Esempi RGB per tutte le classi](plots/rgb_examples.png)
- [Metriche per classe in CSV](metrics_per_class.csv)
- [Manifest, configurazione e metriche](evaluation.json)
- Campioni: `samples/<classe>/MS/` e `samples/<classe>/RGB/`, suffisso `_eval`.

Il MAE delle RMC misura la fedeltà degli spettri medi di classe; non è una misura completa della qualità visiva o della coerenza spettrale pixel per pixel.
