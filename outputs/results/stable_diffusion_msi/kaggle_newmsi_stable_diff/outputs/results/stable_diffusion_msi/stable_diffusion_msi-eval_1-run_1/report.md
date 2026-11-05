# Stable Diffusion multispettrale — evaluation sul test

Checkpoint SD: epoca 10, pesi EMA, SHA-256 `8c21daa667a280e3fd32d00f0de116e3b2618098ec2cdb994be2b674f5207280`.
VAE: epoca 9, SHA-256 `62cf636bbc8a95cf698f00a29f3ff041395df7a5ddcbcf49b02fd8973ac8aa1f`.
Riferimento: 1350 immagini test. Campioni generati: 1000.

| Metrica macro | VAE ricostruzione | SD generazione |
|---|---:|---:|
| rmc_mae | 0.002559 | 0.014272 |
| rmc_rmse | 0.003170 | 0.017796 |
| rmc_sam_deg | 1.271690 | 3.217747 |
| mean_spectrum_w1 | 0.002830 | 0.017300 |
| between_image_std_ratio | 0.981183 | 0.776349 |
| spatial_std_ratio | 0.929536 | 0.902641 |
| negative_fraction | 0.006878 | 0.012248 |
| above_one_fraction | 0.000032 | 0.000004 |
| reconstruction_mae | 0.007133 | — |
| reconstruction_rmse | 0.011581 | — |
| reconstruction_sam_deg | 3.186221 | — |
| sam_valid_pixel_fraction | 1.000000 | — |

Ogni classe pesa allo stesso modo nelle medie macro. Gli errori di ricostruzione sono calcolati su coppie reali/ricostruite; le metriche di generazione confrontano insiemi indipendenti.
La W1 confronta le distribuzioni delle medie di banda delle immagini, poi viene mediata sulle bande. Non misura l'intera distribuzione spettrale congiunta dei pixel.
La variabilità fra immagini è la media sulle bande delle deviazioni standard delle loro medie; quella spaziale è la media delle deviazioni standard dentro le immagini. Entrambi i rapporti hanno riferimento reale 1.
Il SAM delle RMC confronta firme medie di classe. Il SAM di ricostruzione confronta pixel corrispondenti ed esclude vettori con norma <= 1e-8; la frazione valida è esportata.
Le fasce RMC sono ±1 SD fra immagini, non intervalli di confidenza. Le metriche descrivono questa run e questo campionamento, senza stimare la variabilità fra training indipendenti.
I TIFF conservano la riflettanza senza clipping; i PNG usano B4/B3/B2 e lo stesso punto di bianco. I valori fuori [0,1] sono diagnostici, disponibili anche per il reale nel CSV delle bande.

- [RMC per classe](plots/rmc_all_classes.png)
- [RMC macro](plots/rmc_macro.png)
- [Metriche per classe](plots/metrics_comparison.png)
- [Confronto RGB](plots/rgb_comparison.png)
- [Metriche CSV](metrics_per_class.csv)
- [Curve e statistiche per banda](spectral_curves.csv)
- [Statistiche per immagine](image_statistics.csv)
- [Manifest e metriche JSON](evaluation.json)
