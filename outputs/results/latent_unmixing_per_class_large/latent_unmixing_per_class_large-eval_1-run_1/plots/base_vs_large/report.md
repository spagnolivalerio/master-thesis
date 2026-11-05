# Confronto latent unmixing per classe: base e large

**Il large produce immagini più dettagliate nei campioni esaminati, ma una distribuzione delle medie spettrali più concentrata. La variabilità tra immagini scende di circa il 40%; la variabilità spaziale dentro le immagini resta simile.** Il miglioramento complessivo non è uniforme: le RMC medie cambiano poco, mentre la distanza fra le distribuzioni degli spettri medi peggiora.

## Protocollo

- 100 immagini generate per classe e modello, per un totale di 1.000 immagini per ciascun generatore.
- Stesso test set completo di 1.350 immagini: l’identità degli indici nei due checkpoint è stata verificata.
- I due best DDPM sono all’epoca 100 e usano pesi EMA, 50 passi DDIM e CFG 3.
- I primi stadi incorporati nei due checkpoint sono stati valutati sulle stesse immagini test, per distinguere ricostruzione e generazione.
- Le metriche principali qui usano tutto il test set. Lo script evaluation originale usa invece le prime 100 immagini reali per classe: questo spiega il lieve cambiamento dei valori MAE rispetto al riepilogo precedente.

## Modifiche al modello

| Componente | Base | Large |
|---|---:|---:|
| Canali degli encoder | 64 | 128 |
| Parametri per encoder | 304.518 | 1.199.628 |
| Basi per classe | 6 | 12 |
| Coordinate ILR per pixel | 5 | 11 |
| Parametri della U-Net | 27.752.581 | 110.910.731 |

La profondità e la durata del training rimangono uguali. Cambiano contemporaneamente capacità delle reti e rappresentazione latente: il confronto non isola l’effetto della sola U-Net.

## Risultati

| Metrica macro | Base | Large |
|---|---:|---:|
| Std tra immagini / reale, riferimento 1 | 1.112 | 0.672 |
| Std tra immagini sulle bande RGB / reale | 1.035 | 0.614 |
| Std spaziale dentro le immagini / reale | 1.087 | 1.090 |
| Gradiente RGB / reale | 0.738 | 0.976 |
| MAE RMC ↓ | 0.003640 | 0.003493 |
| W1 delle medie spettrali per immagine ↓ | 0.008546 | 0.009756 |

La std tra immagini misura la dispersione dei valori medi di banda delle immagini della stessa classe. Le fasce nei grafici RMC rappresentano questa variabilità. Il rapporto è calcolato per classe, mediando le std sulle bande, e poi mediato uniformemente sulle classi. Non è una misura completa della diversità semantica.

La riduzione macro è -39.6%; l’intervallo bootstrap al 95% sui campioni salvati è [-42.6%, -36.8%]. Il calo riguarda tutte e dieci le classi.

| Classe | Std base / reale | Std large / reale | Variazione |
|---|---:|---:|---:|
| AnnualCrop | 1.144 | 0.821 | -28.2% |
| Forest | 0.904 | 0.541 | -40.2% |
| HerbaceousVegetation | 1.155 | 0.666 | -42.3% |
| Highway | 1.122 | 0.703 | -37.4% |
| Industrial | 1.393 | 0.826 | -40.7% |
| Pasture | 1.131 | 0.663 | -41.3% |
| PermanentCrop | 0.966 | 0.577 | -40.3% |
| Residential | 1.022 | 0.603 | -40.9% |
| River | 1.505 | 0.813 | -46.0% |
| SeaLake | 0.783 | 0.507 | -35.2% |

Un calo non è sempre un peggioramento: River passa da una dispersione eccessiva (1,505) a una sottostimata (0,813), più vicina al reale. Complessivamente però il large sottostima la variabilità in tutte le classi, con rapporti fra circa 0,51 e 0,83.

Il MAE delle RMC migliora solo del 4.0% sul riferimento completo. L’intervallo bootstrap della differenza MAE_large − MAE_base include zero: [-0.002495, 0.001027]. La W1 peggiora del 14.1%: avere una firma media buona non garantisce di coprire bene la distribuzione delle firme delle singole immagini.

## Dove compare la contrazione?

| Primo stadio sulle immagini test reali | Base | Large |
|---|---:|---:|
| RMSE di ricostruzione in riflettanza | 0.009541 | 0.005921 |
| Std tra immagini ricostruite / reale | 0.979 | 0.991 |
| Gradiente RGB ricostruito / reale | 0.685 | 0.827 |

Il large riduce l’RMSE di ricostruzione del 37.9% e conserva circa il 99% della variabilità tra immagini nelle ricostruzioni. Il primo stadio è quindi capace di rappresentarla. La riduzione al 67% compare nei campioni della diffusione; non appare come una perdita obbligata della ricostruzione lineare.

Questo non esclude che il nuovo spazio latente renda il compito del DDPM più difficile: passare da 5 a 11 coordinate ILR cambia la distribuzione da apprendere. Il risultato osservato non identifica da solo la causa della contrazione e non dimostra overfitting o mode collapse.

## Lettura delle immagini e prossima verifica

Nelle tavole, il large mostra più texture su Forest, HerbaceousVegetation e Pasture e contorni più definiti in vari campioni. Il gradiente RGB complessivo si avvicina al reale (0,98 contro 0,74), ma questo indicatore può aumentare anche per texture artificiali e rumore. Non è un punteggio percettivo.

La prima verifica utile sarebbe campionare lo stesso checkpoint large con CFG 1, 2 e 3 e un protocollo comune. La classifier-free guidance regola un compromesso fra qualità e copertura della distribuzione; lo stesso valore numerico non garantisce lo stesso compromesso per due modelli differenti. Questa è un’ipotesi da verificare, non la causa già dimostrata della riduzione. [Ho e Salimans, Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).

## Limiti e file

- Una run di training per variante e 100 campioni per classe. Il bootstrap misura l’incertezza dei campioni, non quella fra training indipendenti.
- I vecchi campioni base non hanno un manifest che certifichi il checkpoint di generazione; sono quelli dell’evaluation della run 1 usati nei confronti precedenti. Il large ha invece manifest con hash del checkpoint.
- Non sono stati eseguiti nuovi sampling a CFG differente né valutazioni percettive FID/KID.

- [Grafici della variabilità](variability_comparison.png)
- [RMC con fasce di variabilità](rmc_and_variability.png)
- [Confronto RGB: prime cinque classi](rgb_comparison_1.png)
- [Confronto RGB: altre cinque classi](rgb_comparison_2.png)
- [Metriche CSV](metrics_per_class.csv)
- [Risultati, metodologia e bootstrap](comparison.json)
