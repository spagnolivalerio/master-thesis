# SD 1.5 Kaggle e latent unmixing per classe

Il confronto dei campioni disponibili favorisce latent unmixing sulle firme spettrali medie e sulla variabilità spettrale fra immagini. Le tavole RGB mostrano inoltre più dettaglio in diverse classi per latent unmixing, mentre entrambi conservano difetti visibili. Questi risultati non isolano l’effetto del numero di parametri.

## Dati e protocollo

- SD: `data/results_kaggle_sd/eurosat_sd15_analysis`, 100 TIFF per ciascuna delle 10 classi, 64×64×13, riflettanza float32. Metadati: SD 1.5, LoRA rank 16, VAE multispettrale esterno, 50 passi, CFG 5.
- Latent unmixing: campioni dell’evaluation esistente della run DDPM 1, 100 immagini per classe. Il suo checkpoint best disponibile è all’epoca 100, pesi EMA, validation MSE 0,0473411; usa il best encoder della run 2, 6 basi per classe, 50 passi e CFG 3.
- Riferimento comune: tutte le 1.350 immagini del test set memorizzato nel checkpoint latent unmixing. Le 130 medie reali classe/banda coincidono esattamente con quelle nel CSV Kaggle; coincidono anche le 130 medie SD ricalcolate dai TIFF.
- Ordinamento TIFF verificato: B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11, B12, B8A. Nei grafici le bande sono ordinate per lunghezza d’onda.
- Le medie macro pesano ogni classe allo stesso modo. Non vengono confrontate immagini accoppiate pixel per pixel.

## Risultati aggregati

| Metrica | SD 1.5 + LoRA | Latent unmixing |
|---|---:|---:|
| MAE RMC ↓ | 0.011425 | 0.003640 |
| RMSE RMC ↓ | 0.014459 | 0.004669 |
| Angolo fra RMC, gradi ↓ | 3.055 | 0.859 |
| W1 delle medie spettrali per immagine ↓ | 0.016723 | 0.008546 |
| Std tra immagini / reale, riferimento 1 | 0.593 | 1.112 |
| Std spaziale per immagine / reale, riferimento 1 | 1.134 | 1.087 |
| Gradiente RGB / reale, riferimento 1 | 0.677 | 0.738 |

La riduzione del MAE RMC con latent unmixing è del 68.1%. La differenza macro MAE_SD − MAE_unmixing ha intervallo bootstrap al 95% [0.004793, 0.008388], ricampionando immagini all’interno delle classi. Non è un intervallo sulla variabilità fra training indipendenti.

## MAE RMC per classe

| Classe | SD 1.5 | Latent unmixing | Minore errore osservato |
|---|---:|---:|---|
| AnnualCrop | 0.017454 | 0.001069 | Latent unmixing |
| Forest | 0.010851 | 0.001892 | Latent unmixing |
| HerbaceousVegetation | 0.011946 | 0.004152 | Latent unmixing |
| Highway | 0.003783 | 0.004645 | SD |
| Industrial | 0.015873 | 0.005701 | Latent unmixing |
| Pasture | 0.017542 | 0.008303 | Latent unmixing |
| PermanentCrop | 0.018886 | 0.002572 | Latent unmixing |
| Residential | 0.004542 | 0.003666 | Latent unmixing |
| River | 0.006553 | 0.002543 | Latent unmixing |
| SeaLake | 0.006822 | 0.001858 | Latent unmixing |

Latent unmixing ha MAE inferiore in 9 classi su 10. Highway è l’unica eccezione, ma il suo intervallo bootstrap sulla differenza include zero. Anche Residential e River hanno intervalli che includono zero: il vantaggio puntuale non è conclusivo per queste singole classi.

## Qualità spaziale e diversità

Le tavole usano tre campioni casuali per classe e sorgente con la stessa scala RGB. Nei campioni esaminati:

- SD appare più sfocato su Residential, PermanentCrop e HerbaceousVegetation; su Industrial compaiono aree luminose tondeggianti e contorni poco definiti.
- Latent unmixing mostra in diversi campioni strade, edifici e margini più distinguibili, ma anche dominanti cromatiche blu e tessiture artificiali. Non raggiunge il dettaglio e la naturalezza dei campioni reali.
- Forest è troppo liscio in entrambi: il gradiente RGB è circa il 34% di quello reale. SeaLake produce frequentemente campioni uniformi, senza rappresentare tutta la varietà delle coste visibile nel test.
- La std tra immagini delle medie spettrali è circa il 59% del reale per SD e il 111% per latent unmixing. È un indizio di ridotta diversità spettrale per SD, non una diagnosi sufficiente di mode collapse né una misura completa della diversità semantica.
- La media dei gradienti RGB vale il 68% del reale per SD e il 74% per latent unmixing. I rapporti cambiano molto fra classi: non costituiscono una graduatoria percettiva e possono crescere anche per rumore.

## Valori di riflettanza

SD presenta 0.715% di valori banda/pixel negativi e 0.00072% superiori a 1. Gli estremi nei TIFF SD sono -0.09912 e 1.66895. Il riferimento reale presenta 0.00412% di valori fuori [0,1].

Le negatività sono particolarmente evidenti in SWIR nelle classi acquatiche: in River risultano negativi il 18,84% dei pixel B11 e il 20,80% dei pixel B12; in SeaLake l’11,41% e il 15,41%. Le RMC medie da sole nascondono questo problema.

I TIFF latent unmixing non hanno valori negativi, ma il codice di salvataggio li taglia a zero: questo non dimostra che il modello non ne produca prima del salvataggio. Applicando lo stesso taglio a SD, il suo MAE RMC diventa 0,011387 contro 0,003640 di latent unmixing: il vantaggio sulle RMC resta sostanzialmente invariato.

## Interpretazione per il prossimo esperimento

Questi risultati non mostrano un vantaggio della pipeline SD esaminata rispetto al latent unmixing per classe. Il confronto coinvolge anche VAE/decoder, pretraining, LoRA, loss e CFG differenti, quindi non permette di attribuire la differenza alla dimensione della U-Net.

Per interpretare l’esperimento large già preparato conviene separare la qualità delle ricostruzioni del primo stadio dalla qualità della generazione. La sfocatura SD osservata è compatibile con varie cause; senza immagini reali ricostruite dal suo VAE non si può stabilire se il limite sia già nel VAE o nella diffusione.

## Limiti

- Mancano log e manifest dello split di training SD: non si può controllare un eventuale overlap fra il suo training e questo test set.
- I campioni latent unmixing sono attribuiti alla run 1 dalla directory e dal flusso dell’evaluation. Manca un manifest con hash ed epoca per certificare che siano stati prodotti proprio dal best oggi disponibile.
- Non sono stati calcolati FID/KID o metriche percettive con un encoder adatto al multispettrale. Il giudizio visivo riguarda le tavole campionate; RMC e W1 coprono statistiche spettrali aggregate, non tutta la coerenza fisica pixel per pixel.

## Artefatti

- [Curve RMC](rmc_comparison.png) e [PDF](rmc_comparison.pdf)
- [Tavola RGB: prime cinque classi](rgb_comparison_1.png)
- [Tavola RGB: altre cinque classi](rgb_comparison_2.png)
- [Grafici delle metriche](metrics_comparison.png)
- [Metriche per classe](metrics_per_class.csv)
- [Curve e negatività per banda](spectral_curves.csv)
- [Risultati, metodologia e file selezionati](comparison.json)
