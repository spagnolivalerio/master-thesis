# FID RGB: reali, latent unmixing e SD 1.5

La FID è una distanza fra due insiemi, non una proprietà assoluta di una singola immagine o del dataset reale. Un insieme confrontato con sé stesso ha FID zero per definizione. Per un riferimento empirico utile è stato confrontato il test set selezionato con un secondo gruppo reale disgiunto dalla validation.

## Risultati

Tutti i confronti misurati usano lo stesso riferimento: 1.000 immagini reali del test set, 100 per ciascuna delle dieci classi. Anche ogni insieme confrontato contiene 1.000 immagini, 100 per classe.

| Insieme confrontato con il test reale | FID RGB ↓ |
|---|---:|
| Secondo insieme reale, dalla validation | **39.0048** |
| Latent unmixing per class base | **77.8577** |
| Latent unmixing per class large | **80.3925** |
| SD 1.5 + LoRA, VAE multispettrale | **132.7453** |

Il valore più basso indica una maggiore somiglianza tra le distribuzioni di feature RGB. Su questi campioni il modello base ottiene un punteggio leggermente migliore del large; entrambi ottengono un punteggio inferiore alla pipeline SD valutata.

Il confronto reale-validation contro reale-test produce circa 39 anche senza immagini generate. È un riferimento empirico influenzato dai campioni finiti: non rappresenta un errore delle immagini reali, un minimo teorico universale o una quantità da sottrarre automaticamente alle altre FID.

## Protocollo

- Riferimento test: prime 100 immagini di ogni classe dagli indici test salvati nei checkpoint; i due modelli latent hanno gli stessi indici test.
- Secondo gruppo reale: prime 100 immagini di ogni classe dagli indici validation. È stata verificata l’assenza di sovrapposizioni con test e training. La validation è stata usata durante il training per selezionare i checkpoint; qui viene usata soltanto come secondo insieme di immagini reali.
- Base e large: tutti i 100 TIFF `_eval` per ciascuna classe delle evaluation complete. Nessun nuovo sampling.
- SD: tutti i 100 TIFF per classe presenti in `data/results_kaggle_sd/eurosat_sd15_analysis/generated_reflectance/`. L’ordine delle tredici bande è stato verificato. I metadati indicano SD 1.5, LoRA rank 16, VAE multispettrale, 50 passi e CFG 5.
- Rendering identico per tutti: B4/B3/B2; i TIFF reali sono divisi per 10.000, quelli generati sono già in riflettanza. Punto di bianco 0,3, clipping in [0,1], arrotondamento uint8 e riconversione float32 in [0,1].
- Inception v3 specifica per FID, con pesi TensorFlow convertiti e feature pool3 a 2.048 dimensioni. Resize bilineare a 299×299, `align_corners=False`, normalizzazione interna in [−1,1].
- Implementazione pytorch-fid 0.3.0, installata soltanto in /tmp. La chiamata a SciPy `sqrtm` è stata adattata alla versione installata rimuovendo l’argomento obsoleto `disp=False`; la formula resta invariata.
- Inferenza float32 su RTX 5070 Ti, TF32 disabilitato; statistiche in float64. Le feature e le statistiche dei reali-test sono state riutilizzate esattamente per tutti i confronti. È stata verificata l’identità dell’hash dei pesi Inception.

## Limiti

La FID RGB non misura direttamente la coerenza fisica delle tredici bande. Non separa fedeltà e diversità e non attribuisce le differenze a specifiche parti del modello.

Con 1.000 immagini per insieme resta il bias da numerosità finita, potenzialmente dipendente dal modello: usare la stessa numerosità controlla il protocollo ma non elimina il bias. La differenza fra base e large di circa 2,5 punti non è accompagnata da un intervallo di confidenza e richiede più campioni prima di sostenere una superiorità generale. [Chong e Forsyth, CVPR 2020](https://arxiv.org/abs/1911.07023).

I valori dipendono dal rendering e dal resize dichiarati, quindi non vanno confrontati direttamente con benchmark che usano altri preprocessing o numerosità. [Parmar et al.](https://arxiv.org/abs/2104.11222).

## File e riproducibilità

- [Risultati, configurazione, hash dei pesi ed elenco esatto di tutti i file](fid_results.json)
- [Punteggi CSV](fid_scores.csv)
- Feature: `real_inception_features.npy`, `real_validation_inception_features.npy`, `base_inception_features.npy`, `large_inception_features.npy`, `sd15_inception_features.npy`.
- Medie e covarianze: corrispondenti file `*_fid_statistics.npz`.
- [Implementazione di riferimento](https://github.com/mseitzer/pytorch-fid).
