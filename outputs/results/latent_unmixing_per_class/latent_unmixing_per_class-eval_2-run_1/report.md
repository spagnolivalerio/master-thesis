# Le abundance generate variano fra classi?

**Sì. Il best reagisce alla classe e modifica sia i pesi di miscela sia le mappe spaziali delle abundance.** Le differenze dipendono sia dal condizionamento della U-Net sia dalle statistiche ILR specifiche di classe. Non si tratta soltanto di decodificare le stesse mappe con basi spettrali diverse.

## Checkpoint e campioni

- Checkpoint: `experiments/latent_unmixing_per_class/ddpm_checkpoints/latent_unmixing_per_class_ddpm-run_1-best.pt`, epoca 100, pesi ema, validation MSE 0.04734111.
- SHA-256: `584d0578cf3893434dbadee1d704ccb1084a4bffc64d6a81d0254102ac26ef58`.
- Analisi statistica: 100 TIFF esistenti per classe, 1.000 immagini complessive, ciascuna con 6 mappe di abundance 64×64.
- Controllo diretto del best: una nuova immagine per classe, stesso identico tensore di rumore iniziale, seed 20261030, 50 passi DDIM e CFG 3. Le abundance originali sono salvate in `samples/<classe>/abundances.npy`, assieme ai TIFF MS float32 e alle visualizzazioni RGB.

## Le medie cambiano fra classi

| Classe | A1 | A2 | A3 | A4 | A5 | A6 | Componenti effettive per pixel |
|---|---:|---:|---:|---:|---:|---:|---:|
| AnnualCrop | 11.5% | 24.2% | 17.7% | 18.3% | 16.5% | 11.8% | 4.53 |
| Forest | 15.9% | 26.1% | 10.5% | 11.0% | 16.5% | 20.0% | 4.89 |
| HerbaceousVegetation | 13.7% | 13.7% | 20.6% | 20.7% | 13.4% | 17.9% | 4.71 |
| Highway | 18.1% | 12.4% | 16.6% | 18.9% | 16.8% | 17.3% | 4.61 |
| Industrial | 17.8% | 14.5% | 21.0% | 12.7% | 14.5% | 19.5% | 4.89 |
| Pasture | 15.2% | 17.9% | 16.1% | 22.1% | 11.0% | 17.7% | 4.87 |
| PermanentCrop | 17.2% | 11.1% | 13.1% | 18.6% | 17.7% | 22.3% | 4.91 |
| Residential | 15.7% | 19.1% | 12.4% | 18.3% | 17.7% | 16.8% | 5.10 |
| River | 21.3% | 26.6% | 12.6% | 10.6% | 12.0% | 16.9% | 4.41 |
| SeaLake | 3.6% | 14.9% | 13.2% | 22.4% | 19.7% | 26.2% | 4.45 |

Le medie non sono uniformi né identiche fra classi. Tutte le componenti hanno peso medio non trascurabile: la più piccola quota media è A1 di SeaLake, circa il 3,6%. Non si osserva un collasso globale su una sola componente.

Il numero effettivo è la media di exp(H), con H = −Σ A_k log(A_k) calcolata per pixel: vale 1 per una miscela pura e 6 per pesi uniformi. I valori osservati sono fra 4,41 e 5,10: le miscele sono generalmente distribuite su più basi, non quasi one-hot.

Le mappe variano anche all’interno delle immagini. La std spaziale media, mediata sulle sei componenti, è circa 0,073 per River, 0,071 per Highway e 0,020 per SeaLake. I valori più bassi di SeaLake sono coerenti con mappe frequentemente più uniformi.

**Gli indici A1…A6 hanno significato locale alla classe.** Ogni classe usa encoder e basi propri, senza allineamento comune degli indici. A2 di Forest e A2 di River non identificano necessariamente lo stesso materiale. La tabella confronta l’uso delle coordinate locali; non esprime percentuali di materiali condivisi.

## Il controllo a rumore iniziale identico conferma il condizionamento

La nuova generazione usa lo stesso rumore per tutte le etichette: le strutture spaziali risultanti cambiano fra classi. Sono stati inoltre separati due meccanismi:

1. **Statistiche ILR:** si mantengono identici i latenti finali di AnnualCrop e si cambiano soltanto media/std ILR inverse in base alla classe.
2. **Condizionamento U-Net:** si usano i latenti generati per le diverse etichette, applicando a tutti le stesse statistiche inverse di AnnualCrop.

Per ogni coppia di classi si misura ½Σ_k|A_k−B_k|. Si considera sia la distanza fra vettori medi dell’immagine sia la media della distanza fra pixel corrispondenti. I valori sono compresi fra 0 e 1.

| Controllo | Distanza media fra vettori medi | Distanza media pixel per pixel |
|---|---:|---:|
| Pipeline completa | 0.1341 | 0.2106 |
| Solo statistiche ILR diverse | 0.1338 | 0.1359 |
| Solo condizionamento U-Net diverso | 0.0577 | 0.1891 |

In questo controllo, le statistiche ILR da sole producono differenze nei vettori medi di entità molto simile alla pipeline completa. La U-Net contribuisce comunque in modo evidente alle mappe spaziali: con statistiche inverse identiche la distanza pixel per pixel rimane 0,1891. I due effetti sono non lineari: questi numeri non sono contributi additivi e non permettono di assegnare percentuali causali.

Anche un forward diretto sullo stesso input rumoroso, prima di qualsiasi trasformazione ILR inversa o decodifica, produce predizioni differenti per le diverse etichette. A timestep 500 la differenza RMS rispetto ad AnnualCrop varia circa da 0,078 a 0,165. Questo conferma che la U-Net utilizza effettivamente il condizionamento.

## Recupero delle abundance dai campioni esistenti

Le abundance dei vecchi campioni non erano state salvate. Sono state recuperate risolvendo il decoder lineare nelle bande positive e imponendo la somma unitaria. Le bande tagliate a zero dal salvataggio sono escluse dal sistema. Il rango della matrice aumentata è completo per tutti i pattern di clipping incontrati, quindi le sei coordinate sono identificabili.

Il massimo errore assoluto nella ricostruzione dei TIFF, riapplicando il decoder e il taglio a zero, è 9.08e-08 in riflettanza. Le sole piccole negatività dovute all’arrotondamento sono corrette prima delle statistiche.

## Limiti

- Le statistiche su 100 immagini per classe descrivono i campioni preesistenti, attribuiti alla run 1 dalla directory. Non esiste un manifest che certifichi l’hash della U-Net che li ha generati. La compatibilità con le basi del checkpoint corrente è invece verificata numericamente.
- La generazione a rumore comune proviene direttamente dal best identificato tramite hash, ma usa un solo seed: è un controllo diagnostico del condizionamento, non un’ablation statistica su molti seed.
- Le basi sono parametri liberi e locali alla classe: la variazione delle abundance non dimostra che siano state identificate frazioni di materiali fisici reali.

## Figure e dati

- [Pesi medi e basi dominanti per classe](plots/abundance_usage_by_class.png)
- [RGB e abundance generate dal best con lo stesso rumore](plots/abundance_maps_same_noise.png)
- [Mappe recuperate dai primi campioni esistenti](plots/abundance_maps_existing.png)
- [Statistiche per classe in CSV](abundance_summary.csv)
- [Risultati e controlli in JSON](abundance_analysis.json)
