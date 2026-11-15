# Latent unmixing per class: flusso dei dati

Documentazione tecnica del codice in questa directory. Per ogni componente sono descritti dati in ingresso, trasformazioni, risultati e file scritti. Le forme usano B = batch, N = classi, C = bande, K = basi, H×W = dimensioni spaziali.

## 1. Sequenza dei dati

Il dataset fornisce TIFF multispettrali e label. Il codice calcola uno split stratificato, stima le statistiche delle bande sul train e standardizza le immagini.

Lo stadio encoder usa immagine standardizzata e label per produrre abundance e ricostruzione. Lo stadio DDPM usa abundance trasformate in coordinate ILR standardizzate come campioni puliti. In generazione, rumore iniziale viene denoised dalla U-Net, convertito in abundance e passato al decoder per ottenere l'immagine multispettrale.

    TIFF MS -> [C,H,W] + label
      -> normalizzazione bande
      -> encoder della label -> abundance [K,H,W]
      -> decoder con basi della label -> ricostruzione [C,H,W]
      -> ILR per classe -> [K-1,H,W]
      -> training DDPM: rumore + timestep -> U-Net -> rumore predetto
      -> sampling: denoising -> ILR -> abundance -> decoder
      -> TIFF MS float32 + preview PNG RGB

Le statistiche bande, le basi iniziali e le statistiche ILR sono stimate usando gli indici di training. Validation è usata per loss e selezione best. Gli indici test sono salvati nei checkpoint e utilizzati dalle evaluation finali.

## 2. Configurazione e forme

config.json fornisce i valori letti dal codice. load_config risolve data.root e i percorsi in paths rispetto alla root del progetto.

| Sezione | Dati usati |
| --- | --- |
| experiment | nome e seed |
| data | dataset, classi, bande, scala riflettanza, split |
| model | numero basi e architettura encoder |
| simplex | epsilon e deviazione standard minima ILR |
| encoder_training | training encoder, precisione, resume |
| ddpm_model / diffusion / cfg | U-Net, scheduler e condizionamento CFG |
| ddpm_training | training DDPM, EMA, preview e resume |
| evaluation | quantità campioni, batch e passi di sampling |
| annotation_evaluation | regione maschera e limiti plot |
| fid_evaluation | batch e worker FID |
| paths | directory di input e output |

EuroSATMSDataset restituisce immagine [C,H,W] e label intera. I batch hanno forma [B,C,H,W], le abundance [B,K,H,W], le coordinate ILR [B,K-1,H,W]. La label identifica la classe dell'immagine e seleziona encoder, statistiche e banco basi. L'ordine canali dei tensori resta quello dei TIFF; i grafici possono riordinare le bande per lunghezza d'onda.

## 3. Dataset e utility (dataset.py, utils.py)

### Lettura, split e loader

[**EuroSATMSDataset.__init__(DATA_ROOT, transform=None)**](../../dataset.py) — Ordina le directory classe e i file TIFF; costruisce classes, class_to_idx, samples, targets e registra transform.

[**EuroSATMSDataset.__len__()**](../../dataset.py) — Restituisce il numero di TIFF.

[**EuroSATMSDataset.__getitem__(index)**](../../dataset.py) — Legge TIFF, converte in float32 e da [H,W,C] a [C,H,W], applica transform se presente; restituisce immagine e label.

[**split_indices(dataset, config)**](utils.py) — Mescola gli indici separatamente per classe usando seed e quote del config. Restituisce liste train, validation e test; le quote train e validation sono arrotondate per difetto e il resto va al test.

[**BalancedBatchSampler(indices, targets, batch_size, num_classes, seed)**](utils.py) — Raggruppa gli indici per classe e determina il numero di batch a partire dalla classe più numerosa. Richiede batch divisibile per numero classi e presenza di tutte le classi.

- [**set_epoch(epoch)**](utils.py) memorizza l'epoca per rendere riproducibile il nuovo ordine.
- [**__len__()**](utils.py) restituisce numero batch.
- [**__iter__()**](utils.py) mescola gli indici di ogni classe, ripete quelli delle classi più piccole per completare i batch e restituisce liste bilanciate con ordine interno mescolato.

### Seed, log e percorsi delle run

[**set_seed(seed)**](utils.py) — Inizializza RNG di Python, NumPy, PyTorch CPU e CUDA.

[**get_random_state()**](utils.py) — Restituisce stati RNG per checkpoint, inclusi quelli CUDA quando disponibili.

[**set_random_state(state)**](utils.py) — Ripristina gli stati RNG forniti.

[**append_log(path, values)**](utils.py) — Crea la directory e appende un record JSONL con timestamp locale e campi values.

[**next_run_directory(root, prefix)**](utils.py) — Crea e restituisce la prima directory libera prefix-run_i.

[**latest_checkpoint(directory, prefix, kind="latest")**](utils.py) — Cerca prefix-run_*-kind.pt e restituisce il file con mtime più recente oppure None.

### Normalizzazione e inizializzazione basi

[**compute_band_stats(dataset, train_indices, config)**](utils.py) — Scorre un Subset train, divide per normalization_max e accumula in float64 somme e quadrati per banda. Restituisce media, deviazione standard e metadati dataset.

[**load_or_compute_band_stats(dataset, train_indices, config)**](utils.py) — Legge encoder_checkpoints/band_stats.json se la root dataset coincide; altrimenti chiama compute_band_stats, salva il JSON e restituisce le nuove statistiche.

[**apply_band_normalization(dataset, statistics, config)**](utils.py) — Imposta dataset.transform alla conversione valore grezzo -> riflettanza -> standardizzazione. Modifica il dataset, non restituisce immagini.

[**denormalize_bands(images, statistics)**](utils.py) — Riceve immagine [C,H,W] o batch [B,C,H,W], inverte la standardizzazione e restituisce riflettanza.

[**sample_class_training_pixels(dataset, train_indices, config)**](utils.py) — Per ogni classe sceglie K immagini train distinte e un pixel casuale da ognuna. Restituisce basi iniziali [N,K,C] standardizzate e metadati origine per immagine/riga/colonna.

### Trasformazione ILR

[**helmert_basis(num_components, dtype, device)**](utils.py) — Crea matrice [K,K-1] per la proiezione ILR; richiede K almeno 2.

[**_raw_ILR(abundances, epsilon)**](utils.py) — Riceve [B,K,H,W], limita abundance a epsilon, calcola logaritmi centrati e proietta sulla base Helmert. Restituisce ILR grezza [B,K-1,H,W].

[**_class_statistics(statistics, labels, reference)**](utils.py) — Seleziona media e deviazione standard per le label e le rende broadcastabili [B,K-1,1,1], usando dtype/device di reference.

[**to_ILR(abundances, labels, statistics)**](utils.py) — Applica _raw_ILR e standardizza ogni coordinata con statistiche della classe; restituisce [B,K-1,H,W].

[**from_ILR(latents, labels, statistics)**](utils.py) — Verifica latenti finiti, inverte standardizzazione e proiezione, applica softmax su K, limita e rinormalizza le abundance. Restituisce [B,K,H,W].

[**compute_ILR_statistics(model, loader, config, device)**](utils.py) — Imposta model eval, codifica batch dal loader fornito, converte abundance in ILR grezza e accumula somme/quadrati/conteggi pixel per classe in float64. Restituisce medie/std ILR, dimensioni e epsilon; limita la deviazione standard con simplex.statistics_epsilon.

## 4. Modello (model.py)

### Unmixing

[**ResidualBlock.__init__(channels, groups)**](model.py) — Crea due sequenze GroupNorm/SiLU/Conv2d con padding reflect.

[**ResidualBlock.forward(inputs)**](model.py) — Applica le convoluzioni e somma inputs; forma invariata [B,F,H,W].

[**AbundanceEncoder.__init__(config)**](model.py) — Crea proiezione input C->hidden, blocchi residui, GroupNorm e proiezione finale hidden->K.

[**AbundanceEncoder.forward(images)**](model.py) — Mappa immagini standardizzate [B,C,H,W] a logits [B,K,H,W] e abundance softmax float32 lungo K; restituisce entrambi.

[**ClassSpecificLatentUnmixingModel.__init__(config, initial_bases)**](model.py) — Verifica inizializzazione [N,K,C], crea un encoder indipendente per classe e registra latent_bases come parametro addestrabile.

[**_validate_labels(images, labels)**](model.py) — Verifica una label per immagine e intervallo [0,N); non restituisce dati e solleva ValueError in caso di errore.

[**encode(images, labels)**](model.py) — Seleziona i sotto-batch in base alla label, invoca l'encoder relativo e ricompone output nell'ordine iniziale. Restituisce logits e abundance [B,K,H,W].

[**decode(abundances, labels)**](model.py) — Seleziona il banco [B,K,C] dalla label e calcola la miscela lineare per pixel e banda. Restituisce ricostruzione [B,C,H,W].

[**forward(images, labels)**](model.py) — Compone encode e decode; restituisce dizionario con logits, abundances, reconstruction e latent_bases.

### Diffusione

[**create_ddpm_model(config)**](model.py) — Crea UNet2DModel con K-1 canali in ingresso/uscita, blocchi e canali configurati e N+1 embedding di classe (N è la classe nulla per CFG). Restituisce la U-Net.

[**create_training_scheduler(config)**](model.py) — Crea DDPMScheduler dai parametri diffusion per aggiungere rumore nel training/validation.

[**create_sampling_scheduler(config)**](model.py) — Crea DDIMScheduler partendo dalla configurazione del scheduler DDPM.

[**generate_ILR(model, scheduler, labels, config, inference_steps, generator=None)**](model.py) — Crea rumore iniziale [B,K-1,H,W]. A ogni passo duplica il latente, ottiene predizioni incondizionata (label N) e condizionata (labels), le combina usando guidance_scale e aggiorna il latente con scheduler.step. Restituisce ILR standardizzata; il chiamante la converte con from_ILR.

## 5. Training encoder (train.py, train_encoder.py)

### Orchestrazione

[**train.main()**](train.py) — Legge --stage (all/encoder/ddpm) e chiama i main degli stadi selezionati, encoder prima di DDPM nel caso all. Non gestisce direttamente i tensori.

### Funzioni encoder

[**validate(model, loader, device, use_fp16)**](train_encoder.py) — Imposta eval e no-grad, passa validation su device, calcola MSE ricostruzione/input e restituisce media pesata per immagini.

[**spectral_axis(config)**](train_encoder.py) — Restituisce lunghezze d'onda ordinate e gli indici canale corrispondenti.

[**save_bases(model, band_stats, classes, config, directory, epoch)**](train_encoder.py) — Denormalizza latent_bases, riordina per il solo grafico e salva directory/epoch_XXXX.png. Il chiamante passa la cartella latent_bases della run.

[**save_rmc(model, loader, band_stats, classes, config, directory, epoch, device)**](train_encoder.py) — Percorre il loader, porta input e ricostruzioni in riflettanza e accumula somme per classe/banda. Salva il confronto RMC e restituisce reflectance_rmse e rmc_mae.

[**train_encoder.main()**](train_encoder.py) — Carica config e decide se iniziare da zero o da latest. Da zero crea split, statistiche banda e basi; in resume ripristina pesi, optimizer, scheduler, scaler, RNG e split. Applica normalizzazione e crea unmixing, AdamW con learning rate distinti per encoder/basi, scheduler cosine, scaler, sampler bilanciato e loader.

Ogni batch percorre immagine/label -> ricostruzione -> MSE -> backward -> clipping -> optimizer. A fine epoca calcola validation, salva RMC e basi. Miglior validation MSE aggiorna best. Ogni epoca appende JSONL e salva latest con stato di resume. Restituisce percorso best. best contiene modello, split, statistiche bande, classi e provenienza basi; latest aggiunge stato optimizer/scheduler/scaler/RNG e nome run.

## 6. Training DDPM (train_ddpm.py)

[**validate(ddpm, unmixing, loader, scheduler, stats, config, device, use_fp16)**](train_ddpm.py) — Imposta DDPM eval, usa RNG locale con seed configurato, codifica input col modello congelato, converte abundance in ILR, campiona rumore/timestep e misura MSE tra rumore predetto e aggiunto. Restituisce loss media per immagine.

[**save_preview(ddpm, unmixing, scheduler, stats, band_stats, config, path, device)**](train_ddpm.py) — Genera un campione per label, inverte ILR, decodifica con le basi della stessa classe, denormalizza, seleziona bande 4/3/2 e salva griglia PNG RGB.

[**train_ddpm.main()**](train_ddpm.py) — Carica encoder best all'avvio o DDPM latest in resume. Recupera pesi unmixing, classi, statistiche bande e split; verifica ordine classi e congela unmixing. Calcola statistiche ILR sul train se non esistono nel checkpoint.

Crea sampler, loader, U-Net, optimizer, scaler, scheduler DDPM e LR scheduler. Ogni batch viene codificato senza gradienti; abundance -> ILR pulita -> rumore/timestep -> latente rumoroso -> dropout casuale della label -> predizione U-Net -> MSE. Divide la loss per il gruppo effettivo di accumulo gradienti; al passo optimizer applica clipping, LR scheduler ed EMA.

Alla fine dell'epoca copia i pesi EMA nella U-Net per validation, preview e best, quindi ripristina i pesi di training. latest conserva pesi training e stato EMA insieme agli stati optimizer/scaler/scheduler, split, statistiche e RNG. Restituisce percorso best; best contiene pesi EMA e unmixing per sampling.

## 7. Sampling ed evaluation standard

### sample.py

[**sample.main()**](sample.py) — Legge --class e --n, carica config e checkpoint best DDPM, ricrea U-Net e unmixing, controlla la classe e chiama evaluation.generate_samples. Imposta suffix vuoto e crea una nuova directory sampling sotto paths.results; --n è la quantità per la classe selezionata.

### evaluation.py

[**parse_args(config)**](evaluation.py) — Costruisce parametri: immagini per classe, batch, passi DDIM e percorso samples facoltativo.

[**next_result_directory(root, run_name)**](evaluation.py) — Crea e restituisce la successiva directory latent_unmixing_per_class-eval_i-run_j.

[**generate_samples(ddpm, unmixing, checkpoint, config, args, root, device)**](evaluation.py) — Seleziona tutte le classi o args.class_name, inizializza un RNG per classe da seed + 10000 + indice, genera latenti ILR a batch, li inverte in abundance e decodifica con basi della stessa classe. Salva TIFF/PNG col suffix args.suffix o _eval se assente.

[**spectra(paths, scale)**](evaluation.py) — Legge TIFF, converte in float64, divide per scale e calcola media spaziale per banda. Restituisce [immagini,C] o solleva ValueError se nessun file.

[**draw(axis, real, generated, class_name, config)**](evaluation.py) — Ordina bande per lunghezza d'onda, disegna medie e deviazioni standard per immagine. Restituisce RMSE e MAE tra curve medie.

[**rmc_euclidean_distance(real, generated)**](evaluation.py) — Calcola le due RMC medie per banda, ne sottrae i 13 valori corrispondenti e restituisce la norma euclidea del vettore differenza. Non salva figure.

[**rmc_sam_degrees(real, generated)**](evaluation.py) — Calcola la SAM in gradi fra i vettori RMC medi reali e generati.

[**save_plots(values, config, directory, title=...)**](evaluation.py) — Riceve mapping classe -> firme reali/generate; calcola limiti comuni, invoca draw, rmc_euclidean_distance e rmc_sam_degrees, salva il grafico RMC aggregato e restituisce metriche per classe. Non salva grafici separati delle differenze RMC.

[**evaluation.main()**](evaluation.py) — Carica config/checkpoint best. Senza --path genera campioni; con --path usa file esistenti. Legge indici test dal checkpoint, raccoglie firme reali/generate, salva plot, stampa macro medie di RMSE, MAE, distanza euclidea RMC e SAM e restituisce directory risultati.

[**save_generated_pair(image, ms_path, rgb_path, band_stats, config)**](utils.py) — Denormalizza immagine [C,H,W], clampa riflettanza negativa a zero e verifica valori finiti. Salva TIFF float32 [H,W,C] e PNG RGB dalle bande 4/3/2 divise per rgb_display_max.

## 8. Ablation delle basi (ablation_test.py)

L'ablation mantiene separati target e base del decoder. Il target condiziona il DDPM e seleziona statistiche ILR; la base seleziona il banco usato per decodificare. Con 10 classi e 100 immagini per coppia produce 10×9×100 = 9.000 immagini, escludendo target=base. Ogni elemento riceve un latente gaussiano iniziale indipendente. I file sono in samples/<target>/{MS,RGB} e includono target/base nel nome.

[**parse_args()**](ablation_test.py) — Legge config, immagini per coppia, batch, passi DDIM e seed. --images-per-class è alias di --images-per-pair.

[**next_result_directory(root, run_name, experiment_name)**](ablation_test.py) — Crea directory evaluation numerata per esperimento e run checkpoint.

[**generate_samples(ddpm, unmixing, checkpoint, config, settings, root, device, generator)**](ablation_test.py) — Per ogni coppia non diagonale genera batch con label target e base costanti. Genera ILR col target, applica statistiche target, decodifica con il banco base, salva TIFF/PNG identificati da target e base. Restituisce conteggi per coppia.

[**real_test_paths(dataset, checkpoint, classes, images_per_pair)**](ablation_test.py) — Raggruppa percorsi TIFF dei test_indices per classe, limita ciascun gruppo alla quantità richiesta e verifica la disponibilità.

[**confusion_matrices(samples_directory, classes, real)**](ablation_test.py) — Elenca i TIFF generati per coppia e calcola le RMC per immagine. Restituisce due matrici [N,N], con righe=target, colonne=base e diagonale zero.

- Matrice vettoriale: distanza euclidea tra la RMC media generata per target/base e la RMC media reale del target. La differenza è calcolata banda per banda sui 13 componenti, nell'ordine dei TIFF, poi se ne calcola la norma.
- Matrice SAM: angolo in gradi fra vettore RMC medio generato per target/base e vettore RMC medio reale del target.

Ogni cella descrive quindi lo scostamento dal target reale con il banco indicato. Non misura la distanza tra due classi generate fra loro.

[**summarize_rmc_metrics(values)**](ablation_test.py) — Calcola per classe RMSE e MAE fra RMC medie, la distanza euclidea fra le due RMC medie e la SAM. Restituisce un dizionario numerico senza creare figure; le immagini generate con le basi non diagonali sono aggregate per classe.

[**save_distance_matrix(matrix, classes, path, title, colorbar_label)**](ablation_test.py) — Salva heatmap target×base con etichette e valori, incluso zero sulla diagonale.

[**ablation_test.main()**](ablation_test.py) — Risolve settings da CLI/config, inizializza seed e carica checkpoint best. Raccoglie firme e percorsi test, ricrea modelli, genera tutte le coppie non diagonali, calcola le matrici e il riepilogo numerico RMC, quindi scrive metrics.json. Le metriche aggregate per classe considerano tutte le basi non diagonali; non viene prodotto un grafico RMC aggregato.

Gli unici grafici prodotti dall'ablation sono target_base_rmc_euclidean_distance.png e target_base_rmc_sam.png dentro plots.

## 9. Valutazioni ausiliarie

### eval/eval_ann.py

[**read_region_spectrum(image_path, mask_path, scale, mask_class, unannotated=False)**](eval/eval_ann.py) — Legge maschera e metadati Classes; seleziona ID della classe o ID 0 per pixel non annotati. Divide il TIFF per scale e restituisce la media spettrale dei pixel. Se non esistono pixel non annotati restituisce None; se manca la classe richiesta solleva ValueError.

[**collect_spectra(image_directory, mask_directory, scale, mask_class, unannotated=False)**](eval/eval_ann.py) — Associa ogni TIFF alla maschera con suffisso _annotation.png e raccoglie una firma media per immagine. Restituisce [immagini,C] e richiede almeno due firme.

[**plot_rmc(real, fake, config, directory, unannotated=False)**](eval/eval_ann.py) — Disegna media e deviazione standard campionaria, salva RMC PNG/PDF e restituisce distanza euclidea fra le RMC medie e SAM in gradi. Non crea heatmap di istogrammi.

[**eval_ann.main()**](eval/eval_ann.py) — Carica i gruppi real/fake configurati, ripete analisi per regione annotata e pixel non annotati, stampa conteggi, distanza euclidea RMC e SAM.

### eval/eval_bases.py

[**next_result_directory(root, experiment_name, run_name)**](eval/eval_bases.py) — Crea directory evaluation numerata.

[**save_plots(embedding, basis_labels, classes, directory)**](eval/eval_bases.py) — Separa coordinate MDS di basi e centroidi; salva grafico combinato e grafico dei soli centroidi.

[**pairwise_mmd(bases)**](eval/eval_bases.py) — Appiattisce basi [N,K,C], sceglie bandwidth² come mediana delle distanze quadrate positive. La funzione interna [kernel(left,right)](eval/eval_bases.py) calcola affinità RBF. Calcola distanza MMD simmetrica fra ogni coppia di banchi e restituisce matrice e bandwidth².

[**save_mmd_results(distances, bandwidth_squared, classes, directory)**](eval/eval_bases.py) — Salva parametri e matrice in JSON/CSV e una heatmap PNG.

[**eval_bases.main()**](eval/eval_bases.py) — Carica checkpoint best, denormalizza basi con band_stats, calcola centroidi, adatta MDS a basi e centroidi congiuntamente, calcola MMD e salva risultati.

### eval/eval_fid.py

[**load_config()**](eval/eval_fid.py) — Carica config locale e rende assoluti i tre percorsi FID.

[**image_paths(directory)**](eval/eval_fid.py) — Restituisce percorsi immagine ordinati con estensioni ammesse.

[**evaluate_class(real_directory, fake_directory, device, settings)**](eval/eval_fid.py) — Conta immagini e richiede almeno due per cartella; calcola FID real-fake e controllo FID real-real. Restituisce nomi, conteggi e valori.

[**format_table(rows)**](eval/eval_fid.py) — Restituisce tabella testuale allineata.

[**save_results(rows, config)**](eval/eval_fid.py) — Scrive e stampa report TXT e CSV con conteggi e metriche.

[**eval_fid.main()**](eval/eval_fid.py) — Scorre le sottocartelle fake, confronta ciascuna con la classe reale corrispondente e salva risultati. Usa immagini RGB, non TIFF MS.

## 10. Output scritti

| Percorso dal config o dalla run | Dati |
| --- | --- |
| encoder_checkpoints/band_stats.json | media/std bande e provenienza |
| encoder_checkpoints/*-latest.pt | stato per resume encoder |
| encoder_checkpoints/*-best.pt | encoder selezionato da validation |
| ddpm_checkpoints/*-latest.pt | pesi training e stato EMA/optimizer/scaler/scheduler/RNG |
| ddpm_checkpoints/*-best.pt | denoiser EMA e unmixing per generare |
| encoder_logs, ddpm_logs | JSONL incrementali |
| encoder_previews, ddpm_previews | figure per epoca |
| paths.results | samples, plots e metriche |
| paths.annotation_plots | figure regioni annotate |
| paths.fid_results | report FID |

__init__.py contiene solo la descrizione del package e non trasforma dati. I percorsi effettivi sono determinati da config e dai chiamanti delle funzioni di salvataggio.
