# Confronto RGB: reali, latent unmixing base, large e SD 1.5

Le tavole mostrano tutte le dieci classi EuroSAT. Le immagini sono ricavate dai
TIFF multispettrali con identica conversione B4/B3/B2, punto di bianco 0,3,
clipping in [0,1] e quantizzazione RGB uint8. Non viene applicato auto-contrasto.
Le immagini vengono visualizzate senza interpolazione che aggiunga sfocatura.

Sono selezionati casualmente tre campioni per classe e sorgente, con seed 42.
Sono immagini indipendenti, non coppie corrispondenti. Il manifest conserva
l'elenco esatto dei file scelti.

- [Panoramica di tutte le classi](overview_all_classes.png) · [PDF](overview_all_classes.pdf)
- [AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial](comparison_classes_1.png) · [PDF](comparison_classes_1.pdf)
- [Pasture, PermanentCrop, Residential, River, SeaLake](comparison_classes_2.png) · [PDF](comparison_classes_2.pdf)
- [Selezione dei campioni e protocollo](selection.json)
