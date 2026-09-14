# Classificazione Fine-Grained su CUB-200-2011

Progetto finale per il corso di **Advanced Machine Learning**, Università degli Studi di Milano-Bicocca (A.A. 2025/2026).

**Autori:** Emma Villa, Nicolò Molteni

## Descrizione

Il progetto affronta la classificazione fine-grained di 200 specie di uccelli sul dataset [Caltech-UCSD Birds-200-2011 (CUB-200-2011)](https://authors.library.caltech.edu/records/cvm3y-5hh21), un benchmark di riferimento per task caratterizzati da bassa varianza inter-classe (specie diverse ma visivamente simili) e alta varianza intra-classe (pose, illuminazione, occlusioni).

Vengono confrontati quattro approcci:

- **Transfer Learning (fine-tuning)** su tre reti pre-addestrate su ImageNet: MobileNetV2, ResNet50V2, InceptionV3
- **Feature Extraction con SVM**, usando InceptionV3 (sia pre-addestrata che dopo fine-tuning) come feature extractor
- **CNN custom** addestrata da zero, con architettura ottimizzata tramite Keras Tuner

## Dataset

- 11.788 immagini, 200 classi di uccelli prevalentemente nordamericani
- Split originale degli autori: Training Set 5.094 immagini, Validation Set 900, Test Set 5.794
- Lieve sbilanciamento tra le classi (41–60 immagini per classe)

## Pipeline

**Pre-elaborazione:** rescaling (lato corto a 256px) + center crop (224×224), normalizzazione in [-1, 1], one-hot encoding delle label.

**Data Augmentation:** layer custom `RandomApply` che applica trasformazioni "strong" con probabilità 0.7 (rotazione, shear, zoom, traslazione, variazioni di contrasto/luminosità), oltre a un random flip orizzontale sempre attivo.

**Fine-tuning a due fasi:**
1. *Head-training*: backbone congelata, training solo del classificatore (5 epoche, learning rate alto)
2. *Fine-tuning*: sblocco selettivo dei layer più profondi della backbone, con Cosine Decay scheduler ed Early Stopping (patience 10)

Regolarizzazione tramite Weight Decay (L2), differenziata per modello in base alla profondità della rete.

## Risultati

| Modello | Test Accuracy | Test Loss | Tempo Training |
|---|---|---|---|
| MobileNetV2 | 65.9% | 1.122 | 14.65 min |
| ResNet50V2 | 75.7% | 1.132 | 23.08 min |
| **InceptionV3** | **81.1%** (train) / **75.8%** (test) | 0.877 | 16.60 min |
| Feature Extraction (SVM) | 66.2% | – | 8.47 min |
| Feature Extraction fine-tuned (SVM) | 76.6% | – | 4.77 min |
| Custom CNN (from scratch) | 30.3% | 3.632 | 42.62 min |

*(Le metriche complete — Top-5 Accuracy, Precision, Recall, F1, curve ROC/PR — sono riportate nella relazione.)*

**InceptionV3** si conferma l'architettura ottimale, grazie alla capacità dei suoi moduli di catturare dettagli locali e forme globali simultaneamente. **MobileNetV2** offre un'alternativa più leggera con prestazioni competitive. Il Transfer Learning supera nettamente l'apprendimento da zero: la CNN custom, pur ottimizzata con Keras Tuner, si ferma al 31.2% dopo 150 epoche.

## Sviluppi futuri

- Class weights per gestire lo sbilanciamento del dataset
- Crop mirato sfruttando le bounding box fornite dal dataset, invece del center crop
- Data augmentation più avanzata (es. CutMix)
- Split alternativo del dataset per ridurre l'overfitting

## Struttura della repository

```
AML-exam/
├── Classificazione_Fine-Grained_CUB-200-2011.ipynb   # Notebook con codice e training
├── Relazione_CUB-200-2011.pdf                        # Relazione completa
├── Presentazione_CUB-200-2011.pdf                     # Slide di presentazione
└── README.md
```

## Tecnologie

Python, TensorFlow/Keras, Keras Tuner, scikit-learn (SVM), NumPy

## Riferimenti

- Howard et al., *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*, 2017
- Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, 2019
- He et al., *Identity Mappings in Deep Residual Networks*, 2016
- Szegedy et al., *Rethinking the Inception Architecture for Computer Vision*, 2015
- Wah et al., *The Caltech-UCSD Birds-200-2011 Dataset*, 2011
