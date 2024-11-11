
# Détection de violons dans une image avec MRCNN




## Auteurs

- Amine OULD KACI [@amine-okc](https://www.github.com/amine-okc)
- Djouher AIT YAKOUB [@djouher-ait](https://www.github.com/djouher-ait)



## Sommaire

- Introduction
- Premier entraînement
    - Configuration
    - Pertes
    - Matrice de confusion
    - Graphique des pertes d'entraînement et de validation
- Deuxième entraînement
    - Configuration
    - Pertes
    - Matrice de confusion
    - Graphique des pertes d'entraînement et de validation
- Troisième entraînement
    - Configuration
    - Pertes
    - Matrice de confusion
    - Graphique des pertes d'entraînement et de validation
- Comparaison des résultats


## Introduction
Le projet porte sur la détection de violons dans les images en utilisant Mask R-CNN. Le modèle est spécifiquement conçu pour analyser des photos, identifier la présence de violons, et distinguer ces objets des distractions environnantes. Malgré l'utilisation d'un dataset relativement modeste, avec environ 160 à 189 images pour l'entraînement et 40 pour les tests, l'objectif est de développer un système performant capable de généraliser efficacement. Le défi réside dans l'optimisation des performances avec ce volume de données limité, en s'assurant que le modèle parvient à détecter les violons de manière fiable tout en réduisant les faux positifs.
## Premier entraînement

### Configuration
Notre première configuration utilise 1 GPU (comme notre machine ne dispose pas d'un GPU, il utilisera le CPU) avec 8 images par GPU et des images redimensionnées à 128x128 pixels seulement. Le modèle détecte 2 classes : le fond et le violon. Les ancres sont ajustées pour des objets petits, et le nombre de ROIs par image est limité à 32 pour mieux s'adapter à des images contenant peu d'objets. La configuration définit également un nombre d'étapes par époque qui est égal à la taille du dataset divisée par le nombre d'images par GPU, ainsi qu'un nombre d'étapes de validation réduit.
```
class ViolinConfig(Config):

    # Give the configuration a recognizable name
    NAME = "violin"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shape


    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128


    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels


    TRAIN_ROIS_PER_IMAGE = 32


    STEPS_PER_EPOCH = 160/8

    VALIDATION_STEPS = 5
```

Pour notre entraînement, nous avons choisi de garder le taux d'apprentissage (Learning Rate) par défaut (0.001), avec 50 epochs. Cependant, ce nombre d'epochs a conduit à un overfitting, ce qui indique que le modèle a trop appris sur les données d'entraînement sans généraliser correctement sur les données de validation. En conséquence, ce paramétrage a entraîné environ 7 heures d'entraînement. Nous envisagerons de réduire le nombre d'époques ou d'utiliser des techniques comme l'arrêt précoce pour améliorer la performance du modèle.
### Pertes
À la fin de l'entraînement, les pertes obtenues par le modèle sont les suivantes :
```
Epoch 50/50
20/20 [==============================] - 512s 26s/step - loss: 0.1983 - rpn_class_loss: 0.0046 - rpn_bbox_loss: 0.0231 -
mrcnn_class_loss: 0.0419 - mrcnn_bbox_loss: 0.0297 - mrcnn_mask_loss: 0.0991 - val_loss: 1.4729 - val_rpn_class_loss: 0.0121 - 
val_rpn_bbox_loss: 0.8625 - val_mrcnn_class_loss: 0.1203 - val_mrcnn_bbox_loss: 0.2036 - val_mrcnn_mask_loss: 0.2744
```
Le modèle a obtenu une perte totale de 0.1983 sur le jeu d'entraînement, avec des pertes relativement faibles pour chaque composant, notamment la perte liée au masque (0.0991) et à la classification (0.0419). En revanche, la perte de validation a atteint 1.4729, avec des valeurs significativement plus élevées, en particulier pour la perte de la bbox (0.8625). Cette différence notable entre les pertes d'entraînement et de validation indique un overfitting, suggérant que le modèle a bien appris les détails du jeu d'entraînement mais n'a pas réussi à généraliser efficacement aux nouvelles données. De plus, le redimensionnement des images à 128x128 pixels a peut-être contribué à cette situation, car cette taille réduite ne permet pas de capturer suffisamment de détails, ce qui limite la capacité du modèle à détecter correctement les objets
### Matrice de confusion

![MC1](https://github.com/amine-okc/mrcnn_violin/blob/main/plots/mc1.png)

Les résultats du modèle montrent :

- 0 True Negative (TN) : Aucun exemple n'a été correctement classé comme négatif (aucun objet non violon détecté).
- 0 False Negative (FN) : Aucun violon n'a été omis ou non détecté.
- 3 False Positive (FP) : Trois objets non violon ont été incorrectement classés comme violon.
- 43 True Positive (TP) : Quarante-trois violons ont été correctement identifiés.
Ces résultats suggèrent que le modèle est très sensible à la détection des violons, mais il a tendance à générer des faux positifs, en détectant des objets non violon comme violons. Cela peut être dû au fait qu'il n'y a pas suffisamment d'images d'entraînement sans violon, ce qui rend difficile pour le modèle d'apprendre à distinguer efficacement les objets qui ne sont pas des violons.

### Graphique des pertes d'entraînement et de validation
![Loss train 1](https://github.com/amine-okc/mrcnn_violin/blob/main/plots/loss1.png)

On remarque que la courbe est décroissante, c'est-à-dire qu'à chaque nouvelle itération d'epoch, la perte en données d'entraînement diminue, celà suggère que le modèle apprend au fur et à mesure des images d'entraînement.

![Validation loss train 1](https://github.com/amine-okc/mrcnn_violin/blob/main/plots/val_loss1.png)

//qqch


## Deuxième entraînement

### Configuration
Notre deuxième configuration utilise toujours 1 GPU avec 8 images par GPU et des images redimensionnées cette fois-ci à 256x256 pixels pour mieux capturer les détails. On a augmenté le nombre de ROIs à 50 par image pour permettre au modèle de mieux se concentrer sur les objets d'intérêt.

```
class ViolinConfig(Config):

    # Give the configuration a recognizable name
    NAME = "violin"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shape


    IMAGE_MIN_DIM = 256 #changed
    IMAGE_MAX_DIM = 256 # changed


    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels


    TRAIN_ROIS_PER_IMAGE = 50 ## changed


    STEPS_PER_EPOCH = 168/IMAGES_PER_GPU

    VALIDATION_STEPS = 10

```

Nous avons toujours gardé le taux d'apprentissage (Learning Rate) par défaut (0.001), avec 30 epochs. Cette configuration permet d'avoir un compromis entre un entraînement suffisamment long pour que le modèle puisse s'adapter aux données et tenter d'éviter un sur-apprentissage (overfitting).
### Pertes
À la fin de l'entraînement, les pertes obtenues par le modèle sont les suivantes :
```
Epoch 30/30
21/21 [==============================] - 1056s 50s/step - loss: 0.2210 - rpn_class_loss: 0.0030 - rpn_bbox_loss: 0.0333 -
mrcnn_class_loss: 0.0504 - mrcnn_bbox_loss: 0.0306 - mrcnn_mask_loss: 0.1036 - val_loss: 1.4754 - val_rpn_class_loss: 0.0097 -
val_rpn_bbox_loss: 0.8770 - val_mrcnn_class_loss: 0.0969 - val_mrcnn_bbox_loss: 0.2440 - val_mrcnn_mask_loss: 0.2478
```
À l'epoch 30 de notre deuxième configuration, le modèle présente une perte d'entraînement (loss) de 0.2210. De plus, comparée à la première configuration, où à l'epoch 30 la perte de validation était de 1.6928, cette réduction indique que l'augmentation de la taille des images à 256x256 pixels et l'augmentation du nombre de ROIs (50 par image) ont permis une meilleure généralisation du modèle. Cela suggère que les détails supplémentaires capturés par les images plus grandes ont aidé à réduire l'overfitting. Cette configuration a permis de réduire le temps d'entraînement, avec un gain de temps notable.
### Matrice de confusion

![MC2](https://github.com/amine-okc/mrcnn_violin/blob/main/plots/mc2.png)

On obtient les mêmes résultats que la première configuration
Cela signifie que, malgré l'amélioration des paramètres d'entraînement (telles que la taille des images et le nombre de ROIs), le modèle continue de rencontrer les mêmes difficultés pour distinguer les violons des autres objets. Les erreurs restent similaires en termes de faux positifs et de vrais positifs, ce qui pourrait suggérer que le modèle manque encore de diversité dans les images sans violon ou qu'il a besoin d'un ajustement supplémentaire pour mieux généraliser.

### Graphique des pertes d'entraînement et de validation
![Loss train 2](https://github.com/amine-okc/mrcnn_violin/blob/main/plots/loss2.png)

On remarque que la courbe est décroissante, c'est-à-dire qu'à chaque nouvelle itération d'epoch, la perte en données d'entraînement diminue.

![Validation loss train 2](https://github.com/amine-okc/mrcnn_violin/blob/main/plots/val_loss2.png)


// qqch

## Troisième entraînement

### Configuration
Dans cette dernière configuration, nous avons choisi de redimensionner les images à 512x512 pixels pour mieux capturer les détails fins des violons. Nous avons maintenu l'entraînement sur 1 GPU avec 8 images par GPU pour équilibrer la charge et permettre au modèle d'apprendre plus efficacement tout en évitant une surcharge de la mémoire GPU. Le nombre de classes reste inchangé à 2 (fond + violon).

Nous avons également augmenté le nombre de ROIs (régions d'intérêt) à 80 par image, ce qui permet d'extraire un plus grand nombre de zones pertinentes pour la détection dans des images plus grandes.

```
class ViolinConfig(Config):

    # Give the configuration a recognizable name
    NAME = "violin"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shape

    IMAGE_MIN_DIM = 512 #changed
    IMAGE_MAX_DIM = 512 # changed

    #RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels


    TRAIN_ROIS_PER_IMAGE = 80 ## changed

    STEPS_PER_EPOCH = 189/IMAGES_PER_GPU

    VALIDATION_STEPS = 30

```

Nous avons opté pour un entraînement sur 20 epochs.
Avec ces ajustements, nous espérons que la précision du modèle sera améliorée, notamment en permettant une meilleure généralisation aux variations des images et aux différents contextes de violons.
### Pertes
À la fin de l'entraînement, les pertes obtenues par le modèle sont les suivantes :
```
Epoch 20/20
24/23 [==============================] - 880s 37s/step - loss: 0.2147 - rpn_class_loss: 0.0017 - rpn_bbox_loss: 0.0316 - 
mrcnn_class_loss: 0.0292 - mrcnn_bbox_loss: 0.0470 - mrcnn_mask_loss: 0.1051 - val_loss: 1.4293 - val_rpn_class_loss: 0.0076 - 
val_rpn_bbox_loss: 0.8581 - val_mrcnn_class_loss: 0.0739 - val_mrcnn_bbox_loss: 0.1854 - val_mrcnn_mask_loss: 0.3043
```
À l'epoch 20, le modèle présente une perte d'entraînement (loss) de 0.2147, ce qui montre que l'apprentissage progresse bien. Les pertes pour les différentes composantes, telles que la classification et le masque, restent raisonnables, indiquant que le modèle est en bonne voie. La perte de validation à 1.4293 est légèrement plus élevée, mais cela est normal dans les premières phases d'entraînement, surtout avec une quantité de données aussi limitée. 
### Matrice de confusion

![MC3](https://github.com/amine-okc/mrcnn_violin/blob/main/plots/mc3.png)

Les résultats du modèle montrent :

- 1 True Negative (TN) : Un objet non-violon classé comme négatif.
- 0 False Negative (FN) : Aucun violon n'a été omis ou non détecté.
- 3 False Positive (FP) : Trois objets non violon ont été incorrectement classés comme violon.
- 44 True Positive (TP) : Quarante-quatre violons ont été correctement identifiés.

Le modèle a montré des progrès, avec un True Negative (TN) détecté, ce qui n’était pas le cas précédemment. Le nombre de True Positives (TP) a également légèrement augmenté (44 contre 43), bien que trois objets non violon aient été incorrectement classés comme violons (3 FP). En résumé, la performance s'est améliorée par rapport à la configuration précédente.



### Graphique des pertes d'entraînement et de validation
![Loss train 3](https://github.com/amine-okc/mrcnn_violin/blob/main/plots/loss3.png)

On remarque que la courbe est décroissante, c'est-à-dire qu'à chaque nouvelle itération d'epoch, la perte en données d'entraînement diminue.

![Validation loss train 3](https://github.com/amine-okc/mrcnn_violin/blob/main/plots/val_loss3.png)


## Comparaison des résultats

### Indicateurs de performance
| Configuration | mAP         | Précision   | Recall      | F1-Score    |
|---------------|-------------|-------------|-------------|-------------|
| Configuration 1       | 0.7323      | 0.9348      | 1.0         | 0.9663      |
| Configuration 2       | 0.8485      | 0.9348      | 1.0         | 0.9663      |
| Configuration 3       | 0.8456      | 0.9362      | 1.0         | 0.9670      |

Les configurations 2 et 3 offrent des résultats très similaires, avec la configuration 3 légèrement meilleure en termes de F1-Score. La configuration 1 est plus faible en termes de mAP, mais elle montre encore de bonnes performances en rappel et précision. Si l'on cherche à améliorer les performances globales, la configuration 3 semble être la meilleure option, avec un bon compromis entre la détection de tous les violons et la précision des résultats.

### Pertes en entraînement et validation

#### Comparaison des pertes (Loss) entre les configurations
![Comparaison Loss](https://github.com/amine-okc/mrcnn_violin/blob/main/plots/comp_loss.png)

#### Comparaison des pertes en validation (Val Loss) entre les configurations
![Comparaison Val Loss](https://github.com/amine-okc/mrcnn_violin/blob/main/plots/comp_val_loss.png)
