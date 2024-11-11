
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
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "violin"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + 1 shape

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 160/8

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5
    
config = ViolinConfig()
config.display()
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
Le modèle a obtenu une perte totale de 0.1983 sur le jeu d'entraînement, avec des pertes relativement faibles pour chaque composant, notamment la perte liée au masque (0.0991) et à la classification (0.0419). En revanche, la perte de validation a atteint 1.4729, avec des valeurs significativement plus élevées, en particulier pour la perte de la boîte de détection (0.8625). Cette différence notable entre les pertes d'entraînement et de validation indique un overfitting, suggérant que le modèle a bien appris les détails du jeu d'entraînement mais n'a pas réussi à généraliser efficacement aux nouvelles données.
### Matrice de confusion

![App Screenshot](https://github.com/amine-okc/mrcnn_violin/blob/main/plots/mc1.png)



### Graphique des pertes d'entraînement et de validation
## Deuxième entraînement

### Configuration
### Pertes
### Matrice de confusion
### Graphique des pertes d'entraînement et de validation
## Troisième entraînement

### Configuration
### Pertes
### Matrice de confusion
### Graphique des pertes d'entraînement et de validation
## Comparaison des résultats