# Big Brother
Facial recognition of people from a videostream

## Why Big Brother ?
Big Brother est un projet effectué dans le cadre du projet GATE de l'école d'ingénieurs française Télécom SudParis. Durant 8 mois, nous étions 9 étudiants à élaborer un système de détection et de reconnaissance d'individus, encadré par un enseignant-chercheur et un entrepreneur. Ne connaissant rien à l'intelligence artificielle, à l'apprentissage profond et au cloud, nous nous somme lancés dans ce défi afin d'acquérir des connaissances. De plus, c'était l'occasion de montrer ce que 9 débutants pouvaient faire en quelques mois et d'alerter sur les dangers potentiels de ces technologies aux mains de professionnels.

## How it works ?
Big Brother est un algorithme de reconnaissance faciale qui fonctionne en plusieurs étapes, chacunes utilisant des technologies différentes :
- une étape de détection faciale -> HOG, MTCNN
- une étape de reconnaissance faciale -> Facenet
- une étape de tracking -> Global Nearest Neighbor

## How to use it ?
0. Requirements :
- python 3.X
- numpy
- opencv-python (pip install opencv-python)
- dlib (pip install dlib)
- h5py (pip install h5py)
- mtcnn (pip install mtcnn)
1. Clone the repo
2. Add images in the known_peoples folder : each image should represent ONE person, with the name of this person as name of the image
3. Launch preprocessing.py : this will create files in the datasets folder (known_peoples.hdf5 and known_peoples_names.txt)
4. Launch webcam_closedbase.py

## To do
- améliorer le tracking : Global Nearest Neighbor
- reconnaissance faciale par webcam en base ouverte
- main.py ?
- rédiger et traduire le readme
- alternance reconnaissance/tracking : autre qu'aléatoire ?
(- tracking basé sur le mouvement ?)
(- objectif initial de bigbro : pedestrian recognition)
(- améliorer le tracking : filtrage, MCMCDA)
