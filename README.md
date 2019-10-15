# Big Brother

Facial recognition of people from a video stream.

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/EuzzjI9s7yE/hqdefault.jpg)]( https://youtu.be/EuzzjI9s7yE)


## Why Big Brother ?

Big Brother is a school project carried out at the Télécom SudParis (a French engineering school). We have developed a facial recognition system with accurate detection and tracking of faces. Having little knowledge at the beginning, we embarked on this challenge to show what 9 beginners could do in a few months and to warn about the potential dangers of these technologies.


## How it works ?

The pipeline is as follows :

image -> face detection, landmarks estimation -> feature extraction -> matching with database features

- Face detection : HOG, MTCNN, RetinaFace (soon)
- Facial landmarks estimation : dlib algorithms
- Feature extraction : FaceNet, ArcFace (soon)
- Tracking : opencv algorithms
- Matching : L2 distance, Hungarian algorithm


## How to use it ?

0. Requirements :
- python 3.X
- os, pkg_resources, time, PIL, numpy, scipy, pandas
- opencv -> pip install opencv-python or conda install opencv-python
- dlib -> pip install dlib or conda install dlib
- if you want to use MTCNN face detection : mtcnn -> pip install mtcnn or conda install mtcnn
1. Clone the repo
2. Download the model weights : http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 and http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 then unzip and put 'dlib_face_recognition_resnet_model_v1.dat' and 'shape_predictor_68_face_landmarks.dat' in './models/'
3. Add images in the './data/known_peoples/' : each image should only contain ONE face, with the name of the person as name of the image
4. Launch ./Preprocessing.py : it will create 'dataset.csv' in './data/', this file contains a feature vector for each person in the dataset and the corresponding name
5. Launch ./Main.py


## To do

- data visualization of the feature space with t-SNE
- make it work in open database (no pre-defined set of people to recognize)
- code profiling and optimization to real-time (jit, cuda, c++)
- use binary tree to efficiently process the database
- mobile/web app (flask?)
- arcface + retinaface


### Our team

![alt text](https://raw.githubusercontent.com/ridouaneg/Big-Brother/master/data/unknown_peoples/image1.jpg)


## Collaboration

This project was achieved as part of Télécom SudParis' GATE project in collaboration with the start-up Watiz.
