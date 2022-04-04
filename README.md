# Engagement Detection

## Table of contents
* [Purpose](#purpose)
* [Demo](#demo)
* [Technologies Used](#technologies-used)
* [Intallation](#installation)

## Purpose
We all have been attending classes online for the past couple of years. This although is not a method of taking the knowledge that we are familiar with contrary to the physical method of sitting in a classroom and making notes and such. 

The main problem that we have identified was the lack of attention many students face while sitting in front of a laptop or a phone. They get distracted repeatedly since they are also in the comfort of their home. This all leads to a decrease in the quality of learning.

So to tackle the above issue, we decided to train and use an ML model to find if a student is engaged or not and, if found, then alert them likewise.
	
## Demo

## Technologies Used

Project is created using:
* Open CV
* Mediapipe
* Keras and tensorflow
* Dlib

## Installation
* First install the dependencies:

```
pip install -r requirements.txt
```

* You can train your own models using following commands:-

```
python train_detect_detection.py
python train_emotion_recognition.py
```

And use these models to run engagement_detection.py

* Or you can directly use trained model
* to run the engagement detction
```
python detect_distraction.py
```


