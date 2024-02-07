# ResNet50-Based Face Recognition System

The repository does not include the model due to its large size. You can find it in the project's OneDrive repository.

## Overview

This system employs ResNet50, a deep learning model, for facial recognition tasks. It generates embeddings of facial images, which are numerical representations capturing distinctive features of faces. These embeddings can be used for various applications such as identity verification. The system is trained using the Labeled Faces in the Wild (LFW) dataset, which is a large-scale database of face images containing images of celebrities and public figures in diverse conditions. A notable feature of this system is its web interface, which allows users to interact with the model via their camera, enabling real-time face recognition.

## Dataset Preprocessing

The dataset preprocessing step is crucial for preparing the data for training. It involves several steps:

1. **Resizing and Saving Images:** Images from the LFW dataset are resized to a uniform size suitable for input to the neural network. Resized images are saved for further processing.

2. **Face Detection and Cropping:** A pre-trained OpenCV face detection model is used to identify and extract faces from images. These faces are then cropped and saved separately.

3. **Ensuring Data Consistency:** The preprocessing step ensures that all images are consistently sized and contain only the face region, which is essential for effective training.

## Model Architecture

The core of this system is built on the ResNet50 model, which is a deep convolutional neural network known for its effectiveness in image recognition tasks. The model architecture is adapted for the face recognition task, and it performs the following key functions:

- **Image Preprocessing:** The model loads and preprocesses images to ensure they meet the required input size and format.
  
- **Triplet Generation:** For training, the model generates triplets of images. Each triplet consists of an anchor image (the image of the target person), a positive image (another image of the same person), and a negative image (an image of a different person). This setup enables the model to learn to differentiate between different individuals.
  
- **Embedding Generation:** The model embeds images using a modified ResNet50 architecture. This process involves passing images through the network to produce embeddings, which are numerical representations of facial features.
  
- **Triplet Loss Optimization:** During training, the model employs triplet loss, a loss function specifically designed for training with triplets. This loss function encourages the model to minimize the distance between embeddings of images from the same person (anchor and positive) while maximizing the distance between embeddings of images from different people (anchor and negative). This helps the model learn to generate embeddings that are discriminative for each identity.

## Training Process

The training process involves feeding the preprocessed dataset into the model and optimizing its parameters using backpropagation. The key steps of the training process include:

- **Compilation:** The model is compiled with appropriate optimizer and loss function settings.
  
- **Training Iterations:** The model is trained over multiple epochs, with each epoch consisting of iterations over batches of training data.
  
- **Loss Monitoring:** The training progress is monitored by tracking the loss value, which indicates how well the model is performing relative to the training data.
  
- **Model Saving:** Once training is complete, the trained model is saved for future use.

```
Model is compiled and ready to be trained.
Starting training...
Epoch 1/10
100/100 [==============================] - 843s 8s/step - loss: 1.9330
Epoch 2/10
100/100 [==============================] - 795s 8s/step - loss: 0.4049
Epoch 3/10
100/100 [==============================] - 795s 8s/step - loss: 0.4049
Epoch 4/10
100/100 [==============================] - 742s 7s/step - loss: 0.3562
Epoch 5/10
100/100 [==============================] - 744s 7s/step - loss: 0.3197
Epoch 6/10
100/100 [==============================] - 703s 7s/step - loss: 0.2769
Epoch 7/10
100/100 [==============================] - 714s 7s/step - loss: 0.2142
Epoch 8/10
100/100 [==============================] - 554s 6s/step - loss: 0.1768
Epoch 9/10
100/100 [==============================] - 600s 6s/step - loss: 0.1704
Epoch 10/10
100/100 [==============================] - 565s 6s/step - loss: 0.1478
Training completed.
Model saved successfully.
```

## Web Interface

The system includes a web interface that enables users to interact with the trained model using their camera. This interface provides a user-friendly way to perform real-time face recognition tasks, allowing for seamless integration into various applications and systems. Users can capture images through their camera, and the system will process these images to perform face recognition.

## Prediction

The prediction process involves using the trained model to compare embeddings of pairs of images and compute a similarity score. The system can predict whether two images depict the same person or different people based on this score. Two scenarios are considered:

- **Prediction Between Different People:** The system loads the trained model and computes embeddings for two images depicting different people. It then calculates the Euclidean distance between the embeddings to determine their dissimilarity.
  
- **Prediction Between the Same Person:** Similarly, the system computes embeddings for two images depicting the same person and calculates the Euclidean distance between the embeddings to measure their similarity.

## Usage

To use the system:

1. **Download Requirements:** Download the `requirements.txt` file, which contains the necessary Python packages.
  
2. **Launch the Application:** Run `app.py` to start the web interface.
  
3. **Interact with the Web UI:** Connect to the web interface and start taking photos using your camera. The system will process these images for face recognition, comparing them with images in the dataset to find the best match.