from flask import Flask, render_template, request
from opencv_face_detection import detect_and_save_face
from PIL import Image
import numpy as np
import uuid

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
import tensorflow as tf
import csv

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def euclidean_distance(embedding1, embedding2):
    """Calculate the Euclidean distance between two embeddings."""
    print(np.linalg.norm(embedding1 - embedding2))
    return np.linalg.norm(embedding1 - embedding2)

# Triplet loss function as defined during training
def triplet_loss(y_true, y_pred, alpha = 0.4):
    total_length = y_pred.shape.as_list()[-1]
    anchor, positive, negative = y_pred[:,:int(1/3*total_length)], y_pred[:,int(1/3*total_length):int(2/3*total_length)], y_pred[:,int(2/3*total_length):]

    pos_dist = tf.reduce_sum(tf.square(anchor-positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor-negative), axis=1)

    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(64, 64))  # Update size if different
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Load the saved model
MODEL = load_model('my_resnet50_triplet_model.h5', custom_objects={'triplet_loss': triplet_loss})
print("Model loaded successfully.")

app = Flask(__name__)

@app.route("/")
def index():
  return render_template("index.html")

@app.route("/dataset")
def dataset():
  return render_template("dataset.html")

@app.route("/about")
def about():
  return render_template("about.html")

@app.route("/take-photo", methods=["POST"])
def take_photo():
    # Take a Photo
    photo = request.files["photo"]

    # Generate UUID for original photo
    unique_filename = "photos/" + str(uuid.uuid4()) + ".jpg"

    # Save the photo
    photo.save(unique_filename)

    # Process the photo to detect and save the face, and generate another UUID for the output photo
    output_filename = 'photos/output_' + str(uuid.uuid4()) + '.jpg'
    detect_and_save_face(unique_filename, output_filename)

    # Resize the output photo to 100x100
    resize_and_save_image(output_filename)

    image_path = output_filename

    # Preprocess the image
    preprocessed_img = preprocess_image(image_path)
    embeddings = MODEL.predict([preprocessed_img, preprocessed_img, preprocessed_img])
    # Extract the embeddings (the first part of the output)
    embedding_output = embeddings[0]  

    # Save image name and embedding to CSV
    with open('face_database.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([image_path, embedding_output.tolist()])

    # Load the CSV into a dict
    face_data = {}
    with open('face_database.csv', mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            print(row)
            image_name, stored_embedding = row
            stored_embedding = eval(stored_embedding)
            distance = euclidean_distance(embedding_output, stored_embedding)
            face_data[image_name] = [stored_embedding, distance]

    print(face_data)
    sorted_face_data = sorted(face_data.items(), key=lambda x: x[1][1])

    # Get top 3 results
    top_3_images = sorted(face_data.items(), key=lambda x: x[1][1])[:3]

    # Plot the images
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (image_name, data) in enumerate(top_3_images):
        img = mpimg.imread(image_name)
        axes[i].imshow(img)
        axes[i].set_title(f'Distance: {data[1]:.2f}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

    # Save plot to file
    plot_filename = 'top_3_images.png'
    fig.savefig(plot_filename)
    plt.close(fig)

    return "ok"

def resize_and_save_image(image_path):
    with Image.open(image_path) as img:
        # Resize the image
        img = img.resize((224, 224), Image.LANCZOS)

        # Save the resized image
        img.save(image_path)

if __name__ == "__main__":
  app.run(debug=True)
