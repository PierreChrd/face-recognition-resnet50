import os, random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, concatenate

# Load and preprocess image
def load_image(path):
    img = image.load_img(path, target_size=(64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

# Generate triplets
def generate_triplets(directory, batch_size=32):
    anchor_images = np.zeros((batch_size, 64, 64, 3))
    positive_images = np.zeros((batch_size, 64, 64, 3))
    negative_images = np.zeros((batch_size, 64, 64, 3))
    
    persons = os.listdir(directory)
    
    while True:
        for i in range(batch_size):
            anchor_person = random.choice(persons)
            negative_person = random.choice([p for p in persons if p != anchor_person])
            
            anchor_path = os.path.join(directory, anchor_person)
            positive_path = anchor_path
            negative_path = os.path.join(directory, negative_person)
            
            anchor_image = random.choice(os.listdir(anchor_path))
            positive_image = random.choice([img for img in os.listdir(positive_path) if img != anchor_image])
            negative_image = random.choice(os.listdir(negative_path))
            
            anchor_images[i] = load_image(os.path.join(anchor_path, anchor_image))
            positive_images[i] = load_image(os.path.join(positive_path, positive_image))
            negative_images[i] = load_image(os.path.join(negative_path, negative_image))

        yield [anchor_images, positive_images, negative_images], np.zeros((batch_size, 3*128))

# Load model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Embedding model
def embedding_model():
    input = Input(shape=(64, 64, 3))  
    x = base_model(input)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)

# Triplet loss
def triplet_loss(y_true, y_pred, alpha = 0.4):
    total_length = y_pred.shape.as_list()[-1]
    anchor, positive, negative = y_pred[:,:int(1/3*total_length)], y_pred[:,int(1/3*total_length):int(2/3*total_length)], y_pred[:,int(2/3*total_length):]

    pos_dist = tf.reduce_sum(tf.square(anchor-positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor-negative), axis=1)

    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    return loss

# Triplet network
anchor_input = Input(shape=(64, 64, 3), name='anchor_input')
positive_input = Input(shape=(64, 64, 3), name='positive_input')
negative_input = Input(shape=(64, 64, 3), name='negative_input')

shared_embedding = embedding_model()

encoded_anchor = shared_embedding(anchor_input)
encoded_positive = shared_embedding(positive_input)
encoded_negative = shared_embedding(negative_input)

merged_vector = concatenate([encoded_anchor, encoded_positive, encoded_negative], axis=-1, name='merged_layer')

model = Model(inputs=[anchor_input, positive_input, negative_input], outputs=merged_vector)
model.compile(optimizer='adam', loss=triplet_loss)

print("Model is compiled and ready to be trained.")

# Train the model
batch_size = 32
steps_per_epoch = 100  

print("Starting training...")
history = model.fit(
    generate_triplets('lfw_filtered', batch_size=batch_size),
    steps_per_epoch=steps_per_epoch,
    epochs=10  
)

print("Training completed.")

model.save('my_resnet50_triplet_model.h5')

print("Model saved successfully.")