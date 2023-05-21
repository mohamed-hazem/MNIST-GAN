import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.chdir(os.path.dirname(__file__))

import tensorflow as tf
import matplotlib.pyplot as plt

images_num = 16
latent_size = 100

model_path = "models/generator.h5"
model = tf.keras.models.load_model(model_path)


while True:
    noise = tf.random.normal([images_num, latent_size])
    images = model(noise)

    plt.figure(figsize=(4, 4))
    for i in range(images_num):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i, :, :, 0], cmap="gray")
        plt.axis("off")
    plt.show()

    x = input("Generate?: ")
    if (x == "q"):
        exit()