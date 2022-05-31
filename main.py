import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

def visualise_image(train_images,index):
    plt.figure()
    plt.imshow(train_images[index])
    plt.colorbar()
    plt.grid(False)
    plt.show()

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)

def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")
    
if __name__ == "__main__":
    ### Load dataset
    fashion_mnist = keras.datasets.fashion_mnist  
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    ### Split data into testing set and training set
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  
    #visualise_image(train_images,1) # Uncomment to visualise entry image

    ### Preprocess data: Scaling greyscale pixel values (0-255) to be between 0 and 1
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    ### Build model using Keras sequential model
    # Input layer has 784 neurons. Use the flatten layer of shape (28,28)
    # Only one hidden layer which has 128 neurons.
    # Output layer has 10 neurons as we have 10 labels.
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
        keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
        keras.layers.Dense(10, activation='softmax') # output layer (3)
    ])

    ### Compile model
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    ### Train model
    model.fit(train_images, train_labels, epochs=10)

    ### Evaluate model
    test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
    print('Test accuracy:', test_acc)
    #Observation: Accuracy on testing set is lower than training set. Overfitting occurs!

    ### Make one prediction
    predictions = model.predict(test_images)
    #Get prediction for the first data point
    predictions[0]
    np.argmax(predictions[0])
    #Check prediction
    test_labels[0]

    ### Make predictions (generalised)
    COLOR = 'white'
    plt.rcParams['text.color'] = COLOR
    plt.rcParams['axes.labelcolor'] = COLOR
    num = get_number()
    image = test_images[num]
    label = test_labels[num]
    predict(model, image, label)