import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#import data and split it into train and test data
mnist = tf.keras.datasets.mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()

#normalize train and test data so that values become between 0 and 1
X_train = tf.keras.utils.normalize(X_train,axis=1)
X_test = tf.keras.utils.normalize(X_test,axis=1)

#Creates a sequential model, meaning layers are added one after another in a straight line.
model = tf.keras.models.Sequential()
#add layers

#Converts each 28×28 image (2D) into a 1D array of 784 pixels.
#Example: a matrix like [[0, 1], [2, 3]] becomes [0, 1, 2, 3].
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

#Adds a hidden layer with 120 neurons.
#Uses ReLU activation to introduce non-linearity.
model.add(tf.keras.layers.Dense(units=120,activation=tf.nn.relu))

#Another hidden layer with 120 neurons and ReLU.
#Helps the model learn more complex patterns.
model.add(tf.keras.layers.Dense(units=120,activation=tf.nn.relu))

#This is the output layer with 10 neurons.
#Each neuron corresponds to a digit (0 to 9).
#Uses softmax, which outputs a probability distribution over the 10 classes.
model.add(tf.keras.layers.Dense(units=10,activation=tf.nn.softmax))

#optimizer='adam': Uses the Adam optimizer, a very popular and efficient version of gradient descent.
#loss='sparse_categorical_crossentropy': This is used for multi-class classification problems.
#"Sparse" means your labels are integers (0 to 9), not one-hot encoded.
#metrics=['accuracy']: Tells Keras to track and show accuracy during training and evaluation.
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Trains the model using the training data (X_train, y_train) for 3 passes over the entire dataset (3 epochs).
#During this step, the model learns by updating its weights and biases.
model.fit(X_train,y_train,epochs=3)

#Tests the trained model on unseen data (X_test, y_test) to measure how well it generalizes.
#Returns the loss and accuracy
loss,accuracy = model.evaluate(X_test,y_test)
print(accuracy)
print(loss)

#Saves the trained model to a file named 'digits.model'.
#This includes:The architecture, The weights, The training configuration
model.save('digits.keras')

#test the model on my images
for x in [5,7]:
    img = cv.imread(f'test_digit/{x}.png', cv.IMREAD_GRAYSCALE)
    img = np.invert(np.array([img]))
    prediction = model.predict(img)
    print(np.argmax(prediction))
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()