import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777' )
    plt.ylim([0, 1])
    predicted_labels = np.argmax(predictions_array)

    thisplot[predicted_labels].set_color('red')
    thisplot[true_label].set_color('blue')

matplotlib.get_backend()

#data = np.random.random((1000 , 32))
#labels = np.random.random((1000 , 10))
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

pp = PdfPages('test.pdf')
plt.savefig(pp,format='pdf')
pp.close()
os.system('evince test.pdf &')

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
    if i == 24:
        pp = PdfPages('test' + str(i) + '.pdf')
        plt.savefig(pp,format='pdf')
        pp.close()
        os.system('evince test' + str(i) + '.pdf &')

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10,activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=1)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
for i in range(0):#range( len(predictions)):
    print(i)
    print("I predict: " + str(class_names[np.argmax(predictions[i])]))
    print("It actually was: " + str(class_names[test_labels[i]]) )
    print("\n")

for i in range(0,0):
    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions,  test_labels)

    pp = PdfPages('test' + str(i) + '.pdf')
    plt.savefig(pp,format='pdf')
    pp.close()
    os.system('evince test' + str(i) + '.pdf &')

num_rows = 5
num_cols = 3
num_images = num_cols * num_rows
plt.figure( figsize=( 2 * 2 * num_cols, 2 * num_rows) )
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 *i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)

pp = PdfPages('massplot.pdf')
plt.savefig(pp,format='pdf')
pp.close()
os.system('evince massplot.pdf &')