from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose, Dense, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist

from numpy import load, argmax, sum, random, clip
from matplotlib import pyplot as plt

def CNN_model(input_shape, output_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3,3), padding='same'))
    model.add(Dropout(0.1))
    model.add(Conv2D(32, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(4,4), padding='same'))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(output_shape, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def autoencoder_model(input_shape):
    model = Sequential()
    #Encoder 
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3,3), padding='same'))
    model.add(Conv2D(32, kernel_size=3, activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    #Decoder
    model.add(Conv2DTranspose(32, kernel_size=3, strides=1, activation='relu', padding='same')) 
    model.add(Conv2DTranspose(32, kernel_size=3, strides=3, activation='relu', padding='same'))
    model.add(Conv2D(1, kernel_size=3, activation='sigmoid', padding='same'))

    model.compile(optimizer='adam', loss='binary_crossentropy')

    return model

def run_CNN():
    N = 4
    CNN = CNN_model((21,21,1), N)
    # (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # X_train = X_train.reshape(60000,28,28,1)
    # X_test = X_test.reshape(10000,28,28,1)

    # print(X_train[0] )

    # print(y_train[0] )

    # y_train = to_categorical(y_train)
    # y_test = to_categorical(y_test)

    # print(y_train[0] )

    X_train = load('ElectricDevices/X_train.npy')
    X_test = load('ElectricDevices/X_test.npy')
    y_train = load('ElectricDevices/Y_train.npy')
    y_test = load('ElectricDevices/Y_test.npy')

    # print(X_train.shape)
    # print(y_train.shape)
    # print(X_test.shape)
    # print(y_test.shape)

    CNN.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=32)

    tags = ['W_Air_cond','W_Computers','W_Audio_TV','W_Lights','W_Kitchen','W_Washing_m','W_Dish_w','W_Gas_boiler','W_Oven_vitro'] 
    classes = {} 

    for tag in tags:

        X_eval = load('ElectricDevices/X_eval_'+tag+'.npy')

        prediction = CNN.predict(X_eval)
        print()
        print(tag)
        print(prediction)
        print('tag=', argmax(sum(prediction, axis=0)) + 1)
        classes[tag] = argmax(sum(prediction, axis=0)) + 1
        # print('tag=', argmax(prediction), argmax(prediction)%N + 1)
        # classes[tag] =  argmax(prediction)%N + 1

    print(classes)

def preprocess(array):
    """
    Normalizes the supplied array and reshapes it into the appropriate format.
    """

    array = array.astype("float32") / 255.0
    array = array.reshape(len(array), 28, 28, 1)
    return array

def noise(array):
    """
    Adds random noise to each image in the supplied array.
    """

    noise_factor = 0.4
    noisy_array = array + noise_factor * random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return clip(noisy_array, 0.0, 1.0)

def display(array1, array2):
    """
    Displays ten random images from each one of the supplied arrays.
    """

    n = 10

    indices = random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    size1 = array1.shape[1] 
    size2 = array2.shape[1] 

    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(size1, size1))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(size2, size2))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()

if __name__ == "__main__":
    #CNN classification 
    # run_CNN

    #Autoencoder

    autoencoder = autoencoder_model((21,21,1))

    autoencoder.summary()

    X_train = load('ElectricDevices/X_train_AE_2.npy')
    Y_train = load('ElectricDevices/Y_train_AE_2.npy')
    X_test = load('ElectricDevices/X_test_AE_2.npy')
    Y_test = load('ElectricDevices/Y_test_AE_2.npy')

    display(X_train, Y_train)

    # autoencoder.fit(X_train, Y_train, validation_split=0.2, epochs=200, batch_size=8, shuffle=True)
    autoencoder.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=128, shuffle=True)

    # predictions = autoencoder.predict(X_train[:,:,:,:])

    # display(Y_train[:,:,:,:], predictions)

    predictions = autoencoder.predict(X_test)

    display(Y_test, predictions)

    # (X_train, _), (X_test, _) = mnist.load_data()
    # #preprocess 
    # X_train = preprocess(X_train)
    # X_test = preprocess(X_test)

    # noisy_X_train = noise(X_train)
    # noisy_X_test = noise(X_test)

    # display(X_train, noisy_X_train)

    # #Train the NN
    # autoencoder = autoencoder_model((28,28,1))

    # autoencoder.fit(noisy_X_train, X_train, validation_data=(noisy_X_test, X_test), epochs=10, batch_size=128, shuffle=True) 

    # predictions = autoencoder.predict(noisy_X_test)

    # display(noisy_X_test, predictions)
