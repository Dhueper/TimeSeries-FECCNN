from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist

from numpy import load, argmax, sum

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

if __name__ == "__main__":
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
