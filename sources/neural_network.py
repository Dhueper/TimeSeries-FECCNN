from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist

def CNN_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Conv2D(32, kernel_size=5, activation='relu'))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Flatten())
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

if __name__ == "__main__":
    CNN = CNN_model()
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(60000,28,28,1)
    X_test = X_test.reshape(10000,28,28,1)

    print(y_train[0] )

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    print(y_train[0] )

    CNN.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=8)