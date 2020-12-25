from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
import numpy as np
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from keras import regularizers
from keras import optimizers

class Train:

    def __init__(self):
        print("hello")


    def process(self,candles, type,volume):
        """
        processing candles to a format/shape consumable for the model
        :param candles: dict/list of Open, High, Low, Close prices
        :return: X: numpy.ndarray, Y: numpy.ndarray
        """

        decimal_figures = 6
        y_change_threshold = 0.0001
        if type=='train':
            if volume == 1:
                X = np.ndarray(shape=(0, 5))
            if volume == 0:
                X = np.ndarray(shape=(0, 4))
            Y = np.ndarray(shape=(0, 1))

            # clean and process data
            previous_close = None

            for candle in candles:
                if volume==1:
                    X = np.append(X,
                                  np.array([[
                                      # High 2 Open Price
                                      round(candle['high'] / candle['open'] - 1, decimal_figures),
                                      # Low 2 Open Price
                                      round(candle['low'] / candle['open'] - 1, decimal_figures),
                                      # Close 2 Open Price
                                      round(candle['close'] / candle['open'] - 1, decimal_figures),
                                      # High 2 Low Price
                                      round(candle['high'] / candle['low'] - 1, decimal_figures),

                                      round(1/candle['tick_volume'], decimal_figures)]]),
                                  axis=0)
                else:
                      X = np.append(X,
                                    np.array([[
                                        # High 2 Open Price
                                        round(candle['high'] / candle['open'] - 1, decimal_figures),
                                        # Low 2 Open Price
                                        round(candle['low'] / candle['open'] - 1, decimal_figures),
                                        # Close 2 Open Price
                                        round(candle['close'] / candle['open'] - 1, decimal_figures),
                                        # High 2 Low Price
                                        round(candle['high'] / candle['low'] - 1, decimal_figures)]]),


                                    axis=0)


                # Compute the Y / Target Variable
                if previous_close is not None:
                    y = 0
                    precise_prediction = round(1 - previous_close / candle['close'], decimal_figures)

                    # positive price change more growth than threshold
                    if precise_prediction > y_change_threshold:
                        y = 1
                    # negative price change with more decline than threshold
                    elif precise_prediction < 0 - y_change_threshold:
                        y = 2
                    # price change in between positive and negative threshold
                    elif precise_prediction < y_change_threshold and precise_prediction > 0 - y_change_threshold:
                        y = 0

                    Y = np.append(Y, np.array([[y]]))
                else:
                    Y = np.append(Y, np.array([[0]]))
                previous_close = candle['close']

            Y = np.delete(Y, 0)
            Y = np.append(Y, np.array([0]))
            Y = to_categorical(Y, num_classes=3)


            return X, Y


    def get_lstm_model(self,layer,volume,lrs):
        model = Sequential()
        model.add(LSTM(units=20, input_shape=(3,1), return_sequences=True))
        model.add(LSTM(units=20))
        model.add(Dense(units=3,
                                activation='softmax'))
        sgd = optimizers.SGD(lr=lrs)
        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])

        return model

    def get_model(self,layer,volume,lrs):
        """
        Here we define our model Layers using Keras
        :return: Keras Model Object
        """
        inputsize = 4
        if volume==1:
            inputsize=5
        model = Sequential()
        if(layer==2):
                model.add(Dense(units=16,
                                activation='relu',
                                input_shape=(inputsize,)))
                # model.add(Dropout(0.2, input_shape=(inputsize,)))

                model.add(Dense(units=16,
                                activation='relu',
                                kernel_regularizer=regularizers.l2(0.001),
                                activity_regularizer=regularizers.l1(0.001)))

                model.add(Dense(units=3,
                                activation='softmax'))
        elif(layer==3):
            model.add(Dense(units=16,
                            activation='relu',
                            input_shape=(inputsize,)))
            model.add(Dense(units=16,
                            activation='relu',
                            kernel_regularizer=regularizers.l2(0.001),
                            activity_regularizer=regularizers.l1(0.001)))
            model.add(Dense(units=16,
                            activation='relu',
                            kernel_regularizer=regularizers.l2(0.001),
                            activity_regularizer=regularizers.l1(0.001)))

            model.add(Dense(units=3,
                            activation='softmax'))



        sgd = optimizers.SGD(lr=lrs)

        model.compile(loss='categorical_crossentropy',
                      optimizer=sgd,
                      metrics=['accuracy'])
        return model

    week = 60 * 60 * 24 * 7
    decimal_figures = 6
    y_change_threshold = 0.001







    def retrain(self,candles,epochs,layer,batch_size,volume,lr,currency):
        """
        Retrains a model for a specific a) trading instrument, b) timeframe, c) input shape
        """

        # get historical data from data service

        X, Y = self.process(candles,"train",volume)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)

        model = self.get_lstm_model(layer,volume,lr)
        fit = model.fit(X_train, Y_train, epochs=epochs, verbose=True)
        score = model.evaluate(X_test, Y_test, batch_size=batch_size)
        print("this is score  =================> :",score)
        print(model.summary())

        # TODO: Save trained model to disk
        # filename = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        model.save("cuurecy "+currency+" lr "+str(lr)+" volume "+str(volume) + " batch size "+str(batch_size)+" epochs "+str(epochs) +"  layers "+ str(layer) + " towLayer  "+ str(score[0]) + "  " + str(score[1]) )


