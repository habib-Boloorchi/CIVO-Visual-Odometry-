import absolute_relative_pose_handler
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Activation
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense

import pandas as pd
import numpy as np


class Dof_trainer():
    def __init__(self):
        self.activation = 'elu'
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=2e-4)
        self.loss_function = 'huber_loss'
        self.epochs = 200
        self.batch_size = 32



    def vo_model(self):
        inputs = Input(shape=(7,))
        x = Dense(256, activation=None)(inputs)
        x = Activation(self.activation)(x)
        x = Dense(128, activation=None)(x)
        x = Activation(self.activation)(x)
        x = Dense(64, activation=None)(x)
        x = Activation(self.activation)(x)
        x = Dense(64, activation=None)(x)
        x = Activation(self.activation)(x)
        x = Dense(32, activation=None)(x)
        x = Activation(self.activation)(x)
        x = Dense(32, activation=None)(x)
        x = Dropout(0.5)(x)
        outputs = Dense(units=1)(x)
        functional_model = Model(inputs=inputs, outputs=outputs)
        return functional_model

    def runner(self, X_train, X_val, y_train, y_val, activation='relu', plot=False):
        self.activation = activation
        y_hat = []
        dofs = ['x', 'y', 'z','rw', 'rx', 'ry', 'rz']
        rpe_train = []
        rpe_val = []
        rpe_ytest_list = []
        rpe_yhat_list = []
        hist_list = []
        value_list = []

        # X_train, X_val, y_train, y_val = self.convert_to_euler(X_train, X_val, y_train, y_val)
        for index, value in enumerate(dofs):
            model = self.vo_model()
            model.compile(loss=self.loss_function, optimizer=self.optimizer,
                          metrics=[tf.keras.metrics.RootMeanSquaredError()])
            hist = model.fit(X_train, y_train[:, index], epochs=self.epochs, batch_size=self.batch_size, verbose=0,
                             validation_data=(X_val, y_val[:, index]))
            hist_list.append(hist)
            value_list.append(value)

            maeq = hist.history['root_mean_squared_error']
            val_maeq = hist.history['val_root_mean_squared_error']
            rpe_train.append(maeq[-1])
            rpe_val.append(val_maeq[-1])

            yhat__for_plot = model.predict(X_val)
            rpe_ytest_list.append(y_val[:, index])
            rpe_yhat_list.append(yhat__for_plot)

            y_hat_temp = model.predict(X_val)
            tf.keras.backend.clear_session()

            y_hat.append(y_hat_temp)
        print('Training is done for all degrees of freedom')
        y_hat = np.array(y_hat)
        y_hat = y_hat.squeeze()
        y_hat = y_hat.transpose()
        y_hat = pd.DataFrame(y_hat, columns=dofs)
        # rpe_train_dofs = np.mean(np.array(rpe_train))
        # rpe_val_dofs = np.mean(np.array(rpe_val))
        if plot:
            self.plot_epochs(hist_list=hist_list, value_list=value_list)
            self.plot_relative_pose(RPE_train_list=rpe_ytest_list, RPE_val_list=rpe_yhat_list, value_list=value_list)
        # print('*********************************Now Absolute Poses*********************************')
        y_val = pd.DataFrame(y_val, columns=dofs)
        return y_hat, y_val

    def plot_epochs(self, hist_list, value_list):
        fig, axs = plt.subplots(4, 2, figsize=(10, 10))
        fig.tight_layout(pad=5.0)
        for hist, value, ax in zip(hist_list, value_list, axs.flat):
            ax.plot(hist.history['root_mean_squared_error'])
            ax.plot(hist.history['val_root_mean_squared_error'])
            ax.set_title('Model accuracy of ' + value)
            ax.set_ylabel('RMSE (Relative Pose Error)')
            ax.legend(['train', 'validation'], loc='upper right')
        plt.show()
        plt.clf()

    def plot_relative_pose(self, RPE_train_list, RPE_val_list, value_list):
        print('******************************************RPE of DOFs***********************************')
        fig, axs = plt.subplots(7, 1, figsize=(10, 10))
        fig.tight_layout(pad=5.0)
        for RPE_train, RPE_val, value, ax in zip(RPE_train_list, RPE_val_list, value_list, axs.flat):
            ax.plot(RPE_train)
            ax.plot(RPE_val)
            ax.set_xlabel('Frame number')
        ax.set_ylabel('Relative Pose')
        fig.legend(['yhat', 'ytest'], loc='upper right')
        plt.show()
        plt.clf()


