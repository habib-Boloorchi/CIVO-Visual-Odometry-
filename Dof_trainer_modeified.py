import absolute_relative_pose_handler
import os
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Input, Activation
from keras.layers import Dropout
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense,concatenate
import datetime
import pandas as pd
import numpy as np
from keras.utils.vis_utils import plot_model
from absolute_relative_pose_handler import abs_rel_handler

class Dof_trainer():
    def __init__(self):
        self.activation = 'elu'
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=2e-4)
        self.loss_function = 'huber_loss'
        self.epochs =100
        self.batch_size = 32
        self.model_folder = "models"

    def vo_model(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        inputs = Input(shape=(7,), name=f"{timestamp}_input_layer")
        x = Dense(256, activation=None, name=f"{timestamp}_dense_1")(inputs)
        x = Activation(self.activation, name=f"{timestamp}_activation_1")(x)
        # x = Dense(256, activation=None, name=f"{timestamp}_dense_2")(x)
        # x = Activation(self.activation, name=f"{timestamp}_activation_2")(x)
        # x = Dense(128, activation=None, name=f"{timestamp}_dense_3")(x)
        # x = Activation(self.activation, name=f"{timestamp}_activation_3")(x)
        # x = Dense(64, activation=None, name=f"{timestamp}_dense_3_5")(x)
        # x = Activation(self.activation, name=f"{timestamp}_activation_3_5")(x)
        # x = Dense(64, activation=None, name=f"{timestamp}_dense_4")(x)
        # x = Activation(self.activation, name=f"{timestamp}_activation_4")(x)
        x = Dense(32, activation=None, name=f"{timestamp}_dense_5")(x)
        x = Activation(self.activation, name=f"{timestamp}_activation_5")(x)
        x = Dense(32, activation=None, name=f"{timestamp}_dense_6")(x)
        x = Dropout(0.5, name=f"{timestamp}_dropout_1")(x)
        outputs = Dense(units=1, name=f"{timestamp}_output_layer")(x)
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
        # Create DataFrames to store histories and predictions
        hist_df = pd.DataFrame()
        y_hat_rel_df = pd.DataFrame()
        y_hat_abs_df = pd.DataFrame()

        abs_rel_handler_instance = absolute_relative_pose_handler.abs_rel_handler()

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
            #save model
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = os.path.join(self.model_folder, f"{self.activation}_{value}_model_{index}_{timestamp}")
            os.makedirs(model_dir, exist_ok=True)

            model.save(os.path.join(model_dir, f"{self.activation}_{value}_model_{timestamp}.h5"))

            yhat__for_plot = model.predict(X_val)
            rpe_ytest_list.append(y_val[:, index])
            rpe_yhat_list.append(yhat__for_plot)

            y_hat_temp = model.predict(X_val)
            epoch_data = {
                'dof': [value] * self.epochs,
                'activation': [self.activation] * self.epochs,
                'epoch': list(range(1, self.epochs + 1)),
                'train_loss': hist.history['loss'],
                'val_loss': hist.history['val_loss'],
                'train_rmse': hist.history['root_mean_squared_error'],
                'val_rmse': hist.history['val_root_mean_squared_error']
            }
            epoch_df = pd.DataFrame(epoch_data)
            hist_df = pd.concat([hist_df, epoch_df], ignore_index=True)
            # epoch_df = pd.DataFrame(epoch_data)
            # hist_df = pd.concat([hist_df, epoch_df], ignore_index=True)

            # Save relative y_hat data
            y_hat_rel_df[f'{value}_{self.activation}_relative'] = y_hat_temp.squeeze()

            # Convert y_hat_temp numpy array to DataFrame
            # y_hat_temp_df = pd.DataFrame(y_hat_temp, columns=['x', 'y', 'z', 'rw', 'rx', 'ry', 'rz'])

            # Save absolute y_hat data
            # y_hat_abs_temp = abs_rel_handler_instance.relative_to_absolute_transform(y_hat_temp_df)
            # y_hat_abs_temp = y_hat_abs_temp.drop(columns=["#timestamp [ns]"]).to_numpy()
            # y_hat_abs_df[f'{value}_{self.activation}_absolute'] = y_hat_abs_temp.squeeze()

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
        return y_hat, y_val ,hist_df, y_hat_rel_df



    def freeze_model(self,model):
        for layer in model.layers:
            layer.trainable = False


    def load_models_freez_and_concatenate(self):

        modelx = tf.keras.models.load_model('models\\elu_x_model_0_20230329_094300\\elu_x_model_20230329_094300.h5')
        for layer in modelx.layers:
            layer.trainable = False
        modely = tf.keras.models.load_model('models\\elu_y_model_1_20230329_094426\\elu_y_model_20230329_094426.h5')
        for layer in modely.layers:
            layer.trainable = False
        modelz = tf.keras.models.load_model('models\\elu_z_model_2_20230329_113440\\elu_z_model_20230329_113440.h5')
        for layer in modelz.layers:
            layer.trainable = False
        modelrw = tf.keras.models.load_model('models\\elu_rw_model_3_20230329_094722\\elu_rw_model_20230329_094722.h5')
        for layer in modelrw.layers:
            layer.trainable = False
        modelrx = tf.keras.models.load_model('models\\elu_z_model_2_20230329_113440\\elu_z_model_20230329_113440.h5')
        for layer in modelrx.layers:
            layer.trainable = False
        modelry = tf.keras.models.load_model('models\\elu_z_model_2_20230329_113440\\elu_z_model_20230329_113440.h5')
        for layer in modelry.layers:
            layer.trainable = False
        modelrz = tf.keras.models.load_model('models\\leaky_relu_rz_model_6_20230329_115334\\leaky_relu_rz_model_20230329_115334.h5')
        for layer in modelrz.layers:
            layer.trainable = False

        def assign_unique_layer_names(model, name_suffix):
            for layer in model.layers:
                layer._name = layer.name + name_suffix

        assign_unique_layer_names(modelx, "_modelx")
        assign_unique_layer_names(modely, "_modely")
        assign_unique_layer_names(modelz, "_modelz")
        assign_unique_layer_names(modelrw, "_modelrw")
        assign_unique_layer_names(modelrx, "_modelrx")
        assign_unique_layer_names(modelry, "_modelry")
        assign_unique_layer_names(modelrz, "_modelrz")
        modelx_output = modelx.output
        modely_output = modely.output
        modelz_output = modelz.output
        modelrw_output = modelrw.output
        modelrx_output = modelrx.output
        modelry_output = modelry.output
        modelrz_output = modelrz.output

        # models = [modelx,modely,modelz,modelrw,modelrx,modelry,modelrz]
        modelx_input = modelx.input
        modelx_input._keras_history.layer._name = "modelx_input"

        modely_input = modely.input
        modely_input._keras_history.layer._name = "modely_input"

        modelz_input = modelz.input
        modelz_input._keras_history.layer._name = "modelz_input"

        modelrw_input = modelrw.input
        modelrw_input._keras_history.layer._name = "modelrw_input"

        modelrx_input = modelrx.input
        modelrx_input._keras_history.layer._name = "modelrx_input"

        modelry_input = modelry.input
        modelry_input._keras_history.layer._name = "modelry_input"

        modelrz_input = modelrz.input
        modelrz_input._keras_history.layer._name = "modelrz_input"

        # combined_model =concatenate([modelx,modely,modelz,modelrw,modelrx,modelry,modelrz], name='Concatenate')
        combined_model = concatenate(
            [modelx_output, modely_output, modelz_output, modelrw_output, modelrx_output, modelry_output,
             modelrz_output], name='Concatenate')

        x = Dense(32, activation=None)(combined_model)
        x = Activation('selu')(x)
        x = Dense(32, activation=None)(x)
        x = Dropout(0.5)(x)

        final_model_output = Dense(7, activation='selu')(x)

        final_model = tf.keras.Model(
            inputs=[modelx_input, modely_input, modelz_input, modelrw_input, modelrx_input, modelry_input,
                    modelrz_input], outputs=final_model_output)
        plot_model(final_model, to_file='i.png')

        return final_model
    def train_concatenated(self,X_train, y_train, X_test, y_test):
        abs_rel= abs_rel_handler()
        dofs = ['x', 'y', 'z', 'rw', 'rx', 'ry', 'rz']
        model =self.load_models_freez_and_concatenate()
        # print (y_test.shape)
        model.compile(loss=self.loss_function, optimizer=self.optimizer,
                      metrics=[tf.keras.metrics.RootMeanSquaredError()])
        X_trains =[X_train,X_train,X_train,X_train,X_train,X_train,X_train]
        X_tests = [X_test,X_test,X_test,X_test,X_test,X_test,X_test]
        model.fit(X_trains, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2,
                         validation_data=(X_tests, y_test))
        model.save('IHV0_model.h5')
        y_hat = model.predict(X_tests)
        y_hat_df = pd.DataFrame(y_hat,columns=dofs)
        y_test_df = pd.DataFrame(y_test,columns=dofs)
        y_test_absolut = abs_rel.relative_to_absolute_transform(file_input=y_test_df)
        y_hat_absolut = abs_rel.relative_to_absolute_transform(file_input=y_hat_df)

        # print(y_hat_absolut)
        y_hat_absolut.to_csv('v12_ytest_abs.csv')
        y_test_absolut.to_csv('v12_ytest_abs.csv')
        self.ate_calculator(y_hat=y_hat_absolut, y_val=y_test_absolut)

    def ate_calculator(self, y_hat, y_val):


        # Assuming y_hat and y_val are dictionaries containing 'x', 'y', 'z' components
        # of the estimated and ground truth trajectories respectively.

        # Calculate error for each component
        error_x = y_hat['x'] - y_val['x']
        error_y = y_hat['y'] - y_val['y']
        error_z = y_hat['z'] - y_val['z']

        # Square the errors
        error_x2 = np.power(error_x, 2)
        error_y2 = np.power(error_y, 2)
        error_z2 = np.power(error_z, 2)

        # Sum the squared errors for each position
        error_sum = error_x2 + error_y2 + error_z2

        # Calculate the mean of the squared errors
        mse = np.mean(error_sum)

        # Take the square root to get the RMSE (RMS ATE)
        rms_ate = np.sqrt(mse)

        print("RMS ATE: ", rms_ate)

        ATE = 0
        print('***********************************ATE**************************************')
        xyz = ['x', 'y', 'z']
        for dof in xyz:
            error = y_hat[dof] - y_val[dof]
            error_2 = np.power(error, 2)
            ate = np.sqrt(np.mean(error_2))
            print('for ate of' + dof + ':  '),
            print(ate)
            ATE += ate
        ATE = ATE / 3
        print(ATE)

        #
        # print(y_hat_df)
        # plt.plot(y_hat_df)
        # plt.plot()
        # plt.title('model accuracy')
        # plt.ylabel('mean absolute error')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()
        # plt.plot(hist.history['loss'])
        # plt.plot(hist.history['val_loss'])
        # plt.title('model accuracy')
        # plt.ylabel('mean absolute error')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()
        # return hist








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


