import tensorflow as tf
from keras.layers import Dense, Input,Activation,Dropout
from keras.models import Model
import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import pytransform3d.transformations as pt
from pytransform3d.transform_manager import TransformManager
from pytransform3d.rotations import matrix_from_quaternion, intrinsic_euler_xyz_from_active_matrix
from absolute_relative_pose_handler import abs_rel_handler
class DOFTrainer:
    def __init__(self, epochs=200, batch_size=64, loss_function='huber_loss', optimizer=tf.keras.optimizers.RMSprop(learning_rate=2e-4)):
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.abs_rel_handler= abs_rel_handler()
        print('we are in DOF Trainer')

    def vo_model(self, activation='relu'):
        model = models.Sequential()
        model.add(layers.Dense(64, activation=activation, input_shape=(7,)))
        model.add(layers.Dense(32, activation=activation))
        model.add(layers.Dense(1, activation='linear'))
        return model

    def dof_trainer(self, X_train, X_val, y_train, y_val, activation_functions):
        dofs = ['x', 'y', 'z', 'rw', 'rx', 'ry', 'rz']

        for index, value in enumerate(dofs):
            for activation in activation_functions:
                model = self.vo_model(activation)
                model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['mae'])

                hist = model.fit(X_train, y_train[:, index], epochs=self.epochs, batch_size=self.batch_size, verbose=1, validation_data=(X_val, y_val[:, index]))

                # Save the trained model
                model.save(f"model_dof{index+1}_activation_{activation}.h5")

                # Save the history
                hist_df = pd.DataFrame(hist.history)
                hist_df.to_csv(f"history_dof{index+1}_activation_{activation}.csv", index=False)

                y_hat = model.predict(X_val)
                y_hat_df = pd.DataFrame(y_hat)
                y_hat_df.to_csv(f"y_hat_dof{index+1}_activation_{activation}.csv", index=False)

                y_val_absolut = abs_rel_handler.relative_to_absolute_transform(y_val)
                y_hat_absolut = abs_rel_handler.relative_to_absolute_transform(y_hat)

                # Calculate ATE
                self.ate_calculator(y_hat=y_hat_absolut, y_val=y_val_absolut)

                # # Calculate ARPE
                # y_val_euler = self.quaternion_2_euler(y_val_absolut)
                # y_hat_euler = self.quaternion_2_euler(y_hat_absolut)
                # self.ate_calculator(y_hat=y_hat_euler, y_val=y_val_euler)

    def ate_calculator(self, y_hat, y_val):
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
    # Implement the remaining methods here


# Example usage:

