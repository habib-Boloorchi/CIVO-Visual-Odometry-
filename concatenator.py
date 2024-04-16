import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Activation, Dropout, Concatenate
from tensorflow.keras.models import Model, model_from_json


class CombinedModelTrainer:
    def __init__(self):
        self.activation = 'elu'
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=2e-4)
        self.loss_function = 'huber_loss'
        self.epochs = 2
        self.batch_size = 32
        self.model_folder = "models"



    def runner(self, X_train, X_val, y_train, y_val, activation='relu', plot=False):
        self.activation = activation
        dofs = ['x', 'y', 'z', 'rw', 'rx', 'ry', 'rz']
        loaded_models = []

        # Load and freeze pre-trained models
        for index, value in enumerate(dofs):
            loaded_model = self.load_single_model(self.activation, value, index)
            loaded_models.append(loaded_model)

        # Concatenate model outputs
        model_outputs = [model.output for model in loaded_models]
        concatenated_outputs = Concatenate()(model_outputs)

        # Add a new Dense layer with SELU activation
        new_output = Dense(1, activation='selu')(concatenated_outputs)
        combined_model = Model(inputs=[model.input for model in loaded_models], outputs=new_output)

        # Compile and train the model
        combined_model.compile(loss=self.loss_function, optimizer=self.optimizer,
                               metrics=[tf.keras.metrics.RootMeanSquaredError()])
        hist = combined_model.fit([X_train] * 7, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0,
                                  validation_data=([X_val] * 7, y_val))

        # ... rest of the code
