#todo
# trad vo:
# gt hndler
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from gt_handler import Ground_truth_handler
from trad_vo_runner import VisualOdometryRunner

class DataLoader:
    def __init__(self,save_data_exists=False):
        self.dataset_name = 'euroc'
        self.input_address = 'D:\\V1_01_easy\\'
        self.visualization = False
        self.save_data_exists=save_data_exists



    def _normalize_quaternions(self, df):
        quaternions = df.iloc[:, :].to_numpy()
        magnitudes = np.linalg.norm(quaternions, axis=1, keepdims=True)
        epsilon = 1e-8  # Add a small constant to avoid division by zero
        normalized_quaternions = quaternions / (magnitudes + epsilon)
        df_normalized = df.copy()
        df_normalized.iloc[:, :] = normalized_quaternions
        return df_normalized

    def preprocess_data(self, X, y):
        # Remove the first column by index
        X = X.drop(X.columns[0], axis=1)
        y = y.drop(y.columns[0], axis=1).to_numpy()

        # # Replace these with the actual column names
        # translation_columns = ['predicted_x', 'predicted_y', 'predicted_z']
        # quaternion_columns = ['predicted_rw', 'predicted_rx', 'predicted_ry', 'predicted_rz']

        # Load the traditional_vo output and ground truth (gt) as DataFrames
        trad_vo_df = X


        # Separate the translation and quaternion columns
        trad_vo_translation = trad_vo_df.iloc[:, :3]
        trad_vo_quaternion = trad_vo_df.iloc[:, 3:]

        # Normalize the translation columns using MinMaxScaler
        min_max_scaler = MinMaxScaler()
        trad_vo_translation_normalized = min_max_scaler.fit_transform(trad_vo_translation)

        # Normalize the quaternion columns

        trad_vo_quaternion_normalized = self._normalize_quaternions(trad_vo_quaternion)

        # Combine the normalized translation and quaternion columns
        X =  np.hstack([    trad_vo_translation_normalized,
    trad_vo_quaternion_normalized.to_numpy()])
        # print('trad_vo_normalized:')
        # print(trad_vo_normalized)

        # print(y)
        # Save the preprocessed data as NumPy arrays
        np.save("trad_vo_normalized.npy", X)
        np.save("gt_normalized.npy", y)

        return X, y

    def get_input_address(self, input_address='D:\\V1_01_easy\\', dataset_name='euroc'):
        self.input_address = input_address
        self.dataset_name = dataset_name

    def trad_vo_loader(self, visulize=False):
        trad_vo_runner = VisualOdometryRunner()
        if self.dataset_name == 'euroc':
            return trad_vo_runner.traditional_vo(input_address=self.input_address, save_as_csv=False)
        else:
            return 0

    def groundtruth_handler(self):
        if self.dataset_name == 'euroc':
            gt_handler = Ground_truth_handler()
            return gt_handler.gt_manipulator_runner(input_address=self.input_address)

    def runner_multiple_addresses(self):
        if self.save_data_exists:
            X_train = np.load('X_train.npy')
            X_test = np.load('X_test.npy')
            y_train = np.load('y_train.npy')
            y_test = np.load('y_test.npy')

        train_addresses = ['D:\\V1_02_medium\\', 'D:\\V2_01_easy\\', 'D:\\V2_02_medium\\']
        test_address = 'D:\\V1_01_easy\\'

        X_train_combined = None
        y_train_combined = None

        for train_address in train_addresses:
            self.get_input_address(input_address=train_address)
            y_df = self.groundtruth_handler()
            X_df = self.trad_vo_loader()
            X_train, y_train = self.preprocess_data(X_df, y_df)

            if X_train_combined is None:
                X_train_combined = X_train
                y_train_combined = y_train
            else:
                X_train_combined = np.vstack((X_train_combined, X_train))
                y_train_combined = np.vstack((y_train_combined, y_train))

        # Preprocess the test data
        self.get_input_address(input_address=test_address)
        y_test_df = self.groundtruth_handler()
        X_test_df = self.trad_vo_loader()
        X_test, y_test = self.preprocess_data(X_test_df, y_test_df)

        # Save combined train data and test data
        np.save("X_train.npy", X_train_combined)
        np.save("y_train.npy", y_train_combined)
        np.save("X_test.npy", X_test)
        np.save("y_test.npy", y_test)

        return X_train_combined, y_train_combined, X_test, y_test


