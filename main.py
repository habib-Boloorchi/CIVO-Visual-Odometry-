from data_loader import DataLoader
import numpy as np
import pandas as pd
from Dof_trainer_modeified import Dof_trainer
from absolute_relative_pose_handler import abs_rel_handler
import datetime




def runner():
    data_exists= True
    abs_rel_handler_instance = abs_rel_handler()
    if data_exists == False:
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test = dataloader.runner_multiple_addresses()
    else:
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
    # Initialize hist_df_all and y_hat_rel_df_all
    hist_df_all = pd.DataFrame()
    y_hat_rel_df_all = pd.DataFrame()

    activation_functions = ['elu', 'leaky_relu', 'selu', 'tanh', 'relu', 'sigmoid']
    trainer = Dof_trainer()
    results = []
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for activation in activation_functions:
        yhat, ytest ,hist_df, y_hat_rel_df= trainer.runner(X_train, X_test, y_train, y_test, activation, plot= False)
        # rpe_translation, rpe_rotation, ate_translation = abs_rel_handler_instance.calculate_metrics(yhat, ytest)

        rpe_translation, rpe_rotation, ate_translation, rpe_translation_axes, ate_translation_axes, total_distance_y_test = abs_rel_handler_instance.calculate_metrics(
            yhat, ytest)
        # Concatenate results
        hist_df_all = pd.concat([hist_df_all, hist_df], ignore_index=True)
        y_hat_rel_df_all = pd.concat([y_hat_rel_df_all, y_hat_rel_df], axis=1)
        # y_hat_abs_df_all = pd.concat([y_hat_abs_df_all, y_hat_abs_df], axis=1)
        results.append({
            'activation': activation,
            'rpe_translation': rpe_translation,
            'rpe_translation_x': rpe_translation_axes[0],
            'rpe_translation_y': rpe_translation_axes[1],
            'rpe_translation_z': rpe_translation_axes[2],
            'rpe_rotation_rx': rpe_rotation[0],
            'rpe_rotation_ry': rpe_rotation[1],
            'rpe_rotation_rz': rpe_rotation[2],
            'ate_translation': ate_translation,
            'ate_translation_x': ate_translation_axes[0],
            'ate_translation_y': ate_translation_axes[1],
            'ate_translation_z': ate_translation_axes[2]
        })

    results_df = pd.DataFrame(results)
    # Save results to CSV files
    hist_df_all.to_csv(f"{timestamp}histories.csv", index=False)
    y_hat_rel_df_all.to_csv(f"{timestamp}predictions_relative.csv", index=False)
    # y_hat_abs_df_all.to_csv('predictions_absolute.csv', index=False)
    results_df.to_csv('results4.csv', index=False)
    print(results_df)


        # print(y)
def runner_cancatenated():
    data_exists = False
    abs_rel_handler_instance = abs_rel_handler()
    test_address = 'D:\\V1_02_medium\\',
    train_addresses = ['D:\\V1_01_easy\\', 'D:\\V2_01_easy\\', 'D:\\V2_02_medium\\']
    if data_exists == False:
        dataloader = DataLoader()
        X_train, y_train, X_test, y_test = dataloader.runner_multiple_addresses()
    else:
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
    trainer = Dof_trainer()
    trainer.train_concatenated(X_train, y_train, X_test, y_test)

    # print(hist)





if __name__ == '__main__':
    # runner()
    runner_cancatenated()