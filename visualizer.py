import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import pytransform3d.rotations as py3drot
class Visualizer:
    def __init__(self):

        self.visualize_trad_vo_video =  True
        self.visualize_trad_vo_trajectory = True
        self.visualize_gt = True
        self.visualize_trad_vo_degs= True
        self.trajectory = np.empty((3, 0))
        self.absolute_trajectory = np.empty((3, 0))
        self.fig0 = plt.figure()
        self.ax = self.fig0.add_subplot(111, projection='3d')
        self.fig1, self.axes = plt.subplots(3, 1, sharex=True)
        self.absolute_cur_R = np.eye(3)
        self.absolute_cur_t = [0., 0., 0.]

    def visualize_features(self, img, px_cur):
        if self.visualize_trad_vo_video:
            if px_cur is not None:
                for point in px_cur:
                    cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
            cv2.imshow("Image with features", img)
            cv2.waitKey(1)

    def visualize_trajectory(self, cur_t, cur_R):
        cur_t = np.reshape(cur_t, (3, 1))  # Use np.reshape if cur_t is a list
        if self.visualize_trad_vo_trajectory:
            self.absolute_cur_t = self.absolute_cur_t + self.absolute_cur_R.dot(cur_t)
            self.absolute_cur_R = cur_R.dot(self.absolute_cur_R)
            self.trajectory = np.hstack((self.trajectory, cur_t))
            self.absolute_trajectory = np.hstack((self.absolute_trajectory, self.absolute_cur_t))

            self.ax.clear()
            self.ax.plot(self.absolute_trajectory[0], self.absolute_trajectory[1], self.absolute_trajectory[2],
                           linestyle='-')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_title("Trajectory")
            # plt.show()
            self.fig0.canvas.draw_idle()
            plt.pause(0.001)

        if self.visualize_trad_vo_degs:
            for ax in self.axes:
                ax.clear()

            self.axes[0].plot(self.trajectory[0], markersize=3, linestyle='-')
            self.axes[0].set_title('X Translation')
            self.axes[0].set_ylabel('X')

            self.axes[1].plot(self.trajectory[1], markersize=3, linestyle='-')
            self.axes[1].set_title('Y Translation')
            self.axes[1].set_ylabel('Y')

            self.axes[2].plot(self.trajectory[2], markersize=3, linestyle='-')
            self.axes[2].set_title('Z Translation')
            self.axes[2].set_ylabel('Z')
            self.axes[2].set_xlabel('Frame')
            self.fig1.canvas.draw_idle()
            plt.tight_layout()
            plt.pause(0.001)
    def visualize_ground_truth(self,gt):
        gt_trajectory = self.gt_df[['x', 'y', 'z']].values.T
        self.ax_gt.plot(gt_trajectory[0], gt_trajectory[1], gt_trajectory[2], linestyle='-')
        self.ax_gt.set_xlabel('X')
        self.ax_gt.set_ylabel('Y')
        self.ax_gt.set_zlabel('Z')
        self.ax_gt.set_title("Ground Truth Trajectory")

        for i, dof in enumerate(['x', 'y', 'z', 'rx', 'ry', 'rz']):
            ax = self.axes[i % 3, i // 3]
            ax.plot(self.gt_df[dof], linestyle='-')
            ax.set_title(f'{dof.upper()} {"Translation" if i < 3 else "Rotation"}')
            if i % 3 == 0:
                ax.set_ylabel(dof.upper())
            if i // 3 == 1:
                ax.set_xlabel('Frame')

        plt.tight_layout()

    def plot_epochs(self,hist_list, value_list):
        # Create a figure with 3x2 subplots
        print(hist_list)
        fig, axs = plt.subplots(4, 2, figsize=(10, 10))
        fig.tight_layout(pad=5.0)
        # Loop over the lists of histories and values
        for hist, value, ax in zip(hist_list, value_list, axs.flat):
            # Plot the training and validation root mean squared error for each history
            ax.plot(hist.history['root_mean_squared_error'])
            ax.plot(hist.history['val_root_mean_squared_error'])
            ax.set_title('model accuracy of ' + value)
            ax.set_ylabel('rmse (Realtive Pose Error)')

            # ax.set_xlabel('Frame Number')
            ax.legend(['train', 'validation'], loc='upper right')
        # Show the plot
        plt.show()
        plt.clf()

    def plot_RPEs(self,RPE_train_list, RPE_val_list, value_list):
        print('******************************************RPE of DOFs***********************************')
        # Create a figure with 3x2 subplots
        fig, axs = plt.subplots(6, 1, figsize=(10, 10))
        fig.tight_layout(pad=5.0)
        # Loop over the lists of histories and values
        for RPE_train, RPE_val, value, ax in zip(RPE_train_list, RPE_val_list, value_list, axs.flat):
            # Plot the training and validation root mean squared error for each history
            ax.plot(RPE_train)
            ax.plot(RPE_val)
            # ax.set_title('rmese of ' + value)

            ax.set_xlabel('Frame number')
            # ax.legend(['train', 'validation'], loc='upper right')
        # Show the plot
        # fig.suptitle('Relative Pose of x,y,z ,rx,ry,rz in euler')
        ax.set_ylabel('Relative')
        fig.legend(['yhat', 'ytest'], loc='upper right')

        plt.show()
        plt.clf()

