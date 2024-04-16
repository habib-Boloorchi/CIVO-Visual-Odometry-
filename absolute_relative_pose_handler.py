from pytransform3d.rotations import matrix_from_quaternion, intrinsic_euler_xyz_from_active_matrix,\
    active_matrix_from_extrinsic_euler_xyz
import numpy as np
import pandas as pd
from pytransform3d import transformations as pt
from pytransform3d.transform_manager import TransformManager
class abs_rel_handler:
    def __init__(self):
        pass
    # def quaternion_2_euler(self,q_Poses):
    #     poses = q_Poses
    #     euler_poses = []
    #     for pose in poses:
    #         # Extract the position and quaternion elements from the row
    #         x, y, z, w, rx, ry, rz = pose
    #         rot_matrix = matrix_from_quaternion([w, rx, ry, rz])
    #         rx, ry, rz = intrinsic_euler_xyz_from_active_matrix(rot_matrix)
    #         e_pose = np.array([x, y, z, rx, ry, rz])
    #         euler_poses.append(e_pose)
    #     euler_poses_array = np.array(euler_poses)
    #     # euler_poses_array
    #     return euler_poses_array
    #
    # def euler_to_quaternion(self,q_Poses):
    #     poses = q_Poses
    #     euler_poses = []
    #     for pose in poses:
    #         # Extract the position and quaternion elements from the row
    #         x, y, z, w, rx, ry, rz = pose
    #         rot_matrix = matrix_from_quaternion([w, rx, ry, rz])
    #         rx, ry, rz = intrinsic_euler_xyz_from_active_matrix(rot_matrix)
    #         e_pose = np.array([x, y, z, rx, ry, rz])
    #         euler_poses.append(e_pose)
    #     euler_poses_array = np.array(euler_poses)
    #     # euler_poses_array
    #     return euler_poses_array

    def absolute_to_relative_transform(self, file_input, plot=False):
        columns = ["#timestamp [ns]", "x", "y", "z", "rw", "rx", "ry", "rz"]
        # body_from_origin = np.array(vicon_body_origin_to_the_body_frame)
        df_output = pd.DataFrame(columns=columns)
        tm = TransformManager()
        # tm.add_transform("o","body",body_from_origin)
        first_data = True

        for counter in range(len(file_input)):
            if first_data:
                first_data = False
                a = file_input.iloc[counter].to_numpy()[1:]
                b = file_input.iloc[counter].to_numpy()[1:]
            else:
                a = file_input.iloc[counter - 1].to_numpy()[1:]
                b = file_input.iloc[counter].to_numpy()[1:]

            oa = pt.transform_from_pq(a)
            ob = pt.transform_from_pq(b)

            tm = TransformManager()

            tm.add_transform("o", "a", oa)
            tm.add_transform("o", "b", ob)
            # tm.add_transform("object", "camera", object2cam)

            ab = tm.get_transform("a", "b")
            position_and_quaternion = pt.pq_from_transform(ab).tolist()
            time_stamp = file_input.iloc[counter][0]
            pqx, pqy, pqz, pqrw, pqrx, pqry, pqrz = position_and_quaternion
            row = [time_stamp, pqx, pqy, pqz, pqrw, pqrx, pqry, pqrz]
            df_output.loc[counter] = row
        return df_output

    def relative_to_absolute_transform(self, file_input, plot=False, input_address='D:\\V1_01_easy\\'):
        path_to_vicon = input_address + 'mav0\\state_groundtruth_estimate0\\data.csv'
        vicon = pd.read_csv(path_to_vicon)
        columns = ["#timestamp [ns]", "x", "y", "z", "rw", "rx", "ry", "rz"]
        df_output = pd.DataFrame(columns=columns)
        tm = TransformManager()
        first_data = True
        position_and_quaternion = []

        for counter in range(len(file_input)):
            if first_data:
                first_data = False
                a = vicon.iloc[0][1:8].to_numpy()
                b = file_input.iloc[counter].to_numpy()
            else:
                a = np.array(position_and_quaternion)
                b = file_input.iloc[counter].to_numpy()

            oa = pt.transform_from_pq(a)
            ab = pt.transform_from_pq(b)
            tm = TransformManager()
            tm.add_transform("o", "a", oa)
            tm.add_transform("a", "b", ab)

            ob = tm.get_transform("o", "b")
            position_and_quaternion = pt.pq_from_transform(ob).tolist()
            time_stamp = vicon.iloc[counter][0]
            pqx, pqy, pqz, pqrw, pqrx, pqry, pqrz = position_and_quaternion
            row = [time_stamp, pqx, pqy, pqz, pqrw, pqrx, pqry, pqrz]
            df_output.loc[counter] = row

        return df_output

    # def calculate_metrics(self,yhat, y_test, input_address='D:\\V2_02_medium\\'):
    #     yhat_df = pd.DataFrame(yhat, columns=["x", "y", "z", "rw","rx", "ry", "rz"])
    #     print(y_test)
    #     y_test_df = pd.DataFrame(y_test, columns=["x", "y", "z", "rw","rx", "ry", "rz"])
    #
    #     yhat_abs = self.relative_to_absolute_transform(yhat_df, input_address=input_address)
    #     y_test_abs = self.relative_to_absolute_transform(y_test, input_address=input_address)
    #
    #     rpe_translation, rpe_rotation = self.compute_rpe(yhat_abs, y_test_abs)
    #     ate_translation = self.compute_ate(yhat_abs, y_test_abs)
    #
    #     return rpe_translation, rpe_rotation, ate_translation
    def calculate_metrics(self, yhat, y_test, input_address='D:\\V1_01_easy\\'):
        yhat_df = pd.DataFrame(yhat, columns=["x", "y", "z", "rw", "rx", "ry", "rz"])
        y_test_df = pd.DataFrame(y_test, columns=["x", "y", "z", "rw", "rx", "ry", "rz"])

        yhat_abs = self.relative_to_absolute_transform(yhat_df, input_address=input_address)
        y_test_abs = self.relative_to_absolute_transform(y_test_df, input_address=input_address)

        rpe_translation, rpe_rotation, rpe_translation_axes = self.compute_rpe(yhat_abs, y_test_abs)
        ate_translation, ate_translation_axes = self.compute_ate(yhat_abs, y_test_abs)
        total_distance_y_test = np.sum(np.abs(np.diff(y_test_abs.iloc[:, 1:4].values, axis=0)), axis=0)
        print ('total distance :'),
        print(total_distance_y_test)
        print()

        return rpe_translation, rpe_rotation, ate_translation, rpe_translation_axes, ate_translation_axes, total_distance_y_test

    def compute_rpe(self, yhat_abs, y_test_abs):
        delta_translations = y_test_abs.iloc[:, 1:4].values - yhat_abs.iloc[:, 1:4].values
        delta_rotations = np.abs(y_test_abs.iloc[:, 4:].values - yhat_abs.iloc[:, 4:].values)

        rpe_translation_axes = np.mean(np.abs(delta_translations), axis=0)
        rpe_translation = np.sqrt(np.sum(rpe_translation_axes ** 2) / 3)

        rpe_rotation = np.mean(delta_rotations, axis=0)  # Calculate mean rotation error for each axis

        return rpe_translation, rpe_rotation, rpe_translation_axes
    def compute_ate(self,yhat_abs, y_test_abs):
        delta_translations = np.linalg.norm(y_test_abs.iloc[:, 1:4].values - yhat_abs.iloc[:, 1:4].values, axis=1)
        ate_translation = np.mean(delta_translations)
        ate_translation_axes = np.mean(np.abs(y_test_abs.iloc[:, 1:4].values - yhat_abs.iloc[:, 1:4].values), axis=0)

        return ate_translation, ate_translation_axes
    # def compute_rpe(self,yhat_abs, y_test_abs):
    #     delta_translations = np.linalg.norm(y_test_abs.iloc[:, 1:4].values - yhat_abs.iloc[:, 1:4].values, axis=1)
    #     delta_rotations = np.abs(np.arccos(np.clip(2 * (y_test_abs.iloc[:, 4] * yhat_abs.iloc[:, 4] +
    #                                                     y_test_abs.iloc[:, 5] * yhat_abs.iloc[:, 5] +
    #                                                     y_test_abs.iloc[:, 6] * yhat_abs.iloc[:, 6] +
    #                                                     y_test_abs.iloc[:, 7] * yhat_abs.iloc[:, 7]) ** 2 - 1, -1, 1)))
    #     rpe_translation = np.mean(delta_translations)
    #     rpe_rotation = np.mean(delta_rotations)
    #     return rpe_translation, rpe_rotation
    # def compute_ate(self,yhat_abs, y_test_abs):
    #     ate_translation = np.mean(np.linalg.norm(y_test_abs.iloc[:, 1:4].values - yhat_abs.iloc[:, 1:4].values, axis=1))
    #     return ate_translation

    # def relative_to_absolute_transform(self,file_input, input_address='D:\\V2_02_medium\\'):
    #     path_to_vicon = input_address + 'mav0\\state_groundtruth_estimate0\\data.csv'
    #     vicon = pd.read_csv(path_to_vicon)
    #     columns = ["#timestamp [ns]", "x", "y", "z", 'rw', "rx", "ry", "rz"]
    #     df_output = pd.DataFrame(columns=columns)
    #     tm = TransformManager()
    #     first_data = True
    #     position_and_quaternion = []
    #
    #     for counter in range(len(file_input)):
    #         if first_data:
    #             first_data = False
    #             a = vicon.iloc[0][1:].to_numpy()
    #             x, y, z, rx, ry, rz = file_input.iloc[counter]
    #             b = pt.pq_from_transform(
    #                 pt.transform_from(active_matrix_from_extrinsic_euler_xyz(np.array([rx, ry, rz])),
    #                                   np.array([x, y, z])))
    #         else:
    #             a = np.array(position_and_quaternion)
    #             x, y, z, rx, ry, rz = file_input.iloc[counter]
    #             b = pt.pq_from_transform(pt.transform_from(active_matrix_from_extrinsic_euler_xyz(np.array([rx,
