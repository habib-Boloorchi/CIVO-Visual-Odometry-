# squeezed_gt = self.groundtruth_squeezer(input_address=input_address,
#                                            output_name='gt11.csv', save_as_csv=False)
#         gt_relative = self.absolute_to_relative_transform(file_input=squeezed_gt)
#         X_train, X_val, y_train, y_val = self.scale_and_convert_data(
#             X=raw_vo, y=gt_relative)
import numpy as np

from absolute_relative_pose_handler import abs_rel_handler
import pandas as pd
class Ground_truth_handler:
    def __init__(self):
        self.np_output = np.ndarray
    def groundtruth_squeezer(self,input_address='D:\\V1_01_easy\\'):
        # dir_path = 'in_process_data\\'
        gt_path = input_address + 'mav0\\state_groundtruth_estimate0\\data.csv'
        cam0_path_csv = input_address + 'mav0\\cam0\\data.csv'
        cam_timestamps = pd.read_csv(cam0_path_csv)
        cam_timestamps.rename(columns={'#timestamp [ns]': '#timestamp'}, inplace=True)
        gt_data = pd.read_csv(gt_path)

        output = pd.merge_asof(cam_timestamps, gt_data, on="#timestamp",
                               direction='nearest').drop('filename', 1).iloc[:, :8]

        return output


    def get_numpy_output_without_timestamps(self):
        return self.np_output
    def gt_manipulator_runner(self,input_address='D:\\V1_01_easy\\'):
        gt = self.groundtruth_squeezer(input_address=input_address)

        # Create an instance of abs_rel_handler class
        abs_rel_handler_instance = abs_rel_handler()

        # Call the method using the instance
        output = abs_rel_handler_instance.absolute_to_relative_transform(file_input=gt)
        self.np_output = output.iloc[:,1:].values
        # print(self.np_output)
        return output

    # def convert_numpy_gt(self):






