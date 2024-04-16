import cv2
import pandas as pd
from tqdm import tqdm
from pinhole_camera import PinholeCamera
from traditional_visual_odometry import TraditionalVisualOdometry
from visualizer import Visualizer

import pytransform3d.rotations as py3drot

class VisualOdometryRunner:
    def __init__(self):
        self.visualization = False


    def traditional_vo(self, input_address='D:\\V1_01_easy\\', output_name='default.csv', save_as_csv=False):
        counter = 0
        output_name = 'in_process_data\\' + output_name
        xlist = []
        ylist = []
        zlist = []
        rwlist = []
        rxlist = []
        rylist = []
        rzlist = []

        cam1 = PinholeCamera(752, 480, 457.587, 456.134, 379.999, 255.238)
        vis = Visualizer()
        vo = TraditionalVisualOdometry(cam1,vis)

        cam1_path = input_address + 'mav0\\cam1\\data\\'
        cam1_path_csv = input_address + 'mav0\\cam0\\data.csv'

        timestamps = pd.read_csv(cam1_path_csv)['#timestamp [ns]']

        for f in tqdm(timestamps):
            counter = counter + 1
            img = cv2.imread(cam1_path + str(f) + '.png', 0)

            vo.update(img)

            cur_t = vo.cur_t
            cur_r = vo.cur_R

            if counter >= 2:
                x0, y0, z0 = cur_t[0][0], cur_t[1][0], cur_t[2][0]

                cur_r0 = py3drot.quaternion_from_matrix(cur_r)

                rw0, rx0, ry0, rz0 = cur_r0
            else:
                x0, y0, z0 = 0., 0., 0.
                rw0, rx0, ry0, rz0 = 0., 0., 0., 0.
            xlist.append(x0)
            ylist.append(y0)
            zlist.append(z0)
            rwlist.append(rw0)
            rxlist.append(rx0)
            rylist.append(ry0)
            rzlist.append(rz0)


        dict = {'#timestamp [ns]': timestamps, 'predicted_x': xlist, 'predicted_y': ylist, 'predicted_z': zlist,
                'predicted_r_w': rwlist, 'predicted_r_x': rxlist, 'predicted_r_y': rylist, 'predicted_r_z': rzlist}
        output = pd.DataFrame(dict)

        if save_as_csv:
            output.to_csv(output_name, index=False)
            print('csv saved')
        else:
            print('Traditional_VisualO_dometry accomplished')

        return output

#
# if __name__ == "__main__":
#     runner = VisualOdometryRunner()
#     runner.traditional_vo(input_address='D:\\V1_01_easy\\', save_as_csv=False)
