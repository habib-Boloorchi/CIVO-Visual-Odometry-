import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import pytransform3d.rotations as py3drot
class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]

class TraditionalVisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = np.eye(3)
        self.cur_t = [0., 0., 0.]
        self.absolute_cur_R = np.eye(3)
        self.absolute_cur_t = [0., 0., 0.]
        self.px_ref = None
        self.px_cur = np.zeros((100, 2))
        self.focal = cam.fx
        self.pp = (cam.cx, cam.cy)
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)
        self.lk_params = dict(winSize=(21, 21),
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
        self.trajectory = np.empty((3, 0))
        self.absolute_trajectory= np.empty((3, 0))
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
    def track_features(self, image_ref, image_cur, px_ref):
        kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **self.lk_params)
        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]
        return kp1, kp2

    def process_frame(self):
        if self.frame_stage == 0:
            self.px_ref = self.detector.detect(self.new_frame)
            self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
            self.frame_stage = 1
        else:
            self.px_ref, self.px_cur = self.track_features(self.last_frame, self.new_frame, self.px_ref)
            E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC,
                                           prob=0.999, threshold=1.0)
            _, self.cur_R, self.cur_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp=self.pp)

            if self.px_ref.shape[0] < 1500:
                self.px_cur = self.detector.detect(self.new_frame)
                self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)

            self.px_ref = self.px_cur

    def update(self, img):
        assert (img.ndim == 2 and img.shape[0] == self.cam.height and img.shape[1] == self.cam.width), \
            "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        self.process_frame()
        self.last_frame = self.new_frame

    def visualize_features(self, img):
        if self.px_cur is not None:
            for point in self.px_cur:
                cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)
        cv2.imshow("Image with features", img)
        cv2.waitKey(1)

    def visualize_trajectory(self):
        # self.t = self.t + self.R.dot(t)
        # self.R = R.dot(self.R)
        self.cur_t = np.reshape(self.cur_t, (3, 1))  # Use np.reshape if self.cur_t is a list
        self.absolute_cur_t = self.absolute_cur_t + self.absolute_cur_R.dot(self.cur_t)
        self.absolute_cur_R= self.cur_R.dot(self.absolute_cur_R)
        self.trajectory = np.hstack((self.trajectory, self.cur_t))
        self.absolute_trajectory = np.hstack((self.absolute_trajectory, self.absolute_cur_t))
        self.ax.clear()
        # self.ax.plot(self.trajectory[0],self.trajectory[1], self.trajectory[2], marker='o', markersize=3, linestyle='-')
        self.ax.plot(self.absolute_trajectory[0],self.absolute_trajectory[1], self.absolute_trajectory[2], marker='o', markersize=3, linestyle='-')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title("Trajectory")
        plt.pause(0.01)
class image_input:
    def __init__(self):
        pass


class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy,
                 k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]


class VisualOdometryRunner:
    def __init__(self):
        pass

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
        vo = TraditionalVisualOdometry(cam1)

        cam1_path = input_address + 'mav0\\cam1\\data\\'
        cam1_path_csv = input_address + 'mav0\\cam0\\data.csv'

        timestamps = pd.read_csv(cam1_path_csv)['#timestamp [ns]']

        for f in tqdm(timestamps):
            counter = counter + 1
            img = cv2.imread(cam1_path + str(f) + '.png', 0)

            vo.update(img)

            cur_t = vo.cur_t
            cur_r = vo.cur_R
            a =np.eye(3)
            # print(a)
            print (cur_r)
            if counter >= 2:
                x0, y0, z0 = cur_t[0][0], cur_t[1][0], cur_t[2][0]

                # cur_r0 = py3drot.matrix_to_quaternion(cur_r)
                [x0], [y0], [z0] = cur_t[0], cur_t[1], cur_t[2]
                cur_r0 = py3drot.quaternion_from_matrix(cur_r)
                # rw0, rx0, ry0, rz0 = cur_r0
                # else:
                #     cur_r0 = [0., 0., 0., 0.]

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

            # Visualize features
            vo.visualize_features(img)

            # Visualize trajectory
            vo.visualize_trajectory()

        dict = {'#timestamp [ns]': timestamps, 'predicted_x': xlist, 'predicted_y': ylist, 'predicted_z': zlist,
                'predicted_r_w': rwlist, 'predicted_r_x': rxlist, 'predicted_r_y': rylist, 'predicted_r_z': rzlist}
        output = pd.DataFrame(dict)

        if save_as_csv:
            output.to_csv(output_name, index=False)
            print('csv saved')
        else:
            print('Traditional_VisualO_dometry accomplished')

        return output


if __name__ == "__main__":
    runner = VisualOdometryRunner()
    runner.traditional_vo(input_address='D:\\V1_01_easy\\', save_as_csv=False)



