import numpy as np
import cv2

class TraditionalVisualOdometry:
    def __init__(self, cam, visualizer,visualize_flag=False):
        self.visualize_flag= visualize_flag
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
        self.detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
        # self.lk_params = dict(winSize=(21, 21),
        #                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.1))
        self.lk_params = dict(winSize=(31, 31),  # Increase the window size for a more stable result
                              maxLevel=4,  # Add this line to limit the number of pyramid levels
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30,
                                        0.01))  # Increase the iteration count and reduce the epsilon value
        self.visualizer = visualizer

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

        # Visualize features and trajectory
        if self.visualize_flag:
            self.visualizer.visualize_features(img, self.px_cur)
            self.visualizer.visualize_trajectory(self.cur_t, self.cur_R)
        # self.visualizer.visualize_Degree_of_freedom(self.cur_t,self.cur_R)
