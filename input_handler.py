import pandas as pd

class ImageInput:
    def __init__(self, input_address):
        self.input_address = input_address

    def get_timestamps(self):
        cam1_path_csv = self.input_address + 'mav0\\cam0\\data.csv'
        timestamps = pd.read_csv(cam1_path_csv)['#timestamp [ns]']
        return timestamps

    def get_image_path(self, timestamp):
        cam1_path = self.input_address + 'mav0\\cam1\\data\\'
        return cam1_path + str(timestamp) + '.png'
class GroundtruthInput:
    def __init__(self):
        pass
    def Euroc_handler(self):
        pass
    def Kitti_handler(self):
        pass
    def TUM_handler(self):
        pass
