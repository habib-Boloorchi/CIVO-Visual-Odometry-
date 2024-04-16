import pandas as pd

class OutputHandler:
    def __init__(self, output_name='default.csv'):
        self.output_name = 'in_process_data\\' + output_name

    def save_as_csv(self, data):
        data.to_csv(self.output_name, index=False)
        print('CSV saved')