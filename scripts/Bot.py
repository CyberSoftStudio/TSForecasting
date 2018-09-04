import scripts.prediction_lib as plib

import numpy as np
import json



class Bot:
    def __init__(self, data):
        self.data = np.array(data)
        self.config = {
            'mult_const': 1,
            'window_size': 256,
            'assurance':0.6,
            'shift': 30,
            'scale': 50,
            'wdname': 'db6',
            'wcname': 'gaus8',
            'extract_alpha': 0.5,
            'model_key': 1,
            'model_json_path': '../models/model_cwt_10k.json',  # model_nclasses_46_1
            'model_h5_path': '../models/model_cwt_10k.h5'
        }

        plib.load_cnn(self.config['model_json_path'], self.config['model_h5_path'])

    def set_data(self, data):
        self.data = data.copy()

    def request_data(self):
        pass

    @staticmethod
    def _merge_config(config, new_config):
        result_config = {key: value for key, value in config.items()}

        for key in new_config.keys():
            if key in config:
                result_config[key] = new_config[key]

        return result_config

    def _set_init_config(self):
        self.config = {
            'mult_const': 1,
            'window_size': 256,
            'assurance':0.6,
            'shift': 30,
            'scale': 50,
            'wdname': 'db6',
            'wcname': 'gaus8',
            'extract_alpha': 0.5,
            'model_key': 1,
            'model_json_path': '../models/model_cwt_10k.json',  # model_nclasses_46_1
            'model_h5_path': '../models/model_cwt_10k.h5'
        }

    def change_config(self, new_config):
        self.config = self._merge_config(self.config, new_config)

    def load_config(self, config_json_file):
        new_config = json.load(open(config_json_file))
        self.config = self._merge_config(self.config, new_config)

    def predict(self):
        mult_const      = self.config['mult_const']
        window_size     = self.config['window_size']
        shift           = self.config['shift']
        scale           = self.config['scale']
        wdname          = self.config['wdname']
        wcname          = self.config['wcname']
        extract_alpha   = self.config['extract_alpha']
        assurance       = self.config['assurance']
        key             = self.config['model_key']

        window = self.data[-window_size:]
        correct_rects, segmentations, _ = plib.predict(
            window,
            scale=scale,
            assurance=assurance,
            wdname=wdname,
            wcname=wcname,
            shift=shift,
            extract_alpha=extract_alpha,
            key=key,
            mult_const=mult_const
        )

        if len(segmentations):
            try:
                interval = plib.predict_interval(segmentations[0].segmentation)
                x, y = correct_rects[0].x * mult_const

                interval['maxy'] *= mult_const
                interval['miny'] *= mult_const
                interval['center'] *= mult_const

                interval['maxy'] += y
                interval['miny'] += y
                interval['center'] += y

                return interval
            except:
                print("Can't predict interval")
                return {}

        return {}

    # def process(self):




