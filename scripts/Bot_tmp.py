import scripts.prediction_lib as plib

import numpy as np
import json



class Bot:
    def __init__(self, data, config):
        self.data = np.array(data)
        self.config = config
        self.mult_const = 1
        self.window_size = 256 * self.mult_const
        self.shift = 30 * self.mult_const
        self.scale = 50 * self.mult_const
        self.wdname = 'db6'
        self.wcname = 'gaus8'
        self.extract_alpha = 0.5
        model_json_path = '../models/model_cwt_10k.json'  # model_nclasses_46_1
        model_h5_path = '../models/model_cwt_10k.h5'
        plib.load_cnn(model_json_path, model_h5_path)

    def set_data(self, data):
        self.data = data.copy()

    def request_data(self):
        pass

    @staticmethod
    def _merge_config(self, config, new_config):
        result_config = {key: value for key, value in config.items()}

        for key in new_config.keys():
            if key in config:
                result_config[key] = new_config[key]

        return result_config

    def change_config(self, new_config):
        self.config = new_config

    def load_config(self, config_json):
        new_config = json.load(open(config_json))
        self.config = self._merge_config(self.config, new_config)

    def predict(self):
        mult_const = self.mult_const
        window_size = self.window_size
        shift = self.shift
        scale = self.scale
        wdname = self.wdname
        wcname = self.wcname
        extract_alpha = self.extract_alpha

        window = self.data[-self.window_size:]
        correct_rects, segmentations, _ = plib.predict(
            window,
            scale=scale,
            assurance=0.5,
            wdname=wdname,
            wcname=wcname,
            shift=shift,
            extract_alpha=extract_alpha,
            key=1,
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




