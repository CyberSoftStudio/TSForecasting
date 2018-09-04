import scripts.prediction_lib as plib

import numpy

class Predictor:
    def __init__(self, prev_data, config):
        self.data = prev_data
        self.config = config
        self.mult_const = 1
        self.window_size = 256 * self.mult_const
        self.shift = 30 * self.mult_const
        self.scale = 50 * self.mult_const
        self.wdname = 'db6'
        self.wcname = 'gaus8'
        self.extract_alpha = 0.5

    def change_config(self, new_config):
        self.config = new_config

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

        return []


