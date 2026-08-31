class Calibration:


    def __init__(self):

        self.pixel_mm=0.05 # mm/pixel



    def set_scale(
        self,
        value
    ):

        self.pixel_mm=value



    def convert(
        self,
        pixel
    ):

        return pixel*self.pixel_mm
