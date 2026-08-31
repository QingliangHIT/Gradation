from PyQt5.QtWidgets import *


class ParameterDialog(QDialog):


    def __init__(self):

        super().__init__()


        self.setWindowTitle(
            "算法参数设置"
        )


        layout=QFormLayout()



        self.pixel=QDoubleSpinBox()

        self.pixel.setValue(
            0.05
        )


        self.threshold=QSpinBox()

        self.threshold.setValue(
            120
        )



        layout.addRow(
            "像素比例(mm/pixel)",
            self.pixel
        )


        layout.addRow(
            "阈值",
            self.threshold
        )


        btn=QPushButton(
            "确定"
        )


        btn.clicked.connect(
            self.accept
        )


        layout.addWidget(
            btn
        )


        self.setLayout(
            layout
        )
