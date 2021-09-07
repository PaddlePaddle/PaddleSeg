from qtpy.QtWidgets import QDockWidget
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt


## 创建文本
def create_text(parent, text_name=None, text_text=None):
    text = QtWidgets.QLabel(parent)
    if text_name is not None:
        text.setObjectName(text_name)
    if text_text is not None:
        text.setText(text_text)
    return text


## 创建按钮
def create_button(parent, btn_name, btn_text, ico_path=None, curt=None):
    # 创建和设置按钮
    sizePolicy = QtWidgets.QSizePolicy(
        QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed
    )
    min_size = QtCore.QSize(0, 40)
    sizePolicy.setHorizontalStretch(0)
    sizePolicy.setVerticalStretch(0)
    btn = QtWidgets.QPushButton(parent)
    sizePolicy.setHeightForWidth(btn.sizePolicy().hasHeightForWidth())
    btn.setSizePolicy(sizePolicy)
    btn.setMinimumSize(min_size)
    btn.setObjectName(btn_name)
    if ico_path is not None:
        btn.setIcon(QtGui.QIcon(ico_path))
    btn.setText(btn_text)
    if curt is not None:
        btn.setShortcut(curt)
    return btn


## 创建滑块区域
def create_slider(
    parent,
    sld_name,
    text_name,
    text,
    default_value=50,
    max_value=100,
    text_rate=0.01,
):
    Region = QtWidgets.QHBoxLayout()
    lab = create_text(parent, None, text)
    Region.addWidget(lab)
    labShow = create_text(parent, text_name, str(default_value * text_rate))
    Region.addWidget(labShow)
    Region.setStretch(0, 1)
    Region.setStretch(1, 10)
    sld = QtWidgets.QSlider(parent)
    sld.setMaximum(max_value)  # 好像只能整数的，这里是扩大了10倍，1 . 10
    sld.setProperty("value", default_value)
    sld.setOrientation(QtCore.Qt.Horizontal)
    sld.setObjectName(sld_name)
    sld.setStyleSheet(
        """
        QSlider::sub-page:horizontal {
            background: #9999F1
        }
        QSlider::handle:horizontal
        {
            background: #3334E3;
            width: 12px;
            border-radius: 4px;
        }
        """
    )
    sld.textLab = labShow
    return sld, Region


class DockWidget(QDockWidget):
    def __init__(self, parent, name, text):
        super().__init__(parent=parent)
        self.setObjectName(name)
        self.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        # 感觉不给关闭好点。可以在显示里面取消显示。有关闭的话显示里面的enable还能判断修改，累了
        self.setFeatures(
            QDockWidget.DockWidgetMovable | 
            QDockWidget.DockWidgetFloatable
        )
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        self.setMinimumWidth(230)
        self.setWindowTitle(text)
        self.setStyleSheet("QDockWidget { background-color:rgb(204,204,248); }")
        self.topLevelChanged.connect(self.changeBackColor)

    def changeBackColor(self, isFloating):
        if isFloating:
            self.setStyleSheet("QDockWidget { background-color:rgb(255,255,255); }")
        else:
            self.setStyleSheet("QDockWidget { background-color:rgb(204,204,248); }")

## 创建dock
def creat_dock(parent, name, text, widget):
    dock = DockWidget(parent, name, text)
    dock.setWidget(widget)
    return dock