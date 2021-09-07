from eiseg.widget.create import creat_dock, create_button, create_slider, create_text
import os.path as osp
from functools import partial

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from eiseg import pjpath, __APPNAME__
from models import ModelsNick
from util import MODELS
from widget import AnnotationScene, AnnotationView
from widget.create import *
from widget.table import TableWidget


class Ui_EISeg(object):
    def __init__(self):
        super(Ui_EISeg, self).__init__()
        self.tr = partial(QtCore.QCoreApplication.translate, "APP_EISeg")

    def setupUi(self, MainWindow):
        ## -- 主窗体设置 --
        MainWindow.setObjectName("MainWindow")
        MainWindow.setMinimumSize(QtCore.QSize(1366, 768))
        MainWindow.setWindowTitle(__APPNAME__)
        CentralWidget = QtWidgets.QWidget(MainWindow)
        CentralWidget.setObjectName("CentralWidget")
        MainWindow.setCentralWidget(CentralWidget)
        ## -----
        ## -- 工具栏 --
        toolBar = QtWidgets.QToolBar(self)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(toolBar.sizePolicy().hasHeightForWidth())
        toolBar.setSizePolicy(sizePolicy)
        toolBar.setMinimumSize(QtCore.QSize(0, 33))
        toolBar.setMovable(True)
        toolBar.setAllowedAreas(QtCore.Qt.BottomToolBarArea | QtCore.Qt.TopToolBarArea)
        toolBar.setObjectName("toolBar")
        self.toolBar = toolBar
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        ## -----
        ## -- 状态栏 --
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        self.statusbar.setStyleSheet("QStatusBar::item {border: none;}")
        MainWindow.setStatusBar(self.statusbar)
        self.statusbar.addPermanentWidget(
            self.show_logo(osp.join(pjpath, "resource/Paddle.png"))
        )
        ## -----
        ## -- 图形区域 --
        ImageRegion = QtWidgets.QHBoxLayout(CentralWidget)
        ImageRegion.setObjectName("ImageRegion")
        # 滑动区域
        self.scrollArea = QtWidgets.QScrollArea(CentralWidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        ImageRegion.addWidget(self.scrollArea)
        # 图形显示
        self.scene = AnnotationScene()
        self.scene.addPixmap(QtGui.QPixmap())
        self.canvas = AnnotationView(self.scene, self)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.canvas.setSizePolicy(sizePolicy)
        self.canvas.setAlignment(QtCore.Qt.AlignCenter)
        self.canvas.setAutoFillBackground(False)
        self.canvas.setStyleSheet("background-color: White")
        self.canvas.setObjectName("canvas")
        self.scrollArea.setWidget(self.canvas)
        ## -----
        ## -- 工作区 --
        p_create_dock = partial(self.creat_dock, MainWindow)
        p_create_button = partial(self.create_button, CentralWidget)
        # 模型加载
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        ModelRegion = QtWidgets.QVBoxLayout()
        ModelRegion.setObjectName("ModelRegion")
        # labShowSet = self.create_text(CentralWidget, "labShowSet", "模型选择")
        # ModelRegion.addWidget(labShowSet)
        combo = QtWidgets.QComboBox(self)
        combo.addItems([self.tr(ModelsNick[m.__name__][0]) for m in MODELS])
        self.comboModelSelect = combo
        ModelRegion.addWidget(self.comboModelSelect)
        # 网络参数
        self.btnParamsSelect = p_create_button(
            "btnParamsLoad",
            self.tr("加载网络参数"),
            osp.join(pjpath, "resource/Model.png"),
            "Ctrl+D",
        )
        ModelRegion.addWidget(self.btnParamsSelect)  # 模型选择
        horizontalLayout.addLayout(ModelRegion)
        self.ModelDock = p_create_dock("ModelDock", self.tr("模型选择"), widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.ModelDock)
        # 数据列表
        # TODO: 数据列表加一个搜索功能
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        ListRegion = QtWidgets.QVBoxLayout()
        ListRegion.setObjectName("ListRegion")
        # labFiles = self.create_text(CentralWidget, "labFiles", "数据列表")
        # ListRegion.addWidget(labFiles)
        self.listFiles = QtWidgets.QListWidget(CentralWidget)
        self.listFiles.setObjectName("ListFiles")
        ListRegion.addWidget(self.listFiles)
        # 保存
        self.btnSave = p_create_button(
            "btnSave",
            self.tr("保存"),
            osp.join(pjpath, "resource/Save.png"),
            "Ctrl+S",
        )
        ListRegion.addWidget(self.btnSave)
        horizontalLayout.addLayout(ListRegion)
        self.DataDock = p_create_dock("DataDock", self.tr("数据列表"), widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.DataDock)
        # 标签列表
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        LabelRegion = QtWidgets.QVBoxLayout()
        LabelRegion.setObjectName("LabelRegion")
        self.labelListTable = TableWidget(CentralWidget)  # QtWidgets.QTableWidget(CentralWidget)
        self.labelListTable.horizontalHeader().hide()
        # 铺满
        self.labelListTable.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.labelListTable.verticalHeader().hide()
        self.labelListTable.setColumnWidth(0, 10)
        # self.labelListTable.setMinimumWidth()
        self.labelListTable.setObjectName("labelListTable")
        self.labelListTable.clearContents()
        self.labelListTable.setRowCount(0)
        self.labelListTable.setColumnCount(4)

        LabelRegion.addWidget(self.labelListTable)
        self.btnAddClass = p_create_button(
            "btnAddClass", self.tr("添加标签"), osp.join(pjpath, "resource/Label.png")
        )
        LabelRegion.addWidget(self.btnAddClass)
        horizontalLayout.addLayout(LabelRegion)
        self.LabelDock = p_create_dock("LabelDock", self.tr("标签列表"), widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.LabelDock)
        ## 滑块设置
        # 分割阈值
        p_create_slider = partial(self.create_slider, CentralWidget)
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        ShowSetRegion = QtWidgets.QVBoxLayout()
        ShowSetRegion.setObjectName("ShowSetRegion")
        self.sldThresh, SegShowRegion = p_create_slider(
            "sldThresh", "labThresh", self.tr("分割阈值：")
        )
        ShowSetRegion.addLayout(SegShowRegion)
        ShowSetRegion.addWidget(self.sldThresh)
        # 透明度
        self.sldOpacity, MaskShowRegion = p_create_slider(
            "sldOpacity", "labOpacity", self.tr("标签透明度："), 75
        )
        ShowSetRegion.addLayout(MaskShowRegion)
        ShowSetRegion.addWidget(self.sldOpacity)
        # 点大小
        self.sldClickRadius, PointShowRegion = p_create_slider(
            "sldClickRadius", "labClickRadius", self.tr("点击可视化半径："), 3, 10, 1
        )
        ShowSetRegion.addLayout(PointShowRegion)
        ShowSetRegion.addWidget(self.sldClickRadius)
        horizontalLayout.addLayout(ShowSetRegion)
        self.ShowSetDock = p_create_dock("ShowSetDock", self.tr("分割设置"), widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.ShowSetDock)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    ## 创建文本
    def create_text(self, parent, text_name=None, text_text=None):
        return create_text(parent, text_name, text_text)

    ## 创建按钮
    def create_button(self, parent, btn_name, btn_text, ico_path=None, curt=None):
        return create_button(parent, btn_name, btn_text, ico_path, curt)

    ## 创建dock
    def creat_dock(self, parent, name, text, layout):
        return creat_dock(parent, name, text, layout)

    ## 显示Logo
    def show_logo(self, logo_path):
        labLogo = QtWidgets.QLabel()
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Maximum
        )
        labLogo.setSizePolicy(sizePolicy)
        labLogo.setMaximumSize(QtCore.QSize(100, 33))
        labLogo.setPixmap(QtGui.QPixmap(logo_path))
        labLogo.setScaledContents(True)
        labLogo.setObjectName("labLogo")
        return labLogo

    ## 创建滑块区域
    def create_slider(
        self,
        parent,
        sld_name,
        text_name,
        text,
        default_value=50,
        max_value=100,
        text_rate=0.01,
    ):
        return create_slider(
            parent,
            sld_name,
            text_name,
            text,
            default_value,
            max_value,
            text_rate,
        )
