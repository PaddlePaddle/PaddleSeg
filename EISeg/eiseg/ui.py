# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os.path as osp
from functools import partial

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtGui import QIcon
from qtpy.QtCore import Qt

from eiseg import pjpath, __APPNAME__, __VERSION__, logger
from eiseg.widget.create import creat_dock, create_button, create_slider, create_text
from widget import AnnotationScene, AnnotationView
from widget.create import *
from widget.table import TableWidget

# log = logging.getLogger(__name__ + ".ui")


class Ui_EISeg(object):
    def __init__(self):
        super(Ui_EISeg, self).__init__()
        self.tr = partial(QtCore.QCoreApplication.translate, "APP_EISeg")

    def setupUi(self, MainWindow):
        ## -- 主窗体设置 --
        MainWindow.setObjectName("MainWindow")
        MainWindow.setMinimumSize(QtCore.QSize(1366, 768))
        MainWindow.setWindowTitle(__APPNAME__ + " " + __VERSION__)
        MainWindow.setWindowIcon(QIcon())  # TODO: 默认图标需要换一个吗，貌似不能不显示图标
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
        # combo = QtWidgets.QComboBox(self)
        # # combo.addItems([self.tr(ModelsNick[m.__name__][0]) for m in MODELS])
        # combo.addItems([self.tr(ModelsNick[m][0]) for m in ModelsNick.keys()])
        # self.comboModelSelect = combo
        # ModelRegion.addWidget(self.comboModelSelect)
        # 网络参数
        self.btnParamsSelect = p_create_button(
            "btnParamsLoad",
            self.tr("加载网络参数"),
            osp.join(pjpath, "resource/Model.png"),
            "Ctrl+D",
        )
        ModelRegion.addWidget(self.btnParamsSelect)  # 模型选择
        self.cheWithMask = QtWidgets.QCheckBox(self)
        self.cheWithMask.setText(self.tr("使用掩膜"))
        self.cheWithMask.setChecked(True)
        ModelRegion.addWidget(self.cheWithMask)  # with_mask
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

        # ListRegion.addWidget(self.btnSave)
        horizontalLayout.addLayout(ListRegion)
        self.DataDock = p_create_dock("DataDock", self.tr("数据列表"), widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.DataDock)
        # 标签列表
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        LabelRegion = QtWidgets.QVBoxLayout()
        LabelRegion.setObjectName("LabelRegion")
        self.labelListTable = TableWidget(
            CentralWidget
        )  # QtWidgets.QTableWidget(CentralWidget)
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
        # 保存
        self.btnSave = p_create_button(
            "btnSave",
            self.tr("保存"),
            osp.join(pjpath, "resource/Save.png"),
            "Ctrl+S",
        )
        ShowSetRegion.addWidget(self.btnSave)
        horizontalLayout.addLayout(ShowSetRegion)
        self.SegSettingDock = p_create_dock("SegSettingDock", self.tr("分割设置"), widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.SegSettingDock)
        ## 专业功能区工作区
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        bandRegion = QtWidgets.QVBoxLayout()
        bandRegion.setObjectName("bandRegion")
        bandSelection = create_text(CentralWidget, "bandSelection", self.tr("波段设置"))
        bandRegion.addWidget(bandSelection)
        text_list = ["R", "G", "B"]
        self.bandCombos = []
        for txt in text_list:
            lab = create_text(CentralWidget, "band" + txt, txt)
            combo = QtWidgets.QComboBox()
            combo.addItems(["band_1"])
            self.bandCombos.append(combo)
            hbandLayout = QtWidgets.QHBoxLayout()
            hbandLayout.setObjectName("hbandLayout")
            hbandLayout.addWidget(lab)
            hbandLayout.addWidget(combo)
            hbandLayout.setStretch(1, 4)
            bandRegion.addLayout(hbandLayout)
        resultSave = create_text(CentralWidget, "resultSave", self.tr("保存设置"))
        bandRegion.addWidget(resultSave)
        self.boundaryRegular = QtWidgets.QCheckBox(self.tr("使用建筑边界简化"))
        self.boundaryRegular.setObjectName("boundaryRegular")
        bandRegion.addWidget(self.boundaryRegular)
        self.shpSave = QtWidgets.QCheckBox(self.tr("额外保存为ESRI Shapefile"))
        self.shpSave.setObjectName("shpSave")
        bandRegion.addWidget(self.shpSave)
        horizontalLayout.addLayout(bandRegion)
        showGeoInfo = create_text(CentralWidget, "showGeoInfo", self.tr("地理信息"))
        bandRegion.addWidget(showGeoInfo)
        self.edtGeoinfo = QtWidgets.QTextEdit(self.tr("无"))
        self.edtGeoinfo.setObjectName("edtGeoinfo")
        self.edtGeoinfo.setReadOnly(True)
        bandRegion.addWidget(self.edtGeoinfo)
        self.RSDock = p_create_dock("RSDock", self.tr("遥感设置"), widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.RSDock)

        ## 医学影像设置
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        MIRegion = QtWidgets.QVBoxLayout()
        MIRegion.setObjectName("MIRegion")
        # mi_text = create_text(CentralWidget, "sliceSelection", self.tr("切片选择"))
        # MIRegion.addWidget(mi_text)
        # self.sldMISlide, slideRegion = p_create_slider(
        #     "sldMISlide", "labMISlide", self.tr("切片选择："), 1, 1, 1
        # )
        # self.sldMISlide.setMinimum(1)
        wwLabel = QtWidgets.QLabel("窗宽")
        self.textWw = QtWidgets.QLineEdit()
        self.textWw.setText("200")
        self.textWw.setValidator(QtGui.QIntValidator())
        self.textWw.setMaxLength(4)

        wcLabel = QtWidgets.QLabel("窗位")
        self.textWc = QtWidgets.QLineEdit()
        self.textWc.setText("0")
        self.textWc.setValidator(QtGui.QIntValidator())
        self.textWc.setMaxLength(4)

        # MIRegion.addLayout(slideRegion)
        # MIRegion.addWidget(self.sldMISlide)
        MIRegion.addWidget(wwLabel)
        MIRegion.addWidget(self.textWw)
        MIRegion.addWidget(wcLabel)
        MIRegion.addWidget(self.textWc)
        horizontalLayout.addLayout(MIRegion)
        self.MedDock = p_create_dock("MedDock", self.tr("医疗设置"), widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.MedDock)
        ## 宫格区域
        widget = QtWidgets.QWidget()
        horizontalLayout = QtWidgets.QHBoxLayout(widget)
        GridRegion = QtWidgets.QVBoxLayout()
        GridRegion.setObjectName("GridRegion")
        # self.btnInitGrid = p_create_button(
        #     "btnInitGrid",
        #     self.tr("创建宫格"),
        #     osp.join(pjpath, "resource/N2.png"),
        #     "",
        # )
        self.btnFinishedGrid = p_create_button(
            "btnFinishedGrid",
            self.tr("完成宫格"),
            osp.join(pjpath, "resource/Save.png"),
            "",
        )
        hbandLayout = QtWidgets.QHBoxLayout()
        hbandLayout.setObjectName("hbandLayout")
        # hbandLayout.addWidget(self.btnInitGrid)
        hbandLayout.addWidget(self.btnFinishedGrid)
        GridRegion.addLayout(hbandLayout)  # 创建宫格
        self.cheSaveEvery = QtWidgets.QCheckBox(self)
        self.cheSaveEvery.setText("保存每个宫格的标签")
        self.cheSaveEvery.setChecked(False)
        GridRegion.addWidget(self.cheSaveEvery)
        self.gridTable = QtWidgets.QTableWidget(CentralWidget)
        self.gridTable.horizontalHeader().hide()
        self.gridTable.verticalHeader().hide()
        # 铺满
        self.gridTable.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.gridTable.verticalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch
        )
        self.gridTable.setObjectName("gridTable")
        self.gridTable.clearContents()
        self.gridTable.setColumnCount(1)
        self.gridTable.setRowCount(1)
        GridRegion.addWidget(self.gridTable)
        horizontalLayout.addLayout(GridRegion)
        self.GridDock = p_create_dock("GridDock", self.tr("宫格切换"), widget)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.GridDock)
        ## -----
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        # log.debug("Set up UI finished")

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
