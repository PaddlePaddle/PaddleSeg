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

import logging
import os
import time
import os.path as osp
from functools import partial
import json
from distutils.util import strtobool
import webbrowser
from easydict import EasyDict as edict

from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtWidgets import QMainWindow, QMessageBox, QTableWidgetItem, QApplication
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtCore import Qt, QByteArray, QVariant, QCoreApplication, QThread, Signal, QTimer
import cv2
import numpy as np
from PIL import Image
import paddle
import paddle.nn.functional as F

from eiseg import pjpath, __APPNAME__, logger
from widget import ShortcutWidget, PolygonAnnotation
from controller import InteractiveController
from ui import Ui_EISeg
import util
from util import COCO
from util import check_cn, normcase

import plugin.remotesensing as rs
from plugin.medical import med
from plugin.remotesensing import Raster
from plugin.n2grid import RSGrids, Grids, checkOpenGrid
from plugin.video import InferenceCore, overlay_davis


# TODO: 研究paddle子线程
class ModelThread(QThread):
    _signal = Signal(dict)

    def __init__(self, controller, param_path):
        super().__init__()
        self.controller = controller
        self.param_path = param_path

    def run(self):
        success, res = self.controller.setModel(self.param_path, False)
        self._signal.emit({
            "success": success,
            "res": res,
            "param_path": self.param_path
        })


class APP_EISeg(QMainWindow, Ui_EISeg):
    IDILE, ANNING, EDITING = 0, 1, 2
    # IDILE：网络，权重，图像三者任一没有加载
    # EDITING：多边形编辑，可以交互式，但是多边形内部不能点
    # ANNING：交互式标注，只能交互式，不能编辑多边形，多边形不接hover

    # 宫格标注背景颜色
    GRID_COLOR = {
        "idle": QtGui.QColor(255, 255, 255),
        "current": QtGui.QColor(192, 220, 243),
        "finised": QtGui.QColor(185, 185, 225),
        "overlying": QtGui.QColor(51, 52, 227),
    }

    def __init__(self, parent=None):
        super(APP_EISeg, self).__init__(parent)

        self.settings = QtCore.QSettings(
            osp.join(pjpath, "config/setting.txt"), QtCore.QSettings.IniFormat)
        currentLang = self.settings.value("language")
        layoutdir = Qt.RightToLeft if currentLang == "Arabic" else Qt.LeftToRight
        self.setLayoutDirection(layoutdir)

        # 初始化界面
        self.setupUi(self)

        # app变量
        self._anning = False  # self.status替代
        self.isDirty = False  # 是否需要保存
        self.image = None  # 可能先加载图片后加载模型，只用于暂存图片
        self.predictor_params = {
            "brs_mode": "NoBRS",
            "with_flip": False,
            "zoom_in_params": {
                "skip_clicks": -1,
                "target_size": (400, 400),
                "expansion_ratio": 1.4,
            },
            "predictor_params": {
                "net_clicks_limit": None,
                "max_size": 800,
                "with_mask": True,
            },
        }
        self.controller = InteractiveController(
            predictor_params=self.predictor_params,
            prob_thresh=self.segThresh, )

        self.video = InferenceCore()
        self.video_images = None
        self.video_masks = None
        # self.controller.labelList = util.LabelList()  # 标签列表
        self.save_status = {
            "gray_scale": True,
            "pseudo_color": True,
            "json": False,
            "coco": True,
            "cutout": True,
        }  # 是否保存这几个格式
        self.outputDir = None  # 标签保存路径
        self.labelPaths = []  # 所有outputdir中的标签文件路径
        self.imagePaths = []  # 文件夹下所有待标注图片路径
        self.currIdx = 0  # 文件夹标注当前图片下标
        self.origExt = False  # 是否使用图片本身拓展名，防止重名覆盖
        if self.save_status["coco"]:
            self.coco = COCO()
        else:
            self.coco = None
        self.colorMap = util.colorMap

        if self.settings.value("cutout_background"):
            self.cutoutBackground = [
                int(c) for c in self.settings.value("cutout_background")
            ]
            if len(self.cutoutBackground) == 3:
                self.cutoutBackground += tuple([255])
        else:
            self.cutoutBackground = [0, 0, 128, 255]

        if self.settings.value("cross_color"):
            self.crossColor = [
                int(c) for c in self.settings.value("cross_color")
            ]
        else:
            self.crossColor = [0, 0, 0, 127]
        self.scene.setPenColor(self.crossColor)

        # widget
        self.dockWidgets = {
            "model": [self.ModelDock],
            "data": [self.DataDock],
            "label": [self.LabelDock],
            "seg": [self.SegSettingDock],
            "rs": [self.RSDock],
            "med": [self.MedDock],
            "grid": [self.GridDock],
            "video": [self.VideoDock],
            "vseg": [self.VSTDock],
            "3d": [self.TDDock],
        }
        # self.display_dockwidget = [True, True, True, True, False, False, False, False, False, False]
        self.dockStatus = self.settings.value(
            "dock_status", QVariant([]), type=list)  # 所有widget是否展示
        if len(self.dockStatus) != len(self.dockWidgets):
            self.dockStatus = [True] * 4 + [False] * (len(self.dockWidgets) - 4)
            self.settings.setValue("dock_status", self.dockStatus)
        else:
            self.dockStatus = [strtobool(s) for s in self.dockStatus]

        self.layoutStatus = self.settings.value("layout_status",
                                                QByteArray())  # 界面元素位置

        self.recentModels = self.settings.value(
            "recent_models", QVariant([]), type=list)
        self.video_recentModels = self.settings.value(
            "video_recent_models", QVariant([]), type=list)
        self.recentFiles = self.settings.value(
            "recent_files", QVariant([]), type=list)

        self.config = util.parse_configs(osp.join(pjpath, "config/config.yaml"))

        # 支持的图像格式
        rs_ext = [".tif", ".tiff"]
        img_ext = []
        for fmt in QtGui.QImageReader.supportedImageFormats():
            fmt = ".{}".format(fmt.data().decode())
            if fmt not in rs_ext:
                img_ext.append(fmt)

        video_ext = [
            ".wmv",
            ".asf",
            ".asx",
            ".rm",
            ".rmvb",
            ".mp4",
            ".3gp",
            ".mov",
            ".m4v",
            ".avi",
            ".dat",
            ".mkv",
            ".flv",
            ".vob",
        ]
        self.video_ext = video_ext

        self.formats = [
            img_ext,  # 自然图像
            [".dcm"],  # 医学影像
            rs_ext,  # 遥感影像
            video_ext,  # 视频
        ]

        # 遥感
        self.raster = None
        self.grid = None
        self.rsRGB = [1, 1, 1]  # 遥感索引

        # 医疗参数
        self.midx = 0  # 医疗切片索引

        # 大图限制
        self.thumbnail_min = 2000

        # 初始化action
        self.initActions()

        # 更新近期记录
        self.loadLayout()  # 放前面
        self.toggleWidget("all", warn=False)
        self.updateModelMenu()
        self.updateVideoModelMenu()
        self.updateRecentFile()
        # self.VideoDock.hide()

        # 窗口
        ## 快捷键
        self.ShortcutWidget = ShortcutWidget(self.actions, pjpath)

        ## 画布
        self.scene.clickRequest.connect(self.canvasClick)
        self.canvas.zoomRequest.connect(self.viewZoomed)
        self.canvas.mousePosChanged.connect(self.scene.onMouseChanged)
        self.annImage = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.annImage)

        ## 按钮点击
        self.btnSave.clicked.connect(self.exportLabel)  # 保存
        self.listFiles.itemDoubleClicked.connect(
            self.imageListClicked)  # 标签列表点击
        self.btnAddClass.clicked.connect(self.addLabel)
        self.btnParamsSelect.clicked.connect(self.changeParam)  # 模型参数选择
        self.btn3DParamsSelect.clicked.connect(self.changePropgationParam)
        self.cheWithMask.stateChanged.connect(self.chooseMode)  # with_mask
        self.btnPropagate.clicked.connect(self.on_propgation)

        ## 滑动
        self.sldOpacity.valueChanged.connect(self.maskOpacityChanged)
        self.sldClickRadius.valueChanged.connect(self.clickRadiusChanged)
        self.sldThresh.valueChanged.connect(self.threshChanged)
        # self.sldBrush.valueChanged.connect(self.brushChanged)
        self.sldWw.valueChanged.connect(self.swwChanged)
        self.sldWc.valueChanged.connect(self.swcChanged)
        self.textWw.returnPressed.connect(self.twwChanged)
        self.textWc.returnPressed.connect(self.twcChanged)

        ## 标签列表点击
        self.labelListTable.cellDoubleClicked.connect(self.labelListDoubleClick)
        self.labelListTable.cellClicked.connect(self.labelListClicked)
        self.labelListTable.cellChanged.connect(self.labelListItemChanged)

        ## 功能区选择
        # self.rsShow.currentIndexChanged.connect(self.rsShowModeChange)  # 显示模型
        for bandCombo in self.bandCombos:
            bandCombo.currentIndexChanged.connect(self.rsBandSet)  # 设置波段
        # self.btnInitGrid.clicked.connect(self.initGrid)  # 打开宫格
        self.btnFinishedGrid.clicked.connect(self.saveGridLabel)

        ## 视频相关
        self.timer.timeout.connect(self.on_time)
        self.videoPlay.clicked.connect(self.on_play)
        self.sldTime.valueChanged.connect(self.sframeChanged)
        self.textTime.returnPressed.connect(self.tframeChanged)
        self.ratio = 20
        self.speedComboBox.currentIndexChanged.connect(self.on_speed)
        self.preFrameButton.clicked.connect(self.turnPreFrame)
        self.nextFrameButton.clicked.connect(self.turnNextFrame)

    def initActions(self):
        tr = partial(QtCore.QCoreApplication.translate, "APP_EISeg")
        action = partial(util.newAction, self)
        start = dir()

        # 打开/加载/保存
        open_image = action(
            tr("&打开图像"),
            self.openImage,
            "open_image",
            "OpenImage",
            tr("打开一张图像进行标注"), )
        open_folder = action(
            tr("&打开文件夹"),
            self.openFolder,
            "open_folder",
            "OpenFolder",
            tr("打开一个文件夹下所有的图像进行标注"), )
        change_output_dir = action(
            tr("&改变标签保存路径"),
            partial(self.changeOutputDir, None),
            "change_output_dir",
            "ChangeOutputDir",
            tr("改变标签保存的文件夹路径"), )
        load_param = action(
            tr("&加载模型参数"),
            self.changeParam,
            "load_param",
            "Model",
            tr("加载一个模型参数"), )
        save = action(
            tr("&保存"),
            self.exportLabel,
            "save",
            "Save",
            tr("保存图像标签"), )
        save_as = action(
            tr("&另存为"),
            partial(
                self.exportLabel, saveAs=True),
            "save_as",
            "SaveAs",
            tr("在指定位置另存为标签"), )
        auto_save = action(
            tr("&自动保存"),
            self.toggleAutoSave,
            "auto_save",
            "AutoSave",
            tr("翻页同时自动保存"),
            checkable=True, )
        # auto_save.setChecked(self.config.get("auto_save", False))

        # 标注
        turn_prev = action(
            tr("&上一张"),
            partial(self.turnImg, -1),
            "turn_prev",
            "Prev",
            tr("翻到上一张图片"), )
        turn_next = action(
            tr("&下一张"),
            partial(self.turnImg, 1),
            "turn_next",
            "Next",
            tr("翻到下一张图片"), )
        finish_object = action(
            tr("&完成当前目标"),
            self.finishObject,
            "finish_object",
            "Ok",
            tr("完成当前目标的标注"), )
        clear = action(
            tr("&清除所有标注"),
            self.clearAll,
            "clear",
            "Clear",
            tr("清除所有标注信息"), )
        undo = action(
            tr("&撤销"),
            self.undoClick,
            "undo",
            "Undo",
            tr("撤销一次点击"), )
        redo = action(
            tr("&重做"),
            self.redoClick,
            "redo",
            "Redo",
            tr("重做一次点击"), )
        del_active_polygon = action(
            tr("&删除多边形"),
            self.delActivePolygon,
            "del_active_polygon",
            "DeletePolygon",
            tr("删除当前选中的多边形"), )
        del_all_polygon = action(
            tr("&删除所有多边形"),
            self.delAllPolygon,
            "del_all_polygon",
            "DeleteAllPolygon",
            tr("删除所有的多边形"), )
        largest_component = action(
            tr("&保留最大连通块"),
            self.toggleLargestCC,
            "largest_component",
            "SaveLargestCC",
            tr("保留最大的连通块"),
            checkable=True, )
        origional_extension = action(
            tr("&标签和图像使用相同拓展名"),
            self.toggleOrigExt,
            "origional_extension",
            "Same",
            tr("标签和图像使用相同拓展名，用于图像中有文件名相同但拓展名不同的情况，防止标签覆盖"),
            checkable=True, )
        save_pseudo = action(
            tr("&伪彩色保存"),
            partial(self.toggleSave, "pseudo_color"),
            "save_pseudo",
            "SavePseudoColor",
            tr("保存为伪彩色图像"),
            checkable=True, )
        save_pseudo.setChecked(self.save_status["pseudo_color"])
        save_grayscale = action(
            tr("&灰度保存"),
            partial(self.toggleSave, "gray_scale"),
            "save_grayscale",
            "SaveGrayScale",
            tr("保存为灰度图像，像素的灰度为对应类型的标签"),
            checkable=True, )
        save_grayscale.setChecked(self.save_status["gray_scale"])
        save_json = action(
            tr("&JSON保存"),
            partial(self.toggleSave, "json"),
            "save_json",
            "SaveJson",
            tr("保存为JSON格式"),
            checkable=True, )
        save_json.setChecked(self.save_status["json"])
        save_coco = action(
            tr("&COCO保存"),
            partial(self.toggleSave, "coco"),
            "save_coco",
            "SaveCOCO",
            tr("保存为COCO格式"),
            checkable=True, )
        save_coco.setChecked(self.save_status["coco"])
        # test func
        self.show_rs_poly = action(
            tr("&显示遥感多边形"),
            None,
            "show_rs_poly",
            "Show",
            tr("显示遥感大图的多边形结果"),
            checkable=True, )
        self.show_rs_poly.setChecked(False)
        self.grid_message = action(
            tr("&启用宫格检测"),
            None,
            "grid_message",
            "Show",
            tr("针对每张图片启用宫格检测"),
            checkable=True, )
        self.grid_message.setChecked(True)
        save_cutout = action(
            tr("&抠图保存"),
            partial(self.toggleSave, "cutout"),
            "save_cutout",
            "SaveCutout",
            tr("只保留前景，背景设置为背景色"),
            checkable=True, )
        save_cutout.setChecked(self.save_status["cutout"])
        set_cutout_background = action(
            tr("&设置抠图背景色"),
            self.setCutoutBackground,
            "set_cutout_background",
            self.cutoutBackground,
            tr("抠图后背景像素的颜色"), )
        close = action(
            tr("&关闭"),
            partial(self.saveImage, True),
            "close",
            "Close",
            tr("关闭当前图像"), )
        quit = action(
            tr("&退出"),
            self.close,
            "quit",
            "Quit",
            tr("退出软件"), )
        export_label_list = action(
            tr("&导出标签列表"),
            partial(self.exportLabelList, None),
            "export_label_list",
            "ExportLabel",
            tr("将标签列表导出成标签配置文件"), )
        import_label_list = action(
            tr("&载入标签列表"),
            partial(self.importLabelList, None),
            "import_label_list",
            "ImportLabel",
            tr("从标签配置文件载入标签列表"), )
        clear_label_list = action(
            tr("&清空标签列表"),
            self.clearLabelList,
            "clear_label_list",
            "ClearLabel",
            tr("清空所有的标签"), )
        clear_recent = action(
            tr("&清除近期文件记录"),
            self.clearRecentFile,
            "clear_recent",
            "ClearRecent",
            tr("清除近期标注文件记录"), )
        model_widget = action(
            tr("&模型选择"),
            partial(self.toggleWidget, 0),
            "model_widget",
            "Net",
            tr("隐藏/展示模型选择面板"),
            checkable=True, )
        data_widget = action(
            tr("&数据列表"),
            partial(self.toggleWidget, 1),
            "data_widget",
            "Data",
            tr("隐藏/展示数据列表面板"),
            checkable=True, )
        label_widget = action(
            tr("&标签列表"),
            partial(self.toggleWidget, 2),
            "label_widget",
            "Label",
            tr("隐藏/展示标签列表面板"),
            checkable=True, )
        segmentation_widget = action(
            tr("&分割设置"),
            partial(self.toggleWidget, 3),
            "segmentation_widget",
            "Setting",
            tr("隐藏/展示分割设置面板"),
            checkable=True, )
        rs_widget = action(
            tr("&遥感设置"),
            partial(self.toggleWidget, 4),
            "rs_widget",
            "RemoteSensing",
            tr("隐藏/展示遥感设置面板"),
            checkable=True, )
        mi_widget = action(
            tr("&医疗设置"),
            partial(self.toggleWidget, 5),
            "mi_widget",
            "MedicalImaging",
            tr("隐藏/展示医疗设置面板"),
            checkable=True, )
        grid_ann_widget = action(
            tr("&N2宫格标注"),
            partial(self.toggleWidget, 6),
            "grid_ann_widget",
            "N2",
            tr("隐藏/展示N^2宫格细粒度标注面板"),
            checkable=True, )
        video_play_widget = action(
            tr("&视频播放"),
            partial(self.toggleWidget, 7),
            "video_play_widget",
            "Video",
            tr("隐藏/展示视频播放面板"),
            checkable=True, )
        video_anno_widget = action(
            tr("&视频标注"),
            partial(self.toggleWidget, 8),
            "video_anno_widget",
            "VideoAnno",
            tr("隐藏/展示视频标注面板"),
            checkable=True, )
        td_widget = action(
            tr("&3D显示"),
            partial(self.toggleWidget, 9),
            "td_widget",
            "3D",
            tr("隐藏/展示3D显示面板"),
            checkable=True, )
        quick_start = action(
            tr("&快速入门"),
            self.quickStart,
            "quick_start",
            "Use",
            tr("主要功能使用介绍"), )
        report_bug = action(
            tr("&反馈问题"),
            self.reportBug,
            "report_bug",
            "ReportBug",
            tr("通过Github Issue反馈使用过程中遇到的问题。我们会尽快进行修复"), )
        edit_shortcuts = action(
            tr("&编辑快捷键"),
            self.editShortcut,
            "edit_shortcuts",
            "Shortcut",
            tr("编辑软件快捷键"), )
        toggle_logging = action(
            tr("&调试日志"),
            self.toggleLogging,
            "toggle_logging",
            "Log",
            tr("用于观察软件执行过程和进行debug。我们不会自动收集任何日志，可能会希望您在反馈问题时间打开此功能，帮助我们定位问题。"),
            checkable=True, )
        toggle_logging.setChecked(bool(self.settings.value("log", False)))
        use_qt_widget = action(
            tr("&使用QT文件窗口"),
            self.useQtWidget,
            "use_qt_widget",
            "Qt",
            tr("如果使用文件选择窗口时遇到问题可以选择使用Qt窗口"),
            checkable=True, )
        # print(
        #     "use_qt_widget",
        #     self.settings.value("use_qt_widget", type=bool),
        # )
        use_qt_widget.setChecked(
            self.settings.value(
                "use_qt_widget", False, type=bool))

        self.actions = util.struct()
        for name in dir():
            if name not in start:
                self.actions.append(eval(name))

        def newWidget(text, icon, showAction):
            widget = QtWidgets.QMenu(text)
            widget.setIcon(util.newIcon(icon))
            widget.aboutToShow.connect(showAction)
            return widget

        recent_files = newWidget(self.tr("近期文件"), "Data", self.updateRecentFile)
        recent_params = newWidget(
            self.tr("近期模型及参数"), "Net", self.updateModelMenu)
        video_recent_params = newWidget(
            self.tr("近期视频传播模型及参数"), "Net", self.updateVideoModelMenu)
        languages = newWidget(self.tr("语言"), "Language", self.updateLanguage)

        self.menus = util.struct(
            recent_files=recent_files,
            recent_params=recent_params,
            video_recent_params=video_recent_params,
            languages=languages,
            fileMenu=(
                open_image,
                open_folder,
                change_output_dir,
                load_param,
                clear_recent,
                recent_files,
                recent_params,
                video_recent_params,
                None,
                save,
                save_as,
                auto_save,
                None,
                turn_next,
                turn_prev,
                close,
                None,
                quit, ),
            labelMenu=(
                export_label_list,
                import_label_list,
                clear_label_list, ),
            functionMenu=(
                largest_component,
                del_active_polygon,
                del_all_polygon,
                None,
                origional_extension,
                save_pseudo,
                save_grayscale,
                save_cutout,
                set_cutout_background,
                None,
                save_json,
                save_coco,
                None,
                # test
                self.show_rs_poly,
                None,
                self.grid_message, ),
            showMenu=(
                model_widget,
                data_widget,
                label_widget,
                segmentation_widget,
                rs_widget,
                mi_widget,
                grid_ann_widget,
                video_play_widget,
                video_anno_widget,
                td_widget, ),
            helpMenu=(
                languages,
                use_qt_widget,
                quick_start,
                report_bug,
                edit_shortcuts,
                toggle_logging, ),
            toolBar=(
                finish_object,
                clear,
                undo,
                redo,
                turn_prev,
                turn_next,
                None,
                save_pseudo,
                save_grayscale,
                save_cutout,
                save_json,
                save_coco,
                origional_extension,
                None,
                largest_component, ), )

        def menu(title, actions=None):
            menu = self.menuBar().addMenu(title)
            if actions:
                util.addActions(menu, actions)
            return menu

        menu(tr("文件"), self.menus.fileMenu)
        menu(tr("标注"), self.menus.labelMenu)
        menu(tr("功能"), self.menus.functionMenu)
        menu(tr("显示"), self.menus.showMenu)
        menu(tr("帮助"), self.menus.helpMenu)
        util.addActions(self.toolBar, self.menus.toolBar)

    def __setColor(self, action, setting_name):
        c = action
        color = QtWidgets.QColorDialog.getColor(
            QtGui.QColor(*c),
            self,
            options=QtWidgets.QColorDialog.ShowAlphaChannel, )
        action = color.getRgb()
        self.settings.setValue(setting_name, [int(c) for c in action])
        return action

    def on_speed(self, sender):
        text = self.speedComboBox.currentText()
        self.ratio = int(20 * float(text[4:-1]))
        if self.timer.isActive():
            self.timer.stop()
            self.timer.start(1000 / self.ratio)

    def setCutoutBackground(self):
        self.cutoutBackground = self.__setColor(self.cutoutBackground,
                                                "cutout_background")
        self.actions.set_cutout_background.setIcon(
            util.newIcon(self.cutoutBackground))

    def editShortcut(self):
        self.ShortcutWidget.center()
        self.ShortcutWidget.show()

    # 多语言
    def updateLanguage(self):
        self.menus.languages.clear()
        langs = os.listdir(osp.join(pjpath, "util/translate"))
        langs = [n.split(".")[0] for n in langs if n.endswith("qm")]
        langs.append("中文")
        for lang in langs:
            if lang == self.currLanguage:
                continue
            entry = util.newAction(
                self,
                lang,
                partial(self.changeLanguage, lang),
                None,
                lang if lang != "Arabic" else "Egypt", )
            self.menus.languages.addAction(entry)

    def changeLanguage(self, lang):
        self.settings.setValue("language", lang)
        self.warn(self.tr("切换语言"), self.tr("切换语言需要重启软件才能生效"))

    # 近期图像
    def updateRecentFile(self):
        menu = self.menus.recent_files
        menu.clear()
        recentFiles = self.settings.value(
            "recent_files", QVariant([]), type=list)
        files = [f for f in recentFiles if osp.exists(f)]
        for i, f in enumerate(files):
            icon = util.newIcon("File")
            action = QtWidgets.QAction(icon, "&【%d】 %s" %
                                       (i + 1, QtCore.QFileInfo(f).fileName()),
                                       self)
            action.triggered.connect(partial(self.openRecentImage, f))
            menu.addAction(action)
        if len(files) == 0:
            menu.addAction(self.tr("无近期文件"))
        self.settings.setValue("recent_files", files)

    def addRecentFile(self, path):
        if not osp.exists(path):
            return
        paths = self.settings.value("recent_files", QVariant([]), type=list)
        if path not in paths:
            paths.append(path)
        if len(paths) > 15:
            del paths[0]
        self.settings.setValue("recent_files", paths)
        self.updateRecentFile()

    def clearRecentFile(self):
        self.settings.remove("recent_files")
        self.statusbar.showMessage(self.tr("已清除最近打开文件"), 10000)

    # 模型加载
    def updateModelMenu(self):
        menu = self.menus.recent_params
        menu.clear()

        self.recentModels = [
            m for m in self.recentModels if osp.exists(m["param_path"])
        ]
        for idx, m in enumerate(self.recentModels):
            icon = util.newIcon("Model")
            action = QtWidgets.QAction(
                icon,
                f"{osp.basename(m['param_path'])}",
                self, )
            action.triggered.connect(
                partial(self.setModelParam, m["param_path"]))
            menu.addAction(action)
        if len(self.recentModels) == 0:
            menu.addAction(self.tr("无近期模型记录"))
        self.settings.setValue("recent_params", self.recentModels)

    def updateVideoModelMenu(self):
        menu = self.menus.video_recent_params
        menu.clear()

        self.video_recentModels = [
            m for m in self.video_recentModels if osp.exists(m["param_path"])
        ]
        for idx, m in enumerate(self.video_recentModels):
            icon = util.newIcon("Model")
            action = QtWidgets.QAction(
                icon,
                f"{osp.basename(m['param_path'])}",
                self, )
            action.triggered.connect(
                partial(self.setVideoModelParam, m["param_path"]))
            menu.addAction(action)
        if len(self.video_recentModels) == 0:
            menu.addAction(self.tr("无近期视频传播模型记录"))
        self.settings.setValue("video_recent_params", self.video_recentModels)

    def setModelParam(self, paramPath):
        res = self.changeParam(paramPath)
        if res:
            return True
        return False

    def setVideoModelParam(self, paramPath):
        res = self.changePropgationParam(paramPath)
        if res:
            return True
        return False

    def changeParam(self, param_path: str=None):
        if not param_path:
            filters = self.tr("Paddle静态模型权重文件(*.pdiparams)")
            start_path = ("." if len(self.recentModels) == 0 else
                          osp.dirname(self.recentModels[-1]["param_path"]))
            if self.settings.value("use_qt_widget", False, type=bool):
                options = QtWidgets.QFileDialog.DontUseNativeDialog
            else:
                options = QtWidgets.QFileDialog.ReadOnly
            param_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("选择传播模型参数") + " - " + __APPNAME__,
                start_path,
                filters,
                options=options, )
            # QtWidgets.QFileDialog.DontUseNativeDialog
        if not param_path:
            return False

        # 中文路径打不开
        if check_cn(param_path):
            self.warn(self.tr("参数路径存在中文"), self.tr("请修改参数路径为非中文路径！"))
            return False

        # success, res = self.controller.setModel(param_path)
        self.load_thread = ModelThread(self.controller, param_path)
        self.load_thread._signal.connect(self.__change_model_callback)
        self.load_thread.start()

    def changePropgationParam(self, param_path: str=None):
        if not param_path:
            filters = self.tr("Paddle静态模型权重文件(*.pdiparams)")
            start_path = (
                ".") if len(self.video_recentModels) == 0 else osp.dirname(
                    self.video_recentModels[-1]["param_path"])
            if self.settings.value("use_qt_widget", False, type=bool):
                options = QtWidgets.QFileDialog.DontUseNativeDialog
            else:
                options = QtWidgets.QFileDialog.ReadOnly
            param_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("选择模型参数") + " - " + __APPNAME__,
                start_path,
                filters,
                options=options, )
            # QtWidgets.QFileDialog.DontUseNativeDialog
        if not param_path:
            return False

        # 中文路径打不开
        if check_cn(param_path):
            self.warn(self.tr("参数路径存在中文"), self.tr("请修改参数路径为非中文路径！"))
            return False

        success, res = self.video.set_model(param_path)

        if success:
            model_dict = {"param_path": param_path}
            if model_dict not in self.video_recentModels:
                self.video_recentModels.insert(0, model_dict)
                if len(self.recentModels) > 10:
                    del self.recentModels[-1]
            else:  # 如果存在移动位置，确保加载最近模型的正确
                self.video_recentModels.remove(model_dict)
                self.video_recentModels.insert(0, model_dict)
            self.settings.setValue("video_recent_models",
                                   self.video_recentModels)
            self.statusbar.showMessage(
                osp.basename(param_path) + self.tr("视频传播模型加载成功"), 10000)
            return True
        else:
            self.warnException(res)
            return False

    def __change_model_callback(self, signal_dict: dict):
        success = signal_dict["success"]
        res = signal_dict["res"]
        param_path = signal_dict["param_path"]
        if success:
            model_dict = {"param_path": param_path}
            if model_dict not in self.recentModels:
                self.recentModels.insert(0, model_dict)
                if len(self.recentModels) > 10:
                    del self.recentModels[-1]
            else:  # 如果存在移动位置，确保加载最近模型的正确
                self.recentModels.remove(model_dict)
                self.recentModels.insert(0, model_dict)
            self.settings.setValue("recent_models", self.recentModels)
            self.statusbar.showMessage(
                osp.basename(param_path) + self.tr(" 模型加载成功"), 10000)
            return True
        else:
            self.warnException(res)
            return False

    def chooseMode(self):
        self.predictor_params["predictor_params"][
            "with_mask"] = self.cheWithMask.isChecked()
        self.controller.reset_predictor(predictor_params=self.predictor_params)
        if self.cheWithMask.isChecked():
            self.statusbar.showMessage(self.tr("掩膜已启用"), 10000)
        else:
            self.statusbar.showMessage(self.tr("掩膜已关闭"), 10000)

    def loadRecentModelParam(self):
        if len(self.recentModels) == 0:
            self.statusbar.showMessage(self.tr("没有最近使用模型信息，请加载模型"), 10000)
            return
        m = self.recentModels[0]
        param_path = m["param_path"]
        self.setModelParam(param_path)

    def loadVideoRecentModelParam(self):
        if len(self.video_recentModels) == 0:
            self.statusbar.showMessage(self.tr("没有最近使用的视频传播模型信息，请加载模型"), 10000)
            return
        m = self.video_recentModels[0]
        param_path = m["param_path"]
        self.setVideoModelParam(param_path)

    # 标签列表
    def importLabelList(self, filePath=None):
        if filePath is None:
            if self.settings.value("use_qt_widget", False, type=bool):
                options = QtWidgets.QFileDialog.DontUseNativeDialog
            else:
                options = QtWidgets.QFileDialog.ReadOnly
            filters = self.tr("标签配置文件") + " (*.txt)"
            filePath, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("选择标签配置文件路径") + " - " + __APPNAME__,
                ".",
                filters,
                options=options, )
        filePath = normcase(filePath)
        if not osp.exists(filePath):
            return
        self.controller.importLabel(filePath)
        logger.info(f"Loaded label list: {self.controller.labelList.labelList}")
        self.refreshLabelList()

    def exportLabelList(self, savePath: str=None):
        if len(self.controller.labelList) == 0:
            self.warn(self.tr("没有需要保存的标签"), self.tr("请先添加标签之后再进行保存！"))
            return
        if savePath is None:
            filters = self.tr("标签配置文件") + "(*.txt)"
            dlg = QtWidgets.QFileDialog(
                self,
                self.tr("保存标签配置文件"),
                ".",
                filters, )
            dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
            if self.settings.value("use_qt_widget", False, type=bool):
                options = QtWidgets.QFileDialog.DontUseNativeDialog
            else:
                options = QtWidgets.QFileDialog.DontUseCustomDirectoryIcons
            dlg.setDefaultSuffix("txt")
            dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            savePath, _ = dlg.getSaveFileName(
                self,
                self.tr("选择保存标签配置文件路径") + " - " + __APPNAME__,
                ".",
                filters,
                options=options, )
        self.controller.exportLabel(savePath)

    def addLabel(self):
        c = self.colorMap.get_color()
        table = self.labelListTable
        idx = table.rowCount()
        table.insertRow(table.rowCount())
        self.controller.addLabel(idx + 1, "", c)
        numberItem = QTableWidgetItem(str(idx + 1))
        numberItem.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(idx, 0, numberItem)
        table.setItem(idx, 1, QTableWidgetItem())
        colorItem = QTableWidgetItem()
        colorItem.setBackground(QtGui.QColor(c[0], c[1], c[2]))
        colorItem.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(idx, 2, colorItem)
        delItem = QTableWidgetItem()
        delItem.setIcon(util.newIcon("Clear"))
        delItem.setTextAlignment(Qt.AlignCenter)
        delItem.setFlags(QtCore.Qt.ItemIsEnabled)
        table.setItem(idx, 3, delItem)
        self.adjustTableSize()
        self.labelListClicked(self.labelListTable.rowCount() - 1, 0)

    def adjustTableSize(self):
        self.labelListTable.horizontalHeader().setDefaultSectionSize(25)
        self.labelListTable.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Fixed)
        self.labelListTable.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.Fixed)
        self.labelListTable.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.Fixed)
        self.labelListTable.setColumnWidth(2, 50)

    def clearLabelList(self):
        if len(self.controller.labelList) == 0:
            return True
        res = self.warn(
            self.tr("清空标签列表?"),
            self.tr("请确认是否要清空标签列表"),
            QMessageBox.Yes | QMessageBox.Cancel, )
        if res == QMessageBox.Cancel:
            return False
        self.controller.labelList.clear()
        if self.controller:
            self.controller.label_list = []
            self.controller.curr_label_number = 0
        self.labelListTable.clear()
        self.labelListTable.setRowCount(0)
        return True

    def refreshLabelList(self):
        table = self.labelListTable
        table.clearContents()
        table.setRowCount(len(self.controller.labelList))
        table.setColumnCount(4)
        for idx, lab in enumerate(self.controller.labelList):
            numberItem = QTableWidgetItem(str(lab.idx))
            numberItem.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(idx, 0, numberItem)
            table.setItem(idx, 1, QTableWidgetItem(lab.name))
            c = lab.color
            colorItem = QTableWidgetItem()
            colorItem.setBackground(QtGui.QColor(c[0], c[1], c[2]))
            colorItem.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(idx, 2, colorItem)
            delItem = QTableWidgetItem()
            delItem.setIcon(util.newIcon("Clear"))
            delItem.setTextAlignment(Qt.AlignCenter)
            delItem.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(idx, 3, delItem)
            self.adjustTableSize()

        cols = [0, 1, 3]
        for idx in cols:
            table.resizeColumnToContents(idx)
        self.adjustTableSize()

    def labelListDoubleClick(self, row, col):
        if col != 2:
            return
        table = self.labelListTable
        color = QtWidgets.QColorDialog.getColor()
        if color.getRgb() == (0, 0, 0, 255):
            return
        table.item(row, col).setBackground(color)
        self.controller.labelList[row].color = color.getRgb()[:3]
        if self.controller:
            self.controller.label_list = self.controller.labelList
        for p in self.scene.polygon_items:
            idlab = self.controller.labelList.getLabelById(p.labelIndex)
            if idlab is not None:
                color = idlab.color
                p.setColor(color, color)
        self.labelListClicked(row, 0)

    @property
    def currLabelIdx(self):
        return self.controller.curr_label_number - 1

    def labelListClicked(self, row, col):
        table = self.labelListTable
        if col == 3:
            labelIdx = int(table.item(row, 0).text())
            if self.status == self.EDITING:
                if self.checkLabel(labelIdx):
                    self.controller.labelList.remove(labelIdx)
                    table.removeRow(row)
                else:
                    self.warn(
                        self.tr("无法删除"),
                        self.tr("当前多边形中存在此标签"), QMessageBox.Yes)
            elif self.status == self.ANNING:
                self.warn(
                    self.tr("无法删除"), self.tr("交互式标注模式无法删除标签"), QMessageBox.Yes)

        if col == 0 or col == 1:
            for cl in range(2):
                for idx in range(len(self.controller.labelList)):
                    table.item(idx,
                               cl).setBackground(QtGui.QColor(255, 255, 255))
                table.item(row, cl).setBackground(QtGui.QColor(48, 140, 198))
                table.item(row, 0).setSelected(True)
            if self.controller:
                self.controller.setCurrLabelIdx(int(table.item(row, 0).text()))
                self.controller.label_list = self.controller.labelList

    def labelListItemChanged(self, row, col):
        self.colorMap.usedColors = self.controller.labelList.colors
        try:
            if col == 1:
                name = self.labelListTable.item(row, col).text()
                self.controller.labelList[row].name = name
        except:
            pass

    # 多边形标注
    def createPoly(self, curr_polygon, color):
        if curr_polygon is None:
            return
        for points in curr_polygon:
            if len(points) < 3:
                continue
            poly = PolygonAnnotation(
                self.controller.labelList[self.currLabelIdx].idx,
                self.controller.image.shape,
                self.delPolygon,
                self.setDirty,
                color,
                color,
                self.opacity, )
            poly.labelIndex = self.controller.labelList[self.currLabelIdx].idx
            self.scene.addItem(poly)
            self.scene.polygon_items.append(poly)
            for p in points:
                poly.addPointLast(QtCore.QPointF(p[0], p[1]))
            self.setDirty(True)

    def delActivePolygon(self):
        for idx, polygon in enumerate(self.scene.polygon_items):
            if polygon.hasFocus():
                res = self.warn(
                    self.tr("确认删除？"),
                    self.tr("确认要删除当前选中多边形标注？"),
                    QMessageBox.Yes | QMessageBox.Cancel, )
                if res == QMessageBox.Yes:
                    self.delPolygon(polygon)

    def delPolygon(self, polygon):
        polygon.remove()
        if self.save_status["coco"]:
            if polygon.coco_id:
                self.coco.delAnnotation(
                    polygon.coco_id,
                    self.coco.imgNameToId[osp.basename(self.imagePath)], )
        self.setDirty(True)

    def delAllPolygon(self):
        for p in self.scene.polygon_items[::-1]:  # 删除所有多边形
            self.delPolygon(p)

    def delActivePoint(self):
        for polygon in self.scene.polygon_items:
            polygon.removeFocusPoint()

    # 图片/标签 io
    def getMask(self):
        if not self.controller or self.controller.image is None:
            return
        s = self.controller.imgShape
        pesudo = np.zeros([s[0], s[1]])
        # 覆盖顺序，从上往下
        # TODO: 是标签数值大的会覆盖小的吗?
        # A: 是列表中上面的覆盖下面的，由于标签可以移动，不一定是大小按顺序覆盖
        # RE: 我们做医学的时候覆盖比较多，感觉一般是数值大的标签覆盖数值小的标签。按照上面覆盖下面的话可能跟常见的情况正好是反过来的，感觉可能从下往上覆盖会比较好
        len_lab = self.labelListTable.rowCount()
        for i in range(len_lab - 1, -1, -1):
            idx = int(self.labelListTable.item(len_lab - i - 1, 0).text())
            for poly in self.scene.polygon_items:
                if poly.labelIndex == idx:
                    pts = np.int32([np.array(poly.scnenePoints)])
                    cv2.fillPoly(pesudo, pts=pts, color=idx)
        return pesudo

    def openRecentImage(self, file_path):
        self.queueEvent(partial(self.loadImage, file_path))
        self.listFiles.addItems([file_path.replace("\\", "/")])
        self.currIdx = self.listFiles.count() - 1
        self.listFiles.setCurrentRow(self.currIdx)  # 移动位置
        self.imagePaths.append(file_path)

    def openImage(self, filePath: str=None):
        # 在triggered.connect中使用不管默认filePath为什么返回值都为False
        if not isinstance(filePath, str) or filePath is False:
            prompts = ["图片", "医学影像", "遥感影像", "视频"]
            filters = ""
            for fmts, p in zip(self.formats, prompts):
                filters += f"{p} ({' '.join(['*' + f for f in fmts])}) ;; "
            filters = filters[:-3]
            recentPath = self.settings.value("recent_files", [])
            if len(recentPath) == 0:
                recentPath = "."
            else:
                recentPath = osp.dirname(recentPath[0])
            if self.settings.value("use_qt_widget", False, type=bool):
                options = QtWidgets.QFileDialog.DontUseNativeDialog
            else:
                options = QtWidgets.QFileDialog.ReadOnly
            filePath, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("选择待标注图片") + " - " + __APPNAME__,
                recentPath,
                filters,
                options=options, )
            if len(filePath) == 0:  # 用户没选就直接关闭窗口
                return
            if osp.splitext(filePath)[-1] in self.video_ext:
                if not paddle.device.is_compiled_with_cuda(
                ):  # TODO: 可以使用GPU却返回False
                    self.warn(
                        self.tr("请在gpu电脑上进行视频标注"),
                        self.tr("准备进行视频标注，由于视频标注需要一定计算，请尽量确保在gpu的电脑上进行操作!"))
        filePath = normcase(filePath)
        if not self.loadImage(filePath):
            return False

        # 3. 添加记录
        self.listFiles.addItems([filePath])
        self.currIdx = self.listFiles.count() - 1
        self.listFiles.setCurrentRow(self.currIdx)  # 移动位置
        self.imagePaths.append(filePath)
        return True

    def openFolder(self, inputDir: str=None):
        # 1. 如果没传文件夹，弹框让用户选
        if not isinstance(inputDir, str):
            recentPath = self.settings.value("recent_files", [])
            if len(recentPath) == 0:
                recentPath = "."
            else:
                recentPath = osp.dirname(recentPath[-1])
            options = (QtWidgets.QFileDialog.ShowDirsOnly |
                       QtWidgets.QFileDialog.DontResolveSymlinks)
            if self.settings.value("use_qt_widget", False, type=bool):
                options = options | QtWidgets.QFileDialog.DontUseNativeDialog
            inputDir = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("选择待标注图片文件夹") + " - " + __APPNAME__,
                recentPath,
                options, )
            if not osp.exists(inputDir):
                return

        # 2. 关闭当前图片，清空文件列表
        self.saveImage(close=True)
        self.imagePaths = []
        self.listFiles.clear()

        # 3. 扫描文件夹下所有图片
        # 3.1 获取所有文件名
        imagePaths = os.listdir(inputDir)
        exts = tuple(f for fmts in self.formats for f in fmts)
        imagePaths = [n for n in imagePaths if n.lower().endswith(exts)]
        imagePaths = [n for n in imagePaths if not n[0] == "."]
        imagePaths.sort()
        if len(imagePaths) == 0:
            return
        # 3.2 设置默认输出路径
        if self.outputDir is None:
            # 没设置为文件夹下的 label 文件夹
            self.outputDir = osp.join(inputDir, "label")
        if not osp.exists(self.outputDir):
            os.makedirs(self.outputDir)
        # 3.3 有重名图片，标签保留原来拓展名
        names = []
        for name in imagePaths:
            name = osp.splitext(name)[0]
            if name not in names:
                names.append(name)
            else:
                self.toggleOrigExt(True)
                break
        imagePaths = [osp.join(inputDir, n) for n in imagePaths]
        for p in imagePaths:
            p = normcase(p)
            self.imagePaths.append(p)
            self.listFiles.addItem(p)

        # 3.4 加载已有的标注
        if self.outputDir is not None and osp.exists(self.outputDir):
            self.changeOutputDir(self.outputDir)
        if len(self.imagePaths) != 0:
            self.currIdx = 0
            self.turnImg(0)
        self.inputDir = inputDir

    def loadImage(self, path):
        if self.controller.model is None:
            self.warn("未检测到模型", "请先加载模型参数")
            return
        # 1. 拒绝None和不存在的路径，关闭当前图像
        if not path:
            return
        path = normcase(path)
        if not osp.exists(path):
            return
        self.saveImage(True)  # 关闭当前图像
        self.eximgsInit()  # TODO: 将grid的部分整合到saveImage里

        # 2. 判断图像类型，打开
        # TODO: 加用户指定类型的功能
        image = None

        # 直接if会报错，因为打开遥感图像后多波段不存在，现在把遥感图像的单独抽出来了
        # 自然图像
        if path.lower().endswith(tuple(self.formats[0])):
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
            image = image[:, :, ::-1]  # BGR转RGB
            if self.grid_message.isChecked():
                if checkOpenGrid(image, self.thumbnail_min):
                    if self.loadGrid(image, False):
                        image, _ = self.grid.getGrid(0, 0)
            else:
                if self.dockWidgets["grid"][0].isVisible() is True:
                    self.grid = Grids(image)
                    self.initGrid()
                    image, _ = self.grid.getGrid(0, 0)

        # 医学影像
        if path.lower().endswith(tuple(self.formats[1])):
            if not self.dockStatus[5]:
                res = self.warn(
                    self.tr("未启用医疗组件"),
                    self.tr("加载医疗影像需启用医疗组件，是否立即启用？"),
                    QMessageBox.Yes | QMessageBox.Cancel, )
                if res == QMessageBox.Cancel:
                    return False
                self.toggleWidget(5)
                if not self.dockStatus[5]:
                    return False
            image = med.dcm_reader(path)  # TODO: 添加多层支持
            if image.shape[-1] != 1:
                self.warn("医学影像打开错误", "暂不支持打开多层医学影像")
                return False

            maxValue = np.max(image)  # 根据数据模态自适应窗宽窗位
            minValue = np.min(image)
            if minValue == 0:
                ww = maxValue
                wc = int(maxValue / 2)
            else:
                ww = maxValue + int(abs(minValue))
                wc = int((minValue + maxValue) / 2)
            self.sldWw.setValue(int(ww))
            self.textWw.setText(str(ww))
            self.sldWc.setValue(int(wc))
            self.textWc.setText(str(wc))

            self.controller.rawImage = self.image = image
            image = med.windowlize(image, self.ww, self.wc)

        # 遥感图像
        if path.lower().endswith(tuple(self.formats[
                2])):  # imghdr.what(path) == "tiff":
            if not self.dockStatus[4]:
                res = self.warn(
                    self.tr("未打开遥感组件"),
                    self.tr("打开遥感图像需启用遥感组件，是否立即启用？"),
                    QMessageBox.Yes | QMessageBox.Cancel, )
                if res == QMessageBox.Cancel:
                    return False
                self.toggleWidget(4)
                if not self.dockStatus[4]:
                    return False
            self.raster = Raster(path)
            gi = self.raster.showGeoInfo()
            self.edtGeoinfo.setText(
                self.tr("● 波段数：") + gi[0] + "\n" + self.tr("● 数据类型：") + gi[1] +
                "\n" + self.tr("● 行数：") + gi[2] + "\n" + self.tr("● 列数：") + gi[
                    3] + "\n" + "● EPSG：" + gi[4])
            if max(self.rsRGB) > self.raster.geoinfo.count:
                self.rsRGB = [1, 1, 1]
            self.raster.setBand(self.rsRGB)
            if self.grid_message.isChecked():
                if self.raster.checkOpenGrid(self.thumbnail_min):
                    if self.loadGrid(self.raster):
                        image, _ = self.raster.getGrid(0, 0)
                    else:
                        image, _ = self.raster.getArray()
                else:
                    image, _ = self.raster.getArray()
            else:
                if self.dockWidgets["grid"][0].isVisible() is True:
                    self.grid = RSGrids(self.raster)
                    self.raster.open_grid = True
                    self.initGrid()
                    image, _ = self.raster.getGrid(0, 0)
                else:
                    image, _ = self.raster.getArray()
            self.updateBandList()
            # self.updateSlideSld(True)
        else:
            self.edtGeoinfo.setText(self.tr("无"))

        # 视频
        if path.lower().endswith(tuple(self.formats[3])):  # mp4
            if not self.dockStatus[7]:
                res = self.warn(
                    self.tr("未启用视频组件"),
                    self.tr("加载视频需启用视频组件，是否立即启用？"),
                    QMessageBox.Yes | QMessageBox.Cancel, )
                if res == QMessageBox.Cancel:
                    return False
                self.toggleWidget(7)
                self.toggleWidget(8)
                if not self.dockStatus[7]:
                    return False
                if not self.dockStatus[8]:
                    return False
            # self.video_masks = None
            self.video_images = self.video.set_video(path)
            self.video_masks = np.zeros(
                (self.video.num_frames, self.video.height, self.video.width),
                dtype=np.uint8)
            self.sldTime.setMaximum(self.video.num_frames - 1)
            image = self.video_images[self.video.cursur]
            self.sldTime.setProperty("value", 0)
            # 清空3d显示
            if self.TDDock.isVisible():
                self.vtkWidget.init()
            # TODO: 处理

            # 如果没找到图片的reader
        if image is None:
            self.warn("打开图像失败", f"未找到{path}文件对应的读取程序")
            return

        self.image = image
        self.controller.setImage(image)
        self.updateImage(True)

        # 2. 加载标签
        self.loadLabel(path)
        self.addRecentFile(path)
        self.imagePath = path
        return True

    def loadLabel(self, imgPath):
        if imgPath == "":
            return None

        if self.video_images is not None:
            videoName = osp.splitext(osp.basename(imgPath))[0]
            maskPath = None
            for path in self.labelPaths:
                if osp.basename(path) == videoName:
                    maskPath = osp.join(path, 'mask')
            if not maskPath:
                return
            for cursur in range(self.video.num_frames):
                h, w = self.video_masks[cursur].shape
                frame_mask = np.zeros([h, w])
                pseudo = cv2.imread(
                    osp.join(maskPath, '{:05d}.png'.format(cursur)))
                for lab in self.controller.labelList:
                    frame_mask[(pseudo == lab.color[::-1])[:, :, 0]] = lab.idx
                self.video_masks[cursur] = frame_mask
            return

        # 1. 读取json格式标签
        if self.save_status["json"]:

            def getName(path):
                return osp.splitext(osp.basename(path))[0]

            imgName = getName(imgPath)
            labelPath = None
            for path in self.labelPaths:
                if not path.endswith(".json"):
                    continue
                if self.origExt:
                    if getName(path) == osp.basename(imgPath):
                        labelPath = path
                        break
                else:
                    if getName(path) == imgName:
                        labelPath = path
                        break
            if not labelPath:
                return

            labels = json.loads(open(labelPath, "r").read())

            for label in labels:
                color = label["color"]
                labelIdx = label["labelIdx"]
                points = label["points"]
                poly = PolygonAnnotation(
                    labelIdx,
                    self.controller.image.shape,
                    self.delPolygon,
                    self.setDirty,
                    color,
                    color,
                    self.opacity, )
                self.scene.addItem(poly)
                self.scene.polygon_items.append(poly)
                for p in points:
                    poly.addPointLast(QtCore.QPointF(p[0], p[1]))

        # 2. 读取coco格式标签
        if self.save_status["coco"]:
            imgId = self.coco.imgNameToId.get(osp.basename(imgPath), None)
            if imgId is None:
                return
            anns = self.coco.imgToAnns[imgId]
            for ann in anns:
                xys = ann["segmentation"][0]
                points = []
                for idx in range(0, len(xys), 2):
                    points.append([xys[idx], xys[idx + 1]])
                labelIdx = ann["category_id"]
                idlab = self.controller.labelList.getLabelById(labelIdx)
                if idlab is not None:
                    color = idlab.color
                    poly = PolygonAnnotation(
                        ann["category_id"],
                        self.controller.image.shape,
                        self.delPolygon,
                        self.setDirty,
                        color,
                        color,
                        self.opacity,
                        ann["id"], )
                    self.scene.addItem(poly)
                    self.scene.polygon_items.append(poly)
                    for p in points:
                        poly.addPointLast(QtCore.QPointF(p[0], p[1]))

    def turnImg(self, delta, list_click=False):
        if (self.grid is None or self.grid.curr_idx is None) or list_click:
            # 1. 检查是否有图可翻，保存标签
            self.currIdx += delta
            if self.currIdx >= len(self.imagePaths) or self.currIdx < 0:
                self.currIdx -= delta
                if delta == 1:
                    self.statusbar.showMessage(self.tr(f"没有后一张图片"))
                else:
                    self.statusbar.showMessage(self.tr(f"没有前一张图片"))
                self.saveImage(False)
                return
            else:
                self.saveImage(True)

            # 2. 打开新图
            self.loadImage(self.imagePaths[self.currIdx])
            self.listFiles.setCurrentRow(self.currIdx)
        else:
            self.turnGrid(delta)
        self.setDirty(False)

    def imageListClicked(self):
        if not self.controller:
            self.warn(self.tr("模型未加载"), self.tr("尚未加载模型，请先加载模型！"))
            self.changeParam()
            if not self.controller:
                return
        if self.controller.is_incomplete_mask:
            self.exportLabel()
        toRow = self.listFiles.currentRow()
        delta = toRow - self.currIdx
        self.turnImg(delta, True)

    def finishObject(self):
        if not self.controller or self.image is None:
            return
        current_mask, curr_polygon = self.controller.finishObject(
            building=self.boundaryRegular.isChecked())
        if curr_polygon is not None:
            self.updateImage()
            if current_mask is not None:
                # current_mask = current_mask.astype(np.uint8) * 255
                # polygon = util.get_polygon(current_mask)
                color = self.controller.labelList[self.currLabelIdx].color
                self.createPoly(curr_polygon, color)
        # 状态改变
        if self.status == self.EDITING:
            self.status = self.ANNING
            for p in self.scene.polygon_items:
                p.setAnning(isAnning=True)
        else:
            self.status = self.EDITING
            for p in self.scene.polygon_items:
                p.setAnning(isAnning=False)
        current_mask = self.getMask()
        if self.video_images is not None:
            if current_mask.max() != 0:
                self.video_masks[self.video.cursur] = current_mask

    def completeLastMask(self):
        # 返回最后一个标签是否完成，false就是还有带点的
        if not self.controller or self.controller.image is None:
            return True
        if not self.controller.is_incomplete_mask:
            return True
        res = self.warn(
            self.tr("完成最后一个目标？"),
            self.tr("是否完成最后一个目标的标注，不完成不会进行保存。"),
            QMessageBox.Yes | QMessageBox.Cancel, )
        if res == QMessageBox.Yes:
            self.finishObject()
            self.exportLabel()
            self.setDirty(False)
            return True
        return False

    def saveImage(self, close=False):
        if self.controller and self.controller.image is not None:
            # 1. 完成正在交互式标注的标签
            self.completeLastMask()
            # 2. 进行保存
            if self.isDirty:
                if self.actions.auto_save.isChecked():
                    self.exportLabel()
                else:
                    res = self.warn(
                        self.tr("保存标签？"),
                        self.tr("标签尚未保存，是否保存标签"),
                        QMessageBox.Yes | QMessageBox.Cancel, )
                    if res == QMessageBox.Yes:
                        self.exportLabel()
                self.setDirty(False)
            if close:
                # 3. 清空多边形标注，删掉图片
                for p in self.scene.polygon_items[::-1]:
                    p.remove()
                self.scene.polygon_items = []
                self.controller.resetLastObject()
                self.updateImage()
                self.controller.image = None
        if close:
            self.annImage.setPixmap(QPixmap())
        if self.video_images is not None and self.video_masks is not None:
            self.reset_video()

    def reset_video(self):
        self.video_images = None
        self.video_masks = None
        self.timer.stop()
        self.textTime.setText(str(0))
        self.videoPlay.setText(self.tr("播放"))
        self.videoPlay.setIcon(
            QtGui.QIcon(osp.join(pjpath, "resource/Play.png")))
        self.ratio = 20
        self.speedComboBox.setCurrentIndex(2)
        self.video.reset()

    def exportLabel(self, saveAs=False, savePath=None, lab_input=None):
        # 1. 需要处于标注状态
        if not self.controller or self.controller.image is None:
            return
        # 2. 完成正在交互式标注的标签
        self.completeLastMask()
        # 3. 确定保存路径
        # 3.1 如果参数指定了保存路径直接存到savePath
        if not savePath:
            if not saveAs and self.outputDir is not None:
                # 3.2 指定了标签文件夹，而且不是另存为：根据标签文件夹和文件名出保存路径
                name, ext = osp.splitext(osp.basename(self.imagePath))
                if not self.origExt:
                    ext = ".png"
                savePath = osp.join(
                    self.outputDir,
                    name + ext, )
                if self.video_images is not None and self.video_masks is not None:
                    savePath = osp.join(self.outputDir, name)
                    os.makedirs(savePath, exist_ok=True)
            else:
                # 3.3 没有指定标签存到哪，或者是另存为：弹框让用户选
                savePath = self.chooseSavePath()

        if savePath is None or not osp.exists(osp.dirname(savePath)):
            return

        if savePath not in self.labelPaths:
            self.labelPaths.append(savePath)

        # 视频帧保存
        if self.video_masks is not None:
            if osp.exists(savePath):
                res = self.warn(
                    self.tr("文件夹已经存在"),
                    self.tr("该文件夹下不为空，您确定继续保存在此路径下吗？"),
                    QMessageBox.Yes | QMessageBox.Cancel, )
                if res == QMessageBox.Cancel:
                    return
            os.makedirs(savePath, exist_ok=True)
            if osp.isdir(savePath):
                mask_dir = osp.join(savePath, 'mask')
                overlay_dir = osp.join(savePath, 'overlay')
                os.makedirs(mask_dir, exist_ok=True)
                os.makedirs(overlay_dir, exist_ok=True)

                progress = QtWidgets.QProgressDialog(self)
                progress.setWindowTitle("请稍等")
                progress.setLabelText("正在保存...")
                progress.setCancelButtonText("取消")
                progress.setMinimumDuration(5)
                progress.setWindowModality(Qt.WindowModal)
                progress.setRange(0, self.video.num_frames)

                for i in range(0, self.video.num_frames):
                    # Save mask
                    h, w = self.video_masks[i].shape
                    mask = self.video_masks[i].astype('uint8')
                    pseudo = np.zeros([h, w, 3])
                    # mask = self.controller.result_mask
                    # print(pseudo.shape, mask.shape)
                    for lab in self.controller.labelList:
                        pseudo[mask == lab.idx, :] = lab.color[::-1]
                    cv2.imwrite(
                        os.path.join(mask_dir, '{:05d}.png'.format(i)), pseudo)

                    # Save overlay
                    overlay = overlay_davis(self.video_images[i],
                                            self.video_masks[i], self.opacity,
                                            self.controller.palette)
                    overlay = Image.fromarray(overlay)
                    overlay.save(
                        os.path.join(overlay_dir, '{:05d}.png'.format(i)))
                    progress.setValue(i)
                    if progress.wasCanceled():
                        # QMessageBox.warning(self, "提示", "保存失败")
                        break
                else:
                    progress.setValue(self.video.num_frames)
                    # QMessageBox.information(self, "提示", "保存成功")

                self.setDirty(False)
                self.statusbar.showMessage(
                    self.tr("视频帧成功保存至") + " " + savePath, 5000)
                return

        if lab_input is None:
            mask_output = self.getMask()
            s = self.controller.imgShape
        else:
            mask_output = lab_input
            s = lab_input.shape

        # BUG: 如果用了多边形标注从多边形生成mask
        # 4.1 保存灰度图
        if self.save_status["gray_scale"]:
            if self.raster is not None:
                # FIXME: when big map saved, self.raster is None,
                #        so adjust polygon can't saved in tif's mask.
                pathHead, _ = osp.splitext(savePath)
                # if self.rsSave.isChecked():
                tifPath = pathHead + "_mask.tif"
                self.raster.saveMask(mask_output, tifPath)
                if self.shpSave.isChecked():
                    shpPath = pathHead + ".shp"
                    # geocode_list = self.mask2poly(mask_output, False)
                    print(rs.save_shp(shpPath, tifPath))
            else:
                ext = osp.splitext(savePath)[1]
                cv2.imencode(ext, mask_output)[1].tofile(savePath)
                # self.labelPaths.append(savePath)

            # 4.2 保存伪彩色
        if self.save_status["pseudo_color"]:
            if self.raster is None:
                pseudoPath, ext = osp.splitext(savePath)
                pseudoPath = pseudoPath + "_pseudo" + ext
                pseudo = np.zeros([s[0], s[1], 3])
                # mask = self.controller.result_mask
                mask = mask_output
                # print(pseudo.shape, mask.shape)
                for lab in self.controller.labelList:
                    pseudo[mask == lab.idx, :] = lab.color[::-1]
                cv2.imencode(ext, pseudo)[1].tofile(pseudoPath)

        # 4.3 保存前景抠图
        if self.save_status["cutout"]:
            if self.raster is None:
                mattingPath, ext = osp.splitext(savePath)
                mattingPath = mattingPath + "_cutout" + ext
                img = np.ones([s[0], s[1], 4], dtype="uint8") * 255
                img[:, :, :3] = self.controller.image.copy()
                img[mask_output == 0] = self.cutoutBackground
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
                cv2.imencode(ext, img)[1].tofile(mattingPath)

        # 4.4 保存json
        if self.save_status["json"]:
            polygons = self.scene.polygon_items
            labels = []
            for polygon in polygons:
                l = self.controller.labelList[polygon.labelIndex - 1]
                label = {
                    "name": l.name,
                    "labelIdx": l.idx,
                    "color": l.color,
                    "points": [],
                }
                for p in polygon.scnenePoints:
                    label["points"].append(p)
                labels.append(label)
            if self.origExt:
                jsonPath = savePath + ".json"
            else:
                jsonPath = osp.splitext(savePath)[0] + ".json"
            open(jsonPath, "w", encoding="utf-8").write(json.dumps(labels))
            self.labelPaths.append(jsonPath)

        # 4.5 保存coco
        if self.save_status["coco"]:
            if not self.coco.hasImage(osp.basename(self.imagePath)):
                imgId = self.coco.addImage(
                    osp.basename(self.imagePath), s[1], s[0])
            else:
                imgId = self.coco.imgNameToId[osp.basename(self.imagePath)]
            for polygon in self.scene.polygon_items:
                points = []
                for p in polygon.scnenePoints:
                    for val in p:
                        points.append(val)

                if not polygon.coco_id:
                    annId = self.coco.addAnnotation(imgId, polygon.labelIndex,
                                                    points)
                    polygon.coco_id = annId
                else:
                    self.coco.updateAnnotation(polygon.coco_id, imgId, points)
            for lab in self.controller.labelList:
                if self.coco.hasCat(lab.idx):
                    self.coco.updateCategory(lab.idx, lab.name, lab.color)
                else:
                    self.coco.addCategory(lab.idx, lab.name, lab.color)
            saveDir = (self.outputDir
                       if self.outputDir is not None else osp.dirname(savePath))
            cocoPath = osp.join(saveDir, "annotations.json")
            open(
                cocoPath, "w",
                encoding="utf-8").write(json.dumps(self.coco.dataset))

        self.setDirty(False)
        self.statusbar.showMessage(self.tr("标签成功保存至") + " " + savePath, 5000)

    def chooseSavePath(self):
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = "Label file (%s)" % " ".join(formats)
        dlg = QtWidgets.QFileDialog(
            self,
            self.tr("保存标签文件路径"),
            osp.dirname(self.imagePath),
            filters, )
        dlg.setDefaultSuffix("png")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        if self.video_masks is not None:
            savePath = dlg.getExistingDirectory(
                self,
                self.tr("选择标签文件保存路径"),
                osp.splitext(osp.basename(self.imagePath))[0], )
            name, ext = osp.splitext(osp.basename(self.imagePath))
            savePath = osp.join(savePath, name)
        else:
            savePath, _ = dlg.getSaveFileName(
                self,
                self.tr("选择标签文件保存路径"),
                osp.splitext(osp.basename(self.imagePath))[0] + ".png", )
        return savePath

    def eximgsInit(self):
        self.gridTable.setRowCount(0)
        self.gridTable.clearContents()
        # 清零
        self.raster = None
        self.grid = None

    def setDirty(self, isDirty):
        self.isDirty = isDirty

    def changeOutputDir(self, outputDir=None):
        # 1. 弹框选择标签路径
        if outputDir is None:
            options = (QtWidgets.QFileDialog.ShowDirsOnly |
                       QtWidgets.QFileDialog.DontResolveSymlinks)
            if self.settings.value("use_qt_widget", False, type=bool):
                options = options | QtWidgets.QFileDialog.DontUseNativeDialog
            outputDir = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("选择标签保存路径") + " - " + __APPNAME__,
                self.settings.value("output_dir", "."),
                options, )
        if not osp.exists(outputDir):
            return False
        self.settings.setValue("output_dir", outputDir)
        self.outputDir = outputDir

        # 2. 加载标签
        # 2.1 如果保存coco格式，加载coco标签
        if self.save_status["coco"]:
            defaultPath = osp.join(self.outputDir, "annotations.json")
            if osp.exists(defaultPath):
                self.initCoco(defaultPath)

        # 2.2 如果保存json格式，获取所有json文件名
        if self.save_status["json"]:
            labelPaths = os.listdir(outputDir)
            labelPaths = [n for n in labelPaths if n.endswith(".json")]
            labelPaths = [osp.join(outputDir, n) for n in labelPaths]
            self.labelPaths = labelPaths

            # 加载对应的标签列表
            lab_auto_save = osp.join(self.outputDir, "autosave_label.txt")
            if osp.exists(lab_auto_save) == False:
                lab_auto_save = osp.join(self.outputDir,
                                         "label/autosave_label.txt")
            if osp.exists(lab_auto_save):
                try:
                    self.importLabelList(lab_auto_save)
                except:
                    pass
        return True

    def maskOpacityChanged(self):
        self.sldOpacity.textLab.setText(str(self.opacity))
        if not self.controller or self.controller.image is None:
            return
        for polygon in self.scene.polygon_items:
            polygon.setOpacity(self.opacity)
        self.updateImage()
        if self.video_images is not None and self.video_masks is not None:
            self.show_current_frame()

    def clickRadiusChanged(self):
        self.sldClickRadius.textLab.setText(str(self.clickRadius))
        if not self.controller or self.controller.image is None:
            return
        self.updateImage()
        if self.video_images is not None and self.video_masks is not None:
            self.show_current_frame()

    def threshChanged(self):
        self.sldThresh.textLab.setText(str(self.segThresh))
        if not self.controller or self.controller.image is None:
            return
        self.controller.prob_thresh = self.segThresh
        self.updateImage()
        if self.video_images is not None and self.video_masks is not None:
            self.show_current_frame()

    # def slideChanged(self):
    #     self.sldMISlide.textLab.setText(str(self.slideMi))
    #     if not self.controller or self.controller.image is None:
    #         return
    #     self.midx = int(self.slideMi) - 1
    #     self.miSlideSet()
    #     self.updateImage()

    def undoClick(self):
        if self.image is None:
            return
        if not self.controller:
            return
        self.controller.undoClick()
        self.updateImage()
        if not self.controller.is_incomplete_mask:
            self.setDirty(False)

    def clearAll(self):
        if not self.controller or self.controller.image is None:
            return
        self.controller.resetLastObject()
        self.updateImage()
        self.setDirty(False)

    def redoClick(self):
        if self.image is None:
            return
        if not self.controller:
            return
        self.controller.redoClick()
        self.updateImage()

    def canvasClick(self, x, y, isLeft):
        c = self.controller
        if c.image is None:
            return
        if not c.inImage(x, y):
            return
        if not c.modelSet:
            self.warn(self.tr("未选择模型", self.tr("尚未选择模型，请先在右上角选择模型")))
            return

        if self.status == self.IDILE:
            return
        currLabel = self.controller.curr_label_number
        if not currLabel or currLabel == 0:
            self.warn(self.tr("未选择当前标签"), self.tr("请先在标签列表中单击点选标签"))
            return

        self.controller.addClick(x, y, isLeft)
        self.updateImage()
        self.status = self.ANNING

    def updateImage(self, reset_canvas=False):
        if not self.controller:
            return
        image = self.controller.get_visualization(
            alpha_blend=self.opacity,
            click_radius=self.clickRadius, )
        height, width, _ = image.shape
        bytesPerLine = 3 * width
        image = QImage(image.data, width, height, bytesPerLine,
                       QImage.Format_RGB888)
        if reset_canvas:
            self.resetZoom(width, height)
        self.annImage.setPixmap(QPixmap(image))

    def update_interact_viz(self):
        height, width, channel = self.viz.shape
        bytesPerLine = 3 * width
        qImg = QImage(self.viz.data, width, height, bytesPerLine,
                      QImage.Format_RGB888)
        self.annImage.setPixmap(QPixmap(qImg))

    def viewZoomed(self, scale):
        self.scene.scale = scale
        self.scene.updatePolygonSize()

    # 界面缩放重置
    def resetZoom(self, width, height):
        # 每次加载图像前设定下当前的显示框，解决图像缩小后不在中心的问题
        self.scene.setSceneRect(0, 0, width, height)
        # 缩放清除
        self.canvas.scale(1 / self.canvas.zoom_all,
                          1 / self.canvas.zoom_all)  # 重置缩放
        self.canvas.zoom_all = 1
        # 最佳缩放
        s_eps = 0.98
        scr_cont = [
            (self.scrollArea.width() * s_eps) / width,
            (self.scrollArea.height() * s_eps) / height,
        ]
        if scr_cont[0] * height > self.scrollArea.height():
            self.canvas.zoom_all = scr_cont[1]
        else:
            self.canvas.zoom_all = scr_cont[0]
        self.canvas.scale(self.canvas.zoom_all, self.canvas.zoom_all)
        self.scene.scale = self.canvas.zoom_all

    def keyReleaseEvent(self, event):
        # print(event.key(), Qt.Key_Control)
        # 释放ctrl的时候刷新图像，对应自适应点大小在缩放后刷新
        if not self.controller or self.controller.image is None:
            return
        if event.key() == Qt.Key_Control:
            self.updateImage()

    def queueEvent(self, function):
        QtCore.QTimer.singleShot(0, function)

    def toggleOrigExt(self, dst=None):
        if dst:
            self.origExt = dst
        else:
            self.origExt = not self.origExt
        self.actions.origional_extension.setChecked(self.origExt)

    def toggleAutoSave(self, save):
        if save and not self.outputDir:
            self.changeOutputDir(None)
        if save and not self.outputDir:
            save = False
        self.actions.auto_save.setChecked(save)
        self.settings.setValue("auto_save", save)

    def toggleSave(self, type):
        self.save_status[type] = not self.save_status[type]
        if type == "coco" and self.save_status["coco"]:
            self.initCoco()
        if type == "coco":
            self.save_status["json"] = not self.save_status["coco"]
            self.actions.save_json.setChecked(self.save_status["json"])
        if type == "json":
            self.save_status["coco"] = not self.save_status["json"]
            self.actions.save_coco.setChecked(self.save_status["coco"])

    def initCoco(self, coco_path: str=None):
        if not coco_path:
            if not self.outputDir or not osp.exists(self.outputDir):
                coco_path = None
            else:
                coco_path = osp.join(self.outputDir, "annotations.json")
        else:
            if not osp.exists(coco_path):
                coco_path = None
        self.coco = COCO(coco_path)
        if self.clearLabelList():
            self.controller.labelList = util.LabelList(self.coco.dataset[
                "categories"])
            self.refreshLabelList()

    def toggleWidget(self, index=None, warn=True):
        # TODO: 输入从数字改成名字

        # 1. 改变
        if isinstance(index, int):
            self.dockStatus[index] = not self.dockStatus[index]

        # 2. 判断widget是否可以开启
        # 2.1 遥感
        if self.dockStatus[4] and not (rs.check_gdal() and rs.check_rasterio()):
            if warn:
                self.warn(
                    self.tr("无法导入GDAL或rasterio"),
                    self.tr("使用遥感工具需要安装GDAL和rasterio！"),
                    QMessageBox.Yes, )
            self.statusbar.showMessage(self.tr("打开遥感工具失败，请安装GDAL和rasterio"))
            self.dockStatus[4] = False

        # 2.2 医疗
        if self.dockStatus[5] and not med.has_sitk():
            if warn:
                self.warn(
                    self.tr("无法导入SimpleITK"),
                    self.tr("使用医疗工具需要安装SimpleITK！"),
                    QMessageBox.Yes, )
            self.statusbar.showMessage(self.tr("打开医疗工具失败，请安装SimpleITK"))
            self.dockStatus[5] = False

        # 2.3 3D显示
        if self.dockStatus[9] and not self.vtkWidget.convert_vtk():
            if warn:
                self.warn(
                    self.tr("无法导入VTK"),
                    self.tr("使用3D显示工具需要安装VTK！"),
                    QMessageBox.Yes, )
            self.statusbar.showMessage(self.tr("打开3D显示工具失败，请安装VTK"))
            self.dockStatus[9] = False

        widgets = list(self.dockWidgets.values())
        for idx, s in enumerate(self.dockStatus):
            self.menus.showMenu[idx].setChecked(s)
            if s:
                for w in widgets[idx]:
                    w.show()
            else:
                for w in widgets[idx]:
                    w.hide()

        self.settings.setValue("dock_status", self.dockStatus)
        # self.display_dockwidget[index] = bool(self.display_dockwidget[index] - 1)
        # self.toggleDockWidgets()
        self.saveLayout()

    # def toggleDockWidgets(self, is_init=False):
    #     if is_init == True:
    #         if self.dockStatus != []:
    #             if len(self.dockStatus) != len(self.menus.showMenu):
    #                 self.settings.remove("dock_status")
    #             else:
    #                 self.display_dockwidget = [strtobool(w) for w in self.dockStatus]
    #         for i in range(len(self.menus.showMenu)):
    #             self.menus.showMenu[i].setChecked(bool(self.display_dockwidget[i]))
    #     else:
    #         self.settings.setValue("dock_status", self.display_dockwidget)
    #     for t, w in zip(self.display_dockwidget, self.dockWidgets.values()):
    #         if t == True:
    #             w.show()
    #         else:
    #             w.hide()

    def rsBandSet(self, idx):
        if self.raster is None:
            return
        for i in range(len(self.bandCombos)):
            self.rsRGB[i] = self.bandCombos[i].currentIndex() + 1  # 从1开始
        self.raster.setBand(self.rsRGB)
        if self.grid is not None:
            if isinstance(self.grid.curr_idx, (list, tuple)):
                row, col = self.grid.curr_idx
                image, _ = self.raster.getGrid(row, col)
            else:
                image, _ = self.raster.getArray()
        else:
            image, _ = self.raster.getArray()
        self.image = image
        self.controller.image = image
        self.updateImage()

    # def miSlideSet(self):
    #     image = rs.slice_img(self.controller.rawImage, self.midx)
    #     self.test_show(image)

    # def changeWorkerShow(self, index):
    #     self.display_dockwidget[index] = bool(self.display_dockwidget[index] - 1)
    #     self.toggleDockWidgets()

    def updateBandList(self, clean=False):
        if clean:
            for i in range(len(self.bandCombos)):
                try:  # 避免打开jpg后再打开tif报错
                    self.bandCombos[i].currentIndexChanged.disconnect()
                except TypeError:
                    pass
                self.bandCombos[i].clear()
                self.bandCombos[i].addItems(["band_1"])
            return
        bands = self.raster.geoinfo.count
        for i in range(len(self.bandCombos)):
            try:  # 避免打开jpg后再打开tif报错
                self.bandCombos[i].currentIndexChanged.disconnect()
            except TypeError:
                pass
            self.bandCombos[i].clear()
            self.bandCombos[i].addItems(
                [("band_" + str(j + 1)) for j in range(bands)])
            try:
                self.bandCombos[i].setCurrentIndex(self.rsRGB[i] - 1)
            except IndexError:
                pass
        for bandCombo in self.bandCombos:
            bandCombo.currentIndexChanged.connect(self.rsBandSet)  # 设置波段

    # def updateSlideSld(self, clean=False):
    #     if clean:
    #         self.sldMISlide.setMaximum(1)
    #         return
    #     C = self.controller.rawImage.shape[-1] if len(self.controller.rawImage.shape) == 3 else 1
    #     self.sldMISlide.setMaximum(C)

    def toggleLargestCC(self, on):
        try:
            self.controller.filterLargestCC(on)
        except:
            pass

    # 宫格标注
    def initGrid(self):
        self.delAllPolygon()
        grid_row_count, grid_col_count = self.grid.createGrids()
        self.gridTable.setRowCount(grid_row_count)
        self.gridTable.setColumnCount(grid_col_count)
        for r in range(grid_row_count):
            for c in range(grid_col_count):
                self.gridTable.setItem(r, c, QtWidgets.QTableWidgetItem())
                self.gridTable.item(r, c).setBackground(self.GRID_COLOR["idle"])
                self.gridTable.item(r, c).setFlags(
                    Qt.ItemIsSelectable)  # 无法高亮选择
        # 初始显示第一个
        self.grid.curr_idx = (0, 0)
        self.gridTable.item(0, 0).setBackground(self.GRID_COLOR["overlying"])
        # 事件注册
        self.gridTable.cellClicked.connect(self.changeGrid)

    def changeGrid(self, row, col):
        # 清除未保存的切换
        # TODO: 这块应该通过dirty判断?
        if self.grid.curr_idx is not None:
            self.saveGrid()  # 切换时自动保存上一块
            last_r, last_c = self.grid.curr_idx
            if self.grid.mask_grids[last_r][last_c] is None:
                self.gridTable.item(
                    last_r, last_c).setBackground(self.GRID_COLOR["idle"])
            else:
                self.gridTable.item(
                    last_r, last_c).setBackground(self.GRID_COLOR["finised"])
        self.delAllPolygon()
        image, mask = self.grid.getGrid(row, col)
        self.controller.setImage(image)
        self.grid.curr_idx = (row, col)
        if mask is None:
            self.gridTable.item(row,
                                col).setBackground(self.GRID_COLOR["current"])
        else:
            self.gridTable.item(row,
                                col).setBackground(self.GRID_COLOR["overlying"])
            self.mask2poly(mask)
        # 刷新
        self.updateImage(True)

    def mask2poly(self, mask, show=True):
        labs = np.unique(mask)[1:]
        colors = []
        for i in range(len(labs)):
            idx = int(labs[i]) - 1
            if idx < len(self.controller.labelList):
                c = self.controller.labelList[idx].color
            else:
                if self.currLabelIdx != -1:
                    c = self.controller.labelList[self.currLabelIdx].color
                else:
                    c = None
            colors.append(c)
        geocode_list = []
        for idx, (l, c) in enumerate(zip(labs, colors)):
            if c is not None:
                curr_polygon = util.get_polygon(
                    ((mask == l).astype(np.uint8) * 255),
                    building=self.boundaryRegular.isChecked(), )
                if show == True:
                    self.createPoly(curr_polygon, c)
                    for p in self.scene.polygon_items:
                        p.setAnning(isAnning=False)
                else:
                    for g in curr_polygon:
                        points = [gi.tolist() for gi in g]
                        geocode_list.append({
                            "name": self.controller.labelList[idx].name,
                            "points": points,
                        })
        return geocode_list

    def saveGrid(self):
        row, col = self.grid.curr_idx
        if self.grid.curr_idx is None:
            return
        self.gridTable.item(row,
                            col).setBackground(self.GRID_COLOR["overlying"])
        # if len(np.unique(self.grid.mask_grids[row][col])) == 1:
        self.grid.mask_grids[row][col] = np.array(self.getMask())
        if self.cheSaveEvery.isChecked():
            if self.outputDir is None:
                self.changeOutputDir()
            _, fullflname = osp.split(self.listFiles.currentItem().text())
            fname, _ = os.path.splitext(fullflname)
            path = osp.join(
                self.outputDir,
                (fname + "_data_" + str(row) + "_" + str(col) + ".tif"))
            im, tf = self.raster.getGrid(row, col)
            h, w = im.shape[:2]
            geoinfo = edict()
            geoinfo.xsize = w
            geoinfo.ysize = h
            geoinfo.dtype = self.raster.geoinfo.dtype
            geoinfo.crs = self.raster.geoinfo.crs
            geoinfo.geotf = tf
            self.raster.saveMask(self.grid.mask_grids[row][col],
                                 path.replace("data", "mask"),
                                 geoinfo)  # 保存mask
            self.raster.saveMask(im, path, geoinfo, 3)  # 保存图像

    def turnGrid(self, delta):
        # 切换下一个宫格
        r, c = self.grid.curr_idx if self.grid.curr_idx is not None else (0, -1)
        c += delta
        if c >= self.grid.grid_count[1]:
            c = 0
            r += 1
            if r >= self.grid.grid_count[0]:
                r = 0
        if c < 0:
            c = self.grid.grid_count[1] - 1
            r -= 1
            if r < 0:
                r = self.grid.grid_count[0] - 1
        self.changeGrid(r, c)

    def closeGrid(self):
        self.grid = None
        self.gridTable.setRowCount(0)
        self.gridTable.clearContents()

    def saveGridLabel(self):
        if self.outputDir is not None:
            name, ext = osp.splitext(osp.basename(self.imagePath))
            if not self.origExt:
                ext = ".png"
            save_path = osp.join(self.outputDir, name + ext)
        else:
            save_path = self.chooseSavePath()
            if save_path == "":
                return
        try:
            self.finishObject()
            self.saveGrid()  # 先保存当前
        except:
            pass
        self.delAllPolygon()  # 清理
        mask = self.grid.splicingList(save_path)
        if self.grid.__class__.__name__ == "RSGrids":
            self.image, is_big = self.raster.getArray()
        else:
            self.image = self.grid.detimg
            is_big = checkOpenGrid(self.image, self.thumbnail_min)
        if is_big is None:
            self.statusbar.showMessage(self.tr("图像过大，已显示缩略图"))
        self.controller.image = self.image
        self.controller._result_mask = mask
        self.exportLabel(savePath=save_path, lab_input=mask)
        # -- RS Show polygon demo --
        if self.show_rs_poly.isChecked():
            h, w = self.image.shape[:2]
            th_mask = cv2.resize(
                mask, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            indexs = np.unique(th_mask)[1:]
            for i in indexs:
                i_mask = np.zeros_like(th_mask, dtype="uint8")
                i_mask[th_mask == i] = 255
                curr_polygon = util.get_polygon(i_mask)
                color = self.controller.labelList[i - 1].color
                self.createPoly(curr_polygon, color)
                for p in self.scene.polygon_items:
                    p.setAnning(isAnning=False)
        # -- RS Show polygon demo --
        # 刷新
        grid_row_count = self.gridTable.rowCount()
        grid_col_count = self.gridTable.colorCount()
        for r in range(grid_row_count):
            for c in range(grid_col_count):
                try:
                    self.gridTable.item(
                        r, c).setBackground(self.GRID_COLOR["idle"])
                except:
                    pass
        self.raster = None
        self.closeGrid()
        self.updateBandList(True)
        self.controller.setImage(self.image)
        self.updateImage(True)
        self.setDirty(False)

    @property
    def opacity(self):
        return self.sldOpacity.value() / 100

    @property
    def clickRadius(self):
        return self.sldClickRadius.value()

    @property
    def segThresh(self):
        return self.sldThresh.value() / 100

    # @property
    # def slideMi(self):
    #     return self.sldMISlide.value()

    def warnException(self, e):
        e = str(e)
        title = e.split("。")[0]
        self.warn(title, e)

    def warn(self, title, text, buttons=QMessageBox.Yes):
        msg = QMessageBox()
        # msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(buttons)
        return msg.exec_()

    @property
    def status(self):
        # TODO: 图片，模型
        if not self.controller:
            return self.IDILE
        c = self.controller
        if c.model is None or c.image is None:
            return self.IDILE
        if self._anning:
            return self.ANNING
        return self.EDITING

    @status.setter
    def status(self, status):
        if status not in [self.ANNING, self.EDITING]:
            return
        if status == self.ANNING:
            self._anning = True
        else:
            self._anning = False

    def loadGrid(self, img, is_rs=True):
        res = self.warn(self.tr("图像过大"), self.tr("图像过大，将启用宫格功能！"), \
                        buttons=QMessageBox.Yes | QMessageBox.No)
        if res == QMessageBox.Yes:
            # 打开宫格功能
            if self.dockWidgets["grid"][0].isVisible() is False:
                # TODO: 改成self.dockStatus
                self.menus.showMenu[-1].setChecked(True)
                # self.display_dockwidget[-1] = True
                self.dockWidgets["grid"][0].show()
            self.grid = RSGrids(img) if is_rs else Grids(img)
            self.initGrid()
            return True
        return False

    # 界面布局
    def loadLayout(self):
        self.restoreState(self.layoutStatus)
        # TODO: 这里检查环境，判断是不是开医疗和遥感widget

    def saveLayout(self):
        # 保存界面
        self.settings.setValue("layout_status", QByteArray(self.saveState()))
        self.settings.setValue(
            "save_status",
            [(k, self.save_status[k]) for k in self.save_status.keys()])
        # # 如果设置了保存路径，把标签也保存下
        # if self.outputDir is not None and len(self.controller.labelList) != 0:
        #     self.exportLabelList(osp.join(self.outputDir, "autosave_label.txt"))

    def closeEvent(self, event):
        self.saveImage()
        self.saveLayout()
        QCoreApplication.quit()
        # sys.exit(0)

    def reportBug(self):
        webbrowser.open(
            "https://github.com/PaddleCV-SIG/EISeg/issues/new/choose")

    def quickStart(self):
        # self.saveImage(True)
        # self.canvas.setStyleSheet(self.note_style)
        webbrowser.open(
            "https://github.com/PaddleCV-SIG/EISeg/tree/release/0.4.0")

    def toggleLogging(self, s):
        if s:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.CRITICAL)
        self.settings.setValue("log", s)

    def toBeImplemented(self):
        self.statusbar.showMessage(self.tr("功能尚在开发"))

    # 医疗
    def wwChanged(self):
        if not self.controller or self.image is None:
            return
        try:  # 那种jpg什么格式的医疗图像调整窗宽等会造成崩溃
            self.textWw.selectAll()
            self.controller.image = med.windowlize(self.controller.rawImage,
                                                   self.ww, self.wc)
            self.updateImage()
        except:
            pass

    def wcChanged(self):
        if not self.controller or self.image is None:
            return
        try:
            self.textWc.selectAll()
            self.controller.image = med.windowlize(self.controller.rawImage,
                                                   self.ww, self.wc)
            self.updateImage()
        except:
            pass

    @property
    def ww(self):
        return int(self.textWw.text())

    @property
    def wc(self):
        return int(self.textWc.text())

    def twwChanged(self):
        if self.ww > self.sldWw.maximum():
            self.textWw.setText(str(self.sldWw.maximum()))
        if self.ww < self.sldWw.minimum():
            self.textWw.setText(str(self.sldWw.minimum()))
        self.sldWw.setProperty("value", self.ww)
        self.wwChanged()

    def swwChanged(self):
        self.textWw.setText(str(self.sldWw.value()))
        self.wwChanged()

    def twcChanged(self):
        if self.wc > self.sldWc.maximum():
            self.textWc.setText(str(self.sldWc.maximum()))
        if self.wc < self.sldWc.minimum():
            self.textWc.setText(str(self.sldWc.minimum()))
        self.sldWc.setProperty("value", self.wc)
        self.wcChanged()

    def swcChanged(self):
        self.textWc.setText(str(self.sldWc.value()))
        self.wcChanged()

    # 视频
    def tframeChanged(self):
        if self.video_images is None:
            return
        if self.video.cursur > self.sldTime.maximum():
            self.textTime.setText(str(self.sldTime.maximum()))
        if self.video.cursur < self.sldTime.minimum():
            self.textTime.setText(str(self.sldTime.minimum()))
        self.sldTime.setProperty("value", int(self.textTime.text()))

    def sframeChanged(self):
        if self.video_images is None:
            return
        self.textTime.setText(str(self.sldTime.value()))
        self.video.cursur = int(self.textTime.text())
        self.controller.setImage(self.video_images[self.video.cursur])
        self.delAllPolygon()
        self.show_current_frame()
        # print('current_frame:',self.video.cursur)

    def turnPreFrame(self):
        if self.video_images is None:
            return
        self.video.cursur -= 1
        if self.video.cursur < 0:
            self.video.cursur = self.video.num_frames - 1
        self.sldTime.setProperty("value", self.video.cursur)

    def turnNextFrame(self):
        if self.video_images is None:
            return
        self.video.cursur += 1
        if self.video.cursur > self.video.num_frames - 1:
            self.video.cursur = 0
        self.sldTime.setProperty("value", self.video.cursur)

    def show_current_frame(self):
        self.viz = overlay_davis(self.video_images[self.video.cursur],
                                 self.video_masks[self.video.cursur],
                                 self.opacity, self.controller.palette)
        self.update_interact_viz()
        self.sldTime.setProperty("value", self.video.cursur)

    def brushChanged(self):
        self.textBrush.setText(str(self.sldBrush.value()))

    def on_time(self):
        self.video.cursur += 1
        if self.video.cursur > self.video.num_frames - 1:
            self.video.cursur = 0
        self.sldTime.setProperty("value", self.video.cursur)

    def on_play(self):
        if self.video_images is None:
            self.warn(self.tr("图片格式无法播放"), self.tr("请先加载视频"))
            return
        if self.timer.isActive():
            self.timer.stop()
            self.videoPlay.setText(self.tr("播放"))
            self.videoPlay.setIcon(
                QtGui.QIcon(osp.join(pjpath, "resource/Play.png")))
        else:
            # self.delAllPolygon()
            self.timer.start(1000 / self.ratio)
            self.videoPlay.setText(self.tr("暂停"))
            self.videoPlay.setIcon(
                QtGui.QIcon(osp.join(pjpath, "resource/Stop.png")))

    def getVideoMask(self):
        if self.video_masks is not None:
            return self.video_masks[self.video.cursur]
        else:
            return None

    def on_propgation(self):
        self.finishObject()
        if self.video_images is None:
            self.warn(self.tr("未加载视频"), self.tr("请先在加载图像按钮中加载视频"))
            return
        if self.video.prop_net_segm is None:
            self.warn(self.tr("传播模型未加载"), self.tr("尚未加载视频传播模型，请先加载模型!"))
            return
        if self.video.fuse_net is None:
            self.warn(self.tr("融合模型未加载"), self.tr("尚未加载视频融合模型，请先加载模型!"))
            return

        current_mask = self.getMask()
        if current_mask is None:
            self.warn(self.tr("未提供传播参考帧"), self.tr("请先在标注传播参考帧再进行传播"))
            return
        if current_mask.max() == 0:
            current_mask = self.video_masks[self.video.cursur]
            # self.warn(self.tr("未新增标注"), self.tr("请先添加新标注再进行传播"))
            # return
        print('-------------start propgation----------------')
        self.statusbar.showMessage(self.tr("开始传播"))
        # set object
        self.video.set_objects(int(max(self.video.k, current_mask.max())))
        self.video.set_images(self.video_images)
        one_hot_mask = F.one_hot(
            paddle.to_tensor(current_mask).astype('int32'),
            int(self.video.k + 1))
        self.one_hot_mask = one_hot_mask.transpose([2, 0, 1]).unsqueeze(1)

        start = time.time()
        self.video_masks = self.video.interact(
            self.one_hot_mask, self.video.cursur, self.progress_total_cb,
            self.progress_step_cb)
        end = time.time()
        print("propagation time cost", end - start)
        self.statusbar.showMessage(self.tr("传播完成!"), 5000)
        # 传播进度条重置
        self.proPropagete.setValue(0)
        self.proPropagete.setFormat('0%')
        self.delAllPolygon()
        self.show_current_frame()
        # 3d显示
        color_map = []
        for lab in self.controller.labelList:
            color_map.append(lab.color)
        if self.TDDock.isVisible():
            self.vtkWidget.show_array(
                np.uint8(self.video_masks), (1., 1., 1.), color_map)

    def progress_step_cb(self):
        self.progress_num += 1
        ratio = self.progress_num / self.progress_max
        self.proPropagete.setValue(int(ratio * 100))
        self.proPropagete.setFormat('%2.1f%%' % (ratio * 100))
        QApplication.processEvents()

    def progress_total_cb(self, total):
        self.progress_max = total
        self.progress_num = -1
        self.progress_step_cb()

    def useQtWidget(self, s):
        self.settings.setValue("use_qt_widget", s)

    def checkLabel(self, labelIndex):
        for p in self.scene.polygon_items:
            if p.labelIndex == labelIndex:
                return False
        return True
