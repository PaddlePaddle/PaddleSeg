import os
import os.path as osp
from functools import partial
import sys
import json
from distutils.util import strtobool

from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtWidgets import QMainWindow, QMessageBox, QTableWidgetItem
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtCore import Qt, QByteArray, QVariant
import cv2
import numpy as np

from eiseg import pjpath, __APPNAME__
from models import ModelsNick
from widget import ShortcutWindow, PolygonAnnotation
from controller import InteractiveController
from ui import Ui_EISeg
import util
from util import MODELS, COCO


class APP_EISeg(QMainWindow, Ui_EISeg):
    IDILE, ANNING, EDITING = 0, 1, 2
    # IDILE：网络，权重，图像三者任一没有加载
    # EDITING：多边形编辑，可以交互式，但是多边形内部不能点
    # ANNING：交互式标注，只能交互式，不能编辑多边形，多边形不接hover

    def __init__(self, parent=None):
        super(APP_EISeg, self).__init__(parent)

        self.settings = QtCore.QSettings(
            osp.join(pjpath, "config/setting.ini"), QtCore.QSettings.IniFormat
        )

        # 初始化界面
        self.setupUi(self)

        # app变量
        self.anning = False
        self.save_status = {
            "gray_scale": True,
            "pseudo_color": True,
            "json": False,
            "coco": True,
            "foreground": True,
        }  # 是否保存这几个格式

        self.image = None  # 可能先加载图片后加载模型，只用于暂存图片
        self.controller = InteractiveController(
            # self.updateImage,
            predictor_params={
                "brs_mode": "NoBRS",
                "zoom_in_params": {
                    "skip_clicks": -1,
                    "target_size": (400, 400),
                    "expansion_ratio": 1.4,
                },
                "predictor_params": {"net_clicks_limit": None, "max_size": 800},
            },
            prob_thresh=self.segThresh,
        )
        self.controller.setModel(MODELS[0].__name__)
        # self.controller.labelList = util.LabelList()  # 标签列表
        self.outputDir = None  # 标签保存路径
        self.labelPaths = []  # 所有outputdir中的标签文件路径
        self.imagePaths = []  # 文件夹下所有待标注图片路径
        self.currIdx = 0  # 文件夹标注当前图片下标
        self.isDirty = False  # 是否需要保存
        self.origExt = False  # 是否使用图片本身拓展名，防止重名覆盖
        self.coco = COCO()
        self.colorMap = util.colorMap
        self.mattingBackground = [0, 0, 128]

        self.rsRGB = [0, 0, 0]  # 遥感RGB索引
        self.midx = 0  # 医疗切片索引
        self.rawimg = None
        self.imagesGrid = []  # 图像宫格
        # worker
        self.display_dockwidget = [True, True, True, True]
        self.dock_widgets = [
            self.ModelDock,
            self.DataDock,
            self.LabelDock,
            self.ShowSetDock
        ]
        self.config = util.parse_configs(osp.join(pjpath, "config/config.yaml"))
        self.recentModels = self.settings.value(
            "recent_models", QVariant([]), type=list
        )
        self.recentFiles = self.settings.value("recent_files", QVariant([]), type=list)
        self.dockStatus = self.settings.value("dock_status", QVariant([]), type=list)
        self.saveStatus = self.settings.value("save_status", QVariant([]), type=list)
        self.layoutStatus = self.settings.value("layout_status", QByteArray())
        self.mattingColor = self.settings.value(
            "matting_color", QVariant([]), type=list
        )

        # 初始化action
        self.initActions()

        # 更新近期记录
        self.toggleDockWidgets(True)
        self.updateModelsMenu()
        self.updateRecentFile()
        self.loadLayout()

        # 窗口
        ## 快捷键
        self.shortcutWindow = ShortcutWindow(self.actions, pjpath)

        ## 画布
        self.scene.clickRequest.connect(self.canvasClick)
        self.canvas.zoomRequest.connect(self.viewZoomed)
        self.annImage = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.annImage)

        ## 按钮点击
        self.btnSave.clicked.connect(self.saveLabel)  # 保存
        self.listFiles.itemDoubleClicked.connect(self.imageListClicked)  # 标签列表点击
        self.comboModelSelect.currentIndexChanged.connect(self.changeModel)  # 模型选择
        self.btnAddClass.clicked.connect(self.addLabel)
        self.btnParamsSelect.clicked.connect(self.changeParam)  # 模型参数选择

        ## 滑动
        self.sldOpacity.valueChanged.connect(self.maskOpacityChanged)
        self.sldClickRadius.valueChanged.connect(self.clickRadiusChanged)
        self.sldThresh.valueChanged.connect(self.threshChanged)

        ## 标签列表点击
        self.labelListTable.cellDoubleClicked.connect(self.labelListDoubleClick)
        self.labelListTable.cellClicked.connect(self.labelListClicked)
        self.labelListTable.cellChanged.connect(self.labelListItemChanged)

        ## 功能区选择
        # self.rsShow.currentIndexChanged.connect(self.rsShowModeChange)  # 显示模型

    def initActions(self):
        tr = partial(QtCore.QCoreApplication.translate, "APP_EISeg")
        action = partial(util.newAction, self)
        self.actions = util.struct()
        start = dir()

        # load status
        if self.saveStatus != []:
            for sv in self.saveStatus:
                self.save_status[sv[0]] = sv[1]

        edit_shortcuts = action(
            tr("&编辑快捷键"),
            self.editShortcut,
            "edit_shortcuts",
            "Shortcut",
            tr("编辑软件快捷键"),
        )
        turn_prev = action(
            tr("&上一张"),
            partial(self.turnImg, -1),
            "turn_prev",
            "Prev",
            tr("翻到上一张图片"),
        )
        turn_next = action(
            tr("&下一张"),
            partial(self.turnImg, 1),
            "turn_next",
            "Next",
            tr("翻到下一张图片"),
        )
        open_image = action(
            tr("&打开图像"),
            self.openImage,
            "open_image",
            "OpenImage",
            tr("打开一张图像进行标注"),
        )
        open_folder = action(
            tr("&打开文件夹"),
            self.openFolder,
            "open_folder",
            "OpenFolder",
            tr("打开一个文件夹下所有的图像进行标注"),
        )
        change_output_dir = action(
            tr("&改变标签保存路径"),
            partial(self.changeOutputDir, None),
            "change_output_dir",
            "ChangeLabelPath",
            tr("改变标签保存的文件夹路径"),
        )
        load_param = action(
            tr("&加载模型参数"),
            self.changeParam,
            "load_param",
            "Model",
            tr("加载一个模型参数"),
        )
        finish_object = action(
            tr("&完成当前目标"),
            self.finishObject,
            "finish_object",
            "Ok",
            tr("完成当前目标的标注"),
        )
        clear = action(
            tr("&清除所有标注"),
            self.undoAll,
            "clear",
            "Clear",
            tr("清除所有标注信息"),
        )
        undo = action(
            tr("&撤销"),
            self.undoClick,
            "undo",
            "Undo",
            tr("撤销一次点击"),
        )
        redo = action(
            tr("&重做"),
            self.redoClick,
            "redo",
            "Redo",
            tr("重做一次点击"),
        )
        save = action(
            tr("&保存"),
            self.saveLabel,
            "save",
            "Save",
            tr("保存图像标签"),
        )
        save_as = action(
            tr("&另存为"),
            partial(self.saveLabel, saveAs=True),
            "save_as",
            "OtherSave",
            tr("指定标签保存路径"),
        )
        auto_save = action(
            tr("&自动保存"),
            self.toggleAutoSave,
            "auto_save",
            "AutoSave",
            tr("翻页同时自动保存"),
            checkable=True,
        )
        # auto_save.setChecked(self.config.get("auto_save", False))
        del_active_polygon = action(
            tr("&删除多边形"),
            self.delActivePolygon,
            "del_active_polygon",
            "RemovePolygon",
            tr("删除当前选中的多边形"),
        )
        largest_component = action(
            tr("&保留最大连通块"),
            self.toggleLargestCC,
            "largest_component",
            "SaveMaxPolygon",
            tr("保留最大的连通块"),
            checkable=True,
        )
        origional_extension = action(
            tr("&标签和图像使用相同拓展名"),
            self.toggleOrigExt,
            "origional_extension",
            "Same",
            tr("标签和图像使用相同拓展名，用于图像中有文件名相同，拓展名不同的情况"),
            checkable=True,
        )
        save_pseudo = action(
            tr("&伪彩色保存"),
            partial(self.toggleSave, "pseudo_color"),
            "save_pseudo",
            "SavePseudoColor",
            tr("保存为伪彩色图像"),
            checkable=True,
        )
        save_pseudo.setChecked(self.save_status["pseudo_color"])
        save_grayscale = action(
            tr("&灰度保存"),
            partial(self.toggleSave, "gray_scale"),
            "save_pseudo",
            "SaveGrayScale",
            tr("保存为灰度图像，像素的灰度为对应类型的标签"),
            checkable=True,
        )
        save_grayscale.setChecked(self.save_status["gray_scale"])
        save_json = action(
            tr("&JSON保存"),
            partial(self.toggleSave, "json"),
            "save_json",
            "SaveJson",
            tr("保存为JSON格式"),
            checkable=True,
        )
        save_json.setChecked(self.save_status["json"])
        save_coco = action(
            tr("&COCO保存"),
            partial(self.toggleSave, "coco"),
            "save_coco",
            "SaveCOCO",
            tr("保存为COCO格式"),
            checkable=True,
        )
        save_coco.setChecked(self.save_status["coco"])
        close = action(
            tr("&关闭"),
            partial(self.saveImage, True),
            "close",
            "End",
            tr("关闭当前图像"),
        )
        save_matting = action(
            tr("&抠图保存"),
            partial(self.toggleSave, "foreground"),
            "save_matting",
            "SaveMatting",
            tr("只保留前景，背景设置为背景色"),
            checkable=True,
        )
        save_matting.setChecked(self.save_status["foreground"])
        set_matting_background = action(
            tr("&设置抠图背景色"),
            self.setMattingBackground,
            "set_matting_background",
            self.mattingBackground,
            tr("抠图后背景像素的颜色"),
        )
        quit = action(
            tr("&退出"),
            self.close,
            "quit",
            "Close",
            tr("退出软件"),
        )
        save_label = action(
            tr("&保存标签列表"),
            partial(self.saveLabelList, None),
            "save_label",
            "ExportLabel",
            tr("将标签保存成标签配置文件"),
        )
        load_label = action(
            tr("&加载标签列表"),
            partial(self.loadLabelList, None),
            "load_label",
            "ImportLabel",
            tr("从标签配置文件中加载标签"),
        )
        clear_label = action(
            tr("&清空标签列表"),
            self.clearLabelList,
            "clear_label",
            "ClearLabel",
            tr("清空所有的标签"),
        )
        clear_recent = action(
            tr("&清除标注记录"),
            self.clearRecentFile,
            "clear_recent",
            "ClearRecent",
            tr("清除近期标注记录"),
        )
        model_worker = action(
            tr("&模型选择"),
            partial(self.changeWorkerShow, 0),
            "model_worker",
            "Net",
            tr("模型选择"),
            checkable=True,
        )
        data_worker = action(
            tr("&数据列表"),
            partial(self.changeWorkerShow, 1),
            "data_worker",
            "Data",
            tr("数据列表"),
            checkable=True,
        )
        label_worker = action(
            tr("&标签列表"),
            partial(self.changeWorkerShow, 2),
            "label_worker",
            "Label",
            tr("标签列表"),
            checkable=True,
        )
        set_worker = action(
            tr("&分割设置"),
            partial(self.changeWorkerShow, 3),
            "set_worker",
            "Setting",
            tr("分割设置"),
            checkable=True,
        )
        for name in dir():
            if name not in start:
                self.actions.append(eval(name))
        recent_files = QtWidgets.QMenu(tr("近期文件"))
        recent_files.setIcon(util.newIcon("Data"))
        recent_files.aboutToShow.connect(self.updateRecentFile)
        recent_params = QtWidgets.QMenu(tr("近期模型及参数"))
        recent_params.setIcon(util.newIcon("Net"))
        recent_params.aboutToShow.connect(self.updateModelsMenu)
        languages = QtWidgets.QMenu(tr("语言"))
        languages.setIcon(util.newIcon("Language"))
        languages.aboutToShow.connect(self.updateLanguage)

        self.menus = util.struct(
            recent_files=recent_files,
            recent_params=recent_params,
            languages=languages,
            fileMenu=(
                open_image,
                open_folder,
                change_output_dir,
                load_param,
                clear_recent,
                recent_files,
                recent_params,
                None,
                save,
                save_as,
                auto_save,
                None,
                turn_next,
                turn_prev,
                close,
                None,
                quit,
            ),
            labelMenu=(
                save_label,
                load_label,
                clear_label,
            ),
            workMenu=(
                largest_component,
                del_active_polygon,
                None,
                origional_extension,
                save_pseudo,
                save_grayscale,
                save_matting,
                set_matting_background,
                None,
                save_json,
                save_coco,
            ),
            showMenu=(
                model_worker,
                data_worker,
                label_worker,
                set_worker,
            ),
            helpMenu=(languages, edit_shortcuts),
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
                save_matting,
                save_json,
                save_coco,
                origional_extension,
                None,
                largest_component,
            ),
        )

        def menu(title, actions=None):
            menu = self.menuBar().addMenu(title)
            if actions:
                util.addActions(menu, actions)
            return menu

        menu(tr("文件"), self.menus.fileMenu)
        menu(tr("标注"), self.menus.labelMenu)
        menu(tr("功能"), self.menus.workMenu)
        menu(tr("显示"), self.menus.showMenu)
        menu(tr("帮助"), self.menus.helpMenu)
        util.addActions(self.toolBar, self.menus.toolBar)
        # foreground backgroud
        if self.settings.value("matting_color"):
            self.mattingBackground = [
                int(c) for c in self.settings.value("matting_color")
            ]
            self.actions.set_matting_background.setIcon(
                util.newIcon(self.mattingBackground)
            )

    def setMattingBackground(self):
        c = self.mattingBackground
        # 添加alpha可选择
        if len(c) == 3:  # RBG保存的ini避免报错，后期可以取消
            c += tuple([255])
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(c[0], c[1], c[2], c[3]), self, 
                                                options=QtWidgets.QColorDialog.ShowAlphaChannel)
        self.mattingBackground = color.getRgb()
        self.settings.setValue(
            "matting_color", [int(c) for c in self.mattingBackground]
        )
        self.actions.set_matting_background.setIcon(
            util.newIcon(self.mattingBackground)
        )

    def editShortcut(self):
        self.shortcutWindow.center()
        self.shortcutWindow.show()

    # 多语言
    def updateLanguage(self):
        self.menus.languages.clear()
        langs = os.listdir(osp.join(pjpath, "util/translate"))
        langs = [n.split(".")[0] for n in langs if n.endswith("qm")]
        langs.append("中文")
        for lang in langs:
            icon = util.newIcon(lang)
            action = QtWidgets.QAction(icon, lang, self)
            action.triggered.connect(partial(self.changeLanguage, lang))
            self.menus.languages.addAction(action)

    def changeLanguage(self, lang):
        self.settings.setValue("language", lang)
        self.warn(self.tr("切换语言"), self.tr("切换语言需要重启软件才能生效"))

    # 近期图像
    def updateRecentFile(self):
        menu = self.menus.recent_files
        menu.clear()
        recentFiles = self.settings.value("recent_files", QVariant([]), type=list)
        files = [f for f in recentFiles if osp.exists(f)]
        for i, f in enumerate(files):
            icon = util.newIcon("File")
            action = QtWidgets.QAction(
                icon, "&【%d】 %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(partial(self.openRecentImage, f))
            menu.addAction(action)
        if len(files) == 0:
            menu.addAction(self.tr("无近期文件"))
        self.settings.setValue("recent_files", files)

    def addRecentFile(self, path):
        path = osp.normcase(path)
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
    def updateModelsMenu(self):
        menu = self.menus.recent_params
        menu.clear()

        self.recentModels = [
            m for m in self.recentModels if osp.exists(m["param_path"])
        ]
        for idx, m in enumerate(self.recentModels):
            icon = util.newIcon("Model")
            action = QtWidgets.QAction(
                icon,
                f"&【{m['model_name']}】 {osp.basename(m['param_path'])}",
                self,
            )
            action.triggered.connect(
                partial(self.setModelParam, m["model_name"], m["param_path"])
            )
            menu.addAction(action)
        if len(self.recentModels) == 0:
            menu.addAction(self.tr("无近期模型记录"))
        self.settings.setValue("recent_params", self.recentModels)

    def setModelParam(self, modelName, paramPath):
        if self.changeModel(ModelsNick[modelName][1]):
            self.comboModelSelect.setCurrentText(self.tr(ModelsNick[modelName][0]))  # 更改显示
            res = self.changeParam(paramPath)
            if res:
                return True
        return False

    def changeModel(self, idx: int or str):
        success, res = self.controller.setModel(MODELS[idx].__name__)
        if not success:
            self.warnException(res)
            return False
        return True

    def changeParam(self, param_path: str = None):
        if not self.controller.modelSet:
            self.warn(self.tr("选择模型结构"), self.tr("尚未选择模型结构，请在右侧下拉菜单进行选择！"))
            return
        if not param_path:
            filters = self.tr("Paddle模型权重文件(*.pdparams)")
            start_path = (
                "."
                if len(self.recentModels) == 0
                else osp.dirname(self.recentModels[-1]["param_path"])
            )
            param_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("选择模型参数") + " - " + __APPNAME__,
                start_path,
                filters,
            )
        if not param_path:
            return False

        success, res = self.controller.setParam(param_path)
        if success:
            model_dict = {
                "param_path": osp.normcase(param_path),
                "model_name": self.controller.modelName,
            }
            if model_dict not in self.recentModels:
                self.recentModels.append(model_dict)
            else:
                # 移动位置确保自动加载的正确
                self.recentModels.remove(model_dict)
                self.recentModels.append(model_dict)
            if len(self.recentModels) > 10:
                del self.recentModels[0]
            self.settings.setValue("recent_models", self.recentModels)
            # self.status = self.ANNING
            return True
        else:
            self.warnException(res)
            return False

    def loadRecentModelParam(self):
        if len(self.recentModels) == 0:
            self.statusbar.showMessage(self.tr("没有最近使用模型信息，请加载模型"), 10000)
            return
        m = self.recentModels[-1]
        model = m["model_name"]
        param_path = m["param_path"]
        self.setModelParam(model, param_path)

    # 标签列表
    def loadLabelList(self, file_path=None):
        if file_path is None:
            filters = self.tr("标签配置文件") + " (*.txt)"
            file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self,
                self.tr("选择标签配置文件路径") + " - " + __APPNAME__,
                ".",
                filters,
            )
        file_path = osp.normcase(file_path)
        if not osp.exists(file_path):
            return
        labelJson = open(file_path, "r").read()
        self.controller.readLabel(file_path)
        self.refreshLabelList()
        self.settings.setValue("label_list_file", file_path)

    def saveLabelList(self, auto_save_path=None):
        if len(self.controller.labelList) == 0:
            self.warn(self.tr("没有需要保存的标签"), self.tr("请先添加标签之后再进行保存！"))
            return
        if auto_save_path is None:
            filters = self.tr("标签配置文件") + "(*.txt)"
            dlg = QtWidgets.QFileDialog(self, self.tr("保存标签配置文件"), ".", filters)
            dlg.setDefaultSuffix("txt")
            dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
            dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
            dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
            savePath, _ = dlg.getSaveFileName(
                self, self.tr("选择保存标签配置文件路径") + " - " + __APPNAME__, ".", filters
            )
        else:
            savePath = auto_save_path
        self.controller.saveLabel(savePath)
        if auto_save_path is None:
            self.settings.setValue("label_list_file", savePath)

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
            0, QtWidgets.QHeaderView.Fixed
        )
        self.labelListTable.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.Fixed
        )
        self.labelListTable.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.Fixed
        )
        self.labelListTable.setColumnWidth(2, 50)

    def clearLabelList(self, display=True):
        if len(self.controller.labelList) == 0:
            return True
        if display:
            res = self.warn(
                self.tr("清空标签列表?"),
                self.tr("请确认是否要清空标签列表"),
                QMessageBox.Yes | QMessageBox.Cancel,
            )
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
            color = self.controller.labelList.getLabelById(p.labelIndex).color
            p.setColor(color, color)
        self.labelListClicked(row, 0)

    @property
    def currLabelIdx(self):
        return self.controller.curr_label_number - 1

    def labelListClicked(self, row, col):
        table = self.labelListTable
        if col == 3:
            labelIdx = int(table.item(row, 0).text())
            self.controller.labelList.remove(labelIdx)
            table.removeRow(row)

        if col == 0 or col == 1:
            for cl in range(2):
                for idx in range(len(self.controller.labelList)):
                    table.item(idx, cl).setBackground(QtGui.QColor(255, 255, 255))
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

    def delActivePolygon(self):
        for idx, polygon in enumerate(self.scene.polygon_items):
            if polygon.hasFocus():
                res = self.warn(
                    self.tr("确认删除？"),
                    self.tr("确认要删除当前选中多边形标注？"),
                    QMessageBox.Yes | QMessageBox.Cancel,
                )
                if res == QMessageBox.Yes:
                    self.delPolygon(polygon)

    def delPolygon(self, polygon):
        polygon.remove()
        if self.save_status["coco"]:
            if polygon.coco_id:
                self.coco.delAnnotation(
                    polygon.coco_id,
                    self.coco.imgNameToId[osp.basename(self.imagePath)],
                )
        self.setDirty()

    def delActivePoint(self):
        for polygon in self.scene.polygon_items:
            polygon.removeFocusPoint()

    # 图片/标签 io
    def getMask(self):
        if not self.controller or self.controller.image is None:
            return
        s = self.controller.image.shape
        img = np.zeros([s[0], s[1]])
        # 覆盖顺序，从上往下
        len_lab = self.labelListTable.rowCount()
        for i in range(len_lab):
            idx = int(self.labelListTable.item(len_lab - i - 1, 0).text())
            color = self.controller.labelList.getLabelById(idx).color
            for poly in self.scene.polygon_items:
                if poly.labelIndex == idx:
                    pts = np.int32([np.array(poly.scnenePoints)])
                    cv2.fillPoly(img, pts=pts, color=idx)
        return img

    def openRecentImage(self, file_path):
        file_path = osp.normcase(file_path)
        self.saveImage(True)  # 清除
        self.queueEvent(partial(self.loadImage, file_path))
        self.listFiles.addItems([file_path])
        self.imagePaths.append(file_path)

    def openImage(self):
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        filters = "Image & Label files (%s)" % " ".join(formats)

        recentPath = self.settings.value("recent_files", [])
        if len(recentPath) == 0:
            recentPath = "."
        else:
            recentPath = osp.dirname(recentPath[-1])

        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("选择待标注图片") + " - " + __APPNAME__,
            recentPath,
            filters,
        )
        if len(file_path) == 0:
            return
        file_path = osp.normcase(file_path)
        self.saveImage(True)  # 清除
        self.queueEvent(partial(self.loadImage, file_path))
        self.listFiles.addItems([file_path])
        self.imagePaths.append(file_path)

    def openFolder(self):
        # 1. 选择文件夹
        recentPath = self.settings.value("recent_files", [])
        if len(recentPath) == 0:
            recentPath = "."
        else:
            recentPath = osp.dirname(recentPath[-1])
        self.inputDir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("选择待标注图片文件夹") + " - " + __APPNAME__,
            recentPath,
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if len(self.inputDir) == 0:
            return

        # 2. 关闭当前图片，清空文件列表
        self.saveImage(close=True)
        self.imagePaths = []
        self.listFiles.clear()

        # 3. 扫描文件夹下所有图片
        # 3.1 获取所有文件名
        imagePaths = os.listdir(self.inputDir)
        exts = QtGui.QImageReader.supportedImageFormats()
        imagePaths = [n for n in imagePaths if n.split(".")[-1] in exts]
        if len(imagePaths) == 0:
            return
        # 3.2 设置默认输出路径为文件夹下的 label 文件夹
        opd = osp.join(self.inputDir, "label")
        self.outputDir = opd
        if not osp.exists(opd):
            os.makedirs(opd)
        # 3.3 有重名标签都保留原来拓展名
        names = []
        for name in imagePaths:
            name = osp.splitext(name)[0]
            if name not in names:
                names.append(name)
            else:
                self.toggleOrigExt(True)
        imagePaths = [osp.join(self.inputDir, n) for n in imagePaths]
        for p in imagePaths:
            if p not in self.imagePaths:
                self.imagePaths.append(p)
                self.listFiles.addItem(osp.normcase(p))

        # 3.4 加载已有的标注
        if self.outputDir is not None and osp.exists(self.outputDir):
            self.changeOutputDir(self.outputDir)
        if len(self.imagePaths) != 0:
            self.currIdx = 0
            self.turnImg(0)

    def loadImage(self, path):
        if not path or not osp.exists(path):
            return
        _, ext = os.path.splitext(path)
        # 1. 读取图片
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        image = image[:, :, ::-1]  # BGR转RGB
        self.image = image
        self.controller.setImage(image)
        self.updateImage(True)

        # 2. 加载标签
        self.loadLabel(path)
        self.addRecentFile(path)
        self.imagePath = path
        # self.status = self.ANNING

    def loadLabel(self, imgPath):
        if imgPath == "":
            return None

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
                    color,
                    color,
                    self.opacity,
                )
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
                color = self.controller.labelList.getLabelById(labelIdx).color
                poly = PolygonAnnotation(
                    ann["category_id"],
                    self.controller.image.shape,
                    self.delPolygon,
                    color,
                    color,
                    self.opacity,
                    ann["id"],
                )
                self.scene.addItem(poly)
                self.scene.polygon_items.append(poly)
                for p in points:
                    poly.addPointLast(QtCore.QPointF(p[0], p[1]))

    def turnImg(self, delta):
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
        self.setClean()

    def imageListClicked(self):
        if not self.controller:
            self.warn(self.tr("模型未加载"), self.tr("尚未加载模型，请先加载模型！"))
            self.changeParam()
            if not self.controller:
                return
        if self.controller.is_incomplete_mask:
            self.saveLabel()
        toRow = self.listFiles.currentRow()
        delta = toRow - self.currIdx
        self.turnImg(delta)

    def finishObject(self):
        if not self.controller or self.image is None:
            return
        current_mask, curr_polygon = self.controller.finishObject()
        if curr_polygon is not None:
            self.updateImage()
            if current_mask is not None:
                color = self.controller.labelList[self.currLabelIdx].color
                for points in curr_polygon:
                    if len(points) < 3:
                        continue
                    poly = PolygonAnnotation(
                        self.controller.labelList[self.currLabelIdx].idx,
                        self.controller.image.shape,
                        self.delPolygon,
                        color,
                        color,
                        self.opacity,
                    )
                    poly.labelIndex = self.controller.labelList[self.currLabelIdx].idx
                    self.scene.addItem(poly)
                    self.scene.polygon_items.append(poly)
                    for p in points:
                        poly.addPointLast(QtCore.QPointF(p[0], p[1]))
                    self.setDirty()
        # 状态改变
        if self.status == self.EDITING:
            self.anning = True
            for p in self.scene.polygon_items:
                p.setAnning(isAnning=True)
        else:
            self.anning = False
            for p in self.scene.polygon_items:
                p.setAnning(isAnning=False)
        self.getMask()

    def completeLastMask(self):
        # 返回最后一个标签是否完成，false就是还有带点的
        if not self.controller or self.controller.image is None:
            return True
        if not self.controller.is_incomplete_mask:
            return True
        res = self.warn(
            self.tr("完成最后一个目标？"),
            self.tr("是否完成最后一个目标的标注，不完成不会进行保存。"),
            QMessageBox.Yes | QMessageBox.Cancel,
        )
        if res == QMessageBox.Yes:
            self.finishObject()
            self.setDirty()
            return True
        return False

    def saveImage(self, close=False):
        if self.controller and self.controller.image is not None:
            # 1. 完成正在交互式标注的标签
            self.completeLastMask()
            # 2. 进行保存
            if self.isDirty:
                if self.actions.auto_save.isChecked():
                    self.saveLabel()
                else:
                    res = self.warn(
                        self.tr("保存标签？"),
                        self.tr("标签尚未保存，是否保存标签"),
                        QMessageBox.Yes | QMessageBox.Cancel,
                    )
                    if res == QMessageBox.Yes:
                        self.saveLabel()
                self.setClean()
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

    def saveLabel(self, saveAs=False, savePath=None):
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
                    name + ext,
                )
            else:
                # 3.3 没有指定标签存到哪，或者是另存为：弹框让用户选
                formats = [
                    "*.{}".format(fmt.data().decode())
                    for fmt in QtGui.QImageReader.supportedImageFormats()
                ]
                filters = "Label file (%s)" % " ".join(formats)
                dlg = QtWidgets.QFileDialog(
                    self,
                    self.tr("保存标签文件路径"),
                    osp.dirname(self.imagePath),
                    filters,
                )
                dlg.setDefaultSuffix("png")
                dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
                dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
                dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
                savePath, _ = dlg.getSaveFileName(
                    self,
                    self.tr("选择标签文件保存路径"),
                    osp.splitext(osp.basename(self.imagePath))[0] + ".png",
                )
        if savePath is None or not osp.exists(osp.dirname(savePath)):
            return

        if savePath not in self.labelPaths:
            self.labelPaths.append(savePath)

        # 4.1 保存灰度图
        if self.save_status["gray_scale"]:
            ext = osp.splitext(savePath)[1]
            cv2.imencode(ext, self.getMask())[1].tofile(savePath)
            # self.labelPaths.append(savePath)

        # 4.2 保存伪彩色
        if self.save_status["pseudo_color"]:
            pseudoPath, ext = osp.splitext(savePath)
            pseudoPath = pseudoPath + "_pseudo" + ext
            s = self.controller.imgShape
            pseudo = np.zeros([s[1], s[0], 3])
            mask = self.getMask()
            for lab in self.controller.labelList:
                pseudo[mask == lab.idx, :] = lab.color[::-1]
            cv2.imencode(ext, pseudo)[1].tofile(pseudoPath)

        # 4.3 保存前景抠图
        if self.save_status["foreground"]:
            mattingPath, ext = osp.splitext(savePath)
            mattingPath = mattingPath + "_foreground" + ext
            h, w = self.controller.image.shape[:2]
            img = np.ones([h, w, 4], dtype="uint8") * 255
            img[:, :, :3] = self.controller.image.copy()
            # 适用以前的RGB三参数版本不报错，后面都用之后这个可以取消
            if len(self.mattingBackground) == 3:
                self.mattingBackground += tuple([255])
            img[self.getMask() == 0] = self.mattingBackground
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
                s = self.controller.imgShape
                imgId = self.coco.addImage(osp.basename(self.imagePath), s[0], s[1])
            else:
                imgId = self.coco.imgNameToId[osp.basename(self.imagePath)]
            for polygon in self.scene.polygon_items:
                points = []
                for p in polygon.scnenePoints:
                    for val in p:
                        points.append(val)

                if not polygon.coco_id:
                    annId = self.coco.addAnnotation(imgId, polygon.labelIndex, points, polygon.bbox.to_array())
                    polygon.coco_id = annId
                else:
                    self.coco.updateAnnotation(polygon.coco_id, imgId, points, polygon.bbox.to_array())
            for lab in self.controller.labelList:
                if self.coco.hasCat(lab.idx):
                    self.coco.updateCategory(lab.idx, lab.name, lab.color)
                else:
                    self.coco.addCategory(lab.idx, lab.name, lab.color)
            saveDir = (
                self.outputDir if self.outputDir is not None else osp.dirname(savePath)
            )
            cocoPath = osp.join(saveDir, "coco.json")
            open(cocoPath, "w", encoding="utf-8").write(json.dumps(self.coco.dataset))

        self.setClean()
        self.statusbar.showMessage(self.tr("标签成功保存至") + " " + savePath, 5000)

    def setClean(self):
        self.isDirty = False

    def setDirty(self):
        self.isDirty = True

    def changeOutputDir(self, outputDir=None):
        # 1. 弹框选择标签路径
        if outputDir is None:
            outputDir = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("选择标签保存路径") + " - " + __APPNAME__,
                self.settings.value("output_dir", "."),
                QtWidgets.QFileDialog.ShowDirsOnly
                | QtWidgets.QFileDialog.DontResolveSymlinks,
            )
        if len(outputDir) == 0 or not osp.exists(outputDir):
            return False
        self.settings.setValue("output_dir", outputDir)
        self.outputDir = outputDir

        # 2. 加载标签
        # 2.1 如果保存coco格式，加载coco标签
        if self.save_status["coco"]:
            self.loadCoco()

        # 2.2 如果保存json格式，获取所有json文件名
        if self.save_status["json"]:
            labelPaths = os.listdir(outputDir)
            labelPaths = [n for n in labelPaths if n.endswith(".json")]
            labelPaths = [osp.join(outputDir, n) for n in labelPaths]
            self.labelPaths = labelPaths

            # 加载对应的标签列表
            lab_auto_save = osp.join(self.outputDir, "autosave_label.txt")
            if osp.exists(lab_auto_save) == False:
                lab_auto_save = osp.join(self.outputDir, "label/autosave_label.txt")
            if osp.exists(lab_auto_save):
                try:
                    self.loadLabelList(lab_auto_save)
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

    def clickRadiusChanged(self):
        self.sldClickRadius.textLab.setText(str(self.clickRadius))
        if not self.controller or self.controller.image is None:
            return
        self.updateImage()

    def threshChanged(self):
        self.sldThresh.textLab.setText(str(self.segThresh))
        if not self.controller or self.controller.image is None:
            return
        self.controller.prob_thresh = self.segThresh
        self.updateImage()

    def slideChanged(self):
        self.sldMISlide.textLab.setText(str(self.slideMi))
        if not self.controller or self.controller.image is None:
            return
        self.midx = int(self.slideMi) - 1
        self.miSlideSet()
        self.updateImage()

    def undoClick(self):
        if self.image is None:
            return
        if not self.controller:
            return
        self.controller.undoClick()
        self.updateImage()
        if not self.controller.is_incomplete_mask:
            self.setClean()

    def undoAll(self):
        if not self.controller or self.controller.image is None:
            return
        self.controller.resetLastObject()
        self.updateImage()
        self.setClean()

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
        if not c.paramSet:
            self.warn(self.tr("未设置参数"), self.tr("尚未设置参数，请先在右上角设置参数"))
            return

        if self.status == self.IDILE:
            return
        currLabel = self.controller.curr_label_number
        if not currLabel or currLabel == 0:
            self.warn(self.tr("未选择当前标签"), self.tr("请先在标签列表中单击点选标签"))
            return

        self.controller.addClick(x, y, isLeft)
        self.updateImage()
        self.anning = True

    def updateImage(self, reset_canvas=False):
        if not self.controller:
            return
        image = self.controller.get_visualization(
            alpha_blend=self.opacity,
            click_radius=self.clickRadius,
        )
        height, width, _ = image.shape
        bytesPerLine = 3 * width
        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        if reset_canvas:
            self.resetZoom(width, height)
        self.annImage.setPixmap(QPixmap(image))

    def viewZoomed(self, scale):
        self.scene.scale = scale
        self.scene.updatePolygonSize()

    # 界面缩放重置
    def resetZoom(self, width, height):
        # 每次加载图像前设定下当前的显示框，解决图像缩小后不在中心的问题
        self.scene.setSceneRect(0, 0, width, height)
        # 缩放清除
        self.canvas.scale(1 / self.canvas.zoom_all, 1 / self.canvas.zoom_all)  # 重置缩放
        self.canvas.zoom_all = 1
        # 最佳缩放
        s_eps = 5e-2
        scr_cont = [
            self.scrollArea.width() / width - s_eps,
            self.scrollArea.height() / height - s_eps,
        ]
        if scr_cont[0] * height > self.scrollArea.height():
            self.canvas.zoom_all = scr_cont[1]
        else:
            self.canvas.zoom_all = scr_cont[0]
        self.canvas.scale(self.canvas.zoom_all, self.canvas.zoom_all)
        self.scene.scale = self.canvas.zoom_all

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
            self.loadCoco()
        if type == "coco":
            self.save_status["json"] = not self.save_status["coco"]
            self.actions.save_json.setChecked(self.save_status["json"])
        if type == "json":
            self.save_status["coco"] = not self.save_status["json"]
            self.actions.save_coco.setChecked(self.save_status["coco"])

    def loadCoco(self, coco_path=None):
        if not coco_path:
            if not self.outputDir or not osp.exists(self.outputDir):
                coco_path = None
            else:
                coco_path = osp.join(self.outputDir, "coco.json")
                # 这里放在外面判断可能会有coco_path为none，exists报错
                if not osp.exists(coco_path):
                    coco_path = None
        self.coco = COCO(coco_path)
        # 避免有coco时不重新加载标签导致报错
        display_cll = False if coco_path is not None else True
        if self.clearLabelList(display_cll):
            self.controller.labelList = util.LabelList(self.coco.dataset["categories"])
            self.refreshLabelList()

    def changeWorkerShow(self, index):
        self.display_dockwidget[index] = bool(self.display_dockwidget[index] - 1)
        self.toggleDockWidgets()


    def toggleDockWidgets(self, is_init=False):
        if is_init == True:
            if self.dockStatus != []:
                if len(self.dockStatus) != len(self.menus.showMenu):
                    self.settings.remove("dock_status")
                else:
                    self.display_dockwidget = [strtobool(w) for w in self.dockStatus]
            for i in range(len(self.menus.showMenu)):
                self.menus.showMenu[i].setChecked(bool(self.display_dockwidget[i]))
        else:
            self.settings.setValue("dock_status", self.display_dockwidget)
        for t, w in zip(self.display_dockwidget, self.dock_widgets):
            if t == True:
                w.show()
            else:
                w.hide()

    def update_bandList(self):
        bands = self.rawimg.shape[-1] if len(self.rawimg.shape) == 3 else 1
        for i in range(len(self.bandCombos)):
            self.bandCombos[i].currentIndexChanged.disconnect()
            self.bandCombos[i].clear()
            self.bandCombos[i].addItems([("band_" + str(j + 1)) for j in range(bands)])
            try:
                self.bandCombos[i].setCurrentIndex(self.rsRGB[i])
            except IndexError:
                pass
        for bandCombo in self.bandCombos:
            bandCombo.currentIndexChanged.connect(self.rsBandSet)  # 设置波段

    def toggleLargestCC(self, on):
        try:
            self.controller.filterLargestCC(on)
        except:
            pass

    @property
    def opacity(self):
        return self.sldOpacity.value() / 100

    @property
    def clickRadius(self):
        return self.sldClickRadius.value()

    @property
    def segThresh(self):
        return self.sldThresh.value() / 100

    @property
    def slideMi(self):
        return self.sldMISlide.value()

    def warnException(self, e):
        e = str(e)
        title = e.split("。")[0]
        self.warn(title, e)

    def warn(self, title, text, buttons=QMessageBox.Yes):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(buttons)
        return msg.exec_()

    @property
    def status(self):
        if not self.controller:
            return self.IDILE
        c = self.controller
        if not c.paramSet or not c.modelSet or c.image is None:
            return self.IDILE
        if self.anning:
            return self.ANNING
        return self.EDITING

    # 加载界面
    def loadLayout(self):
        self.restoreState(self.layoutStatus)

    def closeEvent(self, event):
        # 保存界面
        self.settings.setValue("layout_status", QByteArray(self.saveState()))
        self.settings.setValue(
            "save_status", [(k, self.save_status[k]) for k in self.save_status.keys()]
        )
        # 如果设置了保存路径，把标签也保存下
        if self.outputDir is not None and len(self.controller.labelList) != 0:
            self.saveLabelList(osp.join(self.outputDir, "autosave_label.txt"))
        # 关闭主窗体退出程序，子窗体也关闭
        sys.exit(0)
