import os
import os.path as osp
from functools import partial
import sys
import inspect
import warnings

from qtpy import QtGui, QtCore, QtWidgets
from qtpy.QtWidgets import QMainWindow, QMessageBox, QTableWidgetItem
from qtpy.QtGui import QImage, QPixmap
from qtpy.QtCore import Qt
import paddle
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import models
from controller import InteractiveController
from ui import Ui_EISeg, Ui_Help
from eiseg import pjpath, __APPNAME__
import util
from util.colormap import ColorMask
from util.label import Labeler
from util import MODELS


class APP_EISeg(QMainWindow, Ui_EISeg):
    def __init__(self, parent=None):
        super(APP_EISeg, self).__init__(parent)

        # 初始化界面
        self.setupUi(self)

        # app变量
        self.controller = None
        self.image = None  # 可能先加载图片后加载模型，只用于暂存图片
        self.modelClass = MODELS[0]
        self.outputDir = None  # 标签保存路径
        self.labelPaths = []  # 保存所有从outputdir发现的标签文件路径
        self.filePaths = []  # 文件夹下所有待标注图片路径
        self.currIdx = 0  # 文件夹标注当前图片下标
        self.currentPath = None
        self.isDirty = False
        self.labelList = Labeler()
        self.settings = QtCore.QSettings(
            osp.join(pjpath, "config/setting.ini"), QtCore.QSettings.IniFormat
        )
        self.config = util.parse_configs(osp.join(pjpath, "config/config.yaml"))
        self.recentModels = self.settings.value("recent_models", [])
        self.recentFiles = self.settings.value("recent_files", [])
        if not self.recentFiles:
            self.recentFiles = []
        self.maskColormap = ColorMask(osp.join(pjpath, "config/colormap.txt"))

        # 初始化action
        self.initActions()

        # 更新近期记录
        self.updateModelsMenu()
        self.updateRecentFile()

        # 帮助界面
        self.help_dialog = QtWidgets.QDialog()
        help_ui = Ui_Help()
        help_ui.setupUi(self.help_dialog)

        ## 画布部分
        self.canvas.clickRequest.connect(self.canvasClick)
        self.annImage = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.annImage)

        ## 按钮点击
        self.btnSave.clicked.connect(self.saveLabel)  # 保存
        self.listFiles.itemDoubleClicked.connect(self.listClicked)  # 标签列表点击
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
        self.labelList.readLabel(self.settings.value("label_list_file"))
        self.refreshLabelList()

    def initActions(self):
        def menu(title, actions=None):
            menu = self.menuBar().addMenu(title)
            if actions:
                util.addActions(menu, actions)
            return menu

        action = partial(util.newAction, self)
        shortcuts = self.config["shortcut"]
        turn_prev = action(
            self.tr("&上一张"),
            partial(self.turnImg, -1),
            shortcuts["turn_prev"],
            "Prev",
            self.tr("翻到上一张图片"),
        )
        turn_next = action(
            self.tr("&下一张"),
            partial(self.turnImg, 1),
            shortcuts["turn_next"],
            "Next",
            self.tr("翻到下一张图片"),
        )
        open_image = action(
            self.tr("&打开图像"),
            self.openImage,
            shortcuts["open_image"],
            "OpenImage",
            self.tr("打开一张图像进行标注"),
        )
        open_folder = action(
            self.tr("&打开文件夹"),
            self.openFolder,
            shortcuts["open_folder"],
            "OpenFolder",
            self.tr("打开一个文件夹下所有的图像进行标注"),
        )
        change_output_dir = action(
            self.tr("&改变标签保存路径"),
            self.changeOutputDir,
            shortcuts["change_output_dir"],
            "ChangeLabelPath",
            self.tr("打开一个文件夹下所有的图像进行标注"),
        )
        load_param = action(
            self.tr("&加载模型参数"),
            self.changeParam,
            shortcuts["load_param"],
            "Model",
            self.tr("加载一个模型参数"),
        )
        finish_object = action(
            self.tr("&完成当前目标"),
            self.finishObject,
            shortcuts["finish_object"],
            "Ok",
            self.tr("完成当前目标的标注"),
        )
        clear = action(
            self.tr("&清除所有标注"),
            self.undoAll,
            shortcuts["clear"],
            "Clear",
            self.tr("清除所有标注信息"),
        )
        undo = action(
            self.tr("&撤销"),
            self.undoClick,
            shortcuts["undo"],
            "Undo",
            self.tr("撤销一次点击"),
        )
        redo = action(
            self.tr("&重做"),
            self.redoClick,
            shortcuts["redo"],
            "Redo",
            self.tr("重做一次点击"),
        )
        save = action(
            self.tr("&保存"),
            self.saveLabel,
            "",
            "Save",
            self.tr("保存图像标签"),
        )
        save_as = action(
            self.tr("&另存为"),
            partial(self.saveLabel, True),
            "",
            "OtherSave",
            self.tr("指定标签保存路径"),
        )
        auto_save = action(
            self.tr("&自动保存"),
            self.toggleAutoSave,
            "",
            "AutoSave",
            self.tr("翻页同时自动保存"),
            checkable=True,
        )
        quit = action(
            self.tr("&退出"),
            self.close,
            "",
            "Close",
            self.tr("退出软件"),
        )
        save_label = action(
            self.tr("&保存标签列表"),
            self.saveLabelList,
            "",
            "ExportLabel",
            self.tr("将标签保存成标签配置文件"),
        )
        load_label = action(
            self.tr("&加载标签列表"),
            self.loadLabelList,
            "",
            "ImportLabel",
            self.tr("从标签配置文件中加载标签"),
        )
        clear_label = action(
            self.tr("&清空标签列表"),
            self.clearLabelList,
            "",
            "ClearLabel",
            self.tr("清空所有的标签"),
        )
        clear_recent = action(
            self.tr("&清除标注记录"),
            self.clearRecentFile,
            "",
            "ClearRecent",
            self.tr("清除近期标注记录"),
        )
        recent_files = QtWidgets.QMenu(self.tr("近期文件"))
        recent_files.aboutToShow.connect(self.updateRecentFile)
        recent_params = QtWidgets.QMenu(self.tr("近期模型及参数"))
        recent_params.aboutToShow.connect(self.updateModelsMenu)
        self.actions = util.struct(
            auto_save=auto_save,
            recent_files=recent_files,
            recent_params=recent_params,
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
                None,
                quit,
            ),
            labelMenu=(
                save_label,
                load_label,
                clear_label,
            ),
            toolBar=(
                finish_object,
                clear,
                undo,
                redo,
                turn_prev,
                turn_next,
            ),
        )
        menu("文件", self.actions.fileMenu)
        menu("标注", self.actions.labelMenu)
        util.addActions(self.toolBar, self.actions.toolBar)

    def updateRecentFile(self):
        menu = self.actions.recent_files
        menu.clear()
        recentFiles = self.settings.value("recent_files", [])
        if not recentFiles:
            recentFiles = []
        files = [f for f in recentFiles if osp.exists(f)]
        for i, f in enumerate(files):
            icon = util.newIcon("File")
            action = QtWidgets.QAction(
                icon, "&【%d】 %s" % (i + 1, QtCore.QFileInfo(f).fileName()), self
            )
            action.triggered.connect(partial(self.loadImage, f, True))
            menu.addAction(action)
        if len(files) == 0:
            menu.addAction("无近期文件")
        self.settings.setValue("recent_files", files)

    def addRecentFile(self, path):
        if not osp.exists(path):
            return
        paths = self.settings.value("recent_files")
        if not paths:
            paths = []
        if path not in paths:
            paths.append(path)
        if len(paths) > 15:
            del paths[0]
        self.settings.setValue("recent_files", paths)
        self.updateRecentFile()

    def clearRecentFile(self):
        self.settings.remove("recent_files")
        self.statusbar.showMessage("已清除最近打开文件", 10000)

    def updateModelsMenu(self):
        menu = self.actions.recent_params
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
                partial(self.loadModelParam, m["param_path"], m["model_name"])
            )
            menu.addAction(action)
        self.settings.setValue("recent_params", self.recentModels)

    def changeModel(self, idx):
        self.modelClass = MODELS[idx]

    def changeParam(self):
        if not self.modelClass:
            self.warn("选择模型结构", "尚未选择模型结构，请在右侧下拉菜单进行选择！")
        formats = ["*.pdparams"]
        filters = self.tr("paddle model param files (%s)") % " ".join(formats)
        start_path = (
            "."
            if len(self.recentModels) == 0
            else osp.dirname(self.recentModels[-1]["param_path"])
        )
        param_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("%s - 选择模型参数") % __APPNAME__,
            start_path,
            filters,
        )
        if not osp.exists(param_path):
            return
        res = self.loadModelParam(param_path)
        if res:
            model_dict = {
                "param_path": param_path,
                "model_name": self.modelClass.__name__,
            }
            if model_dict not in self.recentModels:
                self.recentModels.append(model_dict)
                if len(self.recentModels) > 10:
                    del self.recentModels[0]
                self.settings.setValue("recent_models", self.recentModels)

    def loadModelParam(self, param_path, model=None):
        if model is None:
            model = self.modelClass()
        if isinstance(model, str):
            try:
                model = MODELS[model]()
            except KeyError:
                return False
        if inspect.isclass(model):
            model = model()
        if not isinstance(model, models.EISegModel):
            self.warn("选择模型结构", "尚未选择模型结构，请在右侧下拉菜单进行选择")
            return False
        modelIdx = MODELS.idx(model.__name__)
        self.statusbar.showMessage(f"正在加载 {model.__name__} 模型")  # 这里没显示
        model = model.load_param(param_path)
        if model is not None:
            if self.controller is None:
                self.controller = InteractiveController(
                    model,
                    predictor_params={
                        "brs_mode": "NoBRS",
                        "zoom_in_params": {
                            "skip_clicks": -1,
                            "target_size": (400, 400),
                            "expansion_ratio": 1.4,
                        },
                        "predictor_params": {"net_clicks_limit": None, "max_size": 800},
                    },
                    update_image_callback=self._update_image,
                )
                self.controller.prob_thresh = self.segThresh
                if self.image is not None:
                    self.controller.set_image(self.image)
            else:
                self.controller.reset_predictor(model)
            self.statusbar.showMessage(f"{osp.basename(param_path)} 模型加载完成", 20000)
            self.comboModelSelect.setCurrentIndex(modelIdx)
            return True
        else:  # 模型和参数不匹配
            self.warn("模型和参数不匹配", "当前网络结构中的参数与模型参数不匹配，请更换网络结构或使用其他参数！")
            self.statusbar.showMessage("模型和参数不匹配，请重新加载", 20000)
            self.controller = None  # 清空controller
            return False

    def loadRecentModelParam(self):
        if len(self.recentModels) == 0:
            self.statusbar.showMessage("没有最近使用模型信息，请加载模型", 10000)
            return
        m = self.recentModels[-1]
        model = MODELS[m["model_name"]]
        param_path = m["param_path"]
        self.loadModelParam(param_path, model)

    def loadLabelList(self):
        filters = self.tr("标签配置文件 (*.txt)")
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("%s - 选择标签配置文件路径") % __APPNAME__,
            ".",
            filters,
        )
        if not osp.exists(file_path):
            return
        self.labelList.readLabel(file_path)
        self.refreshLabelList()
        self.settings.setValue("label_list_file", file_path)

    def saveLabelList(self):
        if len(self.labelList) == 0:
            self.warn("没有需要保存的标签", "请先添加标签之后再进行保存")
            return
        filters = self.tr("标签配置文件 (*.txt)")
        dlg = QtWidgets.QFileDialog(self, "保存标签配置文件", ".", filters)
        dlg.setDefaultSuffix("txt")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
        savePath, _ = dlg.getSaveFileName(
            self, self.tr("%s - 选择保存标签配置文件路径") % __APPNAME__, ".", filters
        )
        self.labelList.saveLabel(savePath)
        self.settings.setValue("label_list_file", savePath)

    def addLabel(self):
        c = self.maskColormap.get_color()
        table = self.labelListTable
        table.insertRow(table.rowCount())
        idx = table.rowCount() - 1
        self.labelList.add(idx + 1, "", c)
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

    def clearLabelList(self):
        self.labelList.clear()
        if self.controller:
            self.controller.label_list = []
            self.controller.curr_label_number = None
        self.labelListTable.clear()
        self.labelListTable.setRowCount(0)

    def refreshLabelList(self):
        table = self.labelListTable
        table.clearContents()
        table.setRowCount(len(self.labelList))
        table.setColumnCount(4)
        for idx, lab in enumerate(self.labelList):
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
            delItem.setIcon(util.newIcon("clear"))
            delItem.setTextAlignment(Qt.AlignCenter)
            delItem.setFlags(QtCore.Qt.ItemIsEnabled)
            table.setItem(idx, 3, delItem)

        cols = [0, 1, 3]
        for idx in cols:
            table.resizeColumnToContents(idx)

    def labelListDoubleClick(self, row, col):
        if col != 2:
            return
        table = self.labelListTable
        color = QtWidgets.QColorDialog.getColor()
        if color.getRgb() == (0, 0, 0, 255):
            return
        table.item(row, col).setBackground(color)
        self.labelList[row].color = color.getRgb()[:3]
        if self.controller:
            self.controller.label_list = self.labelList

    @property
    def currLabelIdx(self):
        return self.controller.curr_label_number - 1

    def labelListClicked(self, row, col):
        table = self.labelListTable
        if col == 3:
            table.removeRow(row)
            self.labelList.remove(row)
        if col == 0 or col == 1:
            for idx in range(len(self.labelList)):
                table.item(idx, 0).setBackground(QtGui.QColor(255, 255, 255))
            table.item(row, 0).setBackground(QtGui.QColor(48, 140, 198))
            for idx in range(3):
                table.item(row, idx).setSelected(True)
            if self.controller:
                self.controller.change_label_num(int(table.item(row, 0).text()))
                self.controller.label_list = self.labelList

    def labelListItemChanged(self, row, col):
        if col != 1:
            return
        name = self.labelListTable.item(row, col).text()
        self.labelList[row].name = name

    # 图片/标签 io
    def openImage(self):
        formats = [
            "*.{}".format(fmt.data().decode())
            for fmt in QtGui.QImageReader.supportedImageFormats()
        ]
        recentPath = self.settings.value("recent_files", [])
        if len(recentPath) == 0:
            recentPath = "."
        else:
            recentPath = osp.dirname(recentPath[-1])
        filters = self.tr("Image & Label files (%s)") % " ".join(formats)
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("%s - 选择待标注图片") % __APPNAME__,
            recentPath,
            filters,
        )
        if len(file_path) == 0:
            return
        self.queueEvent(partial(self.loadImage, file_path))
        self.listFiles.addItems([file_path])
        self.filePaths.append(file_path)

    def openFolder(self):
        self.inputDir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - 选择待标注图片文件夹") % __APPNAME__,
            "/home/lin/Desktop",
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if len(self.inputDir) == 0:
            return
        filePaths = os.listdir(self.inputDir)
        exts = QtGui.QImageReader.supportedImageFormats()
        filePaths = [n for n in filePaths if n.split(".")[-1] in exts]
        filePaths = [osp.join(self.inputDir, n) for n in filePaths]
        self.filePaths += filePaths
        self.listFiles.addItems(filePaths)
        self.currIdx = 0
        self.turnImg(0)

    def loadImage(self, path, update_list=False):
        if len(path) == 0 or not osp.exists(path):
            return
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), 1)
        image = image[:, :, ::-1]  # BGR转RGB
        self.image = image
        self.currentPath = path
        if self.controller:
            self.controller.set_image(self.image)
        else:
            self.showWarning("未加载模型参数，请先加载模型参数！")
            self.changeParam()
            return 0
        self.controller.set_label(self.loadLabel(path))
        self.addRecentFile(path)
        self.imagePath = path  # 修复使用近期文件的图像保存label报错
        if update_list:
            self.listFiles.addItems([path])
            self.filePaths.append(path)

    def loadLabel(self, imgPath):
        if imgPath == "" or len(self.labelPaths) == 0:
            return None

        def getName(path):
            return osp.basename(path).split(".")[0]

        imgName = getName(imgPath)
        for path in self.labelPaths:
            if getName(path) == imgName:
                labPath = path
                break
        label = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        return label

    def turnImg(self, delta):
        self.currIdx += delta
        if self.currIdx >= len(self.filePaths) or self.currIdx < 0:
            self.currIdx -= delta
            self.statusbar.showMessage(f"没有{'后一张'if delta==1 else '前一张'}图片")
            return
        self.completeLastMask()
        if self.isDirty:
            if self.actions.auto_save.isChecked():
                self.saveLabel()
            else:
                res = self.warn(
                    "保存标签？",
                    "标签尚未保存，是否保存标签",
                    QMessageBox.Yes | QMessageBox.Cancel,
                )
                if res == QMessageBox.Yes:
                    self.saveLabel()

        imagePath = self.filePaths[self.currIdx]
        self.loadImage(imagePath)
        self.imagePath = imagePath
        self.listFiles.setCurrentRow(self.currIdx)
        self.setClean()

    def listClicked(self):
        if self.controller.is_incomplete_mask:
            self.saveLabel()
        toRow = self.listFiles.currentRow()
        delta = toRow - self.currIdx
        self.turnImg(delta)

    def finishObject(self):
        if self.image is None:
            return
        if not self.controller:
            return
        self.controller.finish_object()
        self.setDirty()

    def completeLastMask(self):
        # 返回最后一个标签是否完成，false就是还有带点的
        if not self.controller:
            return True
        if not self.controller.is_incomplete_mask:
            return True
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("完成最后一个目标？")
        msg.setText("是否完成最后一个目标的标注，不完成不会进行保存。")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        res = msg.exec_()
        if res == QMessageBox.Yes:
            self.finishObject()
            self.setDirty()
            return True
        return False

    def saveLabel(self, saveAs=False, savePath=None):
        if not self.controller:
            return
        if self.controller.image is None:
            return
        self.completeLastMask()
        if not savePath:  # 参数没传存到哪
            if not saveAs and self.outputDir is not None:
                # 指定了标签文件夹，而且不是另存为
                savePath = osp.join(
                    self.outputDir, osp.basename(self.imagePath).split(".")[0] + ".png"
                )
            else:
                filters = self.tr("Label files (*.png)")
                dlg = QtWidgets.QFileDialog(
                    self, "保存标签文件路径", osp.dirname(self.imagePath), filters
                )
                dlg.setDefaultSuffix("png")
                dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
                dlg.setOption(QtWidgets.QFileDialog.DontConfirmOverwrite, False)
                dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, False)
                savePath, _ = dlg.getSaveFileName(
                    self,
                    self.tr("选择标签文件保存路径"),
                    osp.basename(self.imagePath).split(".")[0] + ".png",
                )
        if (
            savePath is None
            or len(savePath) == 0
            or not osp.exists(osp.dirname(savePath))
        ):
            return

        cv2.imwrite(savePath, self.controller.result_mask)
        self.statusbar.showMessage(f"标签成功保存至 {savePath}")

    def setClean(self):
        self.isDirty = False

    def setDirty(self):
        self.isDirty = True

    def changeOutputDir(self):
        outputDir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("%s - 选择标签保存路径") % __APPNAME__,
            ".",
            QtWidgets.QFileDialog.ShowDirsOnly
            | QtWidgets.QFileDialog.DontResolveSymlinks,
        )
        if len(outputDir) == 0 or not osp.exists(outputDir):
            return False
        labelPaths = os.listdir(outputDir)
        exts = ["png"]
        labelPaths = [n for n in labelPaths if n.split(".")[-1] in exts]
        labelPaths = [osp.join(outputDir, n) for n in labelPaths]
        self.outputDir = outputDir
        self.labelPaths = labelPaths
        return True

    def maskOpacityChanged(self):
        self.sldOpacity.textLab.setText(str(self.opacity))
        if not self.controller or self.controller.image is None:
            return
        self._update_image()

    def clickRadiusChanged(self):
        self.sldClickRadius.textLab.setText(str(self.clickRadius))
        if not self.controller or self.controller.image is None:
            return

        self._update_image()

    def threshChanged(self):
        self.sldThresh.textLab.setText(str(self.segThresh))
        if not self.controller or self.controller.image is None:
            return
        self.controller.prob_thresh = self.segThresh
        self._update_image()

    def undoClick(self):
        if self.image is None:
            return
        if not self.controller:
            return
        self.controller.undo_click()
        if not self.controller.is_incomplete_mask:
            self.setClean()

    def undoAll(self):
        if not self.controller or self.controller.image is None:
            return
        self.controller.reset_last_object()
        self.setClean()

    def redoClick(self):
        if self.image is None:
            return
        if not self.controller:
            return
        self.controller.redo_click()

    def canvasClick(self, x, y, isLeft):
        if self.controller is None:
            return
        if self.controller.image is None:
            return
        currLabel = self.controller.curr_label_number
        if not currLabel or currLabel == 0:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setWindowTitle("未选择当前标签")
            msg.setText("请先在标签列表中单击点选标签")
            msg.setStandardButtons(QMessageBox.Yes)
            res = msg.exec_()
            return

        self.controller.add_click(x, y, isLeft)

    def _update_image(self, reset_canvas=False):
        if not self.controller:
            return
        image = self.controller.get_visualization(
            alpha_blend=self.opacity,
            click_radius=self.clickRadius,
        )
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        image = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        if reset_canvas:
            self.resetZoom(width, height)
        self.annImage.setPixmap(QPixmap(image))

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

    def queueEvent(self, function):
        # TODO: 研究这个东西是不是真的不影响ui
        QtCore.QTimer.singleShot(0, function)

    def toggleAutoSave(self, save):
        if save and not self.outputDir:
            self.changeOutputDir()
        if save and not self.outputDir:
            save = False
        self.actions.auto_save.setChecked(save)
        self.settings.setValue("auto_save", save)

    def toggleLargestCC(self, on):
        self.controller.filterLargestCC = on

    @property
    def opacity(self):
        return self.sldOpacity.value() / 100

    @property
    def clickRadius(self):
        return self.sldClickRadius.value()

    @property
    def segThresh(self):
        return self.sldThresh.value() / 100

    def warn(self, title, text, buttons=QMessageBox.Yes):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.setStandardButtons(buttons)
        return msg.exec_()
