import logging
import os
import os.path as osp
import pathlib
import time
import json
from functools import partial
import random
import threading

import qt
import ctk
import vtk
import numpy as np
import SimpleITK as sitk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

# when test, wont use any paddle related funcion
HERE = pathlib.Path(__file__).parent.absolute()
TEST = osp.exists(HERE / "TEST")
if not TEST:
    logging.getLogger().setLevel(logging.ERROR)
if not TEST:
    try:
        import paddle
    except ModuleNotFoundError as e:
        if slicer.util.confirmOkCancelDisplay(
                "This module requires 'paddlepaddle' Python package. Click OK to install it now."
        ):
            slicer.util.pip_install("paddlepaddle")
            import paddle

    try:
        import paddleseg
    except ModuleNotFoundError as e:
        if slicer.util.confirmOkCancelDisplay(
                "This module requires 'paddleseg' Python package. Click OK to install it now."
        ):
            slicer.util.pip_install("paddleseg")
            import paddle

    import inference
    import inference.predictor as predictor

# TODO: get some better color map
colors = [
    (0.5019607843137255, 0.6823529411764706, 0.5019607843137255),
    (0.9450980392156862, 0.8392156862745098, 0.5686274509803921),
    (0.6941176470588235, 0.4784313725490196, 0.3960784313725490),
    (0.4352941176470588, 0.7215686274509804, 0.8235294117647058),
    (0.8470588235294118, 0.3960784313725490, 0.3098039215686274),
    (0.8666666666666667, 0.5098039215686274, 0.3960784313725490),
    (0.5647058823529412, 0.9333333333333333, 0.5647058823529412),
    (0.7529411764705882, 0.4078431372549019, 0.3450980392156862),
    (0.8627450980392157, 0.9607843137254902, 0.0784313725490196),
    (0.3058823529411765, 0.2470588235294117, 0.0000000000000000),
    (1.0000000000000000, 0.9803921568627451, 0.8627450980392157),
    (0.9019607843137255, 0.8627450980392157, 0.2745098039215685),
    (0.7843137254901961, 0.7843137254901961, 0.9215686274509803),
    (0.9803921568627451, 0.9803921568627451, 0.8235294117647058),
]

#
# EISegMed3D
#


class EISegMed3D(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "EISegMed3D"  # TODO: make this more human readable by adding spaces
        self.parent.categories = [
            "Interactive Segmentation"
        ]  # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = [
        ]  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Lin Han, Daisy (Baidu Corp.)"]
        # TODO: update with short description of the module and a link to online module documentation
        self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#EISegMed3D">module documentation</a>.
"""
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = """
This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc., Andras Lasso, PerkLab,
and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
"""

        # Additional initialization step after application startup is complete
        slicer.app.connect("startupCompleted()", self.initializeAfterStartup)

    def initializeAfterStartup(self):
        # print("initializeAfterStartup", slicer.app.commandOptions().noMainWindow)
        pass


class Clicker(object):
    def __init__(self):
        self.reset_clicks()

    def get_clicks(self, clicks_limit=None):  # [click1, click2, ...]
        return self.clicks_list[:clicks_limit]

    def add_click(self, click):
        coords = click.coords

        click.index = self.num_pos_clicks + self.num_neg_clicks
        if click.is_positive:
            self.num_pos_clicks += 1
        else:
            self.num_neg_clicks += 1

        self.clicks_list.append(click)

    def reset_clicks(self):
        self.num_pos_clicks = 0
        self.num_neg_clicks = 0
        self.clicks_list = []

    def __len__(self):
        return len(self.clicks_list)


#
# Register sample data sets in Sample Data module
#
def registerSampleData():
    """
    Add data sets to Sample Data module.
    """
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # EISegMed3D1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="placePoint",
        sampleName="placePoint1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "placePoint1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="placePoint1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="placePoint1", )

    # EISegMed3D2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="placePoint",
        sampleName="placePoint2",
        thumbnailFileName=os.path.join(iconsPath, "placePoint2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="placePoint2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="placePoint2", )


#
# EISegMed3DWidget
#


class EISegMed3DWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(
            self)  # needed for parameter node observation

        self.logic = None
        self._parameterNode = None
        # data var
        self._dataFolder = None
        self._scanPaths = []
        self._finishedPaths = []
        self._currScanIdx = None
        self._currVolumeNode = None
        self.dgPositivePointListNode = None
        self.dgPositivePointListNodeObservers = []
        self.dgNegativePointListNode = None
        self.dgNegativePointListNodeObservers = []
        self._prevCatg = None
        self._loadingScans = set()
        self.pb = None

        # status var
        self._turninig = False
        self._dirty = False
        self._syncingCatg = False
        self._usingInteractive = False
        self._updatingGUIFromParameterNode = False
        self._endImportProcessing = False
        self._addingControlPoint = False
        self._lastTurnNextScan = True

        self.init_params()

    def setup(self):
        """
        Called when the user opens the module the first time and the widget is initialized.
        """
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/EISegMed3D.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        # TODO: we may not need logic. user have to interact
        self.logic = EISegMed3DLogic()

        # Connections
        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent,
                         self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent,
                         self.onSceneEndClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.NodeAddedEvent,
                         self.onSceneEndImport)

        # TODO: is syncing settings between node and gui on show/scenestart/... necessary

        # button, slider
        self.ui.loadModelButton.connect("clicked(bool)", self.loadModelClicked)
        self.ui.nextScanButton.connect("clicked(bool)",
                                       lambda p: self.nextScan())
        self.ui.prevScanButton.connect("clicked(bool)",
                                       lambda p: self.prevScan())
        self.ui.finishScanButton.connect("clicked(bool)", self.finishScan)
        self.ui.finishSegmentButton.connect("clicked(bool)",
                                            self.exitInteractiveMode)
        self.ui.opacitySlider.connect("valueChanged(double)",
                                      self.opacityUi2Display)
        self.ui.dataFolderButton.connect("directoryChanged(QString)",
                                         self.loadScans)
        self.ui.skipFinished.connect("clicked(bool)", self.skipFinishedToggled)

        iconPath = HERE / "Resources" / "Icons"
        self.ui.nextScanButton.setIcon(qt.QIcon(iconPath / "next.png"))
        self.ui.prevScanButton.setIcon(qt.QIcon(iconPath / "prev.png"))
        self.ui.finishSegmentButton.setIcon(qt.QIcon(iconPath / "done.png"))
        self.ui.finishScanButton.setIcon(qt.QIcon(iconPath / "save.png"))

        # positive/negative control point
        self.ui.dgPositiveControlPointPlacementWidget.setMRMLScene(
            slicer.mrmlScene)
        self.ui.dgPositiveControlPointPlacementWidget.placeButton(
        ).toolTip = "Add positive points"
        self.ui.dgPositiveControlPointPlacementWidget.placeButton().show()
        self.ui.dgPositiveControlPointPlacementWidget.deleteButton(
        ).setFixedHeight(0)  # diable delete point button
        self.ui.dgPositiveControlPointPlacementWidget.deleteButton(
        ).setFixedWidth(0)
        self.ui.dgPositiveControlPointPlacementWidget.placeButton().connect(
            "clicked(bool)", self.enterInteractiveMode)

        self.ui.dgNegativeControlPointPlacementWidget.setMRMLScene(
            slicer.mrmlScene)
        self.ui.dgNegativeControlPointPlacementWidget.placeButton(
        ).toolTip = "Add negative points"
        self.ui.dgNegativeControlPointPlacementWidget.placeButton().show()
        self.ui.dgNegativeControlPointPlacementWidget.deleteButton(
        ).setFixedHeight(0)
        self.ui.dgNegativeControlPointPlacementWidget.deleteButton(
        ).setFixedWidth(0)
        self.ui.dgNegativeControlPointPlacementWidget.placeButton().connect(
            "clicked(bool)", self.enterInteractiveMode)

        # segment editor
        self.ui.embeddedSegmentEditorWidget.setMRMLScene(slicer.mrmlScene)
        self.ui.embeddedSegmentEditorWidget.setMRMLSegmentEditorNode(
            self.logic.get_segment_editor_node())

        self.initializeFromNode()

        # Set place point widget colors
        self.ui.dgPositiveControlPointPlacementWidget.setNodeColor(
            qt.QColor(0, 255, 0))
        self.ui.dgNegativeControlPointPlacementWidget.setNodeColor(
            qt.QColor(255, 0, 0))

    def init_params(self):
        """init changble parameters here"""
        self.predictor_params_ = {"norm_radius": 2, "spatial_scale": 1.0}
        self.ratio = (
            512 / 880, 512 / 880, 12 /
            12)  # xyz 这个形状与训练的对数据预处理的形状要一致，怎么切换不同模型？ todo： 在模块上设置预处理形状。和模型一致
        self.train_shape = (512, 512, 12)
        self.image_ww = (0, 2650)  # low, high range for image crop
        self.test_iou = False  # the label file need to be set correctly
        self.file_suffix = [".nii",
                            ".nii.gz"]  # files with these suffix will be loaded
        if TEST:
            self.device, self.enable_mkldnn = "cpu", True
        else:
            self.device, self.enable_mkldnn = "gpu", True

    def clearScene(self, clearAllVolumes=False):
        if clearAllVolumes:
            for node in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
                slicer.mrmlScene.RemoveNode(node)

        segmentationNode = self.segmentationNode
        if segmentationNode is not None:
            slicer.mrmlScene.RemoveNode(segmentationNode)

    """ progress bar related """

    def skipFinishedToggled(self, skipFinished):
        self.togglePrevNextBtn(self._currScanIdx)
        if not skipFinished and self._currVolumeNode is None:
            self.turnTo(self._currScanIdx)

    def initPb(self, label="Processing..", windowTitle=None):
        if self.pb is None:
            self.pb = slicer.util.createProgressDialog()
            self.pb.setCancelButtonText("Close")
            self.pb.setAutoClose(True)
            self.pb.show()
            self.pb.activateWindow()
            self.pb.setValue(0)
            self.pbLeft = 100
        else:
            self.pbLeft = 100 - self.pb.value

        if windowTitle is not None:
            self.pb.setWindowTitle(windowTitle)
        self.pb.setLabelText(label)
        slicer.app.processEvents()

    def setPb(self, percentage, label=None, windowTitle=None):
        self.pb.setValue(100 - int(self.pbLeft * (1 - percentage)))
        if label is not None:
            self.pb.setLabelText(label)
        if windowTitle is not None:
            self.pb.setWindowTitle(windowTitle)
        slicer.app.processEvents()

    def closePb(self):
        if self.pb is None:
            return
        pb = self.pb
        self.pb = None
        pb.close()

    """ load/change scan related """

    def loadScans(self, dataFolder):
        """Get all the scans under a folder and turn to the first one"""
        self.initPb("Making sure input valid", "Loading scans")

        # 1. ensure valid input
        if dataFolder is None or len(dataFolder) == 0:
            slicer.util.errorDisplay(
                "Please select a Data Folder first!", autoCloseMsec=5000)
            return

        if not osp.exists(dataFolder):
            slicer.util.errorDisplay(
                f"The Data Folder( {dataFolder} ) doesn't exist!",
                autoCloseMsec=2000)
            return

        self.clearScene()

        self.setPb(0.2, "Searching for scans")

        # 2. list files in assigned directory
        self._dataFolder = dataFolder
        paths = [
            p for p in os.listdir(self._dataFolder)
            if p[p.find("."):] in self.file_suffix
        ]
        paths = [
            p for p in paths if p.split(".")[0][-len("_label"):] != "_label"
        ]
        paths.sort()
        self._scanPaths = [osp.join(self._dataFolder, p) for p in paths]

        if len(paths) == 0:
            self.closePb()
            slicer.util.errorDisplay(
                f"No file ending with {' or '.join(self.file_suffix)} is found under {self._dataFolder}.\nDid you chose the wrong folder?"
            )
            return
        self.setPb(0.5,
                   f"Found {len(paths)} scans in folder {self._dataFolder}")

        self._currScanIdx, self._finishedPaths = self.getProgress()
        self.updateProgressWidgets()

        if len(set(self._scanPaths) - set(
                self._finishedPaths)) == 0 and self.ui.skipFinished.checked:
            self.closePb()
            slicer.util.delayDisplay(
                f"All {len(self._scanPaths)} scans have been annotated!\nUncheck Skip Finished Scans to browse through them.",
                4000, )
            return

        self.setPb(0.6, "Loading Scan and label")
        self._currScanIdx -= 1
        found = self.nextScan(silentFail=True)
        if not found:
            self._currScanIdx += 2
            self.prevScan()

        self.ui.finishScanButton.setEnabled(True)
        logging.info(
            f"All scans found under {self._dataFolder} are{','.join([' '+osp.basename(p) for p in self._scanPaths])}"
        )

    def togglePrevNextBtn(self, currIdx):
        if currIdx is None:
            return
        self.ui.prevScanButton.setEnabled(
            self.getTurnToTaskId(currIdx, "prev") is not None)
        self.ui.nextScanButton.setEnabled(
            self.getTurnToTaskId(currIdx, "next") is not None)

    def nextScan(self, silentFail=False):
        self.saveSegmentation()
        nextIdx = self.getTurnToTaskId(self._currScanIdx, "next")
        if nextIdx is None:
            self.ui.nextScanButton.setEnabled(False)
            if not silentFail:
                slicer.util.errorDisplay(
                    f"This is the last unannotated scan. No next scan")
            return False
        self._lastTurnNextScan = True
        self.turnTo(nextIdx)
        return True

    def prevScan(self, silentFail=False):
        self.saveSegmentation()
        prevIdx = self.getTurnToTaskId(self._currScanIdx, "prev")
        if prevIdx is None:
            self.ui.prevScanButton.setEnabled(False)
            if not silentFail:
                slicer.util.errorDisplay(
                    f"This is the first unannotated scan. No previous scan")
            return False
        self._lastTurnNextScan = False
        self.turnTo(prevIdx)
        return True

    def getTurnToTaskId(self, currIdx, direction, skipFinished=None):
        if skipFinished is None:
            skipFinished = self.ui.skipFinished.checked
        if direction == "next":
            while True:
                if currIdx >= len(self._scanPaths) - 1:
                    return None
                currIdx += 1
                if not skipFinished:
                    break
                if self._scanPaths[currIdx] not in self._finishedPaths:
                    break
            return currIdx
        else:
            while True:
                if currIdx <= 0:
                    return None
                currIdx -= 1
                if not skipFinished:
                    break
                if self._scanPaths[currIdx] not in self._finishedPaths:
                    break
            return currIdx

    def getScan(self, scanPath, wait=True):
        try:
            return slicer.util.getNode(osp.basename(scanPath))
        except slicer.util.MRMLNodeNotFoundException:
            if scanPath in self._loadingScans:  # scan hasn't finished loading
                if wait:
                    timeout = 30
                    while True:
                        if scanPath not in self._loadingScans:
                            return slicer.util.getNode(osp.basename(scanPath))
                    logging.info("waiting", scanPath, timeout)
                    time.sleep(0.1)
                    timeout -= 1
                    if timeout == 0:
                        return None
                else:
                    return None
            else:
                if wait:
                    logging.info(f"loading {scanPath}")
                    self._loadingScans.add(scanPath)
                    node = slicer.util.loadVolume(
                        scanPath,
                        properties={"show": False,
                                    "singleFile": True})
                    node.SetName(osp.basename(scanPath))
                    self._loadingScans.remove(scanPath)
                    return node
                else:

                    def read(path):
                        node = slicer.util.loadVolume(
                            scanPath,
                            properties={"show": False,
                                        "singleFile": True})
                        node.SetName(osp.basename(path))

                    qt.QTimer.singleShot(
                        random.randint(500, 1000), lambda: read(scanPath))

    def manageCache(self, currIdx, skipPreload=False):
        toKeepIdxs = [
            self.getTurnToTaskId(currIdx, "prev"),
            currIdx,
            self.getTurnToTaskId(currIdx, "next"),
        ]
        toKeepPaths = [
            self._scanPaths[idx] for idx in toKeepIdxs if idx is not None
        ]

        allVolumes = slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode")
        for volume in allVolumes:
            if volume.GetName() not in map(osp.basename, toKeepPaths):
                slicer.mrmlScene.RemoveNode(volume)

        for path in toKeepPaths:
            self.getScan(path, wait=False)

    def turnTo(self, turnToIdx, skipPreload=False):
        """
        Turn to the turnToIdx th scan, load scan and label
        """
        if turnToIdx == self._currScanIdx:
            return False
        if self._turninig:
            return

        self._turninig = True
        # 0. clear nodes from previous task and prepare states
        self.initPb("Preparing to load", "Load scan and label")
        self.setPb(0.1)

        if self.segmentation is not None:
            self.saveSegmentation()
        if self._usingInteractive:
            self.exitInteractiveMode()

        if len(self._scanPaths) == 0:
            slicer.util.errorDisplay(
                "No scan found, please load scans first.", autoCloseMsec=2000)
            self.closePb()
            self._turninig = False
            return

        logging.info(
            f"Turning to the {turnToIdx}th scan, path is {self._scanPaths[turnToIdx]}"
        )

        self.ui.dgPositiveControlPointPlacementWidget.setEnabled(False)
        self.ui.dgNegativeControlPointPlacementWidget.setEnabled(False)

        self.clearScene()  # remove segmentation node and control points
        self._currScanIdx = turnToIdx

        slicer.app.processEvents()
        # 1. load new scan & preprocess
        image_path = self._scanPaths[turnToIdx]
        self.setPb(0.2, f"Loading {osp.basename(image_path)}")
        self._currVolumeNode = self.getScan(image_path)
        self._currVolumeNode.SetName(osp.basename(image_path))
        self.manageCache(turnToIdx, skipPreload=skipPreload)

        # 2. load segmentation or create an empty one
        self.setPb(0.8, "Loading segmentation")
        dot_pos = image_path.find(".")
        self._currLabelPath = image_path[:dot_pos] + "_label" + image_path[
            dot_pos:]
        if osp.exists(self._currLabelPath):
            segmentNode = slicer.modules.segmentations.logic(
            ).LoadSegmentationFromFile(self._currLabelPath, False)
        else:
            segmentNode = slicer.mrmlScene.AddNewNodeByClass(
                "vtkMRMLSegmentationNode")

        segmentNode.SetName("EISegMed3DSegmentation")
        segmentNode.SetReferenceImageGeometryParameterFromVolumeNode(
            self._currVolumeNode)
        slicer.app.processEvents()
        slicer.app.processEvents()
        # update category info

        for segment in self.segments:
            if segment.GetNameAutoGenerated():
                segment.SetName(f"Segment_{segment.GetLabelValue()}")
        self._prevCatg = None
        self.catgFile2Segmentation()
        self.catgSegmentation2File()

        for idx, segment in enumerate(self.segments):
            # print(f"setting color for segment name: {segment.GetName()}, color: {colors[idx % len(colors)]}")
            segment.SetColor(colors[idx % len(colors)])
            segment.SetColor(colors[idx % len(colors)])
            segment.SetColor(colors[idx % len(colors)])

        def sync(*args):
            if not self._turninig:
                self.catgSegmentation2File()

        def setDirty(*args):
            if not self._turninig:
                self._dirty = True

        segmentNode.AddObserver(
            segmentNode.GetContentModifiedEvents().GetValue(5), sync)
        segmentNode.AddObserver(
            segmentNode.GetContentModifiedEvents().GetValue(4), sync)
        segmentNode.AddObserver(
            segmentNode.GetContentModifiedEvents().GetValue(1), setDirty)

        # add: 3, 5
        # edit: 5
        # remove: 1, 2, 4 (will be triggered when turn task)

        # 3. create category label from txt and segmentation
        self.setPb(0.8, "Syncing progress")
        self.saveProgress()
        self.updateProgressWidgets()

        # 4. set image
        self.setPb(0.9, "Preprocessing image for interactive segmentation")
        if not TEST:
            self.prepImage()

        # 5. set the editor as current result.
        self.setPb(0.95, "Wrapping up")
        self.ui.embeddedSegmentEditorWidget.setSegmentationNode(segmentNode)
        self.ui.embeddedSegmentEditorWidget.setMasterVolumeNode(
            self._currVolumeNode)

        self.ui.dgPositiveControlPointPlacementWidget.setEnabled(True)
        self.ui.dgNegativeControlPointPlacementWidget.setEnabled(True)

        # 6. change button state
        self.togglePrevNextBtn(self._currScanIdx)

        layoutManager = slicer.app.layoutManager()
        for sliceViewName in layoutManager.sliceViewNames():
            layoutManager.sliceWidget(sliceViewName).mrmlSliceNode(
            ).RotateToVolumePlane(self._currVolumeNode)
        slicer.util.resetSliceViews()

        self.closePb()
        self._turninig = False

    """ category and segmentation management """

    @property
    def segmentationNode(self):
        try:
            return slicer.util.getNode("EISegMed3DSegmentation")
        except slicer.util.MRMLNodeNotFoundException:
            return None

    @property
    def segmentation(self):
        segmentationNode = self.segmentationNode
        if segmentationNode is None:
            return None
        return segmentationNode.GetSegmentation()

    @property
    def segments(self):
        segmentation = self.segmentation
        if segmentation is None:
            return []
        for segId in segmentation.GetSegmentIDs():
            yield segmentation.GetSegment(segId)

    @property
    def configPath(self):
        if self._dataFolder is None:
            return None
        return osp.join(self._dataFolder, "EISegMed3D.json")

    def getConfig(self):
        skeleton = {"labels": [], "finished": [], "leftOff": ""}
        if not osp.exists(self.configPath):
            return skeleton
        try:
            config = json.loads(open(self.configPath, "r").read())
            return config
        except:
            return skeleton

    def getSegmentId(self, segment):
        segmentation = self.segmentation
        for segId in segmentation.GetSegmentIDs():
            if segmentation.GetSegment(segId) == segment:
                return segId

    def getCatgFromFile(self):
        """Parse category info from EISegMed3D.json

        Returns:
            dict: {name: labelValue, ... }
        """
        config = self.getConfig()
        catg = {}
        for info in config.get("labels", []):
            catg[info["name"]] = int(info["labelValue"])
        return catg

    def catgFile2Segmentation(self):
        """Sync category info from EISegMed3D.json to segmentation

        match by labelValue
        - create if missing
        - correct name if segmentation differes from EISegMed3D.json
        """
        if self._syncingCatg:
            return
        self._syncingCatg = True

        # 1. get info from config file
        name2value = self.getCatgFromFile()
        value2name = {value: name for name, value in name2value.items()}

        # 2. set segmentation's names
        segmentValues = []
        for segment in self.segments:
            labelValue = segment.GetLabelValue()
            segmentValues.append(labelValue)
            name = value2name.get(labelValue, None)
            if name is not None:
                segment.SetName(name)

        # 3. create missing categories
        for labelValue in set(value2name.keys()) - set(segmentValues):
            segmentId = self.segmentation.AddEmptySegment(
                "", value2name[labelValue])
            self.segmentation.GetSegment(segmentId).SetLabelValue(labelValue)

        self._syncingCatg = False

    def catgSegmentation2File(self):
        """Sync category info from segmentation to EISegMed3D.json

        match by name
        - sync user change name
        - sync user add
        - sync user delete
        """
        if self._syncingCatg:
            return
        self._syncingCatg = True

        # 1. if no prev catg record, record current
        segmentation = self.segmentation
        if self._prevCatg is None:
            self._prevCatg = {
                segId: segmentation.GetSegment(segId).GetName()
                for segId in segmentation.GetSegmentIDs()
            }

        # 2. change name, add to file or delete
        name2value = self.getCatgFromFile()

        for segmentId in segmentation.GetSegmentIDs():
            segment = segmentation.GetSegment(segmentId)

            # change name
            if segmentId in self._prevCatg.keys() and segment.GetName(
            ) != self._prevCatg[segmentId]:
                del name2value[self._prevCatg[segmentId]]
                name2value[segment.GetName()] = segment.GetLabelValue()

            # user add or this segmentation have more catgs
            if segment.GetName() not in name2value.keys():
                if segment.GetNameAutoGenerated():
                    segment.SetName(f"Segment_{segment.GetLabelValue()}")
                name2value[segment.GetName()] = segment.GetLabelValue()

        # delete
        for segmentId in set(self._prevCatg.keys()) - set(
                segmentation.GetSegmentIDs()):
            logging.info(
                f"deleting segment {segmentId} {self.segmentation.GetSegment(segmentId)}"
            )
            del name2value[self._prevCatg[segmentId]]

        # 3. record catg info
        self._prevCatg = {
            segId: segmentation.GetSegment(segId).GetName()
            for segId in segmentation.GetSegmentIDs()
        }

        # 4. write to file
        config = self.getConfig()
        config["labels"] = [{
            "name": name,
            "labelValue": value
        } for name, value in name2value.items()]
        print(json.dumps(config), file=open(self.configPath, "w"))

        self._syncingCatg = False

    """ task progress related """

    def saveProgress(self):
        config = self.getConfig()
        relpath = lambda path: osp.relpath(path, self._dataFolder)
        config["finished"] = [relpath(p) for p in self._finishedPaths]
        config["leftOff"] = relpath(self._scanPaths[self._currScanIdx])

        print(json.dumps(config), file=open(self.configPath, "w"))

    def getProgress(self):
        config = self.getConfig()
        leftOffIdx = 0
        if "leftOff" in config.keys():
            for idx, p in enumerate(self._scanPaths):
                if p == osp.join(self._dataFolder, config["leftOff"]):
                    leftOffIdx = idx
        return leftOffIdx, [
            osp.join(self._dataFolder, p) for p in config.get("finished", [])
        ]

    def updateProgressWidgets(self):
        self.ui.annProgressBar.setValue(
            int(100 * len(self._finishedPaths) / len(self._scanPaths)))
        self.ui.progressDetail.setText(
            f"Finished: {len(self._finishedPaths)} / Total: {len(self._scanPaths)}"
        )

        def toggleFinished(idx, *args):
            logging.info(idx, *args)
            if self._scanPaths[idx] in self._finishedPaths:
                self._finishedPaths.remove(self._scanPaths[idx])
            else:
                self._finishedPaths.append(self._scanPaths[idx])
            self.saveProgress()
            self.updateProgressWidgets()

        def pathDoubleClicked(row, col):
            if self._currScanIdx == row:
                return
            if col != 1:
                return
            self.turnTo(row, skipPreload=True)

        table = self.ui.progressTable
        table.setRowCount(len(self._scanPaths))
        for idx, path in enumerate(self._scanPaths):
            layout = qt.QVBoxLayout()
            checkbox = qt.QCheckBox()
            checkbox.setChecked(path in self._finishedPaths)
            checkbox.toggled.connect(partial(toggleFinished, idx))
            layout.addWidget(checkbox)
            wrapper = qt.QWidget()
            wrapper.setLayout(layout)
            table.setCellWidget(idx, 0, wrapper)
            table.setItem(
                idx, 1,
                qt.QTableWidgetItem(osp.relpath(path, self._dataFolder)))

        table.cellDoubleClicked.connect(pathDoubleClicked)
        table.resizeColumnsToContents()

        # ugly fix. second colum wont strength after setting data
        self.ui.progressCollapse.toggle()
        self.ui.progressCollapse.toggle()

        self.togglePrevNextBtn(self._currScanIdx)

    """ control point related """

    def createPointListNode(self, name, onMarkupNodeModified, color):
        displayNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsDisplayNode")
        displayNode.SetTextScale(0)
        displayNode.SetSelectedColor(color)

        pointListNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode")
        pointListNode.SetName(name)
        pointListNode.SetAndObserveDisplayNodeID(displayNode.GetID())

        pointListNodeObservers = []
        self.addPointListNodeObserver(pointListNode, onMarkupNodeModified)
        return pointListNode, pointListNodeObservers

    def removePointListNodeObservers(self, pointListNode,
                                     pointListNodeObservers):
        if pointListNode and pointListNodeObservers:
            for observer in pointListNodeObservers:
                pointListNode.RemoveObserver(observer)

    def addPointListNodeObserver(self, pointListNode, onMarkupNodeModified):
        pointListNodeObservers = []
        if pointListNode:
            eventIds = [slicer.vtkMRMLMarkupsNode.PointPositionDefinedEvent]
            for eventId in eventIds:
                pointListNodeObservers.append(
                    pointListNode.AddObserver(eventId, onMarkupNodeModified))
        return pointListNodeObservers

    def getControlPointsXYZ(self, pointListNode, name):
        v = self._currVolumeNode
        RasToIjkMatrix = vtk.vtkMatrix4x4()
        v.GetRASToIJKMatrix(RasToIjkMatrix)

        point_set = []
        n = pointListNode.GetNumberOfControlPoints()
        for i in range(n):
            coord = pointListNode.GetNthControlPointPosition(i)
            world = [0, 0, 0]
            pointListNode.GetNthControlPointPositionWorld(i, world)
            p_Ras = [coord[0], coord[1], coord[2], 1.0]
            p_Ijk = RasToIjkMatrix.MultiplyDoublePoint(p_Ras)
            p_Ijk = [round(i) for i in p_Ijk]
            point_set.append(p_Ijk[0:3])

        logging.info(f"{name} => Current control points: {point_set}")
        return point_set

    def getControlPointXYZ(self, pointListNode, index):
        v = self._currVolumeNode
        RasToIjkMatrix = vtk.vtkMatrix4x4()
        v.GetRASToIJKMatrix(RasToIjkMatrix)

        coord = pointListNode.GetNthControlPointPosition(index)

        world = [0, 0, 0]
        pointListNode.GetNthControlPointPositionWorld(index, world)

        p_Ras = [coord[0], coord[1], coord[2], 1.0]
        p_Ijk = RasToIjkMatrix.MultiplyDoublePoint(p_Ras)
        p_Ijk = [round(i) for i in p_Ijk]

        return p_Ijk[0:3]

    def resetPointList(self, markupsPlaceWidget, pointListNode,
                       pointListNodeObservers):
        if markupsPlaceWidget.placeModeEnabled:
            markupsPlaceWidget.setPlaceModeEnabled(False)

        if pointListNode:
            slicer.mrmlScene.RemoveNode(pointListNode)
            self.removePointListNodeObservers(pointListNode,
                                              pointListNodeObservers)

    def removePointListNodeObservers(self, pointListNode,
                                     pointListNodeObservers):
        if pointListNode and pointListNodeObservers:
            for observer in pointListNodeObservers:
                pointListNode.RemoveObserver(observer)

    def enterInteractiveMode(self):
        if self._usingInteractive:
            return

        segmentation = self.segmentation
        segmentId = self.ui.embeddedSegmentEditorWidget.currentSegmentID()
        segment = segmentation.GetSegment(segmentId)

        if len(segmentation.GetSegmentIDs()) == 0 or len(
                segmentId) == 0:  # no segment or currently no active segment
            segmentId = segmentation.AddEmptySegment("")
        else:
            if (slicer.util.arrayFromSegmentBinaryLabelmap(
                    self.segmentationNode, segmentId,
                    self._currVolumeNode).sum() != 0):
                # TODO: prompt and let user choose whether to create new segment
                segmentId = segmentation.AddEmptySegment("",
                                                         segment.GetName(),
                                                         segment.GetColor())
        self.ui.embeddedSegmentEditorWidget.setCurrentSegmentID(segmentId)

        if not TEST:
            self.setImage()
        self.clicker = Clicker()
        # TODO: scroll to the new segment
        self.ui.embeddedSegmentEditorWidget.setDisabled(True)
        self.ui.finishSegmentButton.setEnabled(True)
        self._usingInteractive = True

    def exitInteractiveMode(self):
        self.ui.dgPositiveControlPointPlacementWidget.deleteAllPoints()
        self.ui.dgNegativeControlPointPlacementWidget.deleteAllPoints()

        self.ui.dgPositiveControlPointPlacementWidget.placeButton().setChecked(
            False)
        self.ui.dgNegativeControlPointPlacementWidget.placeButton().setChecked(
            False)
        self.ui.embeddedSegmentEditorWidget.setDisabled(False)
        self.ui.finishSegmentButton.setEnabled(False)

        self._usingInteractive = False

    """ inference related """

    def loadModelClicked(self):
        model_path, param_path = self.ui.modelPathInput.currentPath, self.ui.paramPathInput.currentPath
        if not model_path or not param_path:
            slicer.util.errorDisplay(
                "Please set the model_path and parameter path before load model."
            )
            return

        self.inference_predictor = predictor.BasePredictor(
            model_path,
            param_path,
            device=self.device,
            enable_mkldnn=self.enable_mkldnn,
            **self.predictor_params_)

        slicer.util.delayDisplay(
            "Sucessfully loaded model to {}!".format(self.device),
            autoCloseMsec=1500)

    def onControlPointAdded(self, observer, eventid):
        if self._addingControlPoint:
            return
        self._addingControlPoint = True
        self.initPb("Entering interactive mode", "Doing Inference")

        if not self._usingInteractive:
            self.enterInteractiveMode()

        # 1. get new point pos and type
        self.setPb(0.1, "Preparing interactive segment")
        posPoints = self.getControlPointsXYZ(self.dgPositivePointListNode,
                                             "positive")
        negPoints = self.getControlPointsXYZ(self.dgNegativePointListNode,
                                             "negative")
        newPointIndex = observer.GetDisplayNode().GetActiveControlPoint()
        logging.info("newPointIndex", newPointIndex)
        newPointPos = self.getControlPointXYZ(observer, newPointIndex)
        isPositivePoint = False if len(
            posPoints) == 0 else newPointPos == posPoints[-1]
        logging.info(
            f"{['Negative', 'Positive'][int(isPositivePoint)]} point added at {newPointPos}"
        )

        # 2. ensure current segment empty, create if not
        segmentation = self.segmentation
        segmentId = self.ui.embeddedSegmentEditorWidget.currentSegmentID()
        segment = segmentation.GetSegment(segmentId)

        logging.info(
            f"Current segment: {self.getSegmentId(segment)} {segment.GetName()} {segment.GetLabelValue()}",
        )

        with slicer.util.tryWithErrorDisplay(
                "Failed to run inference.", waitCursor=True):
            self.setPb(0.2, "Running inference")

            # predict image for test
            if TEST:
                p = newPointPos
                p = [p[2], p[1], p[0]]
                res = slicer.util.arrayFromSegmentBinaryLabelmap(
                    self.segmentationNode, segmentId, self._currVolumeNode)
                mask = np.zeros_like(res)
                mask[p[0] - 10:p[0] + 10, p[1] - 10:p[1] + 10, p[2] - 10:p[2] +
                     10] = 1
            else:
                paddle.device.cuda.empty_cache()
                mask = self.infer_image(
                    newPointPos, isPositivePoint)  # (880, 880, 12) same as res

            self.setPb(0.9, "Wrapping up")
            # set new numpy mask to segmentation
            slicer.util.updateSegmentBinaryLabelmapFromArray(
                mask, self.segmentationNode, segmentId, self._currVolumeNode)

            if self.test_iou:
                label = sitk.ReadImage(self._currLabelPath)
                label = sitk.GetArrayFromImage(label).astype("int32")
                iou = self.get_iou(label, mask, newPointPos)
                logging.info("Current IOU is {}".format(iou))
        self.closePb()
        self._addingControlPoint = False

    def get_iou(self, gt_mask, pred_mask, newPointPos, ignore_label=-1):
        ignore_gt_mask_inv = gt_mask != ignore_label
        pred_mask = pred_mask == 1
        obj_gt_mask = gt_mask == gt_mask[newPointPos[2], newPointPos[1],
                                         newPointPos[0]]

        intersection = np.logical_and(
            np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
        union = np.logical_and(
            np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

        return intersection / union

    def prepImage(self):
        self.origin = sitk.ReadImage(self._scanPaths[self._currScanIdx])
        itk_img_res = inference.crop_wwwc(
            self.origin, max_v=self.image_ww[1], min_v=self.image_ww[0]
        )  # same as the preprocess when you train the model (512, 512, 12) WHD
        itk_img_res, self.new_spacing = inference.resampleImage(
            itk_img_res, out_size=self.train_shape)  #  origin: (880, 880, 12)

        npy_img = sitk.GetArrayFromImage(itk_img_res).astype(
            "float32")  # 12, 512, 512 DHW

        # Exchange dim and normalize
        input_data = np.expand_dims(np.transpose(npy_img, [2, 1, 0]), axis=0)
        if input_data.max() > 0:
            input_data = input_data / input_data.max()
        self.input_data = input_data

    def setImage(self):
        logging.info(
            f"输入网络前数据的形状:{self.input_data.shape}")  # shape (1, 512, 512, 12)
        try:
            self.inference_predictor.set_input_image(self.input_data)
        except AttributeError:
            slicer.util.errorDisplay(
                "Please load model first", autoCloseMsec=1200)

    def infer_image(self,
                    click_position=None,
                    positive_click=True,
                    pred_thr=0.49):
        """
        click_position: one or serveral clicks represent by list like: [[234, 284, 7]]
        positive_click: whether this click is positive or negative
        """
        try:
            paddle.device.set_device(self.device)
        except AttributeError:
            slicer.util.errorDisplay(
                "Model is not loaded. Please load model first")
            return
        except ValueError:
            slicer.util.errorDisplay(
                "The AI-assisted image infer process need to be run on gpu device, please install paddle with GPU enabled."
            )

        tic = time.time()
        self.prepare_click(click_position, positive_click)
        with paddle.no_grad():
            pred_probs = self.inference_predictor.get_prediction_noclicker(
                self.clicker)

        output_data = (pred_probs > pred_thr) * pred_probs  # (12, 512, 512) DHW
        output_data[output_data > 0] = 1

        # Load mask from model infer result, and change from numpy to simpleitk
        output_data = np.transpose(output_data, [2, 1, 0])
        mask_itk_new = sitk.GetImageFromArray(output_data)  # (512, 512, 12) WHD
        mask_itk_new.SetSpacing(self.new_spacing)
        mask_itk_new.SetOrigin(self.origin.GetOrigin())
        mask_itk_new.SetDirection(self.origin.GetDirection())
        mask_itk_new = sitk.Cast(mask_itk_new, sitk.sitkUInt8)

        # if need max connect opponet filter, add it before here.
        Mask, _ = inference.resampleImage(mask_itk_new,
                                          self.origin.GetSize(),
                                          self.origin.GetSpacing(),
                                          sitk.sitkNearestNeighbor)
        Mask.CopyInformation(self.origin)

        npy_img = sitk.GetArrayFromImage(Mask).astype(
            "float32")  # 12, 512, 512 DHW

        logging.info(
            f"预测结果的形状：{output_data.shape}, 预测时间为 {(time.time() - tic) * 1000} ms"
        )  # shape (12, 512, 512) DHW test

        return npy_img

    def prepare_click(self, click_position, positive_click):
        click_position_new = []
        for i, v in enumerate(click_position):
            click_position_new.append(int(self.ratio[i] * click_position[i]))

        if positive_click:
            click_position_new.append(100)
        else:
            click_position_new.append(-100)

        logging.info("The {} click is click on {} (resampled)".format(
            ["negative", "positive"][positive_click],
            click_position_new))  # result is correct

        click = inference.Click(
            is_positive=positive_click, coords=click_position_new)
        self.clicker.add_click(click)
        logging.info("####################### clicker length",
                     len(self.clicker.clicks_list))

    """ saving related """

    def finishScan(self):
        if self._usingInteractive:
            self.exitInteractiveMode()
        self.saveSegmentation()
        self._finishedPaths.append(self._scanPaths[self._currScanIdx])
        self.saveProgress()
        self.updateProgressWidgets()
        if self._lastTurnNextScan:
            self.nextScan()
        else:
            self.prevScan()

    def saveSegmentation(self):
        """
        save segmentation mask to self._dataFolder
        """
        tic = time.time()
        if not self._dirty:
            logging.info("Segmentation not changed, skip saving")
            slicer.app.processEvents()
            return

        catgs = self.getCatgFromFile()
        segmentationNode = self.segmentationNode
        logging.info("segmentationNode", segmentationNode)
        segmentation = segmentationNode.GetSegmentation()

        # 2. prepare save path
        scanPath = self._scanPaths[self._currScanIdx]
        dotPos = scanPath.find(".")
        labelPath = scanPath[:dotPos] + "_label" + scanPath[dotPos:]

        # 3. save
        colorTableNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLColorTableNode")
        colorTableNode.SetTypeToUser()
        colorTableNode.SetNumberOfColors(len(self.segmentation.GetSegmentIDs()))
        colorTableNode.UnRegister(None)
        colorTableNode.SetNamesInitialised(True)

        for segment in self.segments:
            colorTableNode.SetColor(catgs[segment.GetName()],
                                    segment.GetName(), *segment.GetColor(), 1.0)

        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLLabelMapVolumeNode")

        slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
            segmentationNode,
            segmentation.GetSegmentIDs(),
            labelmapVolumeNode,
            self._currVolumeNode,
            segmentation.EXTENT_UNION_OF_EFFECTIVE_SEGMENTS,
            colorTableNode, )

        res = slicer.util.saveNode(labelmapVolumeNode, labelPath)

        # clean up useless nodes
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)

        if res:
            logging.info(f"{labelPath.split('/')[-1]} save successfully.")
        else:
            slicer.util.errorDisplay(f"{labelPath.split('/')[-1]} save failed!")

        self._dirty = False
        logging.info(f"saving took {time.time() - tic}s")

    """ display related """

    def opacityUi2Display(self):
        segmentationNode = self.segmentationNode
        if segmentationNode is None:
            return
        threshold = self.ui.opacitySlider.value
        displayNode = segmentationNode.GetDisplayNode()
        displayNode.SetOpacity3D(threshold)  # Set opacity for 3d render
        displayNode.SetOpacity(threshold)  # Set opacity for 2d

    def opacityDisplay2Ui(self):
        segmentationNode = self.segmentationNode
        if segmentationNode is not None:
            displayNode = segmentationNode.GetDisplayNode()
            if displayNode is not None:
                opacity = displayNode.GetOpacity()
                self.ui.opacitySlider.value = opacity

    """ life cycle related """

    def cleanup(self):
        """
        Called when the application closes and the module widget is destroyed.
        """
        self._turninig = True
        self.clearScene(clearAllVolumes=True)
        self.removeObservers()
        self.resetPointList(
            self.ui.dgPositiveControlPointPlacementWidget,
            self.dgPositivePointListNode,
            self.dgPositivePointListNodeObservers, )
        self.dgPositivePointListNode = None
        self.resetPointList(
            self.ui.dgNegativeControlPointPlacementWidget,
            self.dgNegativePointListNode,
            self.dgNegativePointListNodeObservers, )
        self.dgNegativePointListNode = None

    def enter(self):
        """
        Called each time the user opens this module. Not when reload/switch back.
        """

    def exit(self):
        """
        Called each time the user opens a different module.
        """
        # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
        self.removeObserver(
            self._parameterNode,
            vtk.vtkCommand.ModifiedEvent,
            self.updateGUIFromParameterNode, )

    def onReload(self):
        self.cleanup()
        super().onReload()

    def onSceneEndImport(self, caller, event):
        """
        Called after reload and after scan/segmentation is imported
        """
        if self._endImportProcessing:
            return
        self._endImportProcessing = True

        self._endImportProcessing = False

    def onSceneStartClose(self, caller, event):
        """
        Called just before the scene is closed.
        """
        self._turninig = True
        # Parameter node will be reset, do not use it anymore
        self.saveProgress()

        self.setParameterNode(None)
        self.resetPointList(
            self.ui.dgPositiveControlPointPlacementWidget,
            self.dgPositivePointListNode,
            self.dgPositivePointListNodeObservers, )
        self.dgPositivePointListNode = None
        self.resetPointList(
            self.ui.dgNegativeControlPointPlacementWidget,
            self.dgNegativePointListNode,
            self.dgNegativePointListNodeObservers, )
        self.dgNegativePointListNode = None

    def onSceneEndClose(self, caller, event):
        """
        Called just after the scene is closed.
        """
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeFromNode(self):
        """
        Ensure parameter node exists and observed.
        """
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.
        self.setParameterNode(self.logic.getParameterNode())
        segNode = self.segmentationNode
        if segNode is not None:
            self.ui.opacitySlider.setValue(segNode.GetDisplayNode().GetOpacity(
            ))

    def setParameterNode(self, inputParameterNode):
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        # Unobserve previously selected parameter node and add an observer to the newly selected.
        # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
        # those are reflected immediately in the GUI.
        if self._parameterNode is not None:
            self.removeObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode, )
        self._parameterNode = inputParameterNode
        if self._parameterNode is not None:
            self.addObserver(
                self._parameterNode,
                vtk.vtkCommand.ModifiedEvent,
                self.updateGUIFromParameterNode, )

        # Initial GUI update
        self.updateGUIFromParameterNode()

    def updateGUIFromParameterNode(self, caller=None, event=None):
        """
        This method is called whenever parameter node is changed.
        The module GUI is updated to show the current state of the parameter node.
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
        self._updatingGUIFromParameterNode = True
        if not self.dgPositivePointListNode:
            (
                self.dgPositivePointListNode,
                self.dgPositivePointListNodeObservers,
            ) = self.createPointListNode("P", self.onControlPointAdded,
                                         [0.5, 1, 0.5])
            self.ui.dgPositiveControlPointPlacementWidget.setCurrentNode(
                self.dgPositivePointListNode)
            self.ui.dgPositiveControlPointPlacementWidget.setPlaceModeEnabled(
                False)

        if not self.dgNegativePointListNode:
            (
                self.dgNegativePointListNode,
                self.dgNegativePointListNodeObservers,
            ) = self.createPointListNode("P", self.onControlPointAdded,
                                         [0.5, 1, 0.5])

            self.ui.dgNegativeControlPointPlacementWidget.setCurrentNode(
                self.dgNegativePointListNode)
            self.ui.dgNegativeControlPointPlacementWidget.setPlaceModeEnabled(
                False)

        # All the GUI updates are done
        self._updatingGUIFromParameterNode = False

    def updateParameterNodeFromGUI(self, caller=None, event=None):
        """
        This method is called when the user makes any change in the GUI.
        The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
        """

        if self._parameterNode is None or self._updatingGUIFromParameterNode:
            return

        wasModified = self._parameterNode.StartModify(
        )  # Modify all properties in a single batch

        self._parameterNode.EndModify(wasModified)


#
# EISegMed3DLogic
#
class EISegMed3DLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)

    def setDefaultParameters(self, parameterNode):
        """
        Initialize parameter node with default settings.
        """
        if not parameterNode.GetParameter("Threshold"):
            parameterNode.SetParameter("Threshold", "100.0")

    def process(self,
                inputVolume,
                outputVolume,
                imageThreshold,
                invert=False,
                showResult=True):
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(
            slicer.modules.thresholdscalarvolume,
            None,
            cliParams,
            wait_for_completion=True,
            update_display=showResult, )
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(
            f"Processing completed in {stopTime - startTime:.2f} seconds")

    def get_segment_editor_node(self):
        # Use the Segment Editor module's parameter node for the embedded segment editor widget.
        # This ensures that if the user switches to the Segment Editor then the selected
        # segmentation node, volume node, etc. are the same.
        segmentEditorSingletonTag = "SegmentEditor"
        segmentEditorNode = slicer.mrmlScene.GetSingletonNode(
            segmentEditorSingletonTag, "vtkMRMLSegmentEditorNode")
        if segmentEditorNode is None:
            segmentEditorNode = slicer.mrmlScene.CreateNodeByClass(
                "vtkMRMLSegmentEditorNode")
            segmentEditorNode.UnRegister(None)
            segmentEditorNode.SetSingletonTag(segmentEditorSingletonTag)
            segmentEditorNode = slicer.mrmlScene.AddNode(segmentEditorNode)
        return segmentEditorNode


#
# EISegMed3DTest
#


class EISegMed3DTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_EISegMed3D1()

    def test_EISegMed3D1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("EISegMed3D1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = EISegMed3DLogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
