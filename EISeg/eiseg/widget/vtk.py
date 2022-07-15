# reference: https://blog.csdn.net/weixin_46579211/article/details/118279231
import typing
import numpy as np
from qtpy.QtWidgets import QWidget, QVBoxLayout


# 只有调用convert_vtk才会加载vtk相关的并把这个控件转换为真正的vtk控件
vtk = None
QVTKRenderWindowInteractor = None
vtkImageImportFromArray = None


class VTKWidget(QWidget):
    def __init__(self, parent: typing.Optional["QWidget"]) -> None:
        super().__init__(parent)
        self.setObjectName("vtkWidget")
        self.vlayer = QVBoxLayout(self)
        # default setting
        self.smoothing_iterations = 10
        self.pass_band = 0.005
        self.feature_angle = 120
        self.import_vtk = False
        
    def convert_vtk(self) -> bool:
        try:
            import vtk
            from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
            from vtkmodules.util.vtkImageImportFromArray import vtkImageImportFromArray
            global vtk
            global QVTKRenderWindowInteractor
            global vtkImageImportFromArray
            self.import_vtk = True
            vtk.vtkFileOutputWindow().SetGlobalWarningDisplay(0)
            self.init()
        except:
            self.import_vtk = False
        finally:
            return self.import_vtk

    def init(self, clear: bool = True) -> None:
        if self.import_vtk is False:
            return
        # remove
        item = self.vlayer.itemAt(0)
        self.vlayer.removeItem(item)
        if item is not None and item.widget():
            item.widget().deleteLater()
        # add
        self.renderer = vtk.vtkRenderer()
        self.interactor = QVTKRenderWindowInteractor(self)
        self.interactor.GetRenderWindow().AddRenderer(self.renderer)
        self.vlayer.addWidget(self.interactor)
        if clear:
            # set background
            self.renderer.SetBackground(vtk.vtkNamedColors().GetColor3d("black"))
            self.interactor.Start()

    def show_array(self, data: np.ndarray, spacing: typing.Tuple, color_map: typing.List) -> None:
        if self.import_vtk is False:
            return
        print("color_map:", color_map)
        self.num_block = len(np.unique(data))
        print("num_block:", self.num_block)
        self.reader = vtkImageImportFromArray()
        self.reader.SetArray(data)
        self.reader.SetDataSpacing(spacing)
        mbds = vtk.vtkMultiBlockDataSet()
        mbds.SetNumberOfBlocks(self.num_block - 1)
        for iter in range(1, self.num_block + 1):
            contour = self._get_mc_contour(iter)
            smoother = self._smoothing(
                self.smoothing_iterations, 
                self.pass_band, 
                self.feature_angle, 
                contour
            )
            mbds.SetBlock(iter, smoother.GetOutput())
        self._multidisplay(mbds, color_map)

    def _get_mc_contour(self, setvalue: int) -> typing.Any:
        contour = vtk.vtkDiscreteMarchingCubes()
        contour.SetInputConnection(self.reader.GetOutputPort())
        contour.ComputeNormalsOn()
        contour.SetValue(0, setvalue)
        return contour

    def _smoothing(self, 
                   smoothing_iterations: int, 
                   pass_band: float, 
                   feature_angle: int, 
                   contour: typing.Any) -> typing.Any:
        smoother = vtk.vtkWindowedSincPolyDataFilter()
        smoother.SetInputConnection(contour.GetOutputPort())
        smoother.SetNumberOfIterations(smoothing_iterations)
        smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(feature_angle)
        smoother.SetPassBand(pass_band)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()
        return smoother

    def _multidisplay(self, obj: typing.Any, color_map: typing.List) -> None:
        self.init(False)
        mapper = vtk.vtkCompositePolyDataMapper2()
        mapper.SetInputDataObject(obj)
        cdsa = vtk.vtkCompositeDataDisplayAttributes()
        mapper.SetCompositeDataDisplayAttributes(cdsa)
        # 上色
        color_map.insert(0, [0., 0., 0.])
        for i in range(1, self.num_block + 1):
            r, g, b = color_map[i - 1]
            mapper.SetBlockColor(i, r / 255., g / 255., b / 255.)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.RotateX(180)  # 翻转一下才对
        # Enable user interface interactor.
        self.renderer.AddActor(actor)
        self.interactor.Start()
