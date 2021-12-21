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


from qtpy import QtWidgets, QtCore, QtGui
from qtpy.QtCore import Qt


class AnnotationView(QtWidgets.QGraphicsView):
    zoomRequest = QtCore.Signal(float)

    def __init__(self, *args):
        super(AnnotationView, self).__init__(*args)
        self.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform
        )
        self.setMouseTracking(True)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.NoAnchor)
        self.point = QtCore.QPoint(0, 0)
        self.middle_click = False
        self.zoom_all = 1

    def wheelEvent(self, ev):
        if ev.modifiers() & QtCore.Qt.ControlModifier:
            zoom = 1 + ev.angleDelta().y() / 2880
            self.zoom_all *= zoom
            oldPos = self.mapToScene(ev.pos())
            if self.zoom_all >= 0.02 and self.zoom_all <= 50:  # 限制缩放的倍数
                self.scale(zoom, zoom)
            newPos = self.mapToScene(ev.pos())
            delta = newPos - oldPos
            self.translate(delta.x(), delta.y())
            ev.ignore()
            self.zoomRequest.emit(self.zoom_all)
        else:
            super(AnnotationView, self).wheelEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.middle_click and (
            self.horizontalScrollBar().isVisible()
            or self.verticalScrollBar().isVisible()
        ):
            # 放大到出现滚动条才允许拖动，避免出现抖动
            self._endPos = ev.pos() / self.zoom_all - self._startPos / self.zoom_all
            # 这儿不写为先减后除，这样会造成速度不一致
            self.point = self.point + self._endPos
            self._startPos = ev.pos()
            self.translate(self._endPos.x(), self._endPos.y())
        super(AnnotationView, self).mouseMoveEvent(ev)

    def mousePressEvent(self, ev):
        if ev.buttons() == Qt.MiddleButton:
            self.middle_click = True
            self._startPos = ev.pos()
        super(AnnotationView, self).mousePressEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MiddleButton:
            self.middle_click = False
        super(AnnotationView, self).mouseReleaseEvent(ev)
