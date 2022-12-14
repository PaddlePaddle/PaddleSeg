# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Qt, QPointF
from qtpy.QtGui import QPen, QColor


class AnnotationScene(QtWidgets.QGraphicsScene):
    clickRequest = QtCore.Signal(int, int, bool)
    rectboxReques = QtCore.Signal(int, int, int, int)
    focusRequest = QtCore.Signal(int)

    def __init__(self, parent=None):
        super(AnnotationScene, self).__init__(parent)
        self.creating = False
        self.det_mode = False
        self.polygon_items = []
        self.scale = 1
        # draw cross
        self.coords = None
        self.pen = QPen()
        self.pen.setWidth(2)
        self.pen.setColor(QColor(0, 0, 0, 127))
        # draw rect_box
        self.is_draw = False
        self.is_press = False
        self.rect_box = [0, 0, 0, 0]

    def setPenColor(self, color_list):
        R, G, B, A = color_list
        self.pen.setColor(QColor(R, G, B, A))

    def updatePenSize(self):
        self.pen.setWidth(max(1, int(2 / self.scale + 1e-12)))

    def updatePolygonSize(self):
        for poly in self.polygon_items:
            for grip in poly.m_items:
                grip.updateSize()
            for line in poly.m_lines:
                line.updateWidth()

    def setCreating(self, creating=True):
        self.creating = creating

    def setDetMode(self, det_mode=True):
        self.det_mode = det_mode

    def mousePressEvent(self, ev):
        pos = ev.scenePos()
        if not self.creating and not self.hovering:
            if ev.buttons() in [Qt.LeftButton, Qt.RightButton]:
                x, y = int(pos.x()), int(pos.y())
                self.clickRequest.emit(x, y, ev.buttons() == Qt.LeftButton)
                if self.is_draw:
                    self.rect_box = [x, y, x, y]
                    self.rectboxReques.emit(*self.rect_box)  # 添加一个空矩形用于被删除
                    self.is_press = True
        elif self.hovering:
            for poly in self.polygon_items:
                if poly.polygon_hovering:
                    self.focusRequest.emit(poly.labelIndex)
        super(AnnotationScene, self).mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        pos = ev.scenePos()
        if not self.creating and self.is_draw and self.is_press:
            self.rect_box[2] = int(pos.x())
            self.rect_box[3] = int(pos.y())
            if len(self.polygon_items) != 0:
                self.polygon_items[-1].remove()
            self.rectboxReques.emit(*self.rect_box)
        super(AnnotationScene, self).mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if self.is_press and \
            (self.rect_box[0] == self.rect_box[2] or self.rect_box[1] == self.rect_box[3]):
            if len(self.polygon_items) != 0:
                self.polygon_items[-1].remove()
        self.is_press = False
        return super(AnnotationScene, self).mouseReleaseEvent(ev)

    def drawForeground(self, painter, rect):
        if not self.det_mode:
            return
        if self.coords is not None and self.coords != QPointF(-1, -1):
            painter.setClipRect(rect)
            painter.setPen(self.pen)
            painter.drawLine(
                int(self.coords.x()),
                int(rect.top()), int(self.coords.x()), int(rect.bottom() + 1))
            painter.drawLine(
                int(rect.left()),
                int(self.coords.y()),
                int(rect.right() + 1), int(self.coords.y()))

    def onMouseChanged(self, pointf):
        self.coords = pointf
        self.invalidate()

    @property
    def item_hovering(self):
        for poly in self.polygon_items:
            if poly.item_hovering:
                return True
        return False

    @property
    def polygon_hovering(self):
        for poly in self.polygon_items:
            if poly.polygon_hovering:
                return True
        return False

    @property
    def line_hovering(self):
        for poly in self.polygon_items:
            if poly.line_hovering:
                return True
        return False

    @property
    def hovering(self):
        return self.item_hovering or self.polygon_hovering or self.line_hovering
