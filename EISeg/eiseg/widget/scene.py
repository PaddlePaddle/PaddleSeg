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


from qtpy import QtWidgets, QtCore
from qtpy.QtCore import Qt


class AnnotationScene(QtWidgets.QGraphicsScene):
    clickRequest = QtCore.Signal(int, int, bool)

    def __init__(self, parent=None):
        super(AnnotationScene, self).__init__(parent)
        self.creating = False
        self.polygon_items = []

    def updatePolygonSize(self):
        for poly in self.polygon_items:
            for grip in poly.m_items:
                grip.updateSize()
            for line in poly.m_lines:
                line.updateWidth()

    def setCreating(self, creating=True):
        self.creating = creating

    def mousePressEvent(self, ev):
        pos = ev.scenePos()
        if not self.creating and not self.hovering:
            if ev.buttons() in [Qt.LeftButton, Qt.RightButton]:
                self.clickRequest.emit(
                    int(pos.x()), int(pos.y()), ev.buttons() == Qt.LeftButton
                )
        elif self.creating:
            self.polygon_item.removeLastPoint()
            self.polygon_item.addPointLast(ev.scenePos())
            # movable element
            self.polygon_item.addPointLast(ev.scenePos())
        super(AnnotationScene, self).mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.creating:
            self.polygon_item.movePoint(
                # self.polygon_item.number_of_points() - 1, ev.scenePos()
                len(self.polygon_item) - 1,
                ev.scenePos(),
            )
        super(AnnotationScene, self).mouseMoveEvent(ev)

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
