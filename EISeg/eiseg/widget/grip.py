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

from PyQt5.QtCore import QPointF
from qtpy import QtWidgets, QtGui, QtCore


class GripItem(QtWidgets.QGraphicsPathItem):
    maxSize = 1.5
    minSize = 0.8

    def __init__(self, annotation_item, index, color, img_size):
        super(GripItem, self).__init__()
        self.m_annotation_item = annotation_item
        self.hovering = False
        self.m_index = index
        self.anning = True
        color.setAlphaF(1)
        self.color = color
        self.img_size = img_size

        self.updateSize()
        self.setPath(self.circle)
        self.setBrush(self.color)
        self.setPen(QtGui.QPen(self.color, 1))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(12)
        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

    def setColor(self, color):
        self.setBrush(color)
        self.setPen(QtGui.QPen(color, 1))
        self.color = color

    def setAnning(self, anning=True):
        self.anning = anning
        self.setEnabled(anning)

    # BUG: Scaling causes a crash
    @property
    def size(self):
        return GripItem.minSize
        # if not self.scene():
        #     return GripItem.minSize
        # else:
        #     maxi, mini = GripItem.maxSize, GripItem.minSize
        #     exp = 1 - mini / maxi
        #     size = maxi * (1 - exp ** self.scene().scale)
        #     if size > GripItem.maxSize:
        #         size = GripItem.maxSize
        #     if size < GripItem.minSize:
        #         size = GripItem.minSize
        #     return size

    def updateSize(self, s=2):
        size = self.size
        self.circle = QtGui.QPainterPath()
        self.circle.addEllipse(QtCore.QRectF(-size, -size, size * s, size * s))
        self.square = QtGui.QPainterPath()
        self.square.addRect(QtCore.QRectF(-size, -size, size * s, size * s))
        self.setPath(self.square if self.hovering else self.circle)

    def hoverEnterEvent(self, ev):
        self.setPath(self.square)
        self.setBrush(QtGui.QColor(0, 0, 0, 0))
        self.m_annotation_item.item_hovering = True
        self.hovring = True
        super(GripItem, self).hoverEnterEvent(ev)

    def hoverLeaveEvent(self, ev):
        self.setPath(self.circle)
        self.setBrush(self.color)
        self.m_annotation_item.item_hovering = False
        self.hovring = False
        super(GripItem, self).hoverLeaveEvent(ev)

    def mouseReleaseEvent(self, ev):
        self.setSelected(False)
        super(GripItem, self).mouseReleaseEvent(ev)

    def itemChange(self, change, value):
        tmp_val = value
        if change == QtWidgets.QGraphicsItem.ItemPositionChange and self.isEnabled(
        ):
            if value.x() > self.img_size[1]:
                x = self.img_size[1]
            elif value.x() < 0:
                x = 0
            else:
                x = value.x()
            if value.y() > self.img_size[0]:
                y = self.img_size[0]
            elif value.y() < 0:
                y = 0
            else:
                y = value.y()
            tmp_val = QPointF(x, y)
            self.m_annotation_item.movePoint(self.m_index, tmp_val)
            self.m_annotation_item.setDirty(True)
        return super(GripItem, self).itemChange(change, tmp_val)

    def shape(self):
        path = QtGui.QPainterPath()
        p = self.mapFromScene(self.pos())
        x, y = p.x(), p.y()
        s = self.size
        path.addEllipse(p, s + GripItem.minSize, s + GripItem.minSize)
        return path

    def mouseDoubleClickEvent(self, ev):
        self.m_annotation_item.removeFocusPoint()
