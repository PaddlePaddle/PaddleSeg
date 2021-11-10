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


from qtpy import QtWidgets, QtGui, QtCore


class LineItem(QtWidgets.QGraphicsLineItem):
    fixedWidth = 1

    def __init__(self, annotation_item, idx, color):
        super(LineItem, self).__init__()
        self.polygon_item = annotation_item
        self.idx = idx
        self.color = color
        self.setPen(QtGui.QPen(color, self.width))

        self.setZValue(11)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, True)
        self.setAcceptHoverEvents(True)

    def setColor(self, color):
        self.setPen(QtGui.QPen(color, self.width))
        self.color = color

    @property
    def width(self):
        if not self.scene():
            width = 1
        else:
            width = LineItem.fixedWidth / self.scene().scale
        return width

    def updateWidth(self):
        self.setPen(QtGui.QPen(self.color, self.width))

    def hoverEnterEvent(self, ev):
        self.polygon_item.line_hovering = True
        self.setPen(QtGui.QPen(self.color, self.width * 3))
        super(LineItem, self).hoverEnterEvent(ev)

    def hoverLeaveEvent(self, ev):
        self.polygon_item.line_hovering = False
        self.setPen(QtGui.QPen(self.color, self.width))
        super(LineItem, self).hoverLeaveEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        self.setPen(QtGui.QPen(self.color, self.width))
        self.polygon_item.addPointMiddle(self.idx, ev.pos())
        super(LineItem, self).mouseDoubleClickEvent(ev)

    def shape(self):
        path = QtGui.QPainterPath()
        path.addPolygon(self.boundingPolygon())
        return path

    def boundingPolygon(self):
        w = self.width * 10
        w = max(w, 3)
        p = QtCore.QPointF(w, 0)
        s, e = self.line().p1(), self.line().p2()
        poly = QtGui.QPolygonF([s - p, s + p, e + p, e - p])
        return poly
