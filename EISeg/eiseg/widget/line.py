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
    maxWidth = 1
    minWidth = 0.5

    def __init__(self, annotation_item, idx, color):
        super(LineItem, self).__init__()
        self.polygon_item = annotation_item
        self.idx = idx
        self.color = color
        self.anning = True
        self.setPen(QtGui.QPen(color, self.width))

        self.setZValue(11)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, True)
        # self.setFlag(QtWidgets.QGraphicsItem.ItemClipsToShape, True)
        self.setAcceptHoverEvents(True)
        self.setBoundingRegionGranularity(0.5)
        self.updateWidth()

    def setColor(self, color):
        self.setPen(QtGui.QPen(color, self.width))
        self.color = color

    def setAnning(self, anning=True):
        self.anning = anning
        self.setEnabled(anning)
        self.updateWidth()

    # BUG: Scaling causes a crash
    @property
    def width(self):
        return LineItem.minWidth
        # if not self.scene():
        #     width = LineItem.minWidth
        # else:
        #     maxi, mini = LineItem.maxWidth, LineItem.minWidth
        #     exp = 1 - mini / maxi
        #     width = maxi * (1 - exp ** self.scene().scale)
        #     if width > LineItem.maxWidth:
        #         width = LineItem.maxWidth
        #     if width < LineItem.minWidth:
        #         width = LineItem.minWidth
        # return width

    def updateWidth(self):
        self.setPen(QtGui.QPen(self.color, self.width))

    def hoverEnterEvent(self, ev):
        self.boundingPolygon(True)
        print("hover in")
        if self.anning:
            self.polygon_item.line_hovering = True
            self.setPen(QtGui.QPen(self.color, self.width * 1.4))
        super(LineItem, self).hoverEnterEvent(ev)

    def hoverLeaveEvent(self, ev):
        self.polygon_item.line_hovering = False
        self.setPen(QtGui.QPen(self.color, self.width))
        super(LineItem, self).hoverLeaveEvent(ev)

    def mouseDoubleClickEvent(self, ev):
        print("anning", self.anning)
        if self.anning:
            self.setPen(QtGui.QPen(self.color, self.width))
            self.polygon_item.addPointMiddle(self.idx, ev.pos())
        super(LineItem, self).mouseDoubleClickEvent(ev)

    def shape(self):
        path = QtGui.QPainterPath()
        path.addPolygon(self.boundingPolygon(False))
        return path

    # def shape(self):
    #     path = QtGui.QPainterPath()
    #     path.moveTo(self.line().p1())
    #     path.lineTo(self.line().p2())
    #     path.setPen(QtGui.QPen(self.color, self.width * 3))
    #     return path

    def boundingPolygon(self, debug):
        w = self.width * 1.5
        w = min(w, 2)
        s, e = self.line().p1(), self.line().p2()
        dir = s - e
        dx, dy = -dir.y(), dir.x()
        norm = (dx**2 + dy**2)**(1 / 2)
        if debug:
            print(
                self.width,
                w,
                s.x(),
                s.y(),
                e.x(),
                e.y(),
                dir.x(),
                dir.y(),
                dx,
                dy,
                norm, )
        dx /= (norm + 1e-16)
        dy /= (norm + 1e-16)
        if debug:
            print("dir", dx, dy)
        p = QtCore.QPointF(dx * w, dy * w)
        poly = QtGui.QPolygonF([s - p, s + p, e + p, e - p])
        return poly
