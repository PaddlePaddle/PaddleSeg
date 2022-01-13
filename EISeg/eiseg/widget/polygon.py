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

from . import GripItem, LineItem, BBoxAnnotation


class PolygonAnnotation(QtWidgets.QGraphicsPolygonItem):
    def __init__(
        self,
        labelIndex,
        shape,
        delPolygon,
        setDirty,
        insideColor=[255, 0, 0],
        borderColor=[0, 255, 0],
        opacity=0.5,
        cocoIndex=None,
        parent=None,
    ):
        super(PolygonAnnotation, self).__init__(parent)
        self.points = []
        self.m_items = []
        self.m_lines = []
        self.coco_id = cocoIndex
        self.height, self.width = shape[:2]
        self.delPolygon = delPolygon
        self.setDirty = setDirty

        self.labelIndex = labelIndex
        self.item_hovering = False
        self.polygon_hovering = False
        self.anning = False  # 是否标注模式
        self.line_hovering = False
        self.noMove = False
        self.last_focse = False  # 之前是不是焦点在

        self.setZValue(10)
        self.opacity = opacity
        i = insideColor
        self.insideColor = QtGui.QColor(i[0], i[1], i[2])
        self.insideColor.setAlphaF(opacity)
        self.halfInsideColor = QtGui.QColor(i[0], i[1], i[2])
        self.halfInsideColor.setAlphaF(opacity / 2)
        self.setBrush(self.halfInsideColor)
        b = borderColor
        self.borderColor = QtGui.QColor(b[0], b[1], b[2])
        self.borderColor.setAlphaF(0.8)
        self.setPen(QtGui.QPen(self.borderColor))
        self.setAcceptHoverEvents(True)

        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, True)

        self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))

        # persistent this bbox instance and update when needed
        self.bbox = BBoxAnnotation(labelIndex, self, cocoIndex, self)
        self.bbox.setParentItem(self)

    @property
    def scnenePoints(self):
        points = []
        for p in self.points:
            p = self.mapToScene(p)
            points.append([p.x(), p.y()])
        return points

    def setAnning(self, isAnning=True):
        if isAnning:
            self.setAcceptHoverEvents(False)
            self.last_focse = self.polygon_hovering
            self.polygon_hovering = False
            self.anning = True
            self.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
            self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, False)
            # self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
            self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, False)
            self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, False)
            self.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
            for line in self.m_lines:
                line.setAnning(False)
            for grip in self.m_items:
                grip.setAnning(False)
        else:
            self.setAcceptHoverEvents(True)
            self.anning = False
            if self.last_focse:
                self.polygon_hovering = True
                self.setBrush(self.insideColor)
            else:
                self.setBrush(self.halfInsideColor)
            self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
            # self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
            self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
            self.setFlag(QtWidgets.QGraphicsItem.ItemIsFocusable, True)
            self.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
            for line in self.m_lines:
                line.setAnning(True)
            for grip in self.m_items:
                grip.setAnning(True)

    def addPointMiddle(self, lineIdx, point):
        gripItem = GripItem(self, lineIdx + 1, self.borderColor, (self.height, self.width))
        gripItem.setEnabled(False)
        gripItem.setPos(point)
        self.scene().addItem(gripItem)
        gripItem.updateSize()
        gripItem.setEnabled(True)
        for grip in self.m_items[lineIdx + 1 :]:
            grip.m_index += 1
        self.m_items.insert(lineIdx + 1, gripItem)
        self.points.insert(lineIdx + 1, self.mapFromScene(point))
        self.setPolygon(QtGui.QPolygonF(self.points))
        self.bbox.update()
        for line in self.m_lines[lineIdx + 1 :]:
            line.idx += 1
        line = QtCore.QLineF(self.mapToScene(self.points[lineIdx]), point)
        self.m_lines[lineIdx].setLine(line)
        lineItem = LineItem(self, lineIdx + 1, self.borderColor)
        line = QtCore.QLineF(
            point,
            self.mapToScene(self.points[(lineIdx + 2) % len(self)]),
        )
        lineItem.setLine(line)
        self.m_lines.insert(lineIdx + 1, lineItem)
        self.scene().addItem(lineItem)
        lineItem.updateWidth()

    def addPointLast(self, p):
        grip = GripItem(self, len(self), self.borderColor, (self.height, self.width))
        self.scene().addItem(grip)
        self.m_items.append(grip)
        grip.updateSize()
        grip.setPos(p)
        if len(self) == 0:
            line = LineItem(self, len(self), self.borderColor)
            self.scene().addItem(line)
            self.m_lines.append(line)
            line.setLine(QtCore.QLineF(p, p))
        else:
            self.m_lines[-1].setLine(QtCore.QLineF(self.points[-1], p))
            line = LineItem(self, len(self), self.borderColor)
            self.scene().addItem(line)
            self.m_lines.append(line)
            line.setLine(QtCore.QLineF(p, self.points[0]))

        self.points.append(p)
        self.setPolygon(QtGui.QPolygonF(self.points))
        self.bbox.update()

    def remove(self):
        for grip in self.m_items:
            self.scene().removeItem(grip)
        for line in self.m_lines:
            self.scene().removeItem(line)
        while len(self.m_items) != 0:
            self.m_items.pop()
        while len(self.m_lines) != 0:
            self.m_lines.pop()
        self.scene().polygon_items.remove(self)
        self.scene().removeItem(self)
        self.bbox.remove_from_scene()
        del self.bbox
        del self

    def removeFocusPoint(self):
        focusIdx = None
        for idx, item in enumerate(self.m_items):
            if item.hasFocus():
                focusIdx = idx
                break
        if focusIdx is not None:
            if len(self) <= 3:
                self.delPolygon(self)  # 调用app的删除多边形，为了同时删除coco标签
                return
            del self.points[focusIdx]
            self.setPolygon(QtGui.QPolygonF(self.points))
            self.bbox.update()
            self.scene().removeItem(self.m_items[focusIdx])
            del self.m_items[focusIdx]
            for grip in self.m_items[focusIdx:]:
                grip.m_index -= 1

            self.scene().removeItem(self.m_lines[focusIdx])
            del self.m_lines[focusIdx]
            line = QtCore.QLineF(
                self.mapToScene(self.points[(focusIdx - 1) % len(self)]),
                self.mapToScene(self.points[focusIdx % len(self)]),
            )
            # print((focusIdx - 1) % len(self), len(self.m_lines), len(self))
            self.m_lines[(focusIdx - 1) % len(self)].setLine(line)
            for line in self.m_lines[focusIdx:]:
                line.idx -= 1

    def removeLastPoint(self):
        # TODO: 创建的时候用到，需要删line
        if len(self.points) == 0:
            self.points.pop()
            self.setPolygon(QtGui.QPolygonF(self.points))
            self.bbox.update()
            it = self.m_items.pop()
            self.scene().removeItem(it)
            del it

    def movePoint(self, i, p):
        # print("Move point", i, p)
        if 0 <= i < len(self.points):
            p = self.mapFromScene(p)
            self.points[i] = p
            self.setPolygon(QtGui.QPolygonF(self.points))
            self.bbox.update()
            self.moveLine(i)

    def moveLine(self, i):
        # print("Moving line: ", i, self.noMove)
        if self.noMove:
            return
        points = self.points
        # line[i]
        line = QtCore.QLineF(
            self.mapToScene(points[i]), self.mapToScene(points[(i + 1) % len(self)])
        )
        self.m_lines[i].setLine(line)
        # line[i-1]
        line = QtCore.QLineF(
            self.mapToScene(points[(i - 1) % len(self)]), self.mapToScene(points[i])
        )
        # print((i - 1) % len(self), len(self.m_lines), len(self))
        self.m_lines[(i - 1) % len(self)].setLine(line)

    def move_item(self, i, pos):
        if 0 <= i < len(self.m_items):
            item = self.m_items[i]
            item.setEnabled(False)
            item.setPos(pos)
            item.setEnabled(True)
            self.moveLine(i)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            for i, point in enumerate(self.points):
                self.move_item(i, self.mapToScene(point))
        return super(PolygonAnnotation, self).itemChange(change, value)

    def hoverEnterEvent(self, ev):
        self.polygon_hovering = True
        self.setBrush(self.insideColor)
        super(PolygonAnnotation, self).hoverEnterEvent(ev)

    def hoverLeaveEvent(self, ev):
        self.polygon_hovering = False
        if not self.hasFocus():
            self.setBrush(self.halfInsideColor)
        super(PolygonAnnotation, self).hoverLeaveEvent(ev)

    def focusInEvent(self, ev):
        if not self.anning:
            self.setBrush(self.insideColor)

    def focusOutEvent(self, ev):
        if not self.polygon_hovering and not self.anning:
            self.setBrush(self.halfInsideColor)

    def setColor(self, insideColor, borderColor):
        i = insideColor
        self.insideColor = QtGui.QColor(i[0], i[1], i[2])
        self.insideColor.setAlphaF(self.opacity)
        self.halfInsideColor = QtGui.QColor(i[0], i[1], i[2])
        self.halfInsideColor.setAlphaF(self.opacity / 2)
        self.setBrush(self.halfInsideColor)
        b = borderColor
        self.borderColor = QtGui.QColor(b[0], b[1], b[2])
        self.borderColor.setAlphaF(0.8)
        self.setPen(QtGui.QPen(self.borderColor))
        for grip in self.m_items:
            grip.setColor(self.borderColor)
        for line in self.m_lines:
            line.setColor(self.borderColor)

    def __len__(self):
        return len(self.points)
