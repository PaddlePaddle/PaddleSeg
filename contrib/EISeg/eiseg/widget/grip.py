from qtpy import QtWidgets, QtGui, QtCore

# BUG: item 不能移出图片的范围，需要限制起来
class GripItem(QtWidgets.QGraphicsPathItem):
    fixedSize = 6

    def __init__(self, annotation_item, index, color):
        super(GripItem, self).__init__()
        self.m_annotation_item = annotation_item
        self.hovering = False
        self.m_index = index
        color.setAlphaF(1)
        self.color = color

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

    @property
    def size(self):
        if not self.scene():
            return 2
        else:
            return GripItem.fixedSize / self.scene().scale

    def updateSize(self, size=2):
        size = self.size
        self.circle = QtGui.QPainterPath()
        self.circle.addEllipse(QtCore.QRectF(-size, -size, size * 2, size * 2))
        self.square = QtGui.QPainterPath()
        self.square.addRect(QtCore.QRectF(-size, -size, size * 2, size * 2))
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
        if change == QtWidgets.QGraphicsItem.ItemPositionChange and self.isEnabled():
            self.m_annotation_item.movePoint(self.m_index, value)
        return super(GripItem, self).itemChange(change, value)

    def shape(self):
        s = super(GripItem, self).shape().boundingRect().x() * 1.2  # 缩小激活区域
        path = QtGui.QPainterPath()
        path.addRect(QtCore.QRectF(-s, -s, 2 * s, 2 * s))
        return path

    def mouseDoubleClickEvent(self, ev):
        self.m_annotation_item.removeFocusPoint()
