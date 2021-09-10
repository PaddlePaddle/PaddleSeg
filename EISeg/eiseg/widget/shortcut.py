import os.path as osp
import math
from functools import partial

from PyQt5.QtCore import QPoint
from PyQt5.QtWidgets import QDesktopWidget

from qtpy import QtCore, QtWidgets
from qtpy.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QGridLayout,
    QKeySequenceEdit,
    QMessageBox,
)
from qtpy.QtGui import QIcon
from qtpy import QtCore
from qtpy.QtCore import Qt

from util import save_configs


class RecordShortcutWindow(QKeySequenceEdit):
    def __init__(self, finishCallback, location):
        super().__init__()
        self.finishCallback = finishCallback
        # 隐藏界面
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.move(location)
        self.show()
        self.editingFinished.connect(lambda: finishCallback(self.keySequence()))

    def keyReleaseEvent(self, ev):
        self.finishCallback(self.keySequence())


class ShortcutWindow(QWidget):
    def __init__(self, actions, pjpath):
        super().__init__()
        self.tr = partial(QtCore.QCoreApplication.translate, "ShortcutWindow")
        self.setWindowTitle(self.tr("编辑快捷键"))
        self.setWindowIcon(QIcon(osp.join(pjpath, "resource/Shortcut.png")))
        # self.setFixedSize(self.width(), self.height())
        self.actions = actions
        self.recorder = None
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        self.setLayout(grid)

        actions = self.actions
        for idx, action in enumerate(actions):
            # 2列英文看不清
            grid.addWidget(QLabel(action.iconText()[1:]), idx // 3, idx % 3 * 3)
            shortcut = action.shortcut().toString()
            if len(shortcut) == 0:
                shortcut = self.tr("无")
            button = QPushButton(shortcut)
            button.setFixedWidth(150)
            button.setFixedHeight(30)
            button.clicked.connect(partial(self.recordShortcut, action))
            grid.addWidget(
                button,
                idx // 3,
                idx % 3 * 3 + 1,
            )

    def refreshUi(self):
        actions = self.actions
        for idx, action in enumerate(actions):
            shortcut = action.shortcut().toString()
            if len(shortcut) == 0:
                shortcut = self.tr("无")
            self.layout().itemAtPosition(
                idx // 3,
                idx % 3 * 3 + 1,
            ).widget().setText(shortcut)

    def recordShortcut(self, action):
        # 打开快捷键设置的窗口时，如果之前的还在就先关闭
        if self.recorder is not None:
            self.recorder.close()
        rect = self.geometry()
        x = rect.x()
        y = rect.y() + rect.height()
        self.recorder = RecordShortcutWindow(self.setShortcut, QPoint(x, y))
        self.currentAction = action

    def setShortcut(self, key):
        self.recorder.close()

        for a in self.actions:
            if a.shortcut() == key:
                key = key.toString()
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle(key + " " + self.tr("快捷键冲突"))
                msg.setText(
                    key
                    + " "
                    + self.tr("快捷键已被")
                    + " "
                    + a.data()
                    + " "
                    + self.tr("使用，请设置其他快捷键或先修改")
                    + " "
                    + a.data()
                    + " "
                    + self.tr("的快捷键")
                )
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec_()
                return
        key = "" if key.toString() == "Esc" else key  # ESC不设置快捷键
        self.currentAction.setShortcut(key)
        self.refreshUi()
        save_configs(None, None, self.actions)

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    # 快捷键设置跟随移动
    def moveEvent(self, event):
        p = self.geometry()
        x = p.x()
        y = p.y() + p.height()
        if self.recorder is not None:
            self.recorder.move(x, y)

    def closeEvent(self, event):
        # 关闭时也退出快捷键设置
        if self.recorder is not None:
            self.recorder.close()
