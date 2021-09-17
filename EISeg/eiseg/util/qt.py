from math import sqrt
import os.path as osp

import numpy as np

from eiseg import pjpath
from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from .config import parse_configs

shortcuts = parse_configs(osp.join(pjpath, "config/config.yaml"))["shortcut"]
here = osp.dirname(osp.abspath(__file__))


def newIcon(icon):
    if isinstance(icon, list) or isinstance(icon, tuple):
        pixmap = QtGui.QPixmap(100, 100)
        c = icon
        pixmap.fill(QtGui.QColor(c[0], c[1], c[2]))
        return QtGui.QIcon(pixmap)
    icons_dir = osp.join(here, "../resource")
    return QtGui.QIcon(osp.join(":/", icons_dir, f"{icon}.png"))


def newButton(text, icon=None, slot=None):
    b = QtWidgets.QPushButton(text)
    if icon is not None:
        b.setIcon(newIcon(icon))
    if slot is not None:
        b.clicked.connect(slot)
    return b


def newAction(
    parent,
    text,
    slot=None,
    shortcutName=None,
    icon=None,
    tip=None,
    checkable=False,
    enabled=True,
    checked=False,
):
    """Create a new action and assign callbacks, shortcuts, etc."""
    a = QtWidgets.QAction(text, parent)
    a.setData(shortcutName)
    # a = QtWidgets.QAction("", parent)
    if icon is not None:
        a.setIconText(text.replace(" ", "\n"))
        a.setIcon(newIcon(icon))
    shortcut = shortcuts.get(shortcutName, None)
    if shortcut is not None:
        if isinstance(shortcut, (list, tuple)):
            a.setShortcuts(shortcut)
        else:
            a.setShortcut(shortcut)
    if tip is not None:
        a.setToolTip(tip)
        a.setStatusTip(tip)
    if slot is not None:
        a.triggered.connect(slot)
    if checkable:
        a.setCheckable(True)
    a.setEnabled(enabled)
    a.setChecked(checked)
    return a


def addActions(widget, actions):
    for action in actions:
        if action is None:
            widget.addSeparator()
        elif isinstance(action, QtWidgets.QMenu):
            widget.addMenu(action)
        else:
            widget.addAction(action)


def labelValidator():
    return QtGui.QRegExpValidator(QtCore.QRegExp(r"^[^ \t].+"), None)


class struct(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __len__(self):
        return len(self.__dict__)

    def append(self, action):
        if isinstance(action, QtWidgets.QAction):
            self.__dict__.update({action.data(): action})

    def __iter__(self):
        return list(self.__dict__.values()).__iter__()

    def __getitem__(self, idx):
        return list(self.__dict__.values())[idx]

    def get(self, name):
        return self.__dict__[name]


def fmtShortcut(text):
    mod, key = text.split("+", 1)
    return "<b>%s</b>+<b>%s</b>" % (mod, key)
