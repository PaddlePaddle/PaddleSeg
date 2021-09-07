import sys
import os.path as osp

from qtpy.QtWidgets import QApplication  # 导入PyQt相关模块
from qtpy import QtCore

from eiseg import pjpath
from app import APP_EISeg  # 导入带槽的界面


def main():
    app = QApplication(sys.argv)
    lang = QtCore.QSettings(
        osp.join(pjpath, "config/setting.ini"), QtCore.QSettings.IniFormat
    ).value("language")
    if lang != "中文":
        trans = QtCore.QTranslator(app)
        trans.load(osp.join(pjpath, f"util/translate/{lang}"))
        app.installTranslator(trans)

    window = APP_EISeg()  # 创建对象
    window.showMaximized()  # 全屏显示窗口
    # 加载近期模型
    QApplication.processEvents()
    window.loadRecentModelParam()
    sys.exit(app.exec_())
