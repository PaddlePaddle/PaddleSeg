import sys

from qtpy.QtWidgets import QApplication  # 导入PyQt相关模块
from app import APP_EISeg  # 导入带槽的界面


def main():
    app = QApplication(sys.argv)
    myWin = APP_EISeg()  # 创建对象
    myWin.showMaximized()  # 全屏显示窗口
    # 加载近期模型
    QApplication.processEvents()
    myWin.loadRecentModelParam()
    sys.exit(app.exec_())