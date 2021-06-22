import sys

from qtpy.QtWidgets import QApplication  # 导入PyQt相关模块
from app import APP_IANN  # 导入带槽的界面


def main():
    app = QApplication(sys.argv)
    myWin = APP_IANN()  # 创建对象
    myWin.showMaximized()  # 全屏显示窗口
    # 加载近期模型
    QApplication.processEvents()
    myWin.load_recent_params()
    sys.exit(app.exec_())