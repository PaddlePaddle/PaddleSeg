from qtpy.QtWidgets import (
    QDialog, QLabel, QRadioButton, QButtonGroup, QDialogButtonBox, QVBoxLayout
)


class CustomDialog(QDialog):
    def __init__(self, parent, title, message, btn1_text, btn2_text):
        super().__init__(parent)
        self.setWindowTitle(title)
        msg_lab = QLabel(message)
        self.res = None
        self.btn1 = QRadioButton(btn1_text)
        self.btn2 = QRadioButton(btn2_text)
        self.btngroup = QButtonGroup()
        self.btngroup.addButton(self.btn1)
        self.btngroup.addButton(self.btn2)
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        # 绑定按钮事件
        self.btngroup.buttonClicked.connect(self._check)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        # 将按钮添加到布局
        layout = QVBoxLayout(self)
        layout.addWidget(msg_lab)
        layout.addWidget(self.btn1)
        layout.addWidget(self.btn2)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def _check(self):
        self.res = self.btngroup.checkedButton().text()
        
    @property
    def rs_support(self):
        return (self.res == self.btn1.text())
