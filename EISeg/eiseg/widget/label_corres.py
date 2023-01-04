# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os.path as osp
from functools import partial
from qtpy.QtCore import Qt, QCoreApplication, QSize
from qtpy.QtWidgets import (
    QDesktopWidget, QWidget, QTableWidget, QTableWidgetItem, \
    QSpacerItem, QHeaderView, QPushButton, QVBoxLayout, \
    QHBoxLayout, QCheckBox, QSizePolicy, QComboBox, QMessageBox
)
from qtpy.QtGui import QBrush, QColor
import json
from eiseg import pjpath

from widget.create import create_button
from util.coco.detlabel import COCO_CLASS_DICT


class LabelCorresWidget(QWidget):
    def __init__(self, param_path):
        super(LabelCorresWidget, self).__init__()
        self.tr = partial(QCoreApplication.translate, "APP_EISeg")

        self.userLabDict = None
        self.classLen = len(COCO_CLASS_DICT)
        self.param_path = param_path
        self.resize(500, 300)

        self.setWindowTitle(self.tr("目标检测标签映射设置"))
        self.labelCorrespondenceTable = QTableWidget()
        self.set_table_properties()

        self.btnCompleteLabelMapping = QPushButton(self.tr("完成"))
        self.btnCompleteLabelMapping.clicked.connect(self.completeLabelMapping)
        self.btnClearLabelMapping = QPushButton(self.tr("重置"))
        self.btnClearLabelMapping.clicked.connect(self.resetLabelMapping)

        self.ckbAllClass = QCheckBox(self.tr("全选"))
        self.ckbAllClass.clicked.connect(self.allSelect)
        self.ckbNoneClass = QCheckBox(self.tr("全不选"))
        self.ckbNoneClass.clicked.connect(self.noneSelect)

        v_layout = QVBoxLayout()
        # 查找
        findHorizontalLayout = QHBoxLayout()
        self.cbbCOCOFind = QComboBox()
        self.cbbCOCOFind.setMinimumSize(QSize(0, 32))
        self.cbbCOCOFind.setEditable(True)
        findHorizontalLayout.addWidget(self.cbbCOCOFind)
        self.btnCOCOFind = create_button(
            self,
            "btnCOCOFind",
            self.tr("查找"),
            osp.join(pjpath, "resource/Find.png"), )
        self.btnCOCOFind.setMinimumSize(QSize(0, 32))
        self.btnCOCOFind.setMaximumWidth(68)
        findHorizontalLayout.addWidget(self.btnCOCOFind)
        v_layout.addLayout(findHorizontalLayout)
        v_layout.addWidget(self.labelCorrespondenceTable)  # 主界面
        # 按钮及选择框
        h_layout = QHBoxLayout()
        h_layout.addWidget(self.btnCompleteLabelMapping)
        h_layout.addWidget(self.btnClearLabelMapping)
        h_layout.addSpacerItem(
            QSpacerItem(0, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))
        h_layout.addWidget(self.ckbAllClass)
        h_layout.addWidget(self.ckbNoneClass)
        v_layout.addLayout(h_layout)
        self.setLayout(v_layout)

        # 连接信号槽
        self.btnCOCOFind.clicked.connect(self._cocoFuzzyMatching)  # 模糊搜索
        self.cbbCOCOFind.activated[str].connect(self._findCOCOByText)  # 确定标签的搜索

    def set_table_properties(self):
        """设置表格属性"""
        self.labelCorrespondenceTable.setColumnCount(3)
        self.labelCorrespondenceTable.setRowCount(self.classLen)
        self.labelCorrespondenceTable.setHorizontalHeaderLabels(
            [self.tr("预标注模型标签"), self.tr("自定义标签"), self.tr("是否启用")])
        self.labelCorrespondenceTable.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.labelCorrespondenceTable.verticalHeader().setHidden(True)

        predict_label_list = list(COCO_CLASS_DICT.values())
        for idx, label_name in enumerate(predict_label_list):
            old_lab_item = QTableWidgetItem(str(label_name))
            old_lab_item.setFlags(Qt.ItemIsEnabled)
            new_lab_item = QTableWidgetItem("")
            check_item = QWidget()
            hb = QHBoxLayout()
            ckb = QCheckBox("", check_item)
            ckb.clicked.connect(self._cancelRadio)  # 点击取消全选/全不选
            hb.addWidget(ckb)
            hb.setAlignment(ckb, Qt.AlignCenter)
            check_item.setLayout(hb)
            self.labelCorrespondenceTable.setItem(idx, 0, old_lab_item)
            self.labelCorrespondenceTable.setItem(idx, 1, new_lab_item)
            self.labelCorrespondenceTable.setCellWidget(idx, 2, check_item)
            self.labelCorrespondenceTable.setRowHeight(idx, 40)  # 设置行高
        self._adjustTableSize()
        self._updateFromConfig()  # 更新显示

    def _cancelRadio(self):
        self.ckbNoneClass.setCheckState(Qt.Unchecked)
        self.ckbAllClass.setCheckState(Qt.Unchecked)

    def _adjustTableSize(self):
        self.labelCorrespondenceTable.horizontalHeader().setDefaultSectionSize(
            40)
        self.labelCorrespondenceTable.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.Fixed)
        self.labelCorrespondenceTable.setColumnWidth(2, 100)

    def _updateFromConfig(self):
        """从已有的json文件中恢复设置"""
        if osp.exists(self.param_path):
            with open(self.param_path, "r") as f:
                lc_dict = json.loads(f.read())
                for k, v in lc_dict.items():
                    row = int([
                        k2 for k2, v2 in COCO_CLASS_DICT.items() if v2 == k
                    ][0])
                    if k == v:
                        self.labelCorrespondenceTable.item(row, 1).setText("")
                    else:
                        self.labelCorrespondenceTable.item(row, 1).setText(v)
                    self.labelCorrespondenceTable.cellWidget(
                        row, 2).findChild(QCheckBox).setCheckState(Qt.Checked)

    def completeLabelMapping(self):
        """完成标签对应关系的建立，获取变化"""
        pre_label_list = []
        custom_label_list = []
        table = self.labelCorrespondenceTable

        for i in range(table.rowCount()):
            pre_label_list.append(table.item(i, 0).text())
            if table.cellWidget(i, 2).findChild(QCheckBox).isChecked() == False:
                custom_label_list.append('')
            else:
                if table.item(i, 1) is None or table.item(i, 1).text() == '':
                    custom_label_list.append(table.item(i, 0).text())
                else:
                    custom_label_list.append(table.item(i, 1).text())

        labelCorres = {}
        for idx, label_c in enumerate(custom_label_list):
            if label_c != '':
                labelCorres[pre_label_list[idx]] = label_c
        self.userLabDict = labelCorres
        # 准备写json文件
        dict_json = json.dumps(self.userLabDict)
        with open(self.param_path, 'w+', encoding='utf-8') as f:
            f.write(dict_json)
        self.close()

        # 增加一个说明
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(self.tr("提示"))
        msg.setText(self.tr("当前设置会在下一张推理图中应用"))
        msg.setStandardButtons(QMessageBox.Yes)
        msg.exec_()

        # 更新表格背景颜色
        for row in range(self.labelCorrespondenceTable.rowCount()):
            for col in range(self.labelCorrespondenceTable.colorCount() - 1):
                if self.labelCorrespondenceTable.item(row, col) is not None:
                    self.labelCorrespondenceTable.item(
                        row, col).setBackground(QBrush(QColor(255, 255, 255)))

    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def resetLabelMapping(self):
        for i in range(self.classLen):
            self.labelCorrespondenceTable.item(i, 1).setText("")

    def allSelect(self):
        self.ckbNoneClass.setCheckState(Qt.Unchecked)
        for i in range(self.classLen):
            self.labelCorrespondenceTable.cellWidget(
                i, 2).findChild(QCheckBox).setCheckState(Qt.Checked)

    def noneSelect(self):
        self.ckbAllClass.setCheckState(Qt.Unchecked)
        for i in range(self.classLen):
            self.labelCorrespondenceTable.cellWidget(
                i, 2).findChild(QCheckBox).setCheckState(Qt.Unchecked)

    @staticmethod
    def initJsonConfig(path):
        tmp_dict = dict()
        for v in COCO_CLASS_DICT.values():
            tmp_dict[v] = v
        dict_json = json.dumps(tmp_dict)
        with open(path, 'w+', encoding='utf-8') as f:
            f.write(dict_json)

    def _cocoFuzzyMatching(self):
        # 先更新背景颜色
        for row in range(self.labelCorrespondenceTable.rowCount()):
            for col in range(self.labelCorrespondenceTable.colorCount() - 1):
                if self.labelCorrespondenceTable.item(row, col) is not None:
                    self.labelCorrespondenceTable.item(
                        row, col).setBackground(QBrush(QColor(255, 255, 255)))

        ctext = self.cbbCOCOFind.currentText()
        self.cbbCOCOFind.clear()
        for i in range(self.classLen):
            clab = self.labelCorrespondenceTable.item(i, 0).text()
            if clab.lower().find(ctext.lower()) != -1:
                self.cbbCOCOFind.addItem(clab)
                for cl in range(self.labelCorrespondenceTable.columnCount() -
                                1):
                    self.labelCorrespondenceTable.item(
                        i, cl).setBackground(QBrush(QColor(48, 140, 198)))
                self.labelCorrespondenceTable.verticalScrollBar(
                ).setSliderPosition(i)

    def _findCOCOByText(self):
        ctext = self.cbbCOCOFind.currentText()
        for i in range(self.classLen):
            clab = self.labelCorrespondenceTable.item(i, 0).text()
            if ctext == clab:
                for cl in range(self.labelCorrespondenceTable.columnCount() -
                                1):
                    self.labelCorrespondenceTable.item(
                        i, cl).setBackground(QBrush(QColor(48, 140, 198)))
                self.labelCorrespondenceTable.verticalScrollBar(
                ).setSliderPosition(i)
