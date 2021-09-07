import time
import json
import cv2
import numpy as np
from skimage.measure import label

from inference import clicker
from inference.predictor import get_predictor
import util
from util.vis import draw_with_blend_and_clicks
from util import MODELS, LabelList


class InteractiveController:
    def __init__(
        self,
        predictor_params: dict = None,
        prob_thresh: float = 0.5,
    ):
        """初始化控制器.

        Parameters
        ----------
        predictor_params : dict
            推理器配置
        prob_thresh : float
            区分前景和背景结果的阈值

        """
        self.predictor_params = predictor_params
        self.prob_thresh = prob_thresh
        self.model = None
        self.image = None
        self.predictor = None
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.polygons = []

        # 用于redo
        self.undo_states = []
        self.undo_probs_history = []

        self.curr_label_number = 0
        self._result_mask = None
        self.labelList = LabelList()
        self.lccFilter = False

    def filterLargestCC(self, do_filter: bool):
        """设置是否只保留推理结果中的最大联通块

        Parameters
        ----------
        do_filter : bool
            是否只保存推理结果中的最大联通块
        """
        if not isinstance(do_filter, bool):
            return
        self.lccFilter = do_filter

    def setModel(self, modelName: str):
        """设置推理其模型.

        Parameters
        ----------
        modelName : str
            模型名称，模型类中的__name__属性

        Returns
        -------
        bool, str
            是否成功设置模型, 失败原因

        """
        if not isinstance(modelName, str):
            return False, "模型名应为str类型"
        try:
            self.model = MODELS[modelName]()
        except KeyError as e:
            return False, str(e)
        return True, "模型设置成功"

    def setParam(self, paramPath: str):
        """设置模型使用的推理参数

        Parameters
        ----------
        paramPath : str
            推理参数路径

        Returns
        -------
        bool, str
            是否设置成功, 失败原因

        """
        if not self.modelSet:
            return False, "模型未设置，请先设置模型"
        try:
            self.model.load_param(paramPath)
        except Exception as e:
            return False, str(e)
        return True, "权重设置成功"

    def setImage(self, image: np.array):
        """设置当前标注的图片

        Parameters
        ----------
        image : np.array
            当前标注的图片

        """
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        self.resetLastObject()

    # 标签操作
    def setLabelList(self, labelList: json):
        """设置标签列表，会覆盖已有的标签列表

        Parameters
        ----------
        labelList : json
            标签列表格式为
            {
                {
                    "idx" : int         (like 0 or 1 or 2)
                    "name" : str        (like "car"　or "airplan")
                    "color" : list      (like [255, 0, 0])
                },
                ...
            }

        Returns
        -------
        type
            Description of returned object.

        """
        self.labelList.clear()
        labels = json.loads(labelList)
        for lab in labels:
            self.labelList.add(lab["id"], lab["name"], lab["color"])

    def addLabel(self, id: int, name: str, color: list):
        self.labelList.add(id, name, color)

    def delLabel(self, id: int):
        self.labelList.remove(id)

    def clearLabel(self):
        self.labelList.clear()

    def readLabel(self, path):
        self.labelList.readLabel(path)

    def saveLabel(self, path):
        self.labelList.saveLabel(path)

    # 点击操作
    def addClick(self, x: int, y: int, is_positive: bool):
        """添加一个点并运行推理，保存历史用于undo

        Parameters
        ----------
        x : int
            点击的横坐标
        y : int
            点击的纵坐标
        is_positive : bool
            是否点的是正点

        Returns
        -------
        bool, str
            点击是否添加成功, 失败原因

        """

        # 1. 确定可以点
        if not self.inImage(x, y):
            return False, "点击越界"
        if not self.modelSet:
            return False, "模型未设置"
        if not self.paramSet:
            return False, "参数未设置"
        if not self.imageSet:
            return False, "图像未设置"

        if len(self.states) == 0:  # 保存一个空状态
            self.states.append(
                {
                    "clicker": self.clicker.get_state(),
                    "predictor": self.predictor.get_states(),
                }
            )

        # 2. 添加点击，跑推理
        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker)

        # 3. 保存状态
        self.states.append(
            {
                "clicker": self.clicker.get_state(),
                "predictor": self.predictor.get_states(),
            }
        )
        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][1], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        # 点击之后就不能接着之前的历史redo了
        self.undo_states = []
        self.undo_probs_history = []
        return True, "点击添加成功"

    def undoClick(self):
        """
        undo一步点击
        """
        if len(self.states) <= 1:  # == 1就只剩下一个空状态了，不用再退
            return
        self.undo_states.append(self.states.pop())
        self.clicker.set_state(self.states[-1]["clicker"])
        self.predictor.set_states(self.states[-1]["predictor"])
        self.undo_probs_history.append(self.probs_history.pop())
        if not self.probs_history:
            self.reset_init_mask()

    def redoClick(self):
        """
        redo一步点击
        """
        if len(self.undo_states) == 0:  # 如果还没撤销过
            return
        if len(self.undo_probs_history) >= 1:
            next_state = self.undo_states.pop()
            self.states.append(next_state)
            self.clicker.set_state(next_state["clicker"])
            self.predictor.set_states(next_state["predictor"])
            self.probs_history.append(self.undo_probs_history.pop())

    def finishObject(self):
        """
        结束当前物体标注，准备标下一个
        """
        object_prob = self.current_object_prob
        if object_prob is None:
            return None, None
        object_mask = object_prob > self.prob_thresh
        polygon = util.get_polygon(object_mask.astype(np.uint8) * 255)
        if polygon is not None:
            if self.lccFilter:
                object_mask = self.getLargestCC(object_mask)
            self._result_mask[object_mask] = self.curr_label_number
            self.resetLastObject()
            self.polygons.append([self.curr_label_number, polygon])
        return object_mask, polygon

    # 多边形
    def getPolygon(self):
        return self.polygon

    def setPolygon(self, polygon):
        self.polygon = polygon

    # mask
    def getMask(self):
        s = self.imgShape
        img = np.zeros([s[0], s[1]])
        for poly in self.polygons:
            pts = np.int32([np.array(poly[1])])
            cv2.fillPoly(img, pts=pts, color=poly[0])
        return img

    def setCurrLabelIdx(self, number):
        if not isinstance(number, int):
            return False
        self.curr_label_number = number

    def resetLastObject(self, update_image=True):
        """
        重置控制器状态
        Parameters
            update_image(bool): 是否更新图像
        """
        self.states = []
        self.probs_history = []
        self.undo_states = []
        self.undo_probs_history = []
        # self.current_object_prob = None
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()

    def reset_predictor(self, predictor_params=None):
        """
        重置推理器，可以换推理配置
        Parameters
            predictor_params(dict): 推理配置
        """
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.model.model, **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self.clicker.click_indx_offset = 0

    def getLargestCC(self, mask):
        mask = label(mask)
        if mask.max() == 0:
            return mask
        mask = mask == np.argmax(np.bincount(mask.flat)[1:]) + 1
        return mask

    def get_visualization(self, alpha_blend: float, click_radius: int):
        if self.image is None:
            return None
        # 1. 正在标注的mask
        # results_mask_for_vis = self.result_mask  # 加入之前标完的mask
        results_mask_for_vis = np.zeros_like(self.result_mask)
        results_mask_for_vis *= self.curr_label_number
        if self.probs_history:
            results_mask_for_vis[
                self.current_object_prob > self.prob_thresh
            ] = self.curr_label_number
        if self.lccFilter:
            results_mask_for_vis = (
                self.getLargestCC(results_mask_for_vis) * self.curr_label_number
            )
        vis = draw_with_blend_and_clicks(
            self.image,
            mask=results_mask_for_vis,
            alpha=alpha_blend,
            clicks_list=self.clicker.clicks_list,
            radius=click_radius,
            palette=self.palette,
        )
        return vis

    def inImage(self, x: int, y: int):
        s = self.image.shape
        if x < 0 or y < 0 or x >= s[1] or y >= s[0]:
            print("点击越界")
            return False
        return True

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        return result_mask

    @property
    def palette(self):
        if self.labelList:
            colors = [ml.color for ml in self.labelList]
            colors.insert(0, [0, 0, 0])
        else:
            colors = [[0, 0, 0]]
        return colors

    @property
    def current_object_prob(self):
        """
        获取当前推理标签
        """
        if self.probs_history:
            _, current_prob_additive = self.probs_history[-1]
            return current_prob_additive
        else:
            return None

    @property
    def is_incomplete_mask(self):
        """
        Returns
            bool: 当前的物体是不是还没标完
        """
        return len(self.probs_history) > 0

    @property
    def imgShape(self):
        return self.image.shape[1::-1]

    @property
    def paramSet(self):
        return self.model.paramSet

    @property
    def modelSet(self):
        return self.model is not None

    @property
    def modelName(self):
        return self.model.__name__

    @property
    def imageSet(self):
        return self.image is not None
