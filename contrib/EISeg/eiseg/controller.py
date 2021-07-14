import time

import paddle
import numpy as np
import paddleseg.transforms as T
from skimage.measure import label

from inference import clicker
from inference.predictor import get_predictor
from util.vis import draw_with_blend_and_clicks


class InteractiveController:
    def __init__(self, net, predictor_params, update_image_callback, prob_thresh=0.5):
        self.net = net
        self.prob_thresh = prob_thresh
        self.clicker = clicker.Clicker()
        self.states = []
        self.probs_history = []
        self.undo_states = []
        self.undo_probs_history = []
        self.curr_label_number = 0
        self._result_mask = None
        self.label_list = None  # 存标签编号和颜色的对照
        self._init_mask = None

        self.image = None
        self.predictor = None
        self.update_image_callback = update_image_callback
        self.predictor_params = predictor_params
        self.filterLargestCC = False
        self.reset_predictor()

    def set_image(self, image):
        """设置当前标注的图片

        Parameters
        ----------
        image :
            Description of parameter `image`.
        """
        self.image = image
        self._result_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        self.reset_last_object(update_image=False)
        self.update_image_callback(reset_canvas=True)

    def set_mask(self, mask):
        if len(self.probs_history) > 0:
            self.reset_last_object()

        self._init_mask = mask.astype(np.float32)
        self.probs_history.append((np.zeros_like(self._init_mask), self._init_mask))
        self._init_mask = paddle.to_tensor(self._init_mask).unsqueeze(0).unsqueeze(0)
        self.clicker.click_indx_offset = 1

    def add_click(self, x, y, is_positive):
        """添加一个点
        跑推理，保存历史用于undo
        Parameters
        ----------
        x : type
            Description of parameter `x`.
        y : type
            Description of parameter `y`.
        is_positive : bool
            是否是正点
        Returns
            -------
            bool
                点击是否成功添加
        """
        s = self.image.shape
        if x < 0 or y < 0 or x >= s[1] or y >= s[0]:
            print("点击越界")
            return False
        
        if len(self.states) == 0:  # 保存最初状态
            self.states.append({
                "clicker": self.clicker.get_state(),
                "predictor": self.predictor.get_states(),
            })

        click = clicker.Click(is_positive=is_positive, coords=(y, x))
        self.clicker.add_click(click)
        pred = self.predictor.get_prediction(self.clicker, prev_mask=self._init_mask)
        if self._init_mask is not None and len(self.clicker) == 1:
            pred = self.predictor.get_prediction(
                self.clicker, prev_mask=self._init_mask
            )

        # 保存最新状态
        self.states.append({
                "clicker": self.clicker.get_state(),
                "predictor": self.predictor.get_states(),
            })

        if self.probs_history:
            self.probs_history.append((self.probs_history[-1][0], pred))
        else:
            self.probs_history.append((np.zeros_like(pred), pred))

        self.update_image_callback()

    def set_label(self, label):
        pass

    def undo_click(self):
        """undo一步点击"""
        if len(self.states) <= 1:  # 如果还没点
            return
        self.undo_states.append(self.states.pop())
        self.clicker.set_state(self.states[-1]["clicker"])
        self.predictor.set_states(self.states[-1]["predictor"])
        self.undo_probs_history.append(self.probs_history.pop())
        if not self.probs_history:
            self.reset_init_mask()
        self.update_image_callback()

    def redo_click(self):
        """redo一步点击"""
        if not self.undo_states:  # 如果还没撤销过
            return
        if len(self.undo_probs_history) >= 1:
            next_state = self.undo_states.pop()
            self.states.append(next_state)
            self.clicker.set_state(next_state["clicker"])
            self.predictor.set_states(next_state["predictor"])
            self.probs_history.append(self.undo_probs_history.pop())
            self.update_image_callback()

    def partially_finish_object(self):
        """部分完成
        保存一个mask的状态，这个状态里不存点，看起来比较
        """

        object_prob = self.current_object_prob
        if object_prob is None:
            return

        self.probs_history.append((object_prob, np.zeros_like(object_prob)))
        self.states.append(self.states[-1])

        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        self.update_image_callback()

    def finish_object(self):
        """结束当前物体标注，准备标下一个"""
        object_prob = self.current_object_prob
        if object_prob is None:
            return

        object_mask = object_prob > self.prob_thresh
        self._result_mask[object_mask] = self.curr_label_number
        self.reset_last_object()

    def change_label_num(self, number):
        """修改当前标签的编号
        如果当前有标注到一半的目标，改mask。
        如果没有，下一个目标是这个数
        Parameters
        ----------
        number : int
            换成目标的编号
        """
        assert isinstance(number, int), "标签编号应为整数"
        self.curr_label_number = number
        if self.is_incomplete_mask:
            pass

    def reset_last_object(self, update_image=True):
        """重置控制器状态
        Parameters
        ----------
        update_image : bool
            Description of parameter `update_image`.
        Returns
        -------
        type
            Description of returned object.

        """
        self.states = []
        self.probs_history = []
        self.undo_states = []
        self.undo_probs_history = []
        self.clicker.reset_clicks()
        self.reset_predictor()
        self.reset_init_mask()
        if update_image:
            self.update_image_callback()

    def reset_predictor(self, net=None, predictor_params=None):
        """重置推理器，可以换权重
        Parameters
        ----------
        predictor_params : 网络权重
            新的网络权重
        """
        if net is not None:
            self.net = net
        if predictor_params is not None:
            self.predictor_params = predictor_params
        self.predictor = get_predictor(self.net, **self.predictor_params)
        if self.image is not None:
            self.predictor.set_input_image(self.image)

    def reset_init_mask(self):
        self._init_mask = None
        self.clicker.click_indx_offset = 0

    @property
    def current_object_prob(self):
        if len(self.probs_history) > 0:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        result_mask = self._result_mask.copy()
        if self.probs_history:
            result_mask[self.current_object_prob > self.prob_thresh] = (
                self.object_count + 1
            )
        return result_mask

    def get_visualization(self, alpha_blend, click_radius):
        if self.image is None:
            return None

        results_mask_for_vis = self.result_mask
        if self.probs_history:
            results_mask_for_vis[
                self.current_object_prob > self.prob_thresh
            ] = self.curr_label_number

        vis = draw_with_blend_and_clicks(
            self.image,
            mask=results_mask_for_vis,
            alpha=alpha_blend,
            clicks_list=self.clicker.clicks_list,
            radius=click_radius,
            palette=self.palette,
        )

        if self.probs_history:
            total_mask = self.probs_history[-1][0] > self.prob_thresh
            results_mask_for_vis[np.logical_not(total_mask)] = 0
            vis = draw_with_blend_and_clicks(
                vis,
                mask=results_mask_for_vis,
                alpha=alpha_blend,
                palette=self.palette,
            )

        return vis

    @property
    def palette(self):
        if self.label_list:
            colors = [ml.color for ml in self.label_list]
            colors.insert(0, [0, 0, 0])
        else:
            colors = [[0, 0, 0]]
        return colors

    @property
    def current_object_prob(self):
        """获取当前推理标签"""
        if self.probs_history:
            current_prob_total, current_prob_additive = self.probs_history[-1]
            return np.maximum(current_prob_total, current_prob_additive)
        else:
            return None

    @property
    def is_incomplete_mask(self):
        """
        Returns
        -------
        bool
            当前的物体是不是还没标完
        """
        return len(self.probs_history) > 0

    @property
    def result_mask(self):
        return self._result_mask.copy()

    @property
    def img_size(self):
        return self.image.shape[1::-1]
