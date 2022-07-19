# The video propagation and fusion code was heavily based on https://github.com/hkchengrex/MiVOS
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/hkchengrex/MiVOS/blob/main/LICENSE

import os

import paddle.nn.functional as F
import numpy as np

from .load_model import *
from .util.tensor_util import pad_divide_by, images_to_paddle
from .video_tools import load_video, aggregate_wbg


class InferenceCore:
    """
    images - leave them in original dimension (unpadded), but do normalize them.
            Should be CPU tensors of shape B*T*3*H*W

    mem_profile - How extravagant I can use the GPU memory.
                Usually more memory -> faster speed but I have not drawn the exact relation
                0 - Use the most memory
                1 - Intermediate, larger buffer
                2 - Intermediate, small buffer
                3 - Use the minimal amount of GPU memory
                Note that *none* of the above options will affect the accuracy
                This is a space-time tradeoff, not a space-performance one

    mem_freq - Period at which new memory are put in the bank
                Higher number -> less memory usage
                Unlike the last option, this *is* a space-performance tradeoff
    """

    def __init__(self, mem_profile=2, mem_freq=5):
        self.cursur = 0

        self.mem_freq = mem_freq

        if mem_profile == 0:
            self.q_buf_size = 105
            self.i_buf_size = -1

        elif mem_profile == 1:
            self.q_buf_size = 105
            self.i_buf_size = 105

        elif mem_profile == 2:
            self.q_buf_size = 3
            self.i_buf_size = 3

        else:
            self.q_buf_size = 1
            self.i_buf_size = 1

        self.query_buf = {}
        self.image_buf = {}
        self.interacted = set()

        self.certain_mem_k = None
        self.certain_mem_v = None
        self.prob = None
        self.fuse_net = None
        self.k = 1

    def reset(self):
        self.cursur = 0
        self.query_buf = {}
        self.image_buf = {}
        self.interacted = set()

        self.certain_mem_k = None
        self.certain_mem_v = None
        self.prob = None
        # self.fuse_net = None
        self.k = 1
        self.images = None
        self.masks = None
        self.np_masks = None

    def set_video(self, video_path):
        self.images = load_video(video_path)
        self.num_frames, self.height, self.width = self.images.shape[:3]
        return self.images

    def get_one_frames(self, idx):
        return self.images[idx]

    def check_match(self, param_key, model_key):

        for p, m in zip(param_key, model_key):
            if p != m[0]:
                print(p)
                print(m[0])
                raise Exception("权重和模型不匹配。请确保指定的权重和模型对应")
        return True

    def set_model(self, param_path=None):
        if param_path is None or not os.path.exists(param_path):
            raise Exception(f"权重路径{param_path}不存在。请指定正确的模型路径")

        # param path
        param_path = os.path.abspath(os.path.dirname(param_path))
        # **************memorize**********************
        memory_model_path = os.path.join(param_path,
                                         'static_propagation_memorize.pdmodel')
        memory_param_path = os.path.join(
            param_path, 'static_propagation_memorize.pdiparams')
        self.prop_net_memory = load_model(memory_model_path, memory_param_path)
        # **************segmentation**********************
        segment_model_path = os.path.join(param_path,
                                          "static_propagation_segment.pdmodel")
        segment_param_path = os.path.join(
            param_path, "static_propagation_segment.pdiparams")
        self.prop_net_segm = load_model(segment_model_path, segment_param_path)
        # **************attention**********************
        attn_model_path = os.path.join(param_path,
                                       'static_propagation_attention')
        self.prop_net_attn = jit_load(attn_model_path)
        # **************fusion**********************
        fusion_model_path = os.path.join(param_path, 'static_fusion.pdmodel')
        fusion_param_path = os.path.join(param_path, 'static_fusion.pdiparams')
        self.fuse_net = load_model(fusion_model_path, fusion_param_path)

        return True, "模型设置成功"

    def set_images(self, images):
        # True dimensions
        images = images_to_paddle(images)
        t = images.shape[1]
        h, w = images.shape[-2:]

        # Pad each side to multiples of 16
        images = paddle.to_tensor(images, dtype='float32')
        self.images, self.pad = pad_divide_by(images, 16, images.shape[-2:])
        # Padded dimensions
        nh, nw = self.images.shape[-2:]

        # These two store the same information in different formats
        self.masks = paddle.zeros((t, 1, nh, nw), dtype='int64')
        self.np_masks = np.zeros((t, h, w), dtype=np.int64)
        if self.prob is None:
            self.prob = paddle.zeros(
                (self.k + 1, t, 1, nh, nw), dtype='float32')
            self.prob[0] = 1e-7
        else:
            k, t, c, nh, nw = self.prob.shape
            if (self.k + 1) != k:
                add_obj = abs(self.k + 1 - k)
                new_prob = paddle.zeros([add_obj, t, c, nh, nw]) + 1e-7
                self.prob = paddle.concat([self.prob, new_prob], axis=0)

        self.t, self.h, self.w = t, h, w
        self.nh, self.nw = nh, nw
        self.kh = self.nh // 16
        self.kw = self.nw // 16

    def set_objects(self, num_objects):
        self.k = num_objects

    def get_image_buffered(self, idx):
        if idx not in self.image_buf:
            # Flush buffer
            if len(self.image_buf) > self.i_buf_size:
                self.image_buf = {}
        self.image_buf[idx] = self.images[:, idx]
        result = self.image_buf[idx]
        return result
        # return self.images[:, idx]

    def get_query_kv_mask(self, idx, this_k, this_v):
        # Queries' key/value never change, so we can buffer them here
        if idx not in self.query_buf:
            # Flush buffer
            if len(self.query_buf) > self.q_buf_size:
                self.query_buf = {}
            result = calculate_segmentation(
                self.prop_net_segm,
                self.get_image_buffered(idx).numpy(),
                this_k.numpy(), this_v.numpy())
        mask = result[0]
        quary = result[1]

        return mask, quary

    def do_pass(self, key_k, key_v, idx, forward=True, step_cb=None):
        """
        Do a complete pass that includes propagation and fusion
        key_k/key_v -  memory feature of the starting frame
        idx - Frame index of the starting frame
        forward - forward/backward propagation
        step_cb - Callback function used for GUI (progress bar) only
        """

        # Pointer in the memory bank
        num_certain_keys = self.certain_mem_k.shape[2]
        m_front = num_certain_keys

        # Determine the required size of the memory bank
        if forward:
            closest_ti = min([ti for ti in self.interacted
                              if ti > idx] + [self.t])
            total_m = (closest_ti - idx - 1
                       ) // self.mem_freq + 1 + num_certain_keys
        else:
            closest_ti = max([ti for ti in self.interacted if ti < idx] + [-1])
            total_m = (idx - closest_ti - 1
                       ) // self.mem_freq + 1 + num_certain_keys
        K, CK, _, H, W = key_k.shape
        _, CV, _, _, _ = key_v.shape

        # Pre-allocate keys/values memory
        keys = paddle.empty((K, CK, total_m, H, W), dtype='float32')
        values = paddle.empty((K, CV, total_m, H, W), dtype='float32')

        # Initial key/value passed in
        keys[:, :, 0:num_certain_keys] = self.certain_mem_k
        values[:, :, 0:num_certain_keys] = self.certain_mem_v
        prev_in_mem = True
        last_ti = idx

        # Note that we never reach closest_ti, just the frame before it
        if forward:
            this_range = range(idx + 1, closest_ti)
            step = +1
            end = closest_ti - 1
        else:
            this_range = range(idx - 1, closest_ti, -1)
            step = -1
            end = closest_ti + 1

        for ti in this_range:
            if prev_in_mem:
                this_k = keys[:, :, :m_front]
                this_v = values[:, :, :m_front]
            else:
                this_k = keys[:, :, :m_front + 1]
                this_v = values[:, :, :m_front + 1]

            out_mask, quary_key = self.get_query_kv_mask(ti, this_k, this_v)
            out_mask = aggregate_wbg(paddle.to_tensor(out_mask), keep_bg=True)

            if ti != end:
                keys[:, :, m_front:m_front +
                     1], values[:, :, m_front:m_front + 1] = calculate_memorize(
                         self.prop_net_memory,
                         self.get_image_buffered(ti).numpy(),
                         out_mask[1:].numpy())
                if abs(ti - last_ti) >= self.mem_freq:
                    # Memorize the frame
                    m_front += 1
                    last_ti = ti
                    prev_in_mem = True
                else:
                    prev_in_mem = False

            # In-place fusion, maximizes the use of queried buffer
            # esp. for long sequence where the buffer will be flushed
            if (closest_ti != self.t) and (closest_ti != -1):
                self.prob[:, ti] = self.fuse_one_frame(
                    closest_ti, idx, ti, self.prob[:, ti], out_mask, key_k,
                    quary_key)
            else:
                self.prob[:, ti] = out_mask

            # Callback function for the GUI
            if step_cb is not None:
                step_cb()

        return closest_ti

    def fuse_one_frame(self, tc, tr, ti, prev_mask, curr_mask, mk16, qk16):
        assert (tc < ti < tr or tr < ti < tc)

        prob = paddle.zeros((self.k, 1, self.nh, self.nw), dtype='float32')

        # Compute linear coefficients
        nc = abs(tc - ti) / abs(tc - tr)
        nr = abs(tr - ti) / abs(tc - tr)
        dist = paddle.to_tensor([nc, nr], dtype='float32').unsqueeze(0)
        for k in range(1, self.k + 1):
            attn_map = self.prop_net_attn(mk16[k - 1:k], qk16,
                                          self.pos_mask_diff[k:k + 1],
                                          self.neg_mask_diff[k:k + 1])
            w = calculate_fusion(self.fuse_net,
                                 self.get_image_buffered(ti).numpy(),
                                 prev_mask[k:k + 1].numpy(),
                                 curr_mask[k:k + 1].numpy(),
                                 attn_map.numpy(), dist.numpy())
            w = paddle.to_tensor(w)
            w = F.sigmoid(w)
            prob[k - 1:k] = w
        return aggregate_wbg(prob, keep_bg=True)

    def interact(self, mask, idx, total_cb=None, step_cb=None):
        """
        Interact -> Propagate -> Fuse

        mask - One-hot mask of the interacted frame, background included
        idx - Frame index of the interacted frame
        total_cb, step_cb - Callback functions for the GUI

        Return: all mask results in np format for DAVIS evaluation
        """
        self.interacted.add(idx)
        mask, _ = pad_divide_by(mask, 16, mask.shape[-2:])
        # print('self.k is %d' % self.k)
        self.mask_diff = mask - self.prob[:, idx]
        self.pos_mask_diff = self.mask_diff.clip(0, 1)
        self.neg_mask_diff = (-self.mask_diff).clip(0, 1)

        self.prob[:, idx] = mask

        key_k, key_v = calculate_memorize(self.prop_net_memory,
                                          self.get_image_buffered(idx).numpy(),
                                          mask[1:].numpy())
        key_k = paddle.to_tensor(key_k).astype("float32")
        key_v = paddle.to_tensor(key_v).astype('float32')
        if self.certain_mem_k is None:
            self.certain_mem_k = key_k
            self.certain_mem_v = key_v
        else:
            K, CK, _, H, W = self.certain_mem_k.shape
            CV = self.certain_mem_v.shape[1]
            self.certain_mem_k = paddle.concat(
                [self.certain_mem_k, paddle.zeros((self.k - K, CK, _, H, W))],
                0)
            self.certain_mem_v = paddle.concat(
                [self.certain_mem_v, paddle.zeros((self.k - K, CV, _, H, W))],
                0)
            self.certain_mem_k = paddle.concat([self.certain_mem_k, key_k], 2)
            self.certain_mem_v = paddle.concat([self.certain_mem_v, key_v], 2)
        # self.certain_mem_k = key_k
        # self.certain_mem_v = key_v

        if total_cb is not None:
            # Finds the total num. frames to process
            front_limit = min([ti for ti in self.interacted
                               if ti > idx] + [self.t])
            back_limit = max([ti for ti in self.interacted if ti < idx] + [-1])
            total_num = front_limit - back_limit - 2  # -1 for shift, -1 for center frame

            if total_num > 0:
                total_cb(total_num)

        with paddle.no_grad():
            self.do_pass(key_k, key_v, idx, True, step_cb=step_cb)
            self.do_pass(key_k, key_v, idx, False, step_cb=step_cb)

        # This is a more memory-efficient argmax
        for ti in range(self.t):
            self.masks[ti] = paddle.argmax(self.prob[:, ti], axis=0)
        out_masks = self.masks

        # Trim paddings
        if self.pad[2] + self.pad[3] > 0:
            out_masks = out_masks[:, :, self.pad[2]:-self.pad[3], :]
        if self.pad[0] + self.pad[1] > 0:
            out_masks = out_masks[:, :, :, self.pad[0]:-self.pad[1]]

        self.np_masks = (out_masks.detach().numpy()[:, 0]).astype(np.int64)

        return self.np_masks

    def update_mask_only(self, prob_mask, idx):
        """
        Interaction only, no propagation/fusion
        prob_mask - mask of the interacted frame, background included
        idx - Frame index of the interacted frame

        Return: all mask results in np format for DAVIS evaluation
        """
        mask = paddle.argmax(prob_mask, 0)

        self.masks[idx:idx + 1] = mask

        # Mask - 1 * H * W
        if self.pad[2] + self.pad[3] > 0:
            mask = mask[:, self.pad[2]:-self.pad[3], :]
        if self.pad[0] + self.pad[1] > 0:
            mask = mask[:, :, self.pad[0]:-self.pad[1]]

        mask = (mask.detach().numpy()[0]).astype(np.int64)
        self.np_masks[idx:idx + 1] = mask

        return self.np_masks
