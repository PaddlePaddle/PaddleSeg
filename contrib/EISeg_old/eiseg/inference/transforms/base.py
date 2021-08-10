import paddle.nn.functional as F


class BaseTransform(object):
    def __init__(self):
        self.image_changed = False

    def transform(self, image_nd, clicks_lists):
        raise NotImplementedError

    def inv_transform(self, prob_map):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError


class SigmoidForPred(BaseTransform):
    def transform(self, image_nd, clicks_lists):
        return image_nd, clicks_lists

    def inv_transform(self, prob_map):
        return F.sigmoid(prob_map)

    def reset(self):
        pass

    def get_state(self):
        return None

    def set_state(self, state):
        pass
