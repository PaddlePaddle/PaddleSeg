from .zoom_in import ZoomIn, get_roi_image_nd


class LimitLongestSide(ZoomIn):
    def __init__(self, max_size=800):
        super().__init__(target_size=max_size, skip_clicks=0)

    def transform(self, image_nd, clicks_lists):
        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        image_max_size = max(image_nd.shape[2:4])
        self.image_changed = False

        if image_max_size <= self.target_size:
            return image_nd, clicks_lists
        self._input_image = image_nd

        self._object_roi = (0, image_nd.shape[2] - 1, 0, image_nd.shape[3] - 1)
        self._roi_image = get_roi_image_nd(image_nd, self._object_roi, self.target_size)
        self.image_changed = True

        tclicks_lists = [self._transform_clicks(clicks_lists[0])]
        return self._roi_image, tclicks_lists