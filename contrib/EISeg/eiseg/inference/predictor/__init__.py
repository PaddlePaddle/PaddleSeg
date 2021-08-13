from .base import BasePredictor
from inference.transforms import ZoomIn


def get_predictor(net, brs_mode,
                  with_flip=True,
                  zoom_in_params=dict(),
                  predictor_params=None):

    predictor_params_ = {
        'optimize_after_n_clicks': 1
    }

    if zoom_in_params is not None:
        zoom_in = ZoomIn(**zoom_in_params)
    else:
        zoom_in = None

    if brs_mode == 'NoBRS':
        
        if predictor_params is not None:
            predictor_params_.update(predictor_params)
        predictor = BasePredictor(net, zoom_in=zoom_in, with_flip=with_flip, **predictor_params_)
   
    else:
        raise NotImplementedError('Just support NoBRS mode')
    return predictor