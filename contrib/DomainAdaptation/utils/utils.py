from paddleseg.utils import logger
import paddle
import os


def load_ema_model(model, resume_model):
    if resume_model is not None:
        logger.info('Load ema model from {}'.format(resume_model))
        if os.path.exists(resume_model):
            resume_model = os.path.normpath(resume_model)
            ckpt_path = os.path.join(resume_model, 'model.pdparams')
            para_state_dict = paddle.load(ckpt_path)
            model.set_state_dict(para_state_dict)
        else:
            raise ValueError(
                'Directory of the model needed to resume is not Found: {}'.
                format(resume_model))
    else:
        logger.info('No model needed to resume.')
