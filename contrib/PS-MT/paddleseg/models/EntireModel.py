from paddleseg.models.encoder_decoder import *
from models.losses.semi_supervised_semantic_loss import *

from itertools import chain


@manager.MODELS.add_component
class EntireModel(nn.Layer):
    def __init__(self, backbone, num_classes, sup_loss=None, cons_w_unsup=None, ignore_index=None,
                 datashape=[512,512],
                 pretrained_model=None, semi=False, use_weak_lables=False, weakly_loss_w=0.4):
        super(EntireModel, self).__init__()
        self.encoder1 = EncoderNetwork(num_classes=num_classes, norm_layer=nn.BatchNorm2D,
                                       pretrained_model=None, back_bone=backbone)
        self.decoder1 = DecoderNetwork(num_classes=num_classes, data_shape=datashape)
        self.encoder2 = EncoderNetwork(num_classes=num_classes, norm_layer=nn.BatchNorm2D,
                                       pretrained_model=None, back_bone=backbone)
        self.decoder2 = DecoderNetwork(num_classes=num_classes, data_shape=datashape)
        self.encoder_s = EncoderNetwork(num_classes=num_classes, norm_layer=nn.BatchNorm2D,
                                        pretrained_model=pretrained_model,
                                        back_bone=backbone)
        self.decoder_s = VATDecoderNetwork(num_classes=num_classes, data_shape=datashape)
        self.mode = "semi" if semi else "sup"
        self.sup_loss = sup_loss
        self.ignore_index = ignore_index
        self.unsup_loss_w = cons_w_unsup

    def freeze_teachers_parameters(self):
        for p in self.encoder1.parameters():
            p.stop_gradient = True

        for p in self.decoder1.parameters():
            p.stop_gradient = True

        for p in self.encoder2.parameters():
            p.stop_gradient = True
        for p in self.decoder2.parameters():
            p.stop_gradient = True

    def warm_up_forward(self, id, x):
        if id == 1:
            output_l = self.decoder1(self.encoder1(x))
        elif id == 2:
            output_l = self.decoder2(self.encoder2(x))
        else:
            output_l = self.decoder_s(self.encoder_s(x))
        output_ul = None
        return output_l, output_ul

    def forward(self, x_l=None, x_ul=None,id=0,
                warm_up=False,  semi_p_th=0.6, semi_n_th=0.0):
        if warm_up:
            return self.warm_up_forward(id=id, x=x_l)

        output_l = self.decoder_s(self.encoder_s(x_l), t_model=[self.decoder1, self.decoder2])

        output_ul = self.decoder_s(self.encoder_s(x_ul), t_model=[self.decoder1, self.decoder2])
        return output_l, output_ul

    def get_other_params(self, id):
        if id == 1:
            return chain(self.encoder1.get_module_params(), self.decoder1.parameters())
        elif id == 2:
            return chain(self.encoder2.get_module_params(), self.decoder2.parameters())
        else:
            return chain(self.encoder_s.get_module_params(), self.decoder_s.parameters())

    def get_backbone_params(self, id):
        if id == 1:
            return self.encoder1.get_backbone_params()
        elif id == 2:
            return self.encoder2.get_backbone_params()
        else:
            return self.encoder_s.get_backbone_params()
