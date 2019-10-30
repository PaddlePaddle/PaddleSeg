import paddle
import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA
import math
from paddle.fluid.param_attr import ParamAttr

__all__ = ["HRNet", "HRNet_W18_C", "HRNet_W30_C", "HRNet_W32_C", "HRNet_W40_C", "HRNet_W44_C", "HRNet_W48_C"]


train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}

class HRNet():
    def __init__(self, width=18, has_se=False):
        self.params = train_parameters
        self.width = width
        self.has_se = has_se
        self.channels = {
            18: [[18, 36], [18, 36, 72], [18, 36, 72, 144]],
            30: [[30, 60], [30, 60, 120], [30, 60, 120, 240]],
            32: [[32, 64], [32, 64, 128], [32, 64, 128, 256]],
            40: [[40, 80], [40, 80, 160], [40, 80, 160, 320]],
            44: [[44, 88], [44, 88, 176], [44, 88, 176, 352]],
            48: [[48, 96], [48, 96, 192], [48, 96, 192, 384]],
            60: [[60, 120], [60, 120, 240], [60, 120, 240, 480]]
            }
        

    def net(self, input, class_dim=1000):
        width = self.width
        channels_2, channels_3, channels_4 = self.channels[width]   
        num_modules_2, num_modules_3, num_modules_4 = 1, 4, 3
  
        x = self.conv_bn_layer(input=input, filter_size=3, num_filters=64, stride=2, if_act=False, name='layer1_1')
        x = self.conv_bn_layer(input=x, filter_size=3, num_filters=64, stride=2, if_act=True, name='layer1_2')

        la1 = self.layer1(x, name='layer2')
        tr1 = self.transition_layer([la1], [256], channels_2, name='tr1')
        st2 = self.stage(tr1, num_modules_2, channels_2, name='st2')
        tr2 = self.transition_layer(st2, channels_2, channels_3, name='tr2')
        st3 = self.stage(tr2, num_modules_3, channels_3, name='st3')
        tr3 = self.transition_layer(st3, channels_3, channels_4, name='tr3')
        st4 = self.stage(tr3, num_modules_4, channels_4, name='st4')
        
        #classification
        last_cls = self.last_cls_out(x=st4, name='cls_head')
        y = last_cls[0]
        last_num_filters = [256, 512, 1024]
        for i in range(3):
            y = fluid.layers.elementwise_add(last_cls[i+1], 
                                             self.conv_bn_layer(input=y, filter_size=3, 
                                                                               num_filters=last_num_filters[i], stride=2, 
                                                                               name='cls_head_add'+str(i+1))
                                            )
            
        y = self.conv_bn_layer(input=y, filter_size=1, num_filters=2048, stride=1, name='cls_head_last_conv')
        pool = fluid.layers.pool2d(input=y, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(input=pool, size=class_dim, act='softmax',                                  
                              param_attr=ParamAttr(name='fc_weights', initializer=fluid.initializer.Uniform(-stdv, stdv)),
                              bias_attr=ParamAttr(name='fc_offset'))
        return out

        
    def layer1(self, input, name=None):
        conv = input
        for i in range(4):
            conv = self.bottleneck_block(conv, num_filters=64, downsample=True if i == 0 else False, name=name+'_'+str(i+1))
        return conv
    
    def transition_layer(self, x, in_channels, out_channels, name=None):
        num_in = len(in_channels)
        num_out = len(out_channels)
        out = []
        for i in range(num_out):
            if i < num_in:
                if in_channels[i] != out_channels[i]:
                    residual = self.conv_bn_layer(x[i], filter_size=3, num_filters=out_channels[i], name=name+'_layer_'+str(i+1))
                    out.append(residual)
                else:
                    out.append(x[i])
            else:
                residual = self.conv_bn_layer(x[-1], filter_size=3, num_filters=out_channels[i], stride=2, 
                                              name=name+'_layer_'+str(i+1))
                out.append(residual)
        return out

    def branches(self, x, block_num, channels, name=None):
        out = []
        for i in range(len(channels)):
            residual = x[i]
            for j in range(block_num):
                residual = self.basic_block(residual, channels[i], name=name+'_branch_layer_'+str(i+1)+'_'+str(j+1))
            out.append(residual)
        return out

    def fuse_layers(self, x, channels, multi_scale_output=True, name=None):
        out = []
        for i in range(len(channels) if multi_scale_output else 1):
            residual = x[i]
            for j in range(len(channels)):
                if j > i:
                    y = self.conv_bn_layer(x[j], filter_size=1, num_filters=channels[i], if_act=False, 
                                           name=name+'_layer_'+str(i+1)+'_'+str(j+1))
                    y = fluid.layers.resize_nearest(input=y, scale=2 ** (j - i))
                    residual = fluid.layers.elementwise_add(x=residual, y=y, act=None)
                elif j < i:
                    y = x[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            y = self.conv_bn_layer(y, filter_size=3, num_filters=channels[i], stride=2, if_act=False, 
                                                   name=name+'_layer_'+str(i+1)+'_'+str(j+1)+'_'+str(k+1))
                        else:
                            y = self.conv_bn_layer(y, filter_size=3, num_filters=channels[j], stride=2,
                                                   name=name+'_layer_'+str(i+1)+'_'+str(j+1)+'_'+str(k+1))
                    residual = fluid.layers.elementwise_add(x=residual, y=y, act=None)        

            residual = fluid.layers.relu(residual)
            out.append(residual)
        return out
    
    def high_resolution_module(self, x, channels, multi_scale_output=True, name=None):
        residual = self.branches(x, 4, channels, name=name)
        out = self.fuse_layers(residual, channels, multi_scale_output=multi_scale_output, name=name)
        return out
    
    def stage(self, x, num_modules, channels, multi_scale_output=True, name=None):
        out = x
        for i in range(num_modules):
            if i == num_modules - 1 and multi_scale_output == False:
                out = self.high_resolution_module(out, channels, multi_scale_output=False, name=name+'_'+str(i+1))
            else:
                out = self.high_resolution_module(out, channels, name=name+'_'+str(i+1))

        return out
    
    def last_cls_out(self, x, name=None):
        out = []
        num_filters_list = [128, 256, 512, 1024]
        for i in range(len(x)):
            out.append(self.conv_bn_layer(input=x[i], filter_size=1, num_filters=num_filters_list[i], 
                                          name=name+'conv_'+str(i+1)))
            
        
        return out

    
    def basic_block(self, input, num_filters, stride=1, downsample=False, name=None):
        residual = input
        conv = self.conv_bn_layer(input=input, filter_size=3, num_filters=num_filters, stride=stride, name=name+'_conv1')
        conv = self.conv_bn_layer(input=conv, filter_size=3, num_filters=num_filters, if_act=False, name=name+'_conv2')
        if downsample:
            residual = self.conv_bn_layer(input=input, filter_size=1, num_filters=num_filters, if_act=False, 
                                          name=name+'_downsample')
        if self.has_se:
            conv = self.squeeze_excitation(
                input=conv,
                num_channels=num_filters,
                reduction_ratio=16,
                name='fc'+name)
        return fluid.layers.elementwise_add(x=residual, y=conv, act='relu')
    

    def bottleneck_block(self, input, num_filters, stride=1, downsample=False, name=None):
        residual = input
        conv = self.conv_bn_layer(input=input, filter_size=1, num_filters=num_filters, name=name+'_conv1')
        conv = self.conv_bn_layer(input=conv, filter_size=3, num_filters=num_filters, stride=stride, name=name+'_conv2')
        conv = self.conv_bn_layer(input=conv, filter_size=1, num_filters=num_filters*4, if_act=False, name=name+'_conv3')
        if downsample:
            residual = self.conv_bn_layer(input=input, filter_size=1, num_filters=num_filters*4, if_act=False, 
                                          name=name+'_downsample')
        if self.has_se:
            conv = self.squeeze_excitation(
                input=conv,
                num_channels=num_filters * 4,
                reduction_ratio=16,
                name='fc'+name)
        return fluid.layers.elementwise_add(x=residual, y=conv, act='relu')
        
    def squeeze_excitation(self, input, num_channels, reduction_ratio, name=None):
        pool = fluid.layers.pool2d(
            input=input, pool_size=0, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(input=pool,
                                  size=num_channels / reduction_ratio,
                                  act='relu',
                                  param_attr=fluid.param_attr.ParamAttr(
                                      initializer=fluid.initializer.Uniform(
                                          -stdv, stdv),name=name+'_sqz_weights'),
                                 bias_attr=ParamAttr(name=name+'_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(input=squeeze,
                                     size=num_channels,
                                     act='sigmoid',
                                     param_attr=fluid.param_attr.ParamAttr(
                                         initializer=fluid.initializer.Uniform(
                                             -stdv, stdv),name=name+'_exc_weights'),
                                     bias_attr=ParamAttr(name=name+'_exc_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale
    
    def conv_bn_layer(self,input, filter_size, num_filters, stride=1, padding=1, num_groups=1, if_act=True, name=None):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size-1)//2,
            groups=num_groups,
            act=None,
            param_attr=ParamAttr(initializer=MSRA(), name=name+'_weights'),
            bias_attr=False)
        bn_name = name + '_bn'
        bn = fluid.layers.batch_norm(input=conv,
                                     param_attr = ParamAttr(name=bn_name+"_scale", initializer=fluid.initializer.Constant(1.0)),
                                     bias_attr=ParamAttr(name=bn_name+"_offset", initializer=fluid.initializer.Constant(0.0)),
                                     moving_mean_name=bn_name + '_mean',
                                     moving_variance_name=bn_name + '_variance')
        if if_act:
            bn = fluid.layers.relu(bn)
        return bn
    
def HRNet_W18_C():
    model = HRNet(width=18)
    return model

def HRNet_W30_C():
    model = HRNet(width=30)
    return model

def HRNet_W32_C():
    model = HRNet(width=32)
    return model

def HRNet_W40_C():
    model = HRNet(width=40)
    return model

def HRNet_W44_C():
    model = HRNet(width=44)
    return model

def HRNet_W48_C():
    model = HRNet(width=48)
    return model
    
def HRNet_W60_C():
    model = HRNet(width=60)
    return model
    
def SE_HRNet_W18_C():
    model = HRNet(width=18, has_se=True)
    return model

def SE_HRNet_W30_C():
    model = HRNet(width=30, has_se=True)
    return model

def SE_HRNet_W32_C():
    model = HRNet(width=32, has_se=True)
    return model

def SE_HRNet_W40_C():
    model = HRNet(width=40, has_se=True)
    return model

def SE_HRNet_W44_C():
    model = HRNet(width=44, has_se=True)
    return model

def SE_HRNet_W48_C():
    model = HRNet(width=48, has_se=True)
    return model
    
def SE_HRNet_W60_C():
    model = HRNet(width=60, has_se=True)
    return model
