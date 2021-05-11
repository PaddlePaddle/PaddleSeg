from paddle.fluid.initializer import Constant
from paddle.fluid.param_attr import ParamAttr
import paddle.fluid as fluid


def conv(input,
         num_filters,
         filter_size=None,
         stride=1,
         padding=0,
         dilation=1,
         act=None,
         name='conv'):
    return fluid.layers.conv2d(
        input,
        filter_size=filter_size,
        num_filters=num_filters,
        stride=stride,
        padding=padding,
        dilation=dilation,
        act=act,
        name=name,
        param_attr=name + '_weights',
        bias_attr=name + '_bias')


def conv_transpose(input,
                   num_filters,
                   output_size=None,
                   filter_size=None,
                   stride=1,
                   padding=0,
                   act=None,
                   name='conv_transpose'):
    return fluid.layers.conv2d_transpose(
        input,
        filter_size=filter_size,
        num_filters=num_filters,
        stride=stride,
        padding=padding,
        act=act,
        name=name,
        param_attr=name + '_weights',
        bias_attr=name + '_bias')


EPSILON = 0.0010000000474974513


def bn(input, name):
    bn_id = name.replace('batch_norm', '')
    return fluid.layers.batch_norm(
        input,
        is_test=True,
        epsilon=EPSILON,
        param_attr='bn_scale' + bn_id + '_scale',
        bias_attr='bn_scale' + bn_id + '_offset',
        moving_mean_name=name + '_mean',
        moving_variance_name=name + '_variance',
        name=name)


def max_pool(input, pool_size=2, pool_stride=2, name=None):
    return fluid.layers.pool2d(
        input,
        pool_size=pool_size,
        pool_stride=pool_stride,
        ceil_mode=True,
        pool_type='max',
        exclusive=False,
        name=name)


def SpatialEmbeddings(input):
    conv1 = conv(
        input, filter_size=3, num_filters=13, stride=2, padding=1, name='conv1')
    max_pool1 = fluid.layers.pool2d(
        input, pool_size=2, pool_stride=2, name='max_pool1')
    cat1 = fluid.layers.concat([conv1, max_pool1], axis=1, name='cat1')
    bn_scale1 = bn(cat1, name='batch_norm1')
    relu1 = fluid.layers.relu(bn_scale1)
    conv2 = conv(
        relu1, filter_size=3, num_filters=48, stride=2, padding=1, name='conv2')
    max_pool2 = fluid.layers.pool2d(
        relu1, pool_size=2, pool_stride=2, name='max_pool2')
    cat2 = fluid.layers.concat([conv2, max_pool2], axis=1, name='cat2')
    bn_scale2 = bn(cat2, name='batch_norm2')
    relu2 = fluid.layers.relu(bn_scale2)
    relu3 = conv(
        relu2,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv3',
        act='relu')
    conv4 = conv(
        relu3, filter_size=[1, 3], num_filters=64, padding=[0, 1], name='conv4')
    bn_scale3 = bn(conv4, name='batch_norm3')
    relu4 = fluid.layers.relu(bn_scale3)
    relu5 = conv(
        relu4,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv5',
        act='relu')
    conv6 = conv(
        relu5, filter_size=[1, 3], num_filters=64, padding=[0, 1], name='conv6')
    bn_scale4 = bn(conv6, name='batch_norm4')
    add1 = fluid.layers.elementwise_add(x=bn_scale4, y=relu2, name='add1')
    relu6 = fluid.layers.relu(add1)
    relu7 = conv(
        relu6,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv7',
        act='relu')
    conv8 = conv(
        relu7, filter_size=[1, 3], num_filters=64, padding=[0, 1], name='conv8')
    bn_scale5 = bn(conv8, name='batch_norm5')
    relu8 = fluid.layers.relu(bn_scale5)
    relu9 = conv(
        relu8,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv9',
        act='relu')
    conv10 = conv(
        relu9,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv10')
    bn_scale6 = bn(conv10, name='batch_norm6')
    add2 = fluid.layers.elementwise_add(x=bn_scale6, y=relu6, name='add2')
    relu10 = fluid.layers.relu(add2)
    relu11 = conv(
        relu10,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv11',
        act='relu')
    conv12 = conv(
        relu11,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv12')
    bn_scale7 = bn(conv12, name='batch_norm7')
    relu12 = fluid.layers.relu(bn_scale7)
    relu13 = conv(
        relu12,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv13',
        act='relu')
    conv14 = conv(
        relu13,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv14')
    bn_scale8 = bn(conv14, name='batch_norm8')
    add3 = fluid.layers.elementwise_add(x=bn_scale8, y=relu10, name='add3')
    relu14 = fluid.layers.relu(add3)
    relu15 = conv(
        relu14,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv15',
        act='relu')
    conv16 = conv(
        relu15,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv16')
    bn_scale9 = bn(conv16, name='batch_norm9')
    relu16 = fluid.layers.relu(bn_scale9)
    relu17 = conv(
        relu16,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv17',
        act='relu')
    conv18 = conv(
        relu17,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv18')
    bn_scale10 = bn(conv18, name='batch_norm10')
    add4 = fluid.layers.elementwise_add(x=bn_scale10, y=relu14, name='add4')
    relu18 = fluid.layers.relu(add4)
    relu19 = conv(
        relu18,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv19',
        act='relu')
    conv20 = conv(
        relu19,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv20')
    bn_scale11 = bn(conv20, name='batch_norm11')
    relu20 = fluid.layers.relu(bn_scale11)
    relu21 = conv(
        relu20,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv21',
        act='relu')
    conv22 = conv(
        relu21,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv22')
    bn_scale12 = bn(conv22, name='batch_norm12')
    add5 = fluid.layers.elementwise_add(x=bn_scale12, y=relu18, name='add5')
    relu22 = fluid.layers.relu(add5)
    conv23 = conv(
        relu22,
        filter_size=3,
        num_filters=64,
        stride=2,
        padding=1,
        name='conv23')
    max_pool3 = fluid.layers.pool2d(
        relu22, pool_size=2, pool_stride=2, name='max_pool3')
    cat3 = fluid.layers.concat([conv23, max_pool3], axis=1, name='cat3')
    bn_scale13 = bn(cat3, name='batch_norm13')
    relu23 = fluid.layers.relu(bn_scale13)
    relu24 = conv(
        relu23,
        filter_size=[3, 1],
        num_filters=128,
        padding=[1, 0],
        name='conv24',
        act='relu')
    conv25 = conv(
        relu24,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 1],
        name='conv25')
    bn_scale14 = bn(conv25, name='batch_norm14')
    relu25 = fluid.layers.relu(bn_scale14)
    relu26 = conv(
        relu25,
        filter_size=[3, 1],
        num_filters=128,
        padding=[2, 0],
        dilation=[2, 1],
        name='conv26',
        act='relu')
    conv27 = conv(
        relu26,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 2],
        dilation=[1, 2],
        name='conv27')
    bn_scale15 = bn(conv27, name='batch_norm15')
    add6 = fluid.layers.elementwise_add(x=bn_scale15, y=relu23, name='add6')
    relu27 = fluid.layers.relu(add6)
    relu28 = conv(
        relu27,
        filter_size=[3, 1],
        num_filters=128,
        padding=[1, 0],
        name='conv28',
        act='relu')
    conv29 = conv(
        relu28,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 1],
        name='conv29')
    bn_scale16 = bn(conv29, name='batch_norm16')
    relu29 = fluid.layers.relu(bn_scale16)
    relu30 = conv(
        relu29,
        filter_size=[3, 1],
        num_filters=128,
        padding=[4, 0],
        dilation=[4, 1],
        name='conv30',
        act='relu')
    conv31 = conv(
        relu30,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 4],
        dilation=[1, 4],
        name='conv31')
    bn_scale17 = bn(conv31, name='batch_norm17')
    add7 = fluid.layers.elementwise_add(x=bn_scale17, y=relu27, name='add7')
    relu31 = fluid.layers.relu(add7)
    relu32 = conv(
        relu31,
        filter_size=[3, 1],
        num_filters=128,
        padding=[1, 0],
        name='conv32',
        act='relu')
    conv33 = conv(
        relu32,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 1],
        name='conv33')
    bn_scale18 = bn(conv33, name='batch_norm18')
    relu33 = fluid.layers.relu(bn_scale18)
    relu34 = conv(
        relu33,
        filter_size=[3, 1],
        num_filters=128,
        padding=[8, 0],
        dilation=[8, 1],
        name='conv34',
        act='relu')
    conv35 = conv(
        relu34,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 8],
        dilation=[1, 8],
        name='conv35')
    bn_scale19 = bn(conv35, name='batch_norm19')
    add8 = fluid.layers.elementwise_add(x=bn_scale19, y=relu31, name='add8')
    relu35 = fluid.layers.relu(add8)
    relu36 = conv(
        relu35,
        filter_size=[3, 1],
        num_filters=128,
        padding=[1, 0],
        name='conv36',
        act='relu')
    conv37 = conv(
        relu36,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 1],
        name='conv37')
    bn_scale20 = bn(conv37, name='batch_norm20')
    relu37 = fluid.layers.relu(bn_scale20)
    relu38 = conv(
        relu37,
        filter_size=[3, 1],
        num_filters=128,
        padding=[16, 0],
        dilation=[16, 1],
        name='conv38',
        act='relu')
    conv39 = conv(
        relu38,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 16],
        dilation=[1, 16],
        name='conv39')
    bn_scale21 = bn(conv39, name='batch_norm21')
    add9 = fluid.layers.elementwise_add(x=bn_scale21, y=relu35, name='add9')
    relu39 = fluid.layers.relu(add9)
    relu40 = conv(
        relu39,
        filter_size=[3, 1],
        num_filters=128,
        padding=[1, 0],
        name='conv40',
        act='relu')
    conv41 = conv(
        relu40,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 1],
        name='conv41')
    bn_scale22 = bn(conv41, name='batch_norm22')
    relu41 = fluid.layers.relu(bn_scale22)
    relu42 = conv(
        relu41,
        filter_size=[3, 1],
        num_filters=128,
        padding=[2, 0],
        dilation=[2, 1],
        name='conv42',
        act='relu')
    conv43 = conv(
        relu42,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 2],
        dilation=[1, 2],
        name='conv43')
    bn_scale23 = bn(conv43, name='batch_norm23')
    add10 = fluid.layers.elementwise_add(x=bn_scale23, y=relu39, name='add10')
    relu43 = fluid.layers.relu(add10)
    relu44 = conv(
        relu43,
        filter_size=[3, 1],
        num_filters=128,
        padding=[1, 0],
        name='conv44',
        act='relu')
    conv45 = conv(
        relu44,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 1],
        name='conv45')
    bn_scale24 = bn(conv45, name='batch_norm24')
    relu45 = fluid.layers.relu(bn_scale24)
    relu46 = conv(
        relu45,
        filter_size=[3, 1],
        num_filters=128,
        padding=[4, 0],
        dilation=[4, 1],
        name='conv46',
        act='relu')
    conv47 = conv(
        relu46,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 4],
        dilation=[1, 4],
        name='conv47')
    bn_scale25 = bn(conv47, name='batch_norm25')
    add11 = fluid.layers.elementwise_add(x=bn_scale25, y=relu43, name='add11')
    relu47 = fluid.layers.relu(add11)
    relu48 = conv(
        relu47,
        filter_size=[3, 1],
        num_filters=128,
        padding=[1, 0],
        name='conv48',
        act='relu')
    conv49 = conv(
        relu48,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 1],
        name='conv49')
    bn_scale26 = bn(conv49, name='batch_norm26')
    relu49 = fluid.layers.relu(bn_scale26)
    relu50 = conv(
        relu49,
        filter_size=[3, 1],
        num_filters=128,
        padding=[8, 0],
        dilation=[8, 1],
        name='conv50',
        act='relu')
    conv51 = conv(
        relu50,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 8],
        dilation=[1, 8],
        name='conv51')
    bn_scale27 = bn(conv51, name='batch_norm27')
    add12 = fluid.layers.elementwise_add(x=bn_scale27, y=relu47, name='add12')
    relu51 = fluid.layers.relu(add12)
    relu52 = conv(
        relu51,
        filter_size=[3, 1],
        num_filters=128,
        padding=[1, 0],
        name='conv52',
        act='relu')
    conv53 = conv(
        relu52,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 1],
        name='conv53')
    bn_scale28 = bn(conv53, name='batch_norm28')
    relu53 = fluid.layers.relu(bn_scale28)
    relu54 = conv(
        relu53,
        filter_size=[3, 1],
        num_filters=128,
        padding=[16, 0],
        dilation=[16, 1],
        name='conv54',
        act='relu')
    conv55 = conv(
        relu54,
        filter_size=[1, 3],
        num_filters=128,
        padding=[0, 16],
        dilation=[1, 16],
        name='conv55')
    bn_scale29 = bn(conv55, name='batch_norm29')
    add13 = fluid.layers.elementwise_add(x=bn_scale29, y=relu51, name='add13')
    relu55 = fluid.layers.relu(add13)
    conv_transpose1 = conv_transpose(
        relu55,
        filter_size=3,
        num_filters=64,
        stride=2,
        padding=1,
        name='conv_transpose1')
    conv_transpose4 = conv_transpose(
        relu55,
        filter_size=3,
        num_filters=64,
        stride=2,
        padding=1,
        name='conv_transpose4')
    bn_scale30 = bn(conv_transpose1, name='batch_norm30')
    bn_scale40 = bn(conv_transpose4, name='batch_norm40')
    relu56 = fluid.layers.relu(bn_scale30)
    relu74 = fluid.layers.relu(bn_scale40)
    relu57 = conv(
        relu56,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv56',
        act='relu')
    relu75 = conv(
        relu74,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv72',
        act='relu')
    conv57 = conv(
        relu57,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv57')
    conv73 = conv(
        relu75,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv73')
    bn_scale31 = bn(conv57, name='batch_norm31')
    bn_scale41 = bn(conv73, name='batch_norm41')
    relu58 = fluid.layers.relu(bn_scale31)
    relu76 = fluid.layers.relu(bn_scale41)
    relu59 = conv(
        relu58,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv58',
        act='relu')
    relu77 = conv(
        relu76,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv74',
        act='relu')
    conv59 = conv(
        relu59,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv59')
    conv75 = conv(
        relu77,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv75')
    bn_scale32 = bn(conv59, name='batch_norm32')
    bn_scale42 = bn(conv75, name='batch_norm42')
    add14 = fluid.layers.elementwise_add(x=bn_scale32, y=relu56, name='add14')
    add18 = fluid.layers.elementwise_add(x=bn_scale42, y=relu74, name='add18')
    relu60 = fluid.layers.relu(add14)
    relu78 = fluid.layers.relu(add18)
    relu61 = conv(
        relu60,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv60',
        act='relu')
    relu79 = conv(
        relu78,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv76',
        act='relu')
    conv61 = conv(
        relu61,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv61')
    conv77 = conv(
        relu79,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv77')
    bn_scale33 = bn(conv61, name='batch_norm33')
    bn_scale43 = bn(conv77, name='batch_norm43')
    relu62 = fluid.layers.relu(bn_scale33)
    relu80 = fluid.layers.relu(bn_scale43)
    relu63 = conv(
        relu62,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv62',
        act='relu')
    relu81 = conv(
        relu80,
        filter_size=[3, 1],
        num_filters=64,
        padding=[1, 0],
        name='conv78',
        act='relu')
    conv63 = conv(
        relu63,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv63')
    conv79 = conv(
        relu81,
        filter_size=[1, 3],
        num_filters=64,
        padding=[0, 1],
        name='conv79')
    bn_scale34 = bn(conv63, name='batch_norm34')
    bn_scale44 = bn(conv79, name='batch_norm44')
    add15 = fluid.layers.elementwise_add(x=bn_scale34, y=relu60, name='add15')
    add19 = fluid.layers.elementwise_add(x=bn_scale44, y=relu78, name='add19')
    relu64 = fluid.layers.relu(add15)
    relu82 = fluid.layers.relu(add19)
    conv_transpose2 = conv_transpose(
        relu64,
        filter_size=3,
        num_filters=16,
        stride=2,
        padding=1,
        name='conv_transpose2')
    conv_transpose5 = conv_transpose(
        relu82,
        filter_size=3,
        num_filters=16,
        stride=2,
        padding=1,
        name='conv_transpose5')
    bn_scale35 = bn(conv_transpose2, name='batch_norm35')
    bn_scale45 = bn(conv_transpose5, name='batch_norm45')
    relu65 = fluid.layers.relu(bn_scale35)
    relu83 = fluid.layers.relu(bn_scale45)
    relu66 = conv(
        relu65,
        filter_size=[3, 1],
        num_filters=16,
        padding=[1, 0],
        name='conv64',
        act='relu')
    relu84 = conv(
        relu83,
        filter_size=[3, 1],
        num_filters=16,
        padding=[1, 0],
        name='conv80',
        act='relu')
    conv65 = conv(
        relu66,
        filter_size=[1, 3],
        num_filters=16,
        padding=[0, 1],
        name='conv65')
    conv81 = conv(
        relu84,
        filter_size=[1, 3],
        num_filters=16,
        padding=[0, 1],
        name='conv81')
    bn_scale36 = bn(conv65, name='batch_norm36')
    bn_scale46 = bn(conv81, name='batch_norm46')
    relu67 = fluid.layers.relu(bn_scale36)
    relu85 = fluid.layers.relu(bn_scale46)
    relu68 = conv(
        relu67,
        filter_size=[3, 1],
        num_filters=16,
        padding=[1, 0],
        name='conv66',
        act='relu')
    relu86 = conv(
        relu85,
        filter_size=[3, 1],
        num_filters=16,
        padding=[1, 0],
        name='conv82',
        act='relu')
    conv67 = conv(
        relu68,
        filter_size=[1, 3],
        num_filters=16,
        padding=[0, 1],
        name='conv67')
    conv83 = conv(
        relu86,
        filter_size=[1, 3],
        num_filters=16,
        padding=[0, 1],
        name='conv83')
    bn_scale37 = bn(conv67, name='batch_norm37')
    bn_scale47 = bn(conv83, name='batch_norm47')
    add16 = fluid.layers.elementwise_add(x=bn_scale37, y=relu65, name='add16')
    add20 = fluid.layers.elementwise_add(x=bn_scale47, y=relu83, name='add20')
    relu69 = fluid.layers.relu(add16)
    relu87 = fluid.layers.relu(add20)
    relu70 = conv(
        relu69,
        filter_size=[3, 1],
        num_filters=16,
        padding=[1, 0],
        name='conv68',
        act='relu')
    relu88 = conv(
        relu87,
        filter_size=[3, 1],
        num_filters=16,
        padding=[1, 0],
        name='conv84',
        act='relu')
    conv69 = conv(
        relu70,
        filter_size=[1, 3],
        num_filters=16,
        padding=[0, 1],
        name='conv69')
    conv85 = conv(
        relu88,
        filter_size=[1, 3],
        num_filters=16,
        padding=[0, 1],
        name='conv85')
    bn_scale38 = bn(conv69, name='batch_norm38')
    bn_scale48 = bn(conv85, name='batch_norm48')
    relu71 = fluid.layers.relu(bn_scale38)
    relu89 = fluid.layers.relu(bn_scale48)
    relu72 = conv(
        relu71,
        filter_size=[3, 1],
        num_filters=16,
        padding=[1, 0],
        name='conv70',
        act='relu')
    relu90 = conv(
        relu89,
        filter_size=[3, 1],
        num_filters=16,
        padding=[1, 0],
        name='conv86',
        act='relu')
    conv71 = conv(
        relu72,
        filter_size=[1, 3],
        num_filters=16,
        padding=[0, 1],
        name='conv71')
    conv87 = conv(
        relu90,
        filter_size=[1, 3],
        num_filters=16,
        padding=[0, 1],
        name='conv87')
    bn_scale39 = bn(conv71, name='batch_norm39')
    bn_scale49 = bn(conv87, name='batch_norm49')
    add17 = fluid.layers.elementwise_add(x=bn_scale39, y=relu69, name='add17')
    add21 = fluid.layers.elementwise_add(x=bn_scale49, y=relu87, name='add21')
    relu73 = fluid.layers.relu(add17)
    relu91 = fluid.layers.relu(add21)
    conv_transpose3 = conv_transpose(
        relu73, filter_size=2, num_filters=4, stride=2, name='conv_transpose3')
    conv_transpose6 = conv_transpose(
        relu91, filter_size=2, num_filters=1, stride=2, name='conv_transpose6')
    cat4 = fluid.layers.concat([conv_transpose3, conv_transpose6],
                               axis=1,
                               name='cat4')

    return cat4
