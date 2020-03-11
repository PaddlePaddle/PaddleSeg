from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import paddle.fluid as fluid
import numpy as np


def _cumsum(x):
    y = np.array(x)
    return np.cumsum(y, axis=0)

def create_tmp_var(name, dtype, shape):
    return fluid.default_main_program().current_block().create_var(
                    name=name, dtype=dtype, shape=shape
            )

def lovasz_grad(gt_sorted):
    gts = fluid.layers.reduce_sum(gt_sorted)
    with fluid.device_guard("cpu"):
        intersection = gts - fluid.layers.cumsum(gt_sorted, axis=0)
        union = fluid.layers.cumsum((1.0 - gt_sorted), axis=0) + gts
#    tmp1 = create_tmp_var(name='tmp1_', dtype=gt_sorted.dtype, shape=gt_sorted.shape)
#    tmp2 = create_tmp_var(name='tmp2_', dtype=gt_sorted.dtype, shape=gt_sorted.shape)
#    intersection = gts - fluid.layers.py_func(func=_cumsum, x=gt_sorted, out=tmp1)
#    union = fluid.layers.py_func(func=_cumsum, x=1.0 - gt_sorted, out=tmp2) + gts
#    intersection = gts - fluid.layers.cumsum(gt_sorted, axis=0)
#    union = fluid.layers.cumsum((1.0 - gt_sorted), axis=0) + gts
    jaccard = 1.0 - intersection / union
    len_jaccard = fluid.layers.shape(jaccard)[0]
    jaccard0 = fluid.layers.slice(jaccard, axes=[0], starts=[0], ends=[1])
    jaccard1 = fluid.layers.slice(jaccard, axes=[0], starts=[1], ends=[len_jaccard])
    jaccard2 = fluid.layers.slice(jaccard, axes=[0], starts=[0], ends=[-1])
    jaccard = fluid.layers.concat([jaccard0, jaccard1 - jaccard2], axis=0)
#    jaccard = fluid.layers.concat([jaccard[0:1, :], jaccard[1:, :] - jaccard[:-1, :]], axis=0)
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    if per_image:
        def treat_image(log, lab, igno):
            log, lab = fluid.layers.unsqueeze(log, 0), fluid.layers.unsqueeze(lab, 0)
            log, lab = flatten_binary_scores(log, lab, igno)
            return lovasz_hinge_flat(log, lab)
        losses = []
        i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
        batch_size = fluid.layers.shape(logits)[0]
        cond = fluid.layers.less_than(x=i, y=batch_size)
        while_op = fluid.layers.While(cond=cond)
        one = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)
        with while_op.block():
            log = fluid.layers.slice(input=logits, axes=[0], starts=[0], ends=[batch_size])[i]
            lab = fluid.layers.slice(input=labels, axes=[0], starts=[0], ends=[batch_size])[i]
            igno = fluid.layers.slice(input=ignore, axes=[0], starts=[0], ends=[batch_size])[i]
            loss = treat_image(log, lab, igno)
#             end = fluid.layers.elementwise_add(i, one)
#             log = fluid.layers.slice(logits, axes=[0], starts=[i], ends=[end])
#             lab = fluid.layers.slice(labels, axes=[0], starts=[i], ends=[end])
#             loss = treat_image(log, lab)
            losses.append(loss)
            i = fluid.layers.increment(x=i, value=1, in_place=True)
            fluid.layers.less_than(x=i, y=batch_size, cond=cond)
        losses_tensor = fluid.layers.stack(losses)
        loss = fluid.layers.reduce_mean(losses_tensor)   
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss

def lovasz_hinge_flat(logits, labels):
    shape = fluid.layers.shape(logits)
    y = fluid.layers.zeros_like(shape[0]) 
    
    out_var = fluid.layers.create_tensor("float32") 
    with fluid.layers.control_flow.Switch() as switch:
        with switch.case(fluid.layers.equal(shape[0], y)):
            loss = fluid.layers.reduce_sum(logits) * 0.
            fluid.layers.assign(input=loss, output=out_var)
        with switch.case(fluid.layers.greater_than(shape[0], y)):
            labelsf = fluid.layers.cast(labels, logits.dtype) 
            signs = labelsf * 2 - 1. 
            signs.stop_gradient = True
            errors = 1.0 - fluid.layers.elementwise_mul(logits, signs)
            errors_sorted, perm = fluid.layers.argsort(errors, axis=0, descending=True)
            errors_sorted.stop_gradient = False
            gt_sorted = fluid.layers.gather(labelsf, perm) 

            grad = lovasz_grad(gt_sorted)
            grad.stop_gradient = True
            loss = fluid.layers.reduce_sum(fluid.layers.matmul(
                    x = fluid.layers.relu(errors_sorted), 
                    y = grad,
                    transpose_x=True))
            fluid.layers.assign(input=loss, output=out_var)
    return out_var 

def lovasz_hinge_flat_v2(logits, labels):
    def compute_loss():
        labelsf = fluid.layers.cast(labels, logits.dtype)
        signs = labelsf * 2 - 1
        signs.stop_gradient = True
        errors = 1.0 - fluid.layers.elementwise_mul(logits, signs)
        errors_sorted, perm = fluid.layers.argsort(errors, axis=0, descending=True)
        gt_sorted = fluid.layers.gather(labelsf, perm) 
        grad = lovasz_grad(gt_sorted)
        grad.stop_gradient = True
        loss = fluid.layers.reduce_sum(
                fluid.layers.matmul(
                x = fluid.layers.relu(errors_sorted), 
                y = grad,
                transpose_x=True))
        return loss
    loss = fluid.layers.cond(fluid.layers.shape(logits)[0] == 0,
            lambda: fluid.layers.reduce_sum(logits) * 0,
            compute_loss)
    return loss 
    
def flatten_binary_scores(scores, labels, ignore=None): 
    scores = fluid.layers.reshape(scores, [-1,1])
    labels = fluid.layers.reshape(labels, [-1,1])
    labels.stop_gradient = True
    if ignore is None:
        return scores, labels
    ignore = fluid.layers.cast(ignore, 'int32')
    ignore_mask = fluid.layers.reshape(ignore, (-1,1))
    indexs = fluid.layers.where(ignore_mask == 1) 
    indexs.stop_gradient = True
    vscores = fluid.layers.gather(scores, indexs[:,0]) 
    vlabels = fluid.layers.gather(labels, indexs[:,0])
    return vscores, vlabels

def lovasz_softmax(probas, labels, classes='present', per_image=False,
        ignore=None):
    """
    ignore: [N, 1, H, W] Tensor. Void class labels, ignore pixels which value=0
    """
    if per_image:
        def treat_image(prob, lab, igno):
            prob, lab = fluid.layers.unsqueeze(prob, 0), fluid.layers.unsqueeze(lab, 0)
            prob, lab = flatten_probas(prob, lab, igno)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        batch_size = fluid.layers.shape(probas)[0] 
        sum_loss = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0)
        sum_loss.stop_gradient = False
        
        i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)  
        cond = fluid.layers.less_than(x=i, y=batch_size)
        while_op = fluid.layers.While(cond=cond)
        with while_op.block():
            prob = fluid.layers.slice(input=probas, axes=[0], starts=[0], ends=[batch_size])[i]
            lab = fluid.layers.slice(input=labels, axes=[0], starts=[0], ends=[batch_size])[i]
#            prob = probas[i,:,:,:]
#            lab = labels[i,:,:,:]
            igno = fluid.layers.slice(input=ignore, axes=[0], starts=[0], ends=[batch_size])[i]
            single_loss = treat_image(prob, lab, igno)
#             temp = fluid.layers.elementwise_add(x=sum_loss, y=single_loss)
#             fluid.layers.assign(temp, sum_loss)
            sum_loss = fluid.layers.sum([sum_loss, single_loss])
#             b = fluid.layers.sum([sum_loss, single_loss])
#             fluid.layers.assign(b, sum_loss)
            i = fluid.layers.increment(x=i, value=1, in_place=True)
            fluid.layers.less_than(x=i, y=batch_size, cond=cond)
        
          
        
#         prob = probas[i,:,:,:]
#         lab = labels[i,:,:,:]
#         igno = ignore[i,:,:,:]
#         single_loss = treat_image(prob, lab, igno)
#         sum_loss = fluid.layers.sum([sum_loss, single_loss])
    
        
#         i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)  
#         prob = probas[i,:,:,:]
#         lab = labels[i,:,:,:]
#         igno = ignore[i,:,:,:]
#         single_loss = treat_image(prob, lab, igno)
#         sum_loss = fluid.layers.sum([sum_loss, single_loss])
        
        
        batch_size = fluid.layers.cast(batch_size, dtype='float32')
        batch_size.stop_gradient = True
        loss = fluid.layers.elementwise_div(sum_loss, batch_size)
#        loss = sum_loss / batch_size
#        loss = fluid.layers.cast(loss, dtype='float32')
#        loss = fluid.layers.mean(loss)
        print(sum_loss)
        print(batch_size)
        print(loss)
    else:
        vprobas, vlabels = flatten_probas(probas, labels, ignore)
        loss = lovasz_softmax_flat(vprobas, vlabels,classes=classes)
    return loss

def lovasz_softmax_flat(probas, labels, classes='present'):
    C = probas.shape[1]
    losses = []
    present = []
    classes_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in classes_to_sum:
        fg = fluid.layers.cast(labels == c, probas.dtype)
        if classes == 'present':
            present.append(fluid.layers.cast(
                fluid.layers.reduce_sum(fg) > 0, "int64"))
        if C == 1:
            if len(classes_to_sum) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = fluid.layers.abs(fg - class_pred)
        errors_sorted, perm = fluid.layers.argsort(errors, axis=0, descending=True)
        errors_sorted.stop_gradient = False
        fg_sorted = fluid.layers.gather(fg, perm)

        grad = lovasz_grad(fg_sorted)
        grad.stop_gradient = True
        loss = fluid.layers.reduce_sum(
            fluid.layers.matmul(
                    x = errors_sorted,
                    y = grad,
                    transpose_x = True))
        
        losses.append(loss)
    
    if len(classes_to_sum) == 1:
        return losses[0]
    
    # fluid.layers.where 
    losses_tensor = fluid.layers.stack(losses)
    if classes == 'present':
        present_tensor = fluid.layers.stack(present) 
        index = fluid.layers.where(present_tensor == 1)
        index.stop_gradient = True
        losses_tensor = fluid.layers.gather(losses_tensor, index[:, 0]) 
    loss = fluid.layers.mean(losses_tensor)
    return loss

def flatten_probas(probas, labels, ignore=None):
    if len(probas.shape) == 3:
        probas = fluid.layers.unsqueeze(probas, axis=[1])
    C = probas.shape[1]
    probas = fluid.layers.transpose(probas, [0, 2, 3, 1])  
    probas = fluid.layers.reshape(probas, [-1, C])
    labels = fluid.layers.reshape(labels, [-1, 1])
    if ignore is None:
        return probas, labels
    ignore = fluid.layers.cast(ignore, 'int32')
    ignore_mask = fluid.layers.reshape(ignore, [-1, 1])
    indexs = fluid.layers.where(ignore_mask == 1) 
    indexs.stop_gradient = True
    vprobas = fluid.layers.gather(probas, indexs[:,0]) 
    vlabels = fluid.layers.gather(labels, indexs[:,0])
    return vprobas, vlabels
