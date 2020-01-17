from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import time
import paddle.fluid as fluid


def argsort(input, axis=0, type="descend"):
    if type == "increase":
        input_sorted, perm = fluid.layers.argsort(input, axis=axis)
    elif type == 'descend':
        input_sorted = input * (-1)
        input_sorted, perm = fluid.layers.argsort(input_sorted, axis=axis)
        input_sorted = input_sorted * (-1)
    else:
        raise Exception("only support increase and descend")
    return input_sorted, perm

def lovasz_grad(gt_sorted):
    gts = fluid.layers.reduce_sum(gt_sorted)
    intersection = (fluid.layers.cumsum(gt_sorted, axis=0) - gts) * (-1)
    union = fluid.layers.cumsum((1.0 - gt_sorted), axis=0) + gts
    jaccard = (intersection / union - 1.0) * (-1)
    jaccard = fluid.layers.concat([jaccard[0:1, :], jaccard[1:, :] - jaccard[:-1, :]], axis=0)
    return jaccard

def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    if per_image:
        def treat_image(log, lab):
            log, lab = fluid.layers.unsqueeze(log, 0), fluid.layers.unsqueeze(lab, 0)
            log, lab = flatten_binary_scores(log, lab, ignore)
            return lovasz_hinge_flat(log, lab)
        losses = []
        i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)
        loop_len = fluid.layers.shape(logits)[0]
        cond = fluid.layers.less_than(x=i, y=loop_len)
        while_op = fluid.layers.While(cond=cond)
        one = fluid.layers.fill_constant(shape=[1], dtype='int32', value=1)
        with while_op.block():
            end = fluid.layers.elementwise_add(i, one)
            log = fluid.layers.slice(logits, axes=[0], starts=[i], ends=[end])
            lab = fluid.layers.slice(labels, axes=[0], starts=[i], ends=[end])
            loss = treat_image(log, lab)
            losses.append(loss)
            i = fluid.layers.increment(x=i, value=1, in_place=True)
            fluid.layers.less_than(x=i, y=loop_len, cond=cond)
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
            errors = (fluid.layers.elementwise_mul(logits, signs) - 1.0) * (-1)
            errors_sorted, perm = argsort(errors)
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
        errors = (fluid.layers.elementwise_mul(logits, signs) - 1.0) * (-1)
        errors_sorted, perm = argsort(errors)
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
    if per_image:
        def treat_image(prob, lab):
            prob, lab = fluid.layers.unsqueeze(prob, 0), fluid.layers.unsqueeze(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = []
        # to do: debug
        batch_size = fluid.layers.shape(probas)[0] 
        sum_loss = fluid.layers.fill_constant(shape=[1], dtype='float32', value=0)
        
        i = fluid.layers.fill_constant(shape=[1], dtype='int32', value=0)  
        cond = fluid.layers.less_than(x=i, y=batch_size)
        while_op = fluid.layers.While(cond=cond)
        
        with while_op.block():
            prob = probas[i,:,:,:]
            lab = labels[i,:,:,:]
            loss = treat_image(prob, lab)
            sum_loss = fluid.layers.sum_inplace([sum_loss, loss])
            
            i = fluid.layers.increment(x=i, value=1, in_place=True)
            fluid.layers.less_than(x=i, y=batch_size, cond=cond)
          
        batch_size = fluid.layers.cast(batch_size, dtype='float32')
        batch_size.stop_gradient = True
        
        loss = fluid.layers.elementwise_div(sum_loss, batch_size)
#         loss = sum_loss / batch_size
#         loss = fluid.layers.reduce_mean(lossesa)   
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
        
        errors_sorted, perm = argsort(errors)
        fg_sorted = fluid.layers.gather(fg, perm)

        grad = lovasz_grad(fg_sorted)
        grad.stop_gradient = True
        loss = fluid.layers.reduce_sum(
            fluid.layers.matmul(
                    x = errors_sorted,
                    y = grad,
                    transpose_x = True))
        
#         fg.stop_gradient = True
#         grad = lovasz_grad(fg)
#         grad.stop_gradient = True
#         loss = fluid.layers.reduce_sum(
#             fluid.layers.matmul(
#                     x = errors,
#                     y = grad,
#                     transpose_x = True))
        
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
