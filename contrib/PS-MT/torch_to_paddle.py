import torch
import paddle


# 模型转换
def transfer():
    input_fp = "paddleseg/models/resnet101.pth"
    output_fp = "paddleseg/models/resnet101_paddle.pdparams"

    torch_dict = torch.load(input_fp)

    paddle_dict = {}
    for key in torch_dict:
        name = key
        weight = torch_dict[key].cpu().detach().numpy()
        k_parts = key.split('.')
        if k_parts[-1] == "running_mean":
            name_end = '_mean'
            name = ''
            for i in range(len(k_parts) - 1):
                name += k_parts[i] + '.'
            name += name_end

        elif k_parts[-1] == "running_var":
            name_end = '_variance'
            name = ''
            for i in range(len(k_parts) - 1):
                name += k_parts[i] + '.'
            name += name_end
        paddle_dict[name] = weight

    paddle.save(paddle_dict, output_fp)


# 读paddle模型
def read_paddle():
    model_file = "paddleseg/models/resnet101_paddle.pdparams"
    param_state_dict = paddle.load(model_file)
    with open('paddle.txt', 'w') as f:
        for key in param_state_dict.keys():
            f.write(key + '\n')


if __name__ == '__main__':
    transfer()
    read_paddle()
