import random
import numpy as np
from matplotlib import pyplot as plt


class ModelPerformance():
    """
        Currently, the models implemented in PaddleSeg is mostly on CityScapes with resoulution 1024x512.
        Thus, this class is used to plot model's performance.All resolution is unified to `1024x512`.
        
        However: 
                1. Some models do not have `model.pdparams` yet,
                2. Some models already have `model.pdparams`, but its resolution is not equal to 1024x512.
        Thus, we set some status flags to note them. Please refer to the function `get_model_status()`.

        This class include several funcitons:
                get_model_names()
                get_model_status()
                set_detailed_info()
                do_plot()

        If needs, please modify them synchronously.
        """

    def set_model_info(self):
        """
                Define the detailed information of each implemented models.
                The mIoU is evaluated with the resulotion: 1024x512.  
                The whole processing time includes preprocess_time, Inference Time (ms) and postprocess_time.(ms)
                """
        model_infos = {}
        # large model
        model_info = {}
        model_info['mIoU'] = 79.50
        model_info['preprocess_time'] = 143.3464
        model_info['Inference Time (ms)'] = 94.9185
        model_info['postprocess_time'] = 0.0133
        model_info['FLOPs (G)'] = 564.43
        model_info['Params (M)'] = 67.70
        model_infos['ANN_ResNet101'] = model_info

        model_info = {}
        model_info['mIoU'] = 80.85
        model_info['preprocess_time'] = 141.65
        model_info['Inference Time (ms)'] = 114.00
        model_info['postprocess_time'] = 0.014
        model_info['FLOPs (G)'] = 481.00
        model_info['Params (M)'] = 58.17
        model_infos['DeeplabV3_ResNet101'] = model_info

        model_info = {}
        model_info['mIoU'] = 81.10
        model_info['preprocess_time'] = 147.2358
        model_info['Inference Time (ms)'] = 69.7831
        model_info['postprocess_time'] = 0.016
        model_info['FLOPs (G)'] = 228.44
        model_info['Params (M)'] = 26.79
        model_infos['DeeplabV3p_ResNet50'] = model_info

        model_info = {}
        model_info['mIoU'] = 80.70
        model_info['preprocess_time'] = 130.5809
        model_info['Inference Time (ms)'] = 45.4565
        model_info['postprocess_time'] = 0.012
        model_info['FLOPs (G)'] = 187.50
        model_info['Params (M)'] = 65.94
        model_infos['FCN_HRNetw18'] = model_info

        model_info = {}
        model_info['mIoU'] = 81.01
        model_info['preprocess_time'] = 119.3773
        model_info['Inference Time (ms)'] = 90.2779
        model_info['postprocess_time'] = 0.0127
        model_info['FLOPs (G)'] = 570.74
        model_info['Params (M)'] = 68.73
        model_infos['GCNet_ResNet101'] = model_info

        model_info = {}
        model_info['mIoU'] = 82.15
        model_info['preprocess_time'] = 138.4849
        model_info['Inference Time (ms)'] = 61.8837
        model_info['postprocess_time'] = 0.0143
        model_info['FLOPs (G)'] = 324.66
        model_info['Params (M)'] = 70.47
        model_infos['OCRNet_HRNetw48'] = model_info

        model_info = {}
        model_info['mIoU'] = 80.48
        model_info['preprocess_time'] = 134.2316
        model_info['Inference Time (ms)'] = 115.9394
        model_info['postprocess_time'] = 0.0123
        model_info['FLOPs (G)'] = 686.89
        model_info['Params (M)'] = 86.97
        model_infos['PSPNet_ResNet101'] = model_info

        model_info = {}
        model_info['mIoU'] = 81.26
        model_info['preprocess_time'] = 136.2757
        model_info['Inference Time (ms)'] = 66.8982
        model_info['postprocess_time'] = 0.0125
        model_info['FLOPs (G)'] = 395.10
        model_info['Params (M)'] = 41.71
        model_infos['DecoupledSegnet_ResNet50'] = model_info

        model_info = {}
        model_info['mIoU'] = 80.00
        model_info['preprocess_time'] = 140.4741
        model_info['Inference Time (ms)'] = 80.0515
        model_info['postprocess_time'] = 0.0133
        model_info['FLOPs (G)'] = 512.18
        model_info['Params (M)'] = 61.45
        model_infos['EMANet_ResNet101'] = model_info

        model_info = {}
        model_info['mIoU'] = 81.03
        model_info['preprocess_time'] = 138.9493
        model_info['Inference Time (ms)'] = 97.8105
        model_info['postprocess_time'] = 0.0142
        model_info['FLOPs (G)'] = 575.04
        model_info['Params (M)'] = 69.13
        model_infos['DNLnet_ResNet101'] = model_info

        model_info = {}
        model_info['mIoU'] = 80.27
        model_info['preprocess_time'] = 134.7776
        model_info['Inference Time (ms)'] = 95.0819
        model_info['postprocess_time'] = 0.0146
        model_info['FLOPs (G)'] = 398.48
        model_info['Params (M)'] = 47.52
        model_infos['DANet_ResNet50'] = model_info

        model_info = {}
        model_info['mIoU'] = 80.10
        model_info['preprocess_time'] = 132.4482
        model_info['Inference Time (ms)'] = 81.2459
        model_info['postprocess_time'] = 0.0143
        model_info['FLOPs (G)'] = 474.13
        model_info['Params (M)'] = 56.81
        model_infos['ISANet_ResNet101'] = model_info

        model_info = {}
        model_info['mIoU'] = 78.72
        model_info['preprocess_time'] = 131.6711
        model_info['Inference Time (ms)'] = 69.5135
        model_info['postprocess_time'] = 0.0153
        model_info['FLOPs (G)'] = 136.80
        model_info['Params (M)'] = 13.81
        model_infos['SFNet_ResNet18'] = model_info

        model_info = {}
        model_info['mIoU'] = 81.49
        model_info['Inference Time (ms)'] = 121.35
        model_info['FLOPs (G)'] = 394.37
        model_info['Params (M)'] = 42.03
        model_infos['SFNet_ResNet50'] = model_info

        model_info = {}
        model_info['mIoU'] = 76.54
        model_info['Inference Time (ms)'] = 70.35
        model_info['FLOPs (G)'] = 363.17
        model_info['Params (M)'] = 28.18
        model_infos['PointRend_ResNet50'] = model_info

        model_info = {}
        model_info['mIoU'] = 81.60
        model_info['Inference Time (ms)'] = 47.08
        model_info['FLOPs (G)'] = 113.71
        model_info['Params (M)'] = 27.36
        model_infos['SegFormer_B2'] = model_info

        model_info = {}
        model_info['mIoU'] = 82.47
        model_info['Inference Time (ms)'] = 62.70
        model_info['FLOPs (G)'] = 142.97
        model_info['Params (M)'] = 47.24
        model_infos['SegFormer_B3'] = model_info

        model_info = {}
        model_info['mIoU'] = 82.38
        model_info['Inference Time (ms)'] = 73.26
        model_info['FLOPs (G)'] = 171.05
        model_info['Params (M)'] = 64.01
        model_infos['SegFormer_B4'] = model_info

        model_info = {}
        model_info['mIoU'] = 82.58
        model_info['Inference Time (ms)'] = 84.34
        model_info['FLOPs (G)'] = 199.68
        model_info['Params (M)'] = 84.61
        model_infos['SegFormer_B5'] = model_info

        model_info = {}
        model_info['mIoU'] = 77.29
        model_info['Inference Time (ms)'] = 201.26
        model_info['FLOPs (G)'] = 620.94
        model_info['Params (M)'] = 303.37
        model_infos['SETR-Naive'] = model_info

        model_info = {}
        model_info['mIoU'] = 78.08
        model_info['Inference Time (ms)'] = 212.22
        model_info['FLOPs (G)'] = 727.46
        model_info['Params (M)'] = 307.24
        model_infos['SETR-PUP'] = model_info

        model_info = {}
        model_info['mIoU'] = 76.52
        model_info['Inference Time (ms)'] = 204.87
        model_info['FLOPs (G)'] = 633.88
        model_info['Params (M)'] = 307.05
        model_infos['SETR-MLA'] = model_info

        # lightweight
        model_info = {}
        model_info['mIoU'] = 76.73
        model_info['Inference Time (ms)'] = 15.66
        model_info['FLOPs (G)'] = 13.63
        model_info['Params (M)'] = 3.72
        model_infos['SegFormer_B0'] = model_info

        model_info = {}
        model_info['mIoU'] = 78.35
        model_info['Inference Time (ms)'] = 21.48
        model_info['FLOPs (G)'] = 26.55
        model_info['Params (M)'] = 13.68
        model_infos['SegFormer_B1'] = model_info

        model_info = {}
        model_info['mIoU'] = 74.74
        model_info['Inference Time (ms)'] = 9.10
        model_info['FLOPs (G)'] = 24.83
        model_info['Params (M)'] = 8.29
        model_infos['STDC1-Seg50'] = model_info

        model_info = {}
        model_info['mIoU'] = 77.60
        model_info['Inference Time (ms)'] = 10.88
        model_info['FLOPs (G)'] = 38.05
        model_info['Params (M)'] = 12.33
        model_infos['STDC2-Seg50'] = model_info

        model_info = {}
        model_info['mIoU'] = 73.19
        model_info['preprocess_time'] = 145.7413
        model_info['Inference Time (ms)'] = 7.5558
        model_info['postprocess_time'] = 0.0132
        model_info['FLOPs (G)'] = 16.14
        model_info['Params (M)'] = 2.33
        model_infos['BiseNetV2'] = model_info

        model_info = {}
        model_info['mIoU'] = 69.31
        model_info['preprocess_time'] = 142.109
        model_info['Inference Time (ms)'] = 8.7706
        model_info['postprocess_time'] = 0.0134
        model_info['FLOPs (G)'] = 2.04
        model_info['Params (M)'] = 1.44
        model_infos['Fast-SCNN'] = model_info

        model_info = {}
        model_info['mIoU'] = 79.03
        model_info['preprocess_time'] = 135.8398
        model_info['Inference Time (ms)'] = 12.8848
        model_info['postprocess_time'] = 0.0127
        model_info['FLOPs (G)'] = 35.40
        model_info['Params (M)'] = 4.13
        model_infos['HarDNet'] = model_info

        # model_info = {}
        # model_info['mIoU'] = 65.00
        # model_info['preprocess_time'] = 137.7497
        # model_info['Inference Time (ms)'] = 29.1146
        # model_info['postprocess_time'] = 0.0123  
        # model_info['FLOPs (G)'] = 253.75
        # model_info['Params (M)'] = 13.41
        # model_infos['U-Net'] = model_info

        return model_infos

    def do_plot(self, title, xlabel, model_infos):
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel('mIoU(%)')

        styles = [
            'Hr', '+r', '*r', '*g', 'hg', 'pg', 'sg', '+g', 'hc', '^c', '+c',
            '*c', 'py', '*y', '+y', 'pk', '+k', 'Dk', 'sk', 'dm', '*m', 'Dm',
            'hb', 'pb', 'Db', '+b', 'sy', 'Hy', 'Dy', '^y'
        ]
        num_styles = len(styles)

        # random.shuffle(styles)

        infer_time = []
        mIoUs = []
        marks = []  # Related to the model that on the figure already.
        labels = []  # Model's name whose status is `Correct`.

        index = 0  # The index of the current using style.

        for model in model_infos:
            infer_time.append(model_infos[model][xlabel])
            mIoUs.append(model_infos[model]['mIoU'])
            x, = plt.plot(infer_time[index], mIoUs[index], styles[index])
            marks.append(x)
            labels.append(model)

            index = (index + 1) % num_styles

        # Add some fonts.
        # if 'OCRNet' in model_infos:
        #         plt.text(55, 82.3,'OCRNet', fontdict={'size':11, 'color':'red'}) 
        # if 'BiseNetV2' in model_infos:
        #         plt.text(14, 74.5, 'BiseNetV2', fontdict={'size':11, 'color':'red'})
        # if 'DeepLabV3p' in model_infos:
        #         plt.text(60.1, 80.1, 'DeepLabV3p', fontdict={'size':11, 'color':'green'})
        # if 'PSPNet' in model_infos:
        #         plt.text(110, 79.8, 'PSPNet', fontdict={'size':11, 'color':'c'})
        # if 'DANet' in model_infos:
        #         plt.text(86.2, 79.6, 'DANet', fontdict={'size':11, 'color':'green'})
        # plt.text(12.889,82.54,'OCRNet',fontdict={'size':11,'color':'c'})
        # plt.text(31.365,78.13,'HarDNet',fontdict={'size':11,'color':'m'})
        # plt.text(35.986,81.49,'SFNet',fontdict={'size':11,'color':'green'})

        plt.legend(
            marks, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

        plt.subplots_adjust(right=0.7)
        plt.grid(
            b=None,
            which='major',
            axis='both', )

        # Add a line on the figure. Could connect DeepLab series or UNet series.
        # plt.plot([29.1, 70, 90.3],[67.2, 81.4, 81.6], linewidth=1, color='orange')

        plt.show()


if __name__ == '__main__':
    plot_object = ModelPerformance()

    model_infos = plot_object.set_model_info()
    title = 'Performance of Segmentation Models'
    xlabel = 'Inference Time (ms)'
    plot_object.do_plot(title, xlabel, model_infos)
    xlabel = 'FLOPs (G)'
    plot_object.do_plot(title, xlabel, model_infos)
    xlabel = 'Params (M)'
    plot_object.do_plot(title, xlabel, model_infos)
