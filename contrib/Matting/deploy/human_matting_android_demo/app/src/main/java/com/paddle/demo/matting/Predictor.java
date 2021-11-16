package com.paddle.demo.matting;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.PowerMode;
import com.baidu.paddle.lite.Tensor;
import com.paddle.demo.matting.config.Config;

import com.paddle.demo.matting.preprocess.Preprocess;
import com.paddle.demo.matting.visual.Visualize;

import java.io.File;
import java.io.InputStream;
import java.util.Date;
import java.util.Vector;

public class Predictor {
    private static final String TAG = Predictor.class.getSimpleName();
    protected Vector<String> wordLabels = new Vector<String>();

    Config config = new Config();

    protected Bitmap inputImage = null;//输入图像
    protected Bitmap scaledImage = null;//尺度归一化图像
    protected Bitmap bgImage = null;//背景图像
    protected Bitmap outputImage = null;//输出图像
    protected String outputResult = "";//输出结果
    protected float preprocessTime = 0;//预处理时间
    protected float postprocessTime = 0;//后处理时间

    public boolean isLoaded = false;
    public int warmupIterNum = 0;
    public int inferIterNum = 1;
    protected Context appCtx = null;
    public int cpuThreadNum = 1;
    public String cpuPowerMode = "LITE_POWER_HIGH";
    public String modelPath = "";
    public String modelName = "";
    protected PaddlePredictor paddlePredictor = null;
    protected float inferenceTime = 0;

    public Predictor() {
        super();
    }

    public boolean init(Context appCtx, String modelPath, int cpuThreadNum, String cpuPowerMode) {
        this.appCtx = appCtx;
        isLoaded = loadModel(modelPath, cpuThreadNum, cpuPowerMode);
        return isLoaded;
    }

    public boolean init(Context appCtx, Config config) {

        if (config.inputShape.length != 4) {
            Log.i(TAG, "size of input shape should be: 4");
            return false;
        }
        if (config.inputShape[0] != 1) {
            Log.i(TAG, "only one batch is supported in the matting demo, you can use any batch size in " +
                    "your Apps!");
            return false;
        }
        if (config.inputShape[1] != 1 && config.inputShape[1] != 3) {
            Log.i(TAG, "only one/three channels are supported in the image matting demo, you can use any " +
                    "channel size in your Apps!");
            return false;
        }
        if (!config.inputColorFormat.equalsIgnoreCase("RGB") && !config.inputColorFormat.equalsIgnoreCase("BGR")) {
            Log.i(TAG, "only RGB and BGR color format is supported.");
            return false;
        }
        init(appCtx, config.modelPath, config.cpuThreadNum, config.cpuPowerMode);

        if (!isLoaded()) {
            return false;
        }
        this.config = config;

        return isLoaded;
    }


    public boolean isLoaded() {
        return paddlePredictor != null && isLoaded;
    }

    //加载真值标签label，在本matting实例中用不到（在图像分类和图像分割任务中如果是多类目标那么一般需要提供label）
    protected boolean loadLabel(String labelPath) {
        wordLabels.clear();
        // load word labels from file
        try {
            InputStream assetsInputStream = appCtx.getAssets().open(labelPath);
            int available = assetsInputStream.available();
            byte[] lines = new byte[available];
            assetsInputStream.read(lines);
            assetsInputStream.close();
            String words = new String(lines);
            String[] contents = words.split("\n");
            for (String content : contents) {
                wordLabels.add(content);
            }
            Log.i(TAG, "word label size: " + wordLabels.size());
        } catch (Exception e) {
            Log.e(TAG, e.getMessage());
            return false;
        }
        return true;
    }

    public Tensor getInput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return paddlePredictor.getInput(idx);
    }

    public Tensor getOutput(int idx) {
        if (!isLoaded()) {
            return null;
        }
        return paddlePredictor.getOutput(idx);
    }

    protected boolean loadModel(String modelPath, int cpuThreadNum, String cpuPowerMode) {
        // 如果已经加载过模型则先释放掉
        releaseModel();

        // 加载模型
        if (modelPath.isEmpty()) {
            return false;
        }
        String realPath = modelPath;
        if (!modelPath.substring(0, 1).equals("/")) {
            // read model files from custom file_paths if the first character of mode file_paths is '/'
            // otherwise copy model to cache from assets
            realPath = appCtx.getCacheDir() + "/" + modelPath;
            Utils.copyDirectoryFromAssets(appCtx, modelPath, realPath);
        }
        if (realPath.isEmpty()) {
            return false;
        }
        MobileConfig config = new MobileConfig();
        //修改模型
        config.setModelFromFile(realPath + File.separator + "hrnet_w18.nb");
        config.setThreads(cpuThreadNum);
        if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_LOW);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_FULL")) {
            config.setPowerMode(PowerMode.LITE_POWER_FULL);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_NO_BIND")) {
            config.setPowerMode(PowerMode.LITE_POWER_NO_BIND);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_LOW);
        } else {
            Log.e(TAG, "unknown cpu power mode!");
            return false;
        }
        paddlePredictor = PaddlePredictor.createPaddlePredictor(config);
        this.cpuThreadNum = cpuThreadNum;
        this.cpuPowerMode = cpuPowerMode;
        this.modelPath = realPath;
        this.modelName = realPath.substring(realPath.lastIndexOf("/") + 1);
        return true;
    }

    public boolean runModel() {
        if (!isLoaded()) {
            return false;
        }
        // warm up
        for (int i = 0; i < warmupIterNum; i++){
            paddlePredictor.run();
        }
        // inference
        Date start = new Date();
        for (int i = 0; i < inferIterNum; i++) {
            paddlePredictor.run();
        }
        Date end = new Date();
        inferenceTime = (end.getTime() - start.getTime()) / (float) inferIterNum;
        return true;
    }

    public boolean runModel(Bitmap image) {
        setInputImage(image,bgImage);
        return runModel();
    }

    //正式的推理主函数
    public boolean runModel(Preprocess preprocess, Visualize visualize) {
        if (inputImage == null || bgImage == null) {
            return false;
        }

        // set input shape
        Tensor inputTensor = getInput(0);
        inputTensor.resize(config.inputShape);

        // pre-process image
        Date start = new Date();

        // setInputImage(scaledImage);
        preprocess.init(config);
        preprocess.to_array(scaledImage);
        preprocess.normalize(preprocess.inputData);

        // feed input tensor with pre-processed data
        inputTensor.setData(preprocess.inputData);

        Date end = new Date();
        preprocessTime = (float) (end.getTime() - start.getTime());

        // inference
        runModel();

        start = new Date();
        Tensor outputTensor = getOutput(0);

        // post-process
        this.outputImage = visualize.draw(inputImage, outputTensor,bgImage);
        postprocessTime = (float) (end.getTime() - start.getTime());

        outputResult = new String();

        return true;
    }
    public void releaseModel() {
        paddlePredictor = null;
        isLoaded = false;
        cpuThreadNum = 1;
        cpuPowerMode = "LITE_POWER_HIGH";
        modelPath = "";
        modelName = "";
    }

    public void setConfig(Config config){
        this.config = config;
    }

    public Bitmap inputImage() {
        return inputImage;
    }

    public Bitmap outputImage() {
        return outputImage;
    }

    public String outputResult() {
        return outputResult;
    }

    public float preprocessTime() {
        return preprocessTime;
    }

    public float postprocessTime() {
        return postprocessTime;
    }

    public String modelPath() {
        return modelPath;
    }

    public String modelName() {
        return modelName;
    }

    public int cpuThreadNum() {
        return cpuThreadNum;
    }

    public String cpuPowerMode() {
        return cpuPowerMode;
    }

    public float inferenceTime() {
        return inferenceTime;
    }

    public void setInputImage(Bitmap image,Bitmap bg) {
        if (image == null || bg == null) {
            return;
        }
        // 预处理输入图像
        Bitmap rgbaImage = image.copy(Bitmap.Config.ARGB_8888, true);
        Bitmap scaleImage = Bitmap.createScaledBitmap(rgbaImage, (int) this.config.inputShape[3], (int) this.config.inputShape[2], true);
        this.inputImage = rgbaImage;
        this.scaledImage = scaleImage;

        //预处理背景图像
        this.bgImage = bg.copy(Bitmap.Config.ARGB_8888, true);
    }

}
