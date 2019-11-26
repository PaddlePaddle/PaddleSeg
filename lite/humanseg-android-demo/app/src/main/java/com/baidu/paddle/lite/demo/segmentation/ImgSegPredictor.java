package com.baidu.paddle.lite.demo.segmentation;

import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.baidu.paddle.lite.Tensor;
import com.baidu.paddle.lite.demo.Predictor;
import com.baidu.paddle.lite.demo.segmentation.config.Config;
import com.baidu.paddle.lite.demo.segmentation.preprocess.Preprocess;
import com.baidu.paddle.lite.demo.segmentation.visual.Visualize;

import java.io.InputStream;
import java.util.Date;
import java.util.Vector;

import static android.graphics.Color.blue;
import static android.graphics.Color.green;
import static android.graphics.Color.red;

public class ImgSegPredictor extends Predictor {
    private static final String TAG = ImgSegPredictor.class.getSimpleName();
    protected Vector<String> wordLabels = new Vector<String>();

    Config config;

    protected Bitmap inputImage = null;
    protected Bitmap scaledImage = null;
    protected Bitmap outputImage = null;
    protected String outputResult = "";
    protected float preprocessTime = 0;
    protected float postprocessTime = 0;

    public ImgSegPredictor() {
        super();
    }

    public boolean init(Context appCtx, Config config) {

        if (config.inputShape.length != 4) {
            Log.i(TAG, "size of input shape should be: 4");
            return false;
        }
        if (config.inputShape[0] != 1) {
            Log.i(TAG, "only one batch is supported in the image classification demo, you can use any batch size in " +
                    "your Apps!");
            return false;
        }
        if (config.inputShape[1] != 1 && config.inputShape[1] != 3) {
            Log.i(TAG, "only one/three channels are supported in the image classification demo, you can use any " +
                    "channel size in your Apps!");
            return false;
        }
        if (!config.inputColorFormat.equalsIgnoreCase("RGB") && !config.inputColorFormat.equalsIgnoreCase("BGR")) {
            Log.i(TAG, "only RGB and BGR color format is supported.");
            return false;
        }
        super.init(appCtx, config.modelPath, config.cpuThreadNum, config.cpuPowerMode);
        if (!super.isLoaded()) {
            return false;
        }
        this.config = config;

        return isLoaded;
    }

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
        return super.getInput(idx);
    }

    public Tensor getOutput(int idx) {
        return super.getOutput(idx);
    }

    public boolean runModel(Bitmap image) {
        setInputImage(image);
        return runModel();
    }

    public boolean runModel(Preprocess preprocess, Visualize visualize) {
        if (inputImage == null) {
            return false;
        }

        // set input shape
        Tensor inputTensor = getInput(0);

        inputTensor.resize(config.inputShape);

        // pre-process image
        Date start = new Date();

        preprocess.init(config);

        preprocess.to_array(scaledImage);

        // feed input tensor with pre-processed data

        inputTensor.setData(preprocess.inputData);

        Date end = new Date();
        preprocessTime = (float) (end.getTime() - start.getTime());

        // inference
        super.runModel();

        Tensor outputTensor = getOutput(0);

        // post-process
        this.outputImage = visualize.draw(inputImage,outputTensor);

        postprocessTime = (float) (end.getTime() - start.getTime());

        start = new Date();
        outputResult = new String();
        end = new Date();

        return true;
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

    public void setInputImage(Bitmap image) {
        if (image == null) {
            return;
        }
        // scale image to the size of input tensor
        Bitmap rgbaImage = image.copy(Bitmap.Config.ARGB_8888, true);

        Bitmap scaleImage = Bitmap.createScaledBitmap(rgbaImage, (int) this.config.inputShape[3], (int) this.config.inputShape[2], true);
        this.inputImage = rgbaImage;
        this.scaledImage = scaleImage;
    }

}
