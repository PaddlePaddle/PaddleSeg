package com.paddle.demo.matting.config;

import android.graphics.Bitmap;

public class Config {

    public String modelPath = "";
    public String labelPath = "";
    public String imagePath = "";
    public String bgPath = "";
    public int cpuThreadNum = 1;
    public String cpuPowerMode = "";
    public String inputColorFormat = "";
    public long[] inputShape = new long[]{};


    public void init(String modelPath, String labelPath, String imagePath,String bgPath, int cpuThreadNum,
                     String cpuPowerMode, String inputColorFormat,long[] inputShape){

        this.modelPath = modelPath;
        this.labelPath = labelPath;
        this.imagePath = imagePath;
        this.bgPath = bgPath;
        this.cpuThreadNum = cpuThreadNum;
        this.cpuPowerMode = cpuPowerMode;
        this.inputColorFormat = inputColorFormat;
        this.inputShape = inputShape;
    }

    public void setInputShape(Bitmap inputImage){
        this.inputShape[0] = 1;
        this.inputShape[1] = 3;
        this.inputShape[2] = inputImage.getHeight();
        this.inputShape[3] = inputImage.getWidth();

    }

}
