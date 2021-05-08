package com.baidu.paddle.lite.demo.segmentation.visual;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.util.Log;

import com.baidu.paddle.lite.Tensor;

public class Visualize {
    private static final String TAG = Visualize.class.getSimpleName();

    public Bitmap draw(Bitmap inputImage, Tensor outputTensor){

        final int[] colors_map = {0xFF000000, 0xFFFFFF00};
//        final int[] colors_map = {0xFF000000, 0xFFFFFF00, 0xFFFFFF01, 0xFFFFFF02, 0xFFFFFF03, 0xFFFFFF04, 0xFFFFFF05, 0xFFFFFF06, 0xFFFFFF07,
//                0xFFFFFF08, 0xFFFFFF09, 0xFFFFFF10, 0xFFFFFF11, 0xFFFFFF12, 0xFFFFFF13, 0xFFFFFF14, 0xFFFFFF15, 0xFFFFFF16, 0xFFFFFF17}; // 19 class

        long[] output = outputTensor.getLongData();
        long outputShape[] = outputTensor.shape();
        long outputSize = 1;

        for (long s : outputShape) {
            outputSize *= s;
        }

        int[] objectColor = new int[(int)outputSize];

        for(int i=0;i<output.length;i++){
            objectColor[i] = colors_map[(int)output[i]];
        }

        Bitmap.Config config = inputImage.getConfig();
        Bitmap outputImage = null;
        if(outputShape.length==3){
            outputImage = Bitmap.createBitmap(objectColor, (int)outputShape[2], (int)outputShape[1], config);
            outputImage = Bitmap.createScaledBitmap(outputImage, inputImage.getWidth(), inputImage.getHeight(),true);
        }

        else if (outputShape.length==4){
            outputImage = Bitmap.createBitmap(objectColor, (int)outputShape[3], (int)outputShape[2], config);
        }
        Bitmap bmOverlay = Bitmap.createBitmap(inputImage.getWidth(), inputImage.getHeight() , inputImage.getConfig());
        Canvas canvas = new Canvas(bmOverlay);
        canvas.drawBitmap(inputImage, new Matrix(), null);

        Paint paint = new Paint();
        paint.setAlpha(0x80);
        canvas.drawBitmap(outputImage, 0, 0, paint);

        return bmOverlay;

    }
}
