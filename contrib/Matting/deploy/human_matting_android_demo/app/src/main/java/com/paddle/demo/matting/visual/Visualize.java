package com.paddle.demo.matting.visual;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.util.Log;

import com.baidu.paddle.lite.Tensor;

import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

public class Visualize {
    private static final String TAG = Visualize.class.getSimpleName();

    public Bitmap segdraw(Bitmap inputImage, Tensor outputTensor,Bitmap bg){
        //设置叠加颜色（ARGB格式）  背景采用黑色，前景采用黄色（黄色由红绿两种颜色叠加）
        final int[] colors_map = {0xFF000000, 0xFFFFFF00};
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
        paint.setAlpha(0x80);//采用一半对应的透明度进行叠加
        canvas.drawBitmap(outputImage, 0, 0, paint);

        return bmOverlay;
    }

    public Bitmap draw(Bitmap inputImage, Tensor outputTensor,Bitmap bg){
        float[] output = outputTensor.getFloatData();
        long outputShape[] = outputTensor.shape();
        int outputSize = 1;

        for (long s : outputShape) {
            outputSize *= s;
        }
        List<Float> arralist = new LinkedList<>();
        for (int i=0; i<outputSize;i++){
            arralist.add((float)output[i]);
        }

        Bitmap mALPHA_IMAGE = floatArrayToBitmap(arralist,(int)outputShape[3],(int)outputShape[2]);

        //调整尺寸
        Bitmap alpha = Bitmap.createScaledBitmap(mALPHA_IMAGE,inputImage.getWidth(),inputImage.getHeight(),true);
        Bitmap bgImg = Bitmap.createScaledBitmap(bg,inputImage.getWidth(),inputImage.getHeight(),true);

        //重新合成图像
        Bitmap result = synthetizeBitmap(inputImage,bgImg, alpha);
        return result;
    }

    //将float数组转成bitmap格式的图片
    private Bitmap floatArrayToBitmap(List<Float>  floatArray,int width,int height){
        byte alpha = (byte) 255 ;
        Bitmap bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888) ;
        ByteBuffer byteBuffer = ByteBuffer.allocate(width*height*4*3) ;
        float Maximum = Collections.max(floatArray);
        float minmum = Collections.min(floatArray);
        float delta = Maximum - minmum + 0.00000000001f ;

        int i = 0 ;
        for (float value : floatArray){
            byte temValue = (byte) ((((value-minmum)/delta)*255));
            byteBuffer.put(4*i, temValue) ;
            byteBuffer.put(4*i+1, temValue) ;
            byteBuffer.put(4*i+2, temValue) ;
            byteBuffer.put(4*i+3, alpha) ;
            i++;
        }
        bmp.copyPixelsFromBuffer(byteBuffer) ;
        return bmp ;
    }

    //将原图与背景按照推理得到的alpha图进行合成
    private Bitmap synthetizeBitmap(Bitmap front,Bitmap background, Bitmap alpha){
        int width = front.getWidth();
        int height = front.getHeight();
        Bitmap result=Bitmap.createBitmap(width,height, Bitmap.Config.ARGB_8888);
        int[] frontPixels = new int[width * height];
        int[] backgroundPixels = new int[width * height];
        int[] alphaPixels = new int[width * height];
        front.getPixels(frontPixels,0,width,0,0,width,height);
        background.getPixels(backgroundPixels,0,width,0,0,width,height);
        alpha.getPixels(alphaPixels,0,width,0,0,width,height);
        float frontA = 0,frontR = 0,frontG = 0,frontB = 0;
        float backgroundR = 0,backgroundG = 0,backgroundB = 0;
        float alphaR = 0,alphaG = 0,alphaB = 0;
        int index=0;

        //逐个像素赋值（这种写法比较耗时，后续可以优化）
        for (int row=0; row < height; row++){
            for (int col=0; col < width; col++){
                index = width*row +col;

                //取出前景图像像素值
                frontA=(frontPixels[index]>>24)&0xff;
                frontR=(frontPixels[index]>>16)&0xff;
                frontG=(frontPixels[index]>>8)&0xff;
                frontB=frontPixels[index]&0xff;

                //取出alpha像素值
                alphaR=(alphaPixels[index]>>16)&0xff;
                alphaG=(alphaPixels[index]>>8)&0xff;
                alphaB=alphaPixels[index]&0xff;

                //取出背景图像像素值
                backgroundR=(backgroundPixels[index]>>16)&0xff;
                backgroundG=(backgroundPixels[index]>>8)&0xff;
                backgroundB=backgroundPixels[index]&0xff;

                //重新合成  ImgOut = F * alpha/255 + BG * ( 1 - alpha/255 )
                frontR= frontR*alphaR/255 + backgroundR*(1-alphaR/255);
                frontG=frontG*alphaG/255 + backgroundG*(1-alphaG/255);
                frontB=frontB*alphaB/255 + backgroundB*(1-alphaB/255);
                frontPixels[index]=(int)frontA<<24|((int)frontR<<16)|((int)frontG<<8)|(int)frontB;
            }
        }
        result.setPixels(frontPixels,0,width,0,0,width,height);;
        return result;
    }
}
