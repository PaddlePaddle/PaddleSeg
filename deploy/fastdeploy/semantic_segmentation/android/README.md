English | [简体中文](README_CN.md)
# PaddleSeg Android Demo for Image Segmentation

For real-time portrait segmentation on Android, this demo has good ease of use and openness. You can run your own training model in the demo.

## Environment Preparations

1. Install the Android Studio tool locally, for details see [Android Stuido official website](https://developer.android.com/studio).
2. Get an Android phone and turn on USB debugging mode. How to turn on: ` Phone Settings -> Find Developer Options -> Turn on Developer Options and USB Debug Mode`.

## Deployment Steps

1. Image Segmentation PaddleSeg Demo is located in `fastdeploy/examples/vision/segmentation/paddleseg/android` directory.
2. Please use Android Studio to open paddleseg/android project.
3. Connect your phone to your computer, turn on USB debugging and file transfer mode, and connect your own mobile device on Android Studio (your phone needs to be enabled to allow software installation from USB).

<p align="center">
<img width="1440" alt="image" src="https://user-images.githubusercontent.com/31974251/203257262-71b908ab-bb2b-47d3-9efb-67631687b774.png">
</p>

> **Notes:**
>> If you encounter an NDK configuration error during importing, compiling or running the program, please open ` File > Project Structure > SDK Location` and change `Andriod SDK location` to your locally configured SDK path.

4. Click the Run button to automatically compile the APP and install it to your phone. (The process will automatically download the pre-compiled FastDeploy Android library and model files, internet connection required.)
The success interface is as follows. Figure 1: Install APP on phone; Figure 2: The opening interface, it will automatically recognize the person in the picture and draw the mask; Figure 3: APP setting options, click setting in the upper right corner, and you can set different options.

| APP icon | APP effect | APP setting options
  | ---     | --- | --- |
  | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/203268599-c94018d8-3683-490a-a5c7-a8136a4fa284.jpg">  | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/203267867-7c51b695-65e6-402e-9826-5d6d5864da87.gif"> | <img width="300" height="500" alt="image" src="https://user-images.githubusercontent.com/31974251/197332983-afbfa6d5-4a3b-4c54-a528-4a3e58441be1.jpg"> |  


## PaddleSegModel Java API Introduction  
- Model initialization API: Model initialization API contains two methods, you can initialize directly through the constructor, or call init function at the appropriate program node. PaddleSegModel initialization parameters are described as follows:
  - modelFile: String, path to the model file in paddle format, e.g. model.pdmodel.
  - paramFile: String, path to the parameter file in paddle format, e.g. model.pdiparams.
  - configFile: String, preprocessing configuration file of model inference, e.g. deploy.yml.
  - option: RuntimeOption, optional, model initialization option. If this parameter is not passed, the default runtime option will be used.

```java
// Constructor w/o label file
public PaddleSegModel(); // An empty constructor, which can be initialised by calling init function later.
public PaddleSegModel(String modelFile, String paramsFile, String configFile);
public PaddleSegModel(String modelFile, String paramsFile, String configFile, RuntimeOption option);
// Call init manually w/o label file
public boolean init(String modelFile, String paramsFile, String configFile, RuntimeOption option);
```  
- Model prediction API: Model prediction API includes direct prediction API and API with visualization function. Direct prediction means that no image is saved and no result is rendered to Bitmap, but only the inference result is predicted. Prediction and visualization means to predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap (currently supports Bitmap in format ARGB8888), which can be displayed in camera later.
```java
// Directly predict: do not save images or render result to Bitmap.
public SegmentationResult predict(Bitmap ARGB8888Bitmap)；
// Predict and visualize: predict the result and visualize it, and save the visualized image to the specified path, and render the result to Bitmap.
public SegmentationResult predict(Bitmap ARGB8888Bitmap, String savedImagePath, float weight);
public SegmentationResult predict(Bitmap ARGB8888Bitmap, boolean rendering, float weight); // Only rendering images without saving.
// Modify result, but not return it. Concerning performance, you can use the following interface with CxxBuffer in SegmentationResult.
public boolean predict(Bitmap ARGB8888Bitmap, SegmentationResult result)；
public boolean predict(Bitmap ARGB8888Bitmap, SegmentationResult result, String savedImagePath, float weight);
public boolean predict(Bitmap ARGB8888Bitmap, SegmentationResult result, boolean rendering, float weight);
```
- Set vertical or horizontal mode: For PP-HumanSeg series model, you should call this method to set the vertical mode to true.
```java  
public void setVerticalScreenFlag(boolean flag);
```
- Model resource release API: Calling function release() API can release model resources, and true means successful release, false means failure. Calling function initialized() can determine whether the model is initialized successfully, and true means successful initialization, false means failure.
```java
public boolean release(); // Release native resources.
public boolean initialized(); // Check if initialization is successful.
```

- Runtime Option Setting
```java  
public void enableLiteFp16(); // Enable fp16 precision inference
public void disableLiteFP16(); // Disable fp16 precision inference
public void setCpuThreadNum(int threadNum); // Set number of threads.
public void setLitePowerMode(LitePowerMode mode);  // Set power mode.
public void setLitePowerMode(String modeStr);  // Set power mode by string.
```

- Segmentation Result
```java
public class SegmentationResult {
  public int[] mLabelMap;  //  The predicted label map, each pixel position corresponds to a label HxW.
  public float[] mScoreMap; // The predicted score map, each pixel position corresponds to a score HxW.
  public long[] mShape; // The real shape(H,W) of label map.
  public boolean mContainScoreMap = false; // Whether score map is included.
  // You can choose to use CxxBuffer directly instead of copying it to JAVA layer through JNI.
  // This method can improve performance to some extent.
  public void setCxxBufferFlag(boolean flag); // Set whether the mode is CxxBuffer.
  public boolean releaseCxxBuffer(); // Release CxxBuffer manually!!!
  public boolean initialized(); // Check if the result is valid.
}  
```  
Other reference: C++/Python corresponding SegmentationResult description: [api/vision_results/segmentation_result.md](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/api/vision_results/segmentation_result.md).


- Model calling example 1: Using constructor and the default RuntimeOption:
```java  
import java.nio.ByteBuffer;
import android.graphics.Bitmap;
import android.opengl.GLES20;

import com.baidu.paddle.fastdeploy.vision.SegmentationResult;
import com.baidu.paddle.fastdeploy.vision.segmentation.PaddleSegModel;

// Initialise model.
PaddleSegModel model = new PaddleSegModel(
  "portrait_pp_humansegv2_lite_256x144_inference_model/model.pdmodel",
  "portrait_pp_humansegv2_lite_256x144_inference_model/model.pdiparams",
  "portrait_pp_humansegv2_lite_256x144_inference_model/deploy.yml");

// If the camera is in portrait mode, the PP-HumanSeg series needs to change the mark.
model.setVerticalScreenFlag(true);

// Read Bitmaps: The following is the pseudo code of reading the Bitmap.
ByteBuffer pixelBuffer = ByteBuffer.allocate(width * height * 4);
GLES20.glReadPixels(0, 0, width, height, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixelBuffer);
Bitmap ARGB8888ImageBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
ARGB8888ImageBitmap.copyPixelsFromBuffer(pixelBuffer);

// Model inference.
SegmentationResult result = new SegmentationResult();
result.setCxxBufferFlag(true);

model.predict(ARGB8888ImageBitmap, result);  

// Release CxxBuffer.
result.releaseCxxBuffer();

// Or return SegmentationResult directly.
SegmentationResult result = model.predict(ARGB8888ImageBitmap);

// Release model resources.  
model.release();
```  

- Model calling example 2: Call init function manually at the appropriate program node and customize RuntimeOption.
```java  
// import id.
import com.baidu.paddle.fastdeploy.RuntimeOption;
import com.baidu.paddle.fastdeploy.LitePowerMode;
import com.baidu.paddle.fastdeploy.vision.SegmentationResult;
import com.baidu.paddle.fastdeploy.vision.segmentation.PaddleSegModel;
// Create empty model.
PaddleSegModel model = new PaddleSegModel();  
// Model path.
String modelFile = "portrait_pp_humansegv2_lite_256x144_inference_model/model.pdmodel";
String paramFile = "portrait_pp_humansegv2_lite_256x144_inference_model/model.pdiparams";
String configFile = "portrait_pp_humansegv2_lite_256x144_inference_model/deploy.yml";
// Specify RuntimeOption.
RuntimeOption option = new RuntimeOption();
option.setCpuThreadNum(2);
option.setLitePowerMode(LitePowerMode.LITE_POWER_HIGH);
option.enableLiteFp16();  
// If the camera is in portrait mode, the PP-HumanSeg series needs to change the mark.
model.setVerticalScreenFlag(true);
// Initialise with the init function.
model.init(modelFile, paramFile, configFile, option);
// Read Bitmap, predict model, release resources, id.
```
For details, please refer to [SegmentationMainActivity](./app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/segmentation/SegmentationMainActivity.java).

##  Replace FastDeploy SDK and model  
 Steps to replace the FastDeploy prediction libraries and model are very simple. The location of the prediction library is `app/libs/fastdeploy-android-sdk-xxx.aar`, where `xxx` indicates the version of the prediction library you are currently using. The location of the model is, `app/src/main/assets/models/portrait_pp_humansegv2_lite_256x144_inference_model`.
- Replace FastDeploy Android SDK: Download or compile the latest FastDeploy Android SDK, unzip it and put it in the `app/libs` directory. For details please refer to:
     - [Use FastDeploy Java SDK on Android](https://github.com/PaddlePaddle/FastDeploy/tree/develop/java/android)

- Steps for replacing the PaddleSeg model.  
  - Put your PaddleSeg model in `app/src/main/assets/models`;  
  - Modify the model path in `app/src/main/res/values/strings.xml`, such as:
```xml
<!-- Modify this path for your model, e.g. models/human_pp_humansegv1_lite_192x192_inference_model -->
<string name="SEGMENTATION_MODEL_DIR_DEFAULT">models/human_pp_humansegv1_lite_192x192_inference_model</string>  
```  

## Other Documenets
If you are interested in more FastDeploy Java API documents and how to access the FastDeploy C++ API via JNI, you can refer to the following:
- [Use FastDeploy Java SDK on Android](https://github.com/PaddlePaddle/FastDeploy/tree/develop/java/android)
- [Use FastDeploy C++ SDK on Android](https://github.com/PaddlePaddle/FastDeploy/blob/develop/docs/cn/faq/use_cpp_sdk_on_android.md)  
