package com.baidu.paddle.lite.demo.segmentation;

import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Menu;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.baidu.paddle.lite.demo.CommonActivity;
import com.baidu.paddle.lite.demo.R;
import com.baidu.paddle.lite.demo.Utils;
import com.baidu.paddle.lite.demo.segmentation.config.Config;
import com.baidu.paddle.lite.demo.segmentation.preprocess.Preprocess;
import com.baidu.paddle.lite.demo.segmentation.visual.Visualize;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;

public class ImgSegActivity extends CommonActivity {
    private static final String TAG = ImgSegActivity.class.getSimpleName();

    protected TextView tvInputSetting;
    protected ImageView ivInputImage;
    protected TextView tvOutputResult;
    protected TextView tvInferenceTime;

    // model config
    Config config = new Config();

    protected ImgSegPredictor predictor = new ImgSegPredictor();

    Preprocess preprocess = new Preprocess();

    Visualize visualize = new Visualize();

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_img_seg);
        tvInputSetting = findViewById(R.id.tv_input_setting);
        ivInputImage = findViewById(R.id.iv_input_image);
        tvInferenceTime = findViewById(R.id.tv_inference_time);
        tvOutputResult = findViewById(R.id.tv_output_result);
        tvInputSetting.setMovementMethod(ScrollingMovementMethod.getInstance());
        tvOutputResult.setMovementMethod(ScrollingMovementMethod.getInstance());
    }

    @Override
    public boolean onLoadModel() {
        return super.onLoadModel() && predictor.init(ImgSegActivity.this, config);
    }

    @Override
    public boolean onRunModel() {
        return super.onRunModel() && predictor.isLoaded() && predictor.runModel(preprocess,visualize);
    }

    @Override
    public void onLoadModelSuccessed() {
        super.onLoadModelSuccessed();
        // load test image from file_paths and run model
        try {
            if (config.imagePath.isEmpty()) {
                return;
            }
            Bitmap image = null;
            // read test image file from custom file_paths if the first character of mode file_paths is '/', otherwise read test
            // image file from assets
            if (!config.imagePath.substring(0, 1).equals("/")) {
                InputStream imageStream = getAssets().open(config.imagePath);
                image = BitmapFactory.decodeStream(imageStream);
            } else {
                if (!new File(config.imagePath).exists()) {
                    return;
                }
                image = BitmapFactory.decodeFile(config.imagePath);
            }
            if (image != null && predictor.isLoaded()) {
                predictor.setInputImage(image);
                runModel();
            }
        } catch (IOException e) {
            Toast.makeText(ImgSegActivity.this, "Load image failed!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }

    @Override
    public void onLoadModelFailed() {
        super.onLoadModelFailed();
    }

    @Override
    public void onRunModelSuccessed() {
        super.onRunModelSuccessed();
        // obtain results and update UI
        tvInferenceTime.setText("Inference time: " + predictor.inferenceTime() + " ms");
        Bitmap outputImage = predictor.outputImage();
        if (outputImage != null) {
            ivInputImage.setImageBitmap(outputImage);
        }
        tvOutputResult.setText(predictor.outputResult());
        tvOutputResult.scrollTo(0, 0);
    }

    @Override
    public void onRunModelFailed() {
        super.onRunModelFailed();
    }

    @Override
    public void onImageChanged(Bitmap image) {
        super.onImageChanged(image);
        // rerun model if users pick test image from gallery or camera
        if (image != null && predictor.isLoaded()) {
//            predictor.setConfig(config);
            predictor.setInputImage(image);
            runModel();
        }
    }

    @Override
    public void onImageChanged(String path) {
        super.onImageChanged(path);
        Bitmap image = BitmapFactory.decodeFile(path);
        predictor.setInputImage(image);
            runModel();
    }
    public void onSettingsClicked() {
        super.onSettingsClicked();
        startActivity(new Intent(ImgSegActivity.this, ImgSegSettingsActivity.class));
    }

    @Override
    public boolean onPrepareOptionsMenu(Menu menu) {
        boolean isLoaded = predictor.isLoaded();
        menu.findItem(R.id.open_gallery).setEnabled(isLoaded);
        menu.findItem(R.id.take_photo).setEnabled(isLoaded);
        return super.onPrepareOptionsMenu(menu);
    }

    @Override
    protected void onResume() {
        Log.i(TAG,"begin onResume");
        super.onResume();

        SharedPreferences sharedPreferences = PreferenceManager.getDefaultSharedPreferences(this);
        boolean settingsChanged = false;
        String model_path = sharedPreferences.getString(getString(R.string.ISG_MODEL_PATH_KEY),
                getString(R.string.ISG_MODEL_PATH_DEFAULT));
        String label_path = sharedPreferences.getString(getString(R.string.ISG_LABEL_PATH_KEY),
                getString(R.string.ISG_LABEL_PATH_DEFAULT));
        String image_path = sharedPreferences.getString(getString(R.string.ISG_IMAGE_PATH_KEY),
                getString(R.string.ISG_IMAGE_PATH_DEFAULT));
        settingsChanged |= !model_path.equalsIgnoreCase(config.modelPath);
        settingsChanged |= !label_path.equalsIgnoreCase(config.labelPath);
        settingsChanged |= !image_path.equalsIgnoreCase(config.imagePath);
        int cpu_thread_num = Integer.parseInt(sharedPreferences.getString(getString(R.string.ISG_CPU_THREAD_NUM_KEY),
                getString(R.string.ISG_CPU_THREAD_NUM_DEFAULT)));
        settingsChanged |= cpu_thread_num != config.cpuThreadNum;
        String cpu_power_mode =
                sharedPreferences.getString(getString(R.string.ISG_CPU_POWER_MODE_KEY),
                        getString(R.string.ISG_CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !cpu_power_mode.equalsIgnoreCase(config.cpuPowerMode);
        String input_color_format =
                sharedPreferences.getString(getString(R.string.ISG_INPUT_COLOR_FORMAT_KEY),
                        getString(R.string.ISG_INPUT_COLOR_FORMAT_DEFAULT));
        settingsChanged |= !input_color_format.equalsIgnoreCase(config.inputColorFormat);
        long[] input_shape =
                Utils.parseLongsFromString(sharedPreferences.getString(getString(R.string.ISG_INPUT_SHAPE_KEY),
                        getString(R.string.ISG_INPUT_SHAPE_DEFAULT)), ",");

        settingsChanged |= input_shape.length != config.inputShape.length;

        if (!settingsChanged) {
            for (int i = 0; i < input_shape.length; i++) {
                settingsChanged |= input_shape[i] != config.inputShape[i];
            }
        }

        if (settingsChanged) {
            config.init(model_path,label_path,image_path,cpu_thread_num,cpu_power_mode,
                    input_color_format,input_shape);
            preprocess.init(config);
            // update UI
            tvInputSetting.setText("Model: " + config.modelPath.substring(config.modelPath.lastIndexOf("/") + 1) + "\n" + "CPU" +
                    " Thread Num: " + Integer.toString(config.cpuThreadNum) + "\n" + "CPU Power Mode: " + config.cpuPowerMode);
            tvInputSetting.scrollTo(0, 0);
            // reload model if configure has been changed
            loadModel();
        }
    }

    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.releaseModel();
        }
        super.onDestroy();
    }
}
