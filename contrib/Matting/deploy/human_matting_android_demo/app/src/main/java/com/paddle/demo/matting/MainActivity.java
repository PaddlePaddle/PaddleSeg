package com.paddle.demo.matting;

import android.Manifest;
import android.app.ProgressDialog;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.Message;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.util.Log;
import android.view.Display;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.paddle.demo.matting.config.Config;
import com.paddle.demo.matting.preprocess.Preprocess;
import com.paddle.demo.matting.visual.Visualize;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = MainActivity.class.getSimpleName();

    //定义图像来源
    public static final int OPEN_GALLERY_REQUEST_CODE = 0;//本地相册
    public static final int TAKE_PHOTO_REQUEST_CODE = 1;//摄像头拍摄

    //定义模型推理相关变量
    public static final int REQUEST_LOAD_MODEL = 0;
    public static final int REQUEST_RUN_MODEL = 1;
    public static final int RESPONSE_LOAD_MODEL_SUCCESSED = 0;
    public static final int RESPONSE_LOAD_MODEL_FAILED = 1;
    public static final int RESPONSE_RUN_MODEL_SUCCESSED = 2;
    public static final int RESPONSE_RUN_MODEL_FAILED = 3;

    protected ProgressDialog pbLoadModel = null;
    protected ProgressDialog pbRunModel = null;

    //定义操作流程线程句柄
    protected HandlerThread worker = null; // 工作线程（加载和运行模型）
    protected Handler receiver = null; // 接收来自工作线程的消息
    protected Handler sender = null; // 发送消息给工作线程

    protected TextView tvInputSetting;//输入信息面板
    protected ImageView ivInputImage;//输入图像面板
    protected TextView tvOutputResult;//输出结果面板
    protected TextView tvInferenceTime;//推理时间面板

    // 模型配置
    Config config = new Config();

    protected Predictor predictor = new Predictor();

    Preprocess preprocess = new Preprocess();

    Visualize visualize = new Visualize();

    //定义图像存储路径
    Uri photoUri;

    private Button btnSaveImg;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        //绑定保存图像按钮
        btnSaveImg = (Button) findViewById(R.id.save_img);

        //定义消息接收线程
        receiver = new Handler() {
            @Override
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case RESPONSE_LOAD_MODEL_SUCCESSED:
                        pbLoadModel.dismiss();
                        onLoadModelSuccessed();
                        break;
                    case RESPONSE_LOAD_MODEL_FAILED:
                        pbLoadModel.dismiss();
                        Toast.makeText(MainActivity.this, "Load model failed!", Toast.LENGTH_SHORT).show();
                        onLoadModelFailed();
                        break;
                    case RESPONSE_RUN_MODEL_SUCCESSED:
                        pbRunModel.dismiss();
                        onRunModelSuccessed();
                        break;
                    case RESPONSE_RUN_MODEL_FAILED:
                        pbRunModel.dismiss();
                        Toast.makeText(MainActivity.this, "Run model failed!", Toast.LENGTH_SHORT).show();
                        onRunModelFailed();
                        break;
                    default:
                        break;
                }
            }
        };

        //定义工作线程
        worker = new HandlerThread("Predictor Worker");
        worker.start();

        //定义发送消息线程
        sender = new Handler(worker.getLooper()) {
            public void handleMessage(Message msg) {
                switch (msg.what) {
                    case REQUEST_LOAD_MODEL:
                        // load model and reload test image
                        if (onLoadModel()) {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_LOAD_MODEL_FAILED);
                        }
                        break;
                    case REQUEST_RUN_MODEL:
                        // run model if model is loaded
                        if (onRunModel()) {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_SUCCESSED);
                        } else {
                            receiver.sendEmptyMessage(RESPONSE_RUN_MODEL_FAILED);
                        }
                        break;
                    default:
                        break;
                }
            }
        };

        tvInputSetting = findViewById(R.id.tv_input_setting);
        ivInputImage = findViewById(R.id.iv_input_image);
        tvInferenceTime = findViewById(R.id.tv_inference_time);
        tvOutputResult = findViewById(R.id.tv_output_result);
        tvInputSetting.setMovementMethod(ScrollingMovementMethod.getInstance());
        tvOutputResult.setMovementMethod(ScrollingMovementMethod.getInstance());
    }

    public boolean onLoadModel() {
        return predictor.init(MainActivity.this, config);
    }

    public boolean onRunModel() {
        return predictor.isLoaded() && predictor.runModel(preprocess,visualize);
    }

    public void onLoadModelFailed() {

    }
    public void onRunModelFailed() {
    }

    public void loadModel() {
        pbLoadModel = ProgressDialog.show(this, "", "加载模型中...", false, false);
        sender.sendEmptyMessage(REQUEST_LOAD_MODEL);
    }

    public void runModel() {
        pbRunModel = ProgressDialog.show(this, "", "推理中...", false, false);
        sender.sendEmptyMessage(REQUEST_RUN_MODEL);
    }

    public void onLoadModelSuccessed() {
        // load test image from file_paths and run model
        try {
            if (config.imagePath.isEmpty()||config.bgPath.isEmpty()) {
                return;
            }
            Bitmap image = null;
            Bitmap bg = null;

            //加载待抠图像（如果是拍照或者本地相册读取，则第一个字符为“/”。否则就是从默认路径下读取图片）
            if (!config.imagePath.substring(0, 1).equals("/")) {
                InputStream imageStream = getAssets().open(config.imagePath);
                image = BitmapFactory.decodeStream(imageStream);
            } else {
                if (!new File(config.imagePath).exists()) {
                    return;
                }
                image = BitmapFactory.decodeFile(config.imagePath);
            }

            //加载背景图像
            if (!config.bgPath.substring(0, 1).equals("/")) {
                InputStream imageStream = getAssets().open(config.bgPath);
                bg = BitmapFactory.decodeStream(imageStream);
            } else {
                if (!new File(config.bgPath).exists()) {
                    return;
                }
                bg = BitmapFactory.decodeFile(config.bgPath);
            }

            if (image != null && bg != null && predictor.isLoaded()) {
                predictor.setInputImage(image,bg);
                runModel();
            }
        } catch (IOException e) {
            Toast.makeText(MainActivity.this, "Load image failed!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }
    }

    public void onRunModelSuccessed() {
        // 获取抠图结果并更新UI
        tvInferenceTime.setText("推理耗时: " + predictor.inferenceTime() + " ms");
        Bitmap outputImage = predictor.outputImage();
        if (outputImage != null) {
            ivInputImage.setImageBitmap(outputImage);
        }
        tvOutputResult.setText(predictor.outputResult());
        tvOutputResult.scrollTo(0, 0);
    }


    public void onImageChanged(Bitmap image) {
        Bitmap bg = null;
        try {
            //加载背景图像
            if (!config.bgPath.substring(0, 1).equals("/")) {
                InputStream imageStream = getAssets().open(config.bgPath);
                bg = BitmapFactory.decodeStream(imageStream);
            } else {
                if (!new File(config.bgPath).exists()) {
                    return;
                }
                bg = BitmapFactory.decodeFile(config.bgPath);
            }
        } catch (IOException e) {
            Toast.makeText(MainActivity.this, "加载背景图失败!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }

        // rerun model if users pick test image from gallery or camera
        //设置预测器图像
        if (image != null && predictor.isLoaded()) {
            predictor.setInputImage(image,bg);
            runModel();
        }
    }

    public void onImageChanged(String path) {
        Bitmap bg = null;
        try {
            //加载背景图像
            if (!config.bgPath.substring(0, 1).equals("/")) {
                InputStream imageStream = getAssets().open(config.bgPath);
                bg = BitmapFactory.decodeStream(imageStream);
            } else {
                if (!new File(config.bgPath).exists()) {
                    return;
                }
                bg = BitmapFactory.decodeFile(config.bgPath);
            }
        } catch (IOException e) {
            Toast.makeText(MainActivity.this, "加载背景图失败!", Toast.LENGTH_SHORT).show();
            e.printStackTrace();
        }

        //设置预测器图像
        Bitmap image = BitmapFactory.decodeFile(path);
        predictor.setInputImage(image,bg);
            runModel();
    }

    //打开设置页面
    public void onSettingsClicked() {
        startActivity(new Intent(MainActivity.this, SettingsActivity.class));
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_action_options, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case android.R.id.home:
                finish();
                break;
            case R.id.open_gallery:
                if (requestAllPermissions()) {
                    openGallery();
                }
                break;
            case R.id.take_photo:
                if (requestAllPermissions()) {
                    takePhoto();
                }
                break;
            case R.id.settings:
                if (requestAllPermissions()) {
                    // make sure we have SDCard r&w permissions to load model from SDCard
                    onSettingsClicked();
                }
                break;
        }
        return super.onOptionsItemSelected(item);
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Permission Denied", Toast.LENGTH_SHORT).show();
        }
    }

    private Bitmap getBitMapFromPath(String imageFilePath) {
        Display currentDisplay = getWindowManager().getDefaultDisplay();
        int dw = currentDisplay.getWidth();
        int dh = currentDisplay.getHeight();
        // Load up the image's dimensions not the image itself
        BitmapFactory.Options bmpFactoryOptions = new BitmapFactory.Options();
        bmpFactoryOptions.inJustDecodeBounds = true;
        Bitmap bmp = BitmapFactory.decodeFile(imageFilePath, bmpFactoryOptions);
        int heightRatio = (int) Math.ceil(bmpFactoryOptions.outHeight
                / (float) dh);
        int widthRatio = (int) Math.ceil(bmpFactoryOptions.outWidth
                / (float) dw);

        // If both of the ratios are greater than 1,
        // one of the sides of the image is greater than the screen
        if (heightRatio > 1 && widthRatio > 1) {
            if (heightRatio > widthRatio) {
                // Height ratio is larger, scale according to it
                bmpFactoryOptions.inSampleSize = heightRatio;
            } else {
                // Width ratio is larger, scale according to it
                bmpFactoryOptions.inSampleSize = widthRatio;
            }
        }
        // Decode it for real
        bmpFactoryOptions.inJustDecodeBounds = false;
        bmp = BitmapFactory.decodeFile(imageFilePath, bmpFactoryOptions);
        return bmp;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        //if (resultCode == RESULT_OK && data != null) {
        if (resultCode == RESULT_OK) {
            switch (requestCode) {
                case OPEN_GALLERY_REQUEST_CODE:
                    try {
                        ContentResolver resolver = getContentResolver();
                        Uri uri = data.getData();
                        Bitmap image = MediaStore.Images.Media.getBitmap(resolver, uri);
                        //判断图像尺寸（如果图像过大推理会奔溃）
                        int width = image.getWidth();
                        int height = image.getHeight();
                        Bitmap scaleImage;
                        if(width > 800) {
                            int new_width = 800;
                            int new_height = (int)(height*1.0/width*new_width);
                            scaleImage = Bitmap.createScaledBitmap(image, new_width, new_height,true);
                        }
                        else{
                            scaleImage = image.copy(Bitmap.Config.ARGB_8888, true);
                        }
                        String[] proj = {MediaStore.Images.Media.DATA};
                        Cursor cursor = managedQuery(uri, proj, null, null, null);
                        cursor.moveToFirst();
                        onImageChanged(scaleImage);
                    } catch (IOException e) {
                        Log.e(TAG, e.toString());
                    }
                    break;

                case TAKE_PHOTO_REQUEST_CODE:
                    //Bitmap image = (Bitmap) data.getParcelableExtra("data");//获取缩略图
                    Bitmap image = null;
                    if (photoUri == null)
                        return;
                    //通过Uri和selection来获取真实的图片路径
                    Cursor cursor = getContentResolver().query(photoUri, null, null, null, null);
                    if (cursor != null) {
                        if (cursor.moveToFirst()){
                            String path = cursor.getString(cursor.getColumnIndex(MediaStore.Images.Media.DATA));
                            //获得图片
                            image = getBitMapFromPath(path);
                        }
                        cursor.close();
                    }
                    photoUri = null;
                    onImageChanged(image);
                    break;
                default:
                    break;
            }
        }
    }
    private boolean requestAllPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE)
                != PackageManager.PERMISSION_GRANTED || ContextCompat.checkSelfPermission(this,
                Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
                            Manifest.permission.CAMERA},
                    0);
            return false;
        }
        return true;
    }

    private void openGallery() {
        Intent intent = new Intent(Intent.ACTION_PICK, null);
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(intent, OPEN_GALLERY_REQUEST_CODE);
    }

    private void takePhoto() {
        Intent takePhotoIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        ContentValues values = new ContentValues();
        photoUri = getContentResolver().insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        takePhotoIntent.putExtra(android.provider.MediaStore.EXTRA_OUTPUT, photoUri);
        if (takePhotoIntent.resolveActivity(getPackageManager()) != null) {
            startActivityForResult(takePhotoIntent, TAKE_PHOTO_REQUEST_CODE);
        }
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
        String model_path = sharedPreferences.getString(getString(R.string.MODEL_PATH_KEY),
                getString(R.string.MODEL_PATH_DEFAULT));
        String label_path = sharedPreferences.getString(getString(R.string.LABEL_PATH_KEY),
                getString(R.string.LABEL_PATH_DEFAULT));
        String image_path = sharedPreferences.getString(getString(R.string.IMAGE_PATH_KEY),
                getString(R.string.IMAGE_PATH_DEFAULT));
        String bg_path = sharedPreferences.getString(getString(R.string.BG_PATH_KEY),
                getString(R.string.BG_PATH_DEFAULT));
        settingsChanged |= !model_path.equalsIgnoreCase(config.modelPath);
        settingsChanged |= !label_path.equalsIgnoreCase(config.labelPath);
        settingsChanged |= !image_path.equalsIgnoreCase(config.imagePath);
        settingsChanged |= !bg_path.equalsIgnoreCase(config.bgPath);
        int cpu_thread_num = Integer.parseInt(sharedPreferences.getString(getString(R.string.CPU_THREAD_NUM_KEY),
                getString(R.string.CPU_THREAD_NUM_DEFAULT)));
        settingsChanged |= cpu_thread_num != config.cpuThreadNum;
        String cpu_power_mode =
                sharedPreferences.getString(getString(R.string.CPU_POWER_MODE_KEY),
                        getString(R.string.CPU_POWER_MODE_DEFAULT));
        settingsChanged |= !cpu_power_mode.equalsIgnoreCase(config.cpuPowerMode);
        String input_color_format =
                sharedPreferences.getString(getString(R.string.INPUT_COLOR_FORMAT_KEY),
                        getString(R.string.INPUT_COLOR_FORMAT_DEFAULT));
        settingsChanged |= !input_color_format.equalsIgnoreCase(config.inputColorFormat);
        long[] input_shape =
                Utils.parseLongsFromString(sharedPreferences.getString(getString(R.string.INPUT_SHAPE_KEY),
                        getString(R.string.INPUT_SHAPE_DEFAULT)), ",");

        settingsChanged |= input_shape.length != config.inputShape.length;

        if (!settingsChanged) {
            for (int i = 0; i < input_shape.length; i++) {
                settingsChanged |= input_shape[i] != config.inputShape[i];
            }
        }

        if (settingsChanged) {
            config.init(model_path,label_path,image_path,bg_path,cpu_thread_num,cpu_power_mode,
                    input_color_format,input_shape);
            preprocess.init(config);
            // 更新UI
            tvInputSetting.setText("算法模型: " + config.modelPath.substring(config.modelPath.lastIndexOf("/") + 1));
            tvInputSetting.scrollTo(0, 0);
            // 如果配置发生改变则重新加载模型并预测
            loadModel();
        }
    }

    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.releaseModel();
        }
        worker.quit();
        super.onDestroy();
    }

    //保存图像
    public void clickSaveImg(View view){
        //取出图像
        Bitmap outputImage = predictor.outputImage();
        if(outputImage==null) {
            Toast.makeText(getApplicationContext(),"当前没有图像",Toast.LENGTH_SHORT).show();
            return;
        }

        //保存图像
        File f = new File(Environment.getExternalStorageDirectory() + "/newphoto.jpg");
        if (f.exists()) {
            f.delete();
        }
        try {
            FileOutputStream out = new FileOutputStream(f);
            outputImage.compress(Bitmap.CompressFormat.JPEG, 80, out);
            out.flush();
            out.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // 把文件插入到系统图库
        try {
            MediaStore.Images.Media.insertImage(this.getContentResolver(),
                    f.getAbsolutePath(), Environment.getExternalStorageDirectory() + "/newphoto.jpg", null);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        // 通知图库更新
        sendBroadcast(new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE,
                Uri.parse("file://" + "/sdcard/matting/")));

        Toast.makeText(getApplicationContext(),"保存成功",Toast.LENGTH_SHORT).show();
    }
}
