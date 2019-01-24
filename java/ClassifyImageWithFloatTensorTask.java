/*
 * Copyright (c) 2016-2018 Qualcomm Technologies, Inc.
 * All Rights Reserved.
 * Confidential and Proprietary - Qualcomm Technologies, Inc.
 */
package com.qualcomm.qti.snpe.imageclassifiers.tasks;

import android.graphics.Bitmap;
import android.os.SystemClock;
import android.util.Log;
import android.util.Pair;

import com.qualcomm.qti.snpe.FloatTensor;
import com.qualcomm.qti.snpe.NeuralNetwork;
import com.qualcomm.qti.snpe.Tensor;
import com.qualcomm.qti.snpe.imageclassifiers.Model;
import com.qualcomm.qti.snpe.imageclassifiers.ModelOverviewFragmentController;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

public class ClassifyImageWithFloatTensorTask extends AbstractClassifyImageTask {

    private static final String LOG_TAG = ClassifyImageWithFloatTensorTask.class.getSimpleName();

    public ClassifyImageWithFloatTensorTask(ModelOverviewFragmentController controller,
                             NeuralNetwork network, Bitmap image, Model model) {
        super(controller, network, image, model);
    }

    @Override
    protected String[] doInBackground(Bitmap... params) {
        final List<String> result = new LinkedList<>();

        final FloatTensor tensor = mNeuralNetwork.createFloatTensor(
                mNeuralNetwork.getInputTensorsShapes().get(mInputLayer));

        loadMeanImageIfAvailable(mModel.meanImage, tensor.getSize());

        final int[] dimensions = tensor.getShape();
        final boolean isGrayScale = (dimensions[dimensions.length -1] == 1);
        float[] rgbBitmapAsFloat;
        if (!isGrayScale) {
            rgbBitmapAsFloat = loadRgbBitmapAsFloat(mImage);
        } else {
            rgbBitmapAsFloat = loadGrayScaleBitmapAsFloat(mImage);
        }
        tensor.write(rgbBitmapAsFloat, 0, rgbBitmapAsFloat.length);

        final Map<String, FloatTensor> inputs = new HashMap<>();
        inputs.put(mInputLayer, tensor);

        final long javaExecuteStart = SystemClock.elapsedRealtime();
        final Map<String, FloatTensor> outputs = mNeuralNetwork.execute(inputs);
        final long javaExecuteEnd = SystemClock.elapsedRealtime();
        mJavaExecuteTime = javaExecuteEnd - javaExecuteStart;

        final float[] classesArray=new float[100];
        final float[] nms_scoresArray=new float[100];
        final float[] nms_classesArray=new float[100];
        final float[] nms_boxesArray=new float[400];

        for (Map.Entry<String, FloatTensor> output : outputs.entrySet()) {
            Log.d("syy","output_key:"+output.getKey());
            Log.d("syy","output_value:"+output.getValue());


            if (output.getKey().equals("detection_classes:0")){
               FloatTensor classesTensor=output.getValue();
               classesTensor.read(classesArray,0,classesArray.length);
            }
            if (output.getKey().equals("Postprocessor/BatchMultiClassNonMaxSuppression_scores")){
                FloatTensor nms_scoresTensor=output.getValue();
                nms_scoresTensor.read(nms_scoresArray,0,nms_scoresArray.length);
            }
            if (output.getKey().equals("Postprocessor/BatchMultiClassNonMaxSuppression_classes")){
                FloatTensor nms_classTensor=output.getValue();
                nms_classTensor.read(nms_classesArray,0,nms_classesArray.length);
            }
            if (output.getKey().equals("Postprocessor/BatchMultiClassNonMaxSuppression_boxes")){
                FloatTensor nms_boxesTensor=output.getValue();
                nms_boxesTensor.read(nms_boxesArray,0,nms_boxesArray.length);
            }

//            if (output.getKey().equals(mOutputLayer)) {
//                FloatTensor outputTensor = output.getValue();
//
//                final float[] array = new float[outputTensor.getSize()];
//                outputTensor.read(array, 0, array.length);
//
//                for (Pair<Integer, Float> pair : topK(1, array)) {
//                    result.add(mModel.labels[pair.first]);
//                    result.add(String.valueOf(pair.second));
//                    Log.d("syy","detection_classes:="+pair.second);
//                }
//            }
        }
        for (int i =0;i<nms_scoresArray.length;i++){
            if (nms_scoresArray[i]>0.9){
                result.add(mModel.labels[(int) nms_classesArray[i]]);
                result.add(String.valueOf(nms_scoresArray[i]));
                Log.d("syy","classes: "+classesArray[i]);
                Log.d("syy","scores: "+nms_scoresArray[i]);
                Log.d("syy","boxes: "+ nms_boxesArray[4*i+1]*300+" "+nms_boxesArray[4*i]*300+" "+nms_boxesArray[4*i+3]*300+" "+nms_boxesArray[4*i+2]*300);
            }
        }

        releaseTensors(inputs, outputs);

        return result.toArray(new String[result.size()]);
    }

    @SafeVarargs
    private final void releaseTensors(Map<String, ? extends Tensor>... tensorMaps) {
        for (Map<String, ? extends Tensor> tensorMap: tensorMaps) {
            for (Tensor tensor: tensorMap.values()) {
                tensor.release();
            }
        }
    }
}
