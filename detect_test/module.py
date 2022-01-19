# -*- encoding: utf-8 -*-
"""
@File    :   module.py    
@Contact :   thgpddl@163.com

@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/1/19 9:09   thgpddl      1.0         None
"""
import argparse
import os
import json
import base64
import shutil

import paddlehub as hub
from paddlehub.module.module import runnable, moduleinfo, serving

import cv2
from detect_test.inference import *


@moduleinfo(
    name="detect_test",
    version="1.0.0",
    summary="This is a PaddleHub Module. Just for test.",
    author="thgpddl",
    author_email="",
    type="cv/detection",
)
class DetectTest:
    def __init__(self):
        model_dir = os.path.join(self.directory, 'Assets/model')
        self.img_path = os.path.join(self.directory, 'Assets', "image", "temp.jpg")
        self.parser = argparse.ArgumentParser(description="run the module", add_help=True)
        self.parser.add_argument('--input_iameg_path', type=str, default=None, help="a url of image")
        # self.vocab = load_vocab(os.path.join(self.directory, 'vocab.list'))
        pred_config = PredictConfig(model_dir)
        detector_func = 'Detector'
        self.detector = eval(detector_func)(pred_config,
                                            model_dir,
                                            device='cpu',
                                            run_mode='paddle',
                                            batch_size=1,
                                            trt_min_shape=1,
                                            trt_max_shape=1280,
                                            trt_opt_shape=640,
                                            trt_calib_mode=False,
                                            cpu_threads=1,
                                            enable_mkldnn=False)

    def predict(self, context=None, input_type="cvmat"):
        """
        Args:
            input_type: "cvmat":使用hubModule时，context为cv2读取的图片矩阵
                        “base64”：使用serving请求时，context为base64数据
            context:

        Returns:im：图片；im：results：标签，置信度，坐标

        """
        if input_type == "cvmat":
            cv2.imwrite(self.img_path, cv2.cvtColor(context, cv2.COLOR_RGB2BGR))
        if input_type == "base64":
            debase64_data = base64.b64decode(context.encode('utf-8'))  # 解码
            with open(self.img_path, 'wb') as f:
                f.write(debase64_data)
        im, im_result = predict_image(self.detector, [self.img_path], 1)
        return json.dumps({"boxes": im_result['boxes'].tolist()})

    @serving
    def predict_serving(self, image64):
        return self.predict(input_type='base64', context=image64)
