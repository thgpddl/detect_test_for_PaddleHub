import os
import yaml
from functools import reduce

import numpy as np
import math
from paddle.inference import Config
from paddle.inference import create_predictor

from detect_test.preprocess import preprocess, Resize, NormalizeImage, Permute, PadStride, LetterBoxResize, WarpAffine
from detect_test.visualize import visualize_box_mask

# Global dictionary
SUPPORT_MODELS = {
    'YOLO',
    'RCNN',
    'SSD',
    'Face',
    'FCOS',
    'SOLOv2',
    'TTFNet',
    'S2ANet',
    'JDE',
    'FairMOT',
    'DeepSORT',
    'GFL',
    'PicoDet',
    'CenterNet',
}


class Detector(object):

    def __init__(self,
                 pred_config,
                 model_dir,
                 device='CPU',
                 run_mode='paddle',
                 batch_size=1,
                 trt_min_shape=1,
                 trt_max_shape=1280,
                 trt_opt_shape=640,
                 trt_calib_mode=False,
                 cpu_threads=1,
                 enable_mkldnn=False):
        self.pred_config = pred_config
        self.predictor, self.config = load_predictor(
            model_dir,
            run_mode=run_mode,
            batch_size=batch_size,
            min_subgraph_size=self.pred_config.min_subgraph_size,
            device=device,
            use_dynamic_shape=self.pred_config.use_dynamic_shape,
            trt_min_shape=trt_min_shape,
            trt_max_shape=trt_max_shape,
            trt_opt_shape=trt_opt_shape,
            trt_calib_mode=trt_calib_mode,
            cpu_threads=cpu_threads,
            enable_mkldnn=enable_mkldnn)
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0

    def preprocess(self, image_list):
        preprocess_ops = []
        for op_info in self.pred_config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            preprocess_ops.append(eval(op_type)(**new_op_info))

        input_im_lst = []
        input_im_info_lst = []
        for im_path in image_list:
            im, im_info = preprocess(im_path, preprocess_ops)
            input_im_lst.append(im)
            input_im_info_lst.append(im_info)
        inputs = create_inputs(input_im_lst, input_im_info_lst)
        return inputs

    def postprocess(self,
                    np_boxes,
                    np_masks,
                    inputs,
                    np_boxes_num,
                    threshold=0.5):
        # postprocess output of predictor
        results = {}
        results['boxes'] = np_boxes
        results['boxes_num'] = np_boxes_num
        if np_masks is not None:
            results['masks'] = np_masks
        return results

    def predict(self, image_list, threshold=0.5, repeats=1, add_timer=False):

        inputs = self.preprocess(image_list)
        np_boxes, np_masks = None, None
        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])
        # model prediction
        for i in range(repeats):
            self.predictor.run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_handle(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()
            boxes_num = self.predictor.get_output_handle(output_names[1])
            np_boxes_num = boxes_num.copy_to_cpu()
            if self.pred_config.mask:
                masks_tensor = self.predictor.get_output_handle(output_names[2])
                np_masks = masks_tensor.copy_to_cpu()

        # postprocess
        results = []
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:
            print('[WARNNING] No object detected.')
            results = {'boxes': np.zeros([0, 6]), 'boxes_num': [0]}
        else:
            results = self.postprocess(
                np_boxes, np_masks, inputs, np_boxes_num, threshold=threshold)
        return results


def create_inputs(imgs, im_info):

    inputs = {}

    im_shape = []
    scale_factor = []
    if len(imgs) == 1:
        inputs['image'] = np.array((imgs[0],)).astype('float32')
        inputs['im_shape'] = np.array(
            (im_info[0]['im_shape'],)).astype('float32')
        inputs['scale_factor'] = np.array(
            (im_info[0]['scale_factor'],)).astype('float32')
        return inputs

    for e in im_info:
        im_shape.append(np.array((e['im_shape'],)).astype('float32'))
        scale_factor.append(np.array((e['scale_factor'],)).astype('float32'))

    inputs['im_shape'] = np.concatenate(im_shape, axis=0)
    inputs['scale_factor'] = np.concatenate(scale_factor, axis=0)

    imgs_shape = [[e.shape[1], e.shape[2]] for e in imgs]
    max_shape_h = max([e[0] for e in imgs_shape])
    max_shape_w = max([e[1] for e in imgs_shape])
    padding_imgs = []
    for img in imgs:
        im_c, im_h, im_w = img.shape[:]
        padding_im = np.zeros(
            (im_c, max_shape_h, max_shape_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = img
        padding_imgs.append(padding_im)
    inputs['image'] = np.stack(padding_imgs, axis=0)
    return inputs


class PredictConfig():

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        print("model_dir", model_dir)
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.check_model(yml_conf)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.mask = False
        self.use_dynamic_shape = yml_conf['use_dynamic_shape']
        if 'mask' in yml_conf:
            self.mask = yml_conf['mask']
        self.tracker = None
        if 'tracker' in yml_conf:
            self.tracker = yml_conf['tracker']
        if 'NMS' in yml_conf:
            self.nms = yml_conf['NMS']
        if 'fpn_stride' in yml_conf:
            self.fpn_stride = yml_conf['fpn_stride']

    def check_model(self, yml_conf):
        for support_model in SUPPORT_MODELS:
            if support_model in yml_conf['arch']:
                return True
        raise ValueError("Unsupported arch: {}, expect {}".format(yml_conf[
                                                                      'arch'], SUPPORT_MODELS))


def load_predictor(model_dir,
                   run_mode='paddle',
                   batch_size=1,
                   device='CPU',
                   min_subgraph_size=3,
                   use_dynamic_shape=False,
                   trt_min_shape=1,
                   trt_max_shape=1280,
                   trt_opt_shape=640,
                   trt_calib_mode=False,
                   cpu_threads=1,
                   enable_mkldnn=False):

    if device != 'GPU' and run_mode != 'paddle':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}"
                .format(run_mode, device))
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))
    if device == 'GPU':
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    elif device == 'XPU':
        config.enable_lite_engine()
        config.enable_xpu(10 * 1024 * 1024)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
                pass

    precision_map = {
        'trt_int8': Config.Precision.Int8,
        'trt_fp32': Config.Precision.Float32,
        'trt_fp16': Config.Precision.Half
    }
    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=1 << 25,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=trt_calib_mode)

        if use_dynamic_shape:
            min_input_shape = {
                'image': [batch_size, 3, trt_min_shape, trt_min_shape]
            }
            max_input_shape = {
                'image': [batch_size, 3, trt_max_shape, trt_max_shape]
            }
            opt_input_shape = {
                'image': [batch_size, 3, trt_opt_shape, trt_opt_shape]
            }
            config.set_trt_dynamic_shape_info(min_input_shape, max_input_shape,
                                              opt_input_shape)
            print('trt set dynamic shape done!')

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)
    return predictor, config


def visualize(image_list, results, labels, threshold=0.5):
    # visualize the predict result
    start_idx = 0
    for idx, image_file in enumerate(image_list):
        im_bboxes_num = results['boxes_num'][idx]
        im_results = {}
        if 'boxes' in results:
            im_results['boxes'] = results['boxes'][start_idx:start_idx +
                                                             im_bboxes_num, :]
        if 'masks' in results:
            im_results['masks'] = results['masks'][start_idx:start_idx +
                                                             im_bboxes_num, :]
        if 'segm' in results:
            im_results['segm'] = results['segm'][start_idx:start_idx +
                                                           im_bboxes_num, :]
        if 'label' in results:
            im_results['label'] = results['label'][start_idx:start_idx +
                                                             im_bboxes_num]
        if 'score' in results:
            im_results['score'] = results['score'][start_idx:start_idx +
                                                             im_bboxes_num]

        start_idx += im_bboxes_num
        im = visualize_box_mask(
            image_file, im_results, labels, threshold=threshold)
        return im,im_results


def predict_image(detector, image_list, batch_size=1):
    batch_loop_cnt = math.ceil(float(len(image_list)) / batch_size)
    for i in range(batch_loop_cnt):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, len(image_list))
        batch_image_list = image_list[start_index:end_index]

        results = detector.predict(batch_image_list, 0.5)
        # 直接返回结果图
        return visualize(
            batch_image_list,
            results,
            detector.pred_config.labels,
            threshold=0.5)


# def detmain(input_image):
#     cv2.imwrite("temp.jpg", cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR))
#     pred_config = PredictConfig("/home/heyujun/heyujun/PaddleDetection/outputs/inference/")
#     detector_func = 'Detector'
#
#     detector = eval(detector_func)(pred_config,
#                                    "/home/heyujun/heyujun/PaddleDetection/outputs/inference/",
#                                    device='cpu',
#                                    run_mode='paddle',
#                                    batch_size=1,
#                                    trt_min_shape=1,
#                                    trt_max_shape=1280,
#                                    trt_opt_shape=640,
#                                    trt_calib_mode=False,
#                                    cpu_threads=1,
#                                    enable_mkldnn=False)
#
#     return predict_image(detector, ["temp.jpg"], 1)


# def detmain2(image_url):
#     pred_config = PredictConfig("/home/heyujun/heyujun/PaddleDetection/outputs/inference/")
#     detector_func = 'Detector'
#
#     detector = eval(detector_func)(pred_config,
#                                    "/home/heyujun/heyujun/PaddleDetection/outputs/inference/",
#                                    device='cpu',
#                                    run_mode='paddle',
#                                    batch_size=1,
#                                    trt_min_shape=1,
#                                    trt_max_shape=1280,
#                                    trt_opt_shape=640,
#                                    trt_calib_mode=False,
#                                    cpu_threads=1,
#                                    enable_mkldnn=False)
#
#     return predict_image(detector, [image_url], 1)