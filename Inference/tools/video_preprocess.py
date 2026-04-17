import os
import torch
import numpy as np
import json
import pickle
import argparse
import copy

from torch import nn
from tqdm import tqdm

from mmdet.apis import init_detector
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def parse_arguments():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--detection_checkpoint_path', type=str, default='./cache/mmdetection/checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth', help="The directory of the detection checkpoint path.")
    parser.add_argument('--detection_config_path', type=str, default='./cache/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py', help="The directory of the detection configuration path.")
    parser.add_argument('--video_data_path', type=str, default='./inference_data/raw_video', help="The directory of the video data path.")
    parser.add_argument('--video_feats_path', type=str, default='video_feats_test.pkl', help="The directory of the video features path.")
    parser.add_argument('--frames_path', type=str, default='MIA/datasets/human_annotations/screenshots', help="The directory of human-annotated frames with bbox.")
    parser.add_argument('--speaker_annotation_path', type=str, default='MIA/datasets/human_annotations/speaker_annotations.json', help="The original file of annotated speaker ids.")
    parser.add_argument(
        '--TalkNet_speaker_path',
        type=str,
        default=os.path.join(PROJECT_ROOT, 'preprocessed_output', 'detect_speaker'),
        help="The output directory of TalkNet model.",
    )
    parser.add_argument("--use_TalkNet", action="store_true", help="whether using the annotations from TalkNet to get video features.")
    parser.add_argument("--roi_feat_size", type = int, default=7, help="The size of Faster R-CNN region of interest.")
    parser.add_argument("--frame_stride", type=int, default=1, help="Process every Nth frame to speed up extraction.")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames to process per video (0 = no limit).")
    parser.add_argument("--log_every", type=int, default=10, help="Print progress every N processed frames.")
    parser.add_argument("--roi_batch_size", type=int, default=2, help="Batch size for Faster R-CNN ROI feature extraction.")

    args = parser.parse_args()

    return args

class VideoFeature:
    
    def __init__(self, args, detector_model=None):
        if detector_model is None:
            self.model, self.device = self._init_detection_model(args)
        else:
            self.model = detector_model
            self.device = next(self.model.parameters()).device
        self.avg_pool = nn.AvgPool2d(args.roi_feat_size)
        self._init_roi_pipeline()

    def _init_roi_pipeline(self):
        cfg = copy.deepcopy(self.model.cfg)
        cfg.data.test.pipeline[0].type = 'LoadImageFromFile'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
        self._roi_test_pipeline = Compose(cfg.data.test.pipeline)

    def _get_feats(self, args):
        
        if args.use_TalkNet:
            self.bbox_feats = self._get_TalkNet_features(args)
        
        else:
            self.bbox_feats = self._get_Annotated_features(args)
            
    def _save_feats(self, args):
        
        video_feats_path = os.path.join(args.video_data_path, args.video_feats_path)

        with open(video_feats_path, 'wb') as f:
            pickle.dump(self.bbox_feats, f)   

    def _init_detection_model(self, args):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, but GPU-only mode is enabled.")
        device_name = 'cuda:0'
        model = init_detector(args.detection_config_path, args.detection_checkpoint_path, device=device_name)
        device = next(model.parameters()).device  # model device
        return model, device

    def _get_TalkNet_features(self, args):
        
        '''
        Input: 
            args.TalkNet_speaker_path

        Output:
        The format of video features
        {
            'video_clip_id_a':[frame_a_feat, frame_b_feat, ..., frame_N_feat],
            'video_clip_id_b':[xxx]
        }
        '''

        video_feats = {}
        error_cnt = 0
        error_path = 0
        
        for video_clip_name in tqdm(os.listdir(args.TalkNet_speaker_path), desc = 'Video'):
            frames_path = os.path.join(args.TalkNet_speaker_path, video_clip_name, 'pyframes')
            bestperson_path = os.path.join(args.TalkNet_speaker_path, video_clip_name, 'pywork', 'best_persons.npy')
            
            if not os.path.exists(bestperson_path):
                error_path += 1
                continue

            bestpersons = np.load(bestperson_path)
            
            total_frames = len(bestpersons)
            frame_indices = list(range(0, total_frames, args.frame_stride))
            if args.max_frames > 0:
                frame_indices = frame_indices[:args.max_frames]
            valid_items = []
            for frame in frame_indices:
                bbox = bestpersons[frame]
                if (bbox[0] == 0) and (bbox[1] == 0) and (bbox[2] == 0) and (bbox[3] == 0):
                    error_cnt += 1
                    continue
                frame_name = str('%06d' % frame)
                frame_path = os.path.join(frames_path, frame_name + '.jpg')
                valid_items.append((frame, frame_path, bbox.tolist()))

            if len(valid_items) == 0:
                continue

            if video_clip_name not in video_feats:
                video_feats[video_clip_name] = []

            batch_size = max(1, int(getattr(args, "roi_batch_size", 2)))
            batch_starts = list(range(0, len(valid_items), batch_size))
            frame_iter = tqdm(batch_starts, desc='FrameBatch', total=len(batch_starts), mininterval=1.0)
            processed = 0
            for start_idx in frame_iter:
                batch_items = valid_items[start_idx:start_idx + batch_size]
                if args.log_every > 0 and processed % args.log_every == 0:
                    print("Video {}: processing frame {}".format(video_clip_name, batch_items[0][0]), flush=True)
                frame_iter.set_postfix_str("frame={}".format(batch_items[-1][0]))

                batch_paths = [item[1] for item in batch_items]
                batch_rois = [item[2] for item in batch_items]
                bbox_feats = self._extract_roi_feats_batch(self.model, self.device, batch_paths, batch_rois)
                bbox_feats = self._average_pooling(bbox_feats).detach().cpu().numpy()

                for feat in bbox_feats:
                    video_feats[video_clip_name].append(feat)
                processed += len(batch_items)
            
        print('The number of error annotations is {}'.format(error_cnt))
        print('The number of error paths is {}'.format(error_path))
        
        return video_feats
            
    def _get_Annotated_features(self, args):

        '''
        Input: 
            args.video_data_path 
            args.speaker_annotation_path
            args.frames_path

        Output:
        The format of video features
        {
            'video_clip_id_a':[frame_a_feat, frame_b_feat, ..., frame_N_feat],
            'video_clip_id_b':[xxx]
        }
        '''

        speaker_annotation_path = os.path.join(args.video_data_path, args.speaker_annotation_path)
        speaker_annotations = json.load(open(speaker_annotation_path, 'r'))

        video_feats = {}
        error_cnt = 0

        try:
            for key in tqdm(speaker_annotations.keys(), desc = 'Frame'):
                
                if 'bbox' not in speaker_annotations[key].keys():
                    error_cnt += 1
                    continue
                
                roi = speaker_annotations[key]['bbox'][:4]
                roi.insert(0, 0.)

                frame_name = '_'.join(key.strip('.jpg').split('_')[:-1])
                frame_path = os.path.join(args.frames_path, frame_name + '.jpg')
                
                bbox_feat = self._extract_roi_feats(self.model, self.device, frame_path, roi)
                bbox_feat = self._average_pooling(bbox_feat)
                bbox_feat = bbox_feat.detach().cpu().numpy()
                
                video_clip_name = '_'.join(key.strip('.jpg').split('_')[:-2])

                if video_clip_name not in video_feats.keys():
                    video_feats[video_clip_name] = [bbox_feat]
                
                else:
                    video_feats[video_clip_name].append(bbox_feat)

        except Exception as e:
                print(e)

        print('The number of error annotations is {}'.format(error_cnt))

        return video_feats         

    def _extract_roi_feats(self, model, device, file_path, roi):
        if len(roi) == 5:
            roi = roi[1:]
        return self._extract_roi_feats_batch(model, device, [file_path], [roi])

    def _extract_roi_feats_batch(self, model, device, file_paths, rois):
        datas = []
        for file_path in file_paths:
            data = dict(img_info=dict(filename=file_path), img_prefix=None)
            try:
                data = self._roi_test_pipeline(data)
            except Exception as e:
                raise RuntimeError(
                    "roi feature pipeline failed: file_path={}, keys={}".format(
                        file_path,
                        list(data.keys()),
                    )
                ) from e
            datas.append(data)

        data = collate(datas, samples_per_gpu=len(datas))
        data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
        data['img'] = [img.data[0] for img in data['img']]
        device_id = device.index if device.index is not None else 0
        data = scatter(data, [device_id])[0]

        img = data['img'][0]
        with torch.no_grad():
            x = model.extract_feat(img)
            roi_tensor = torch.as_tensor(rois, dtype=torch.float32, device=device)
            batch_inds = torch.arange(len(rois), dtype=torch.float32, device=device).unsqueeze(1)
            roi_tensor = torch.cat([batch_inds, roi_tensor], dim=1)
            bbox_feat = model.roi_head.bbox_roi_extractor(
                x[:model.roi_head.bbox_roi_extractor.num_inputs], roi_tensor
            )
        return bbox_feat

    def _average_pooling(self, x):
        """
        Args:
        x: dtype: numpy.ndarray
        """
        x = self.avg_pool(x)
        x = x.flatten(1)
        return x

if __name__ == '__main__':

    args = parse_arguments()
    
    args.use_TalkNet = True
    video_data = VideoFeature(args)
    video_data._get_feats(args)
    video_data._save_feats(args)
    
