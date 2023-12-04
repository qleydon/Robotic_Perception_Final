import numpy as np
import os
import sys
import time
import argparse
import yaml, json
from PIL import Image

import matplotlib.pyplot as plt
import cv2

import torch
import torch.utils.data as data
import torch.optim as optim

sys.path.insert(0, '.')
from modules.model import MDNet, BCELoss, set_optimizer
from modules.sample_generator import SampleGenerator
from modules.utils import overlap_ratio
from data_prov import RegionExtractor
from bbreg import BBRegressor
from gen_config_avi import gen_config

from FastSAM.fastSAM_inference import FastSAM_segmentation
from FastSAM.fastsam.model import FastSAM
from FastSAM.fastsam.prompt import FastSAMPrompt

opts = yaml.safe_load(open('tracking/options.yaml','r'))

avi = False

def forward_samples(model, image, samples, out_layer='conv3'):
    model.eval()
    extractor = RegionExtractor(image, samples, opts)
    for i, regions in enumerate(extractor):
        if opts['use_gpu']:
            regions = regions.cuda()
        with torch.no_grad():
            feat = model(regions, out_layer=out_layer)
        if i==0:
            feats = feat.detach().clone()
        else:
            feats = torch.cat((feats, feat.detach().clone()), 0)
    return feats


def train(model, criterion, optimizer, pos_feats, neg_feats, maxiter, in_layer='fc4'):
    model.train()

    batch_pos = opts['batch_pos']
    batch_neg = opts['batch_neg']
    batch_test = opts['batch_test']
    batch_neg_cand = max(opts['batch_neg_cand'], batch_neg)

    pos_idx = np.random.permutation(pos_feats.size(0))
    neg_idx = np.random.permutation(neg_feats.size(0))
    while(len(pos_idx) < batch_pos * maxiter):
        pos_idx = np.concatenate([pos_idx, np.random.permutation(pos_feats.size(0))])
    while(len(neg_idx) < batch_neg_cand * maxiter):
        neg_idx = np.concatenate([neg_idx, np.random.permutation(neg_feats.size(0))])
    pos_pointer = 0
    neg_pointer = 0

    for i in range(maxiter):

        # select pos idx
        pos_next = pos_pointer + batch_pos
        pos_cur_idx = pos_idx[pos_pointer:pos_next]
        pos_cur_idx = pos_feats.new(pos_cur_idx).long()
        pos_pointer = pos_next

        # select neg idx
        neg_next = neg_pointer + batch_neg_cand
        neg_cur_idx = neg_idx[neg_pointer:neg_next]
        neg_cur_idx = neg_feats.new(neg_cur_idx).long()
        neg_pointer = neg_next

        # create batch
        batch_pos_feats = pos_feats[pos_cur_idx]
        batch_neg_feats = neg_feats[neg_cur_idx]

        # hard negative mining
        if batch_neg_cand > batch_neg:
            model.eval()
            for start in range(0, batch_neg_cand, batch_test):
                end = min(start + batch_test, batch_neg_cand)
                with torch.no_grad():
                    score = model(batch_neg_feats[start:end], in_layer=in_layer)
                if start==0:
                    neg_cand_score = score.detach()[:, 1].clone()
                else:
                    neg_cand_score = torch.cat((neg_cand_score, score.detach()[:, 1].clone()), 0)

            _, top_idx = neg_cand_score.topk(batch_neg)
            batch_neg_feats = batch_neg_feats[top_idx]
            model.train()

        # forward
        pos_score = model(batch_pos_feats, in_layer=in_layer)
        neg_score = model(batch_neg_feats, in_layer=in_layer)

        # optimize
        loss = criterion(pos_score, neg_score)
        model.zero_grad()
        loss.backward()
        if 'grad_clip' in opts:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opts['grad_clip'])
        optimizer.step()


def run_mdnet(video, init_bbox, frame=None, gt=None, output='', display=False):
    SAM_model = FastSAM("C:/Y6_S1/Robotics/Robotic_Perception_Final/FastSAM/weights/FastSAM-x.pt")

    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # Init bbox
    target_bbox = np.array(init_bbox)
    result = np.zeros((video_length, 4))
    result_bb = np.zeros((video_length, 4))
    result[0] = target_bbox
    result_bb[0] = target_bbox

    if gt is not None:
        overlap = np.zeros(video_length)
        overlap[0] = 1

    # Init model
    model = MDNet(opts['model_path'])
    if opts['use_gpu']:
        model = model.cuda()

    # Init criterion and optimizer 
    criterion = BCELoss()
    model.set_learnable_params(opts['ft_layers'])
    init_optimizer = set_optimizer(model, opts['lr_init'], opts['lr_mult'])
    update_optimizer = set_optimizer(model, opts['lr_update'], opts['lr_mult'])

    tic = time.time()
    # Load first image
    
    image = frame
    # Draw pos/neg samples
    pos_examples = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])(
                        target_bbox, opts['n_pos_init'], opts['overlap_pos_init'])

    neg_examples = np.concatenate([
                    SampleGenerator('uniform', image.size, opts['trans_neg_init'], opts['scale_neg_init'])(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init']),
                    SampleGenerator('whole', image.size)(
                        target_bbox, int(opts['n_neg_init'] * 0.5), opts['overlap_neg_init'])])
    neg_examples = np.random.permutation(neg_examples)

    # Extract pos/neg features
    pos_feats = forward_samples(model, image, pos_examples)
    neg_feats = forward_samples(model, image, neg_examples)

    # Initial training
    train(model, criterion, init_optimizer, pos_feats, neg_feats, opts['maxiter_init'])
    del init_optimizer, neg_feats
    torch.cuda.empty_cache()

    # Train bbox regressor
    bbreg_examples = SampleGenerator('uniform', image.size, opts['trans_bbreg'], opts['scale_bbreg'], opts['aspect_bbreg'])(
                        target_bbox, opts['n_bbreg'], opts['overlap_bbreg'])
    bbreg_feats = forward_samples(model, image, bbreg_examples)
    bbreg = BBRegressor(image.size)
    bbreg.train(bbreg_feats, bbreg_examples, target_bbox)
    del bbreg_feats
    torch.cuda.empty_cache()

    # Init sample generators for update
    sample_generator = SampleGenerator('gaussian', image.size, opts['trans'], opts['scale'])
    pos_generator = SampleGenerator('gaussian', image.size, opts['trans_pos'], opts['scale_pos'])
    neg_generator = SampleGenerator('uniform', image.size, opts['trans_neg'], opts['scale_neg'])

    # Init pos/neg features for update
    neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_init'])
    neg_feats = forward_samples(model, image, neg_examples)
    pos_feats_all = [pos_feats]
    neg_feats_all = [neg_feats]

    spf_total = time.time() - tic

    # Segmentation
    result_s = result_bb[0]
    result_s =result_s.astype(int).tolist()
    #print(result_bb)
    SAM_result = FastSAM_segmentation(frame, result_s, SAM_model)
    # Display
    #savefig = savefig_dir != ''
    if display: # or savefig:
        cv2.imshow("Tracking",SAM_result)

        #p1 = (int(result_bb[0,0]), int(result_bb[0,1]))
        #p2 = (int(result_bb[0,0] + result_bb[0,2]), int(result_bb[0,1] + result_bb[0,3]))
        #cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        #cv2.imshow("Tracking", frame)
        #if savefig:
        #    fig.savefig(os.path.join(savefig_dir, '0000.jpg'), dpi=dpi)

    # Main loop
    for i in range(1, video_length):
        tic = time.time()
        # Load image
        ret, frame = video.read()
        if not ret:
            print('something went wrong')
            break
        image = frame # will need to resize

        # Estimate target bbox
        samples = sample_generator(target_bbox, opts['n_samples'])
        sample_scores = forward_samples(model, image, samples, out_layer='fc6')

        top_scores, top_idx = sample_scores[:, 1].topk(5)
        top_idx = top_idx.cpu()
        target_score = top_scores.mean()
        target_bbox = samples[top_idx]
        if top_idx.shape[0] > 1:
            target_bbox = target_bbox.mean(axis=0)
        success = target_score > 0
        
        # Expand search area at failure
        if success:
            sample_generator.set_trans(opts['trans'])
        else:
            sample_generator.expand_trans(opts['trans_limit'])

        # Bbox regression
        if success:
            bbreg_samples = samples[top_idx]
            if top_idx.shape[0] == 1:
                bbreg_samples = bbreg_samples[None,:]
            bbreg_feats = forward_samples(model, image, bbreg_samples)
            bbreg_samples = bbreg.predict(bbreg_feats, bbreg_samples)
            bbreg_bbox = bbreg_samples.mean(axis=0)
        else:
            bbreg_bbox = target_bbox

        # Save result
        result[i] = target_bbox
        result_bb[i] = bbreg_bbox

        # Data collect
        if success:
            pos_examples = pos_generator(target_bbox, opts['n_pos_update'], opts['overlap_pos_update'])
            pos_feats = forward_samples(model, image, pos_examples)
            pos_feats_all.append(pos_feats)
            if len(pos_feats_all) > opts['n_frames_long']:
                del pos_feats_all[0]

            neg_examples = neg_generator(target_bbox, opts['n_neg_update'], opts['overlap_neg_update'])
            neg_feats = forward_samples(model, image, neg_examples)
            neg_feats_all.append(neg_feats)
            if len(neg_feats_all) > opts['n_frames_short']:
                del neg_feats_all[0]

        # Short term update
        if not success:
            nframes = min(opts['n_frames_short'], len(pos_feats_all))
            pos_data = torch.cat(pos_feats_all[-nframes:], 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        # Long term update
        elif i % opts['long_interval'] == 0:
            pos_data = torch.cat(pos_feats_all, 0)
            neg_data = torch.cat(neg_feats_all, 0)
            train(model, criterion, update_optimizer, pos_data, neg_data, opts['maxiter_update'])

        torch.cuda.empty_cache()
        spf = time.time() - tic
        spf_total += spf
        # Segmentation
        result_s = result_bb[i]
        result_s =result_s.astype(int).tolist()
        SAM_result = FastSAM_segmentation(frame, result_s, SAM_model)
        # Display
        if display:
            plt.pause(.01)
            cv2.imshow("Tracking", SAM_result)
            output.write(SAM_result)
            k = cv2.waitKey(1) & 0xff
            if k == 27 : break

            #p1 = (int(result_bb[i,0]), int(result_bb[i,1]))
            #p2 = (int(result_bb[i,0] + result_bb[i,2]), int(result_bb[i,1] + result_bb[i,3]))
            #cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

            #plt.pause(.01)
            #cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
            #cv2.putText(frame, "PyMDNet Tracker", (100,20), 
            #cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
            #cv2.imshow("Tracking", frame)
            #output.write(frame)
            #k = cv2.waitKey(1) & 0xff
            #if k == 27 : break
                
            
        if gt is None:
            print('Frame {:d}/{:d}, Score {:.3f}, Time {:.3f}'
                .format(i, video_length, target_score, spf))
        else:
            overlap[i] = overlap_ratio(gt[i], result_bb[i])[0]
            print('Frame {:d}/{:d}, Overlap {:.3f}, Score {:.3f}, Time {:.3f}'
                .format(i, video_length, overlap[i], target_score, spf))

    if display:
        video.release()
        output.release()
        cv2.destroyAllWindows()

    if gt is not None:
        print('meanIOU: {:.3f}'.format(overlap.mean()))
    fps = video_length / spf_total
    return result, result_bb, fps


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--seq', default='', help='input seq')
    parser.add_argument('-j', '--json', default='', help='input json')
    parser.add_argument('-a', '--avi', default='', help='input avi')
    parser.add_argument('-f', '--savefig', action='store_true')
    parser.add_argument('-d', '--display', action='store_true')
    parser.add_argument('-b', '--bbox', default='', help='select_bbox')

    args = parser.parse_args()
    assert args.seq != '' or args.json != '' or args.avi !=0

    np.random.seed(0)
    torch.manual_seed(0)

    # Generate sequence config
    video, frame, init_bbox, gt, output, display, result_path = gen_config(args)

    # Run tracker
    result, result_bb, fps = run_mdnet(video, init_bbox, frame=frame, gt=gt, output=output, display=display)

    # Save result
    res = {}
    res['res'] = result_bb.round().tolist()
    res['type'] = 'rect'
    res['fps'] = fps
    json.dump(res, open(result_path, 'w'), indent=2)
