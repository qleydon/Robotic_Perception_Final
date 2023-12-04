import argparse
import sys
sys.path.insert(0, '.')
from FastSAM.fastsam.model import FastSAM
from FastSAM.fastsam.prompt import FastSAMPrompt

import ast
import torch
from PIL import Image
from FastSAM.utils.tools import convert_box_xywh_to_xyxy

def FastSAM_segmentation(input, bbox, model=None):
    input = input[:, :, ::-1]
    bbox = [convert_box_xywh_to_xyxy(bbox)]
    #print("bbox before,",bbox)
    if model == None:
        model = FastSAM("./FastSAM-x.pt")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    everything_results = model(
        input,              #
        device=device,      #
        retina_masks=True,  #draw high-resolution segmentation masks
        imgsz=1056,         #image size
        conf=0.4,           #object confidence threshold
        iou=0.9             #iou threshold for filtering the annotations    
        )
    
    prompt_process = FastSAMPrompt(input, everything_results, device=device)
    print(bbox)
    if bbox[0][2]!=0 and bbox[0][3]!=0:
        ann = prompt_process.box_prompt(bboxes=bbox)
        bboxes = bbox
    
    result = prompt_process.plot(
        annotations=ann,
        output_path="./FastSAM/output/test.jpg",
        bboxes = bboxes,
        withContours=False,
        better_quality=False,
    )

    return result

    

    