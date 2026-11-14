"""Evaluate native NeuralPlane RGB on every held-out camera after reconstruction."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils.eval_utils import eval_setup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--scene',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    capture=json.loads((a.output/'input'/a.scene/'input_manifest.json').read_text())
    config=a.output/'native'/a.scene/'neuralplane/benchmark/config.yml'
    _,pipeline,_,_=eval_setup(config)
    dslr=a.data/a.scene/'dslr'
    selected=set(capture['test_names'])
    poses={}
    with (dslr/'colmap/images.txt').open() as stream:
        for line in stream:
            f=line.split()
            if not f or line.startswith('#'):
                continue
            if len(f)!=10:
                raise ValueError('invalid camera header')
            if f[9] in selected:
                qw,qx,qy,qz,tx,ty,tz=map(float,f[1:8])
                w2c=np.eye(4)
                w2c[:3,:3]=Rotation.from_quat([qx,qy,qz,qw]).as_matrix()
                w2c[:3,3]=[tx,ty,tz]
                c2w=np.linalg.inv(w2c)
                c2w[:3,1:3]*=-1
                poses[f[9]]=c2w
            next(stream)
    if set(poses)!=selected:
        raise ValueError('test-camera set mismatch')
    K=np.array(capture['K'])
    width,height=capture['resolution']
    render,gt=a.output/'renders/test',a.output/'gt/test'
    render.mkdir(parents=True,exist_ok=True)
    gt.mkdir(parents=True,exist_ok=True)
    for name in sorted(selected):
        camera=Cameras(fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]),
            width=width,height=height,camera_to_worlds=torch.tensor(poses[name][None,:3,:4],dtype=torch.float32),
            camera_type=CameraType.PERSPECTIVE).to(pipeline.device)
        with torch.inference_mode():
            rgb=pipeline.model.get_outputs_for_camera(camera)['rgb'].clamp(0,1).cpu().numpy()
        Image.fromarray(np.round(rgb*255).astype(np.uint8)).save(render/(Path(name).stem+'.png'))
        with Image.open(dslr/'resized_undistorted_images'/name) as image:
            image.convert('RGB').resize((width,height),Image.Resampling.BICUBIC).save(gt/(Path(name).stem+'.png'))

if __name__=='__main__':
    main()
