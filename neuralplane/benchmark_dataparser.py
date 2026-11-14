"""Canonical training-only RGB cameras without synthetic depth or scene normalization."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type
import json
import numpy as np
import torch
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.scannet_dataparser import ScanNet, ScanNetDataParserConfig as NativeConfig
from nerfstudio.data.scene_box import SceneBox

@dataclass
class BenchmarkParser(ScanNet):
    def _generate_dataparser_outputs(self, split='train'):
        manifest = self.config.data / 'input_manifest.json'
        if not manifest.exists():
            return super()._generate_dataparser_outputs(split)
        capture = json.loads(manifest.read_text())
        poses = np.array([f['c2w_opencv'] for f in capture['train']],dtype=np.float32)
        poses[:,:3,1:3] *= -1  # OpenCV -> OpenGL camera; unchanged metric world
        K = np.array(capture['K'])
        width,height = capture['resolution']
        images = [self.config.data/'color'/f'{i:06d}.png' for i in range(len(poses))]
        cameras = Cameras(fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]),
                          width=width,height=height,camera_to_worlds=torch.from_numpy(poses[:,:3,:4]),
                          camera_type=CameraType.PERSPECTIVE)
        # AABB from supplied training cameras only, no scan bounds.
        positions=poses[:,:3,3]
        aabb=np.stack([positions.min(0)-1,positions.max(0)+1])
        return DataparserOutputs(image_filenames=images,cameras=cameras,
            scene_box=SceneBox(aabb=torch.tensor(aabb,dtype=torch.float32)),
            dataparser_scale=1.0,dataparser_transform=torch.eye(4)[:3],metadata={})

@dataclass
class ScanNetDataParserConfig(NativeConfig):
    _target: Type = field(default_factory=lambda: BenchmarkParser)
    center_method: str = 'none'
    auto_scale_poses: bool = False
    load_3D_points: bool = False
