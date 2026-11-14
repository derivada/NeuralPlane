"""Run the released NeuralPlane algorithm on training-only prepared RGB cameras."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import yaml
import trimesh
import torch
from scripts.benchmark_data import read_capture, materialize, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--scene',required=True)
    p.add_argument('--dataset',required=True)
    p.add_argument('--resolution',type=int,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output=a.output.resolve()
    capture=read_capture(a.data,a.scene,a.dataset,a.resolution)
    # Keep numeric filenames: hLoc cache, segment keys and native iteration order agree.
    scene_dir=materialize(capture,a.output/'input'/a.scene)
    config=a.output/'benchmark.yaml'
    if not config.exists():
        content=dict(_BASE_=str(Path('configs/base.yaml').resolve()),TIMESTAMP='benchmark',
            DATA=dict(SOURCE=scene_dir.parent,SCENE_ID=a.scene,OUTPUT=a.output/'native',BENCHMARK=True),
            MACHINE=dict(SEED=42))
        config.write_text(yaml.dump(content))
    for module in ('geometry_init','local_planar_primitives','np_train','export'):
        marker=a.output/f'{module}.done'
        if marker.exists():
            continue
        subprocess.run([sys.executable,'-m',f'neuralplane.scripts.{module}','--config',str(config)],check=True)
        marker.write_text('completed\n')
    export=a.output/'native'/a.scene/'neuralplane/benchmark/export'
    mesh_path=export/'planar_mesh.ply'
    mesh=trimesh.load(mesh_path,process=False)
    shutil.copy2(mesh_path,a.output/'planar_mesh.ply')
    ckpts=sorted((export.parent/'nerfstudio_models').glob('*.ckpt'))
    if not ckpts:
        raise RuntimeError('native trained checkpoint missing')
    # Only floating trained model parameters; optimizer/scaler/scheduler excluded.
    from nerfstudio.utils.eval_utils import eval_setup
    _, pipeline, _, _ = eval_setup(export.parent/'config.yml')
    count=sum(parameter.numel() for parameter in pipeline.parameters())
    del pipeline
    planes=json.loads((export/'planes.json').read_text())
    planes.update(num_vertices=len(mesh.vertices),num_faces=len(mesh.faces),num_params_model=count,
                  storage_basis='native NeuralPlane pipeline parameters; optimizer and buffers excluded')
    (a.output/'planes.json').write_text(json.dumps(planes,indent=2))
    checkpoint_hashes={str(p):sha(p) for p in Path('checkpoints').glob('*') if p.is_file()}
    subprocess.run([sys.executable,'scripts/render_benchmark.py','--data',str(a.data),'--scene',a.scene,'--output',str(a.output)],check=True)
    (a.output/'condition.json').write_text(json.dumps(dict(condition='neuralplane_native_snu_sam_train_triangulation',
        checkpoint_sha256=checkpoint_hashes,trained_checkpoint_sha256=sha(ckpts[-1]),
        iterations=4000,native_cues='SNU ScanNet normals, SAM ViT-H, NetVLAD/LoFTR training-only triangulation',
        gt_used=False,shared_priors_used=False),indent=2))

if __name__=='__main__':
    main()
