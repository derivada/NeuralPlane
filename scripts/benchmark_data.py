"""Training-only canonical prepared-data reader. Never opens depth, priors or GT."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

EXPECTED = {
    "replica8": dict(total_frames=2000, frame_stride=10, sampled_frames=200,
                     train_frames=175, test_frames=25, sensor_depth_used=False,
                     gt_mesh_source="nice_slam_cull_replica_mesh"),
    "mushroom_kinect": dict(dataset="MuSHRoom", capture="kinect/long_capture",
                            split="published test.txt", sensor_depth_used=False),
    "scannetpp_dslr10": {},
}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_capture(root, scene, dataset, resolution):
    if dataset not in EXPECTED:
        raise ValueError(f"unsupported dataset: {dataset}")
    expected_res = 1 if dataset == "mushroom_kinect" else 2
    if resolution != expected_res:
        raise ValueError(f"{dataset} requires resolution={expected_res}")
    source = Path(root) / scene
    protocol = source / "protocol.json"
    if EXPECTED[dataset]:
        found = json.loads(protocol.read_text())
        errors = {k: (v, found.get(k)) for k, v in EXPECTED[dataset].items() if found.get(k) != v}
        if errors:
            raise ValueError(f"prepared protocol mismatch: {errors}")
    dslr = source / "dslr"
    paths = [dslr / "nerfstudio/transforms_undistorted.json",
             dslr / "train_test_lists.json", dslr / "colmap/images.txt"]
    meta, split = [json.loads(p.read_text()) for p in paths[:2]]
    bad = {Path(f["file_path"]).name for key in ("frames", "test_frames")
           for f in meta.get(key, []) if f.get("is_bad")}
    train, test = set(split["train"]) - bad, set(split["test"]) - bad
    if not train or not test or train & test:
        raise ValueError("empty or overlapping official split")
    if EXPECTED[dataset].get("train_frames") and len(train) != EXPECTED[dataset]["train_frames"]:
        raise ValueError("unexpected training-view count")
    poses = {}
    # Read only image headers, never COLMAP point observations or sparse geometry.
    with paths[2].open() as stream:
        for line in stream:
            fields = line.split()
            if not fields or line.startswith("#"):
                continue
            if len(fields) != 10:
                raise ValueError("invalid COLMAP image header")
            name = fields[9]
            if name in train:
                qw, qx, qy, qz, tx, ty, tz = map(float, fields[1:8])
                w2c = np.eye(4)
                w2c[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                w2c[:3, 3] = [tx, ty, tz]
                poses[name] = np.linalg.inv(w2c)
            next(stream)  # skip 2D observations without parsing them
    if set(poses) != train:
        raise ValueError(f"missing training poses: {sorted(train-set(poses))}")
    width, height = round(meta["w"] / resolution), round(meta["h"] / resolution)
    sx, sy = width / meta["w"], height / meta["h"]
    K = np.array([[meta["fl_x"]*sx, 0, meta["cx"]*sx],
                  [0, meta["fl_y"]*sy, meta["cy"]*sy], [0,0,1]], dtype=float)
    frames = []
    for name in sorted(train):
        path = (dslr / "resized_undistorted_images" / name).resolve(strict=True)
        with Image.open(path) as image:
            if image.size != (meta["w"], meta["h"]):
                raise ValueError(f"RGB/calibration size mismatch: {path}")
        pose = poses[name]
        if not np.isfinite(pose).all() or not np.allclose(pose[:3,:3].T @ pose[:3,:3], np.eye(3), atol=1e-5):
            raise ValueError(f"invalid rigid camera: {name}")
        frames.append(dict(name=name, path=str(path), rgb_sha256=sha(path), c2w_opencv=pose.tolist()))
    return dict(dataset=dataset, scene=scene, resolution=[width,height], K=K.tolist(),
                train=frames, test_names=sorted(test), excluded_bad=sorted(bad),
                source_sha256={str(p.relative_to(source)):sha(p) for p in paths},
                protocol_sha256=sha(protocol) if protocol.exists() else None,
                sensor_depth_used=False, gt_used=False, shared_priors_used=False,
                pose_scale=1.0, coordinate_system="metric source world / OpenCV camera")

def materialize(capture, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "input_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != capture:
        raise RuntimeError("input provenance changed; choose a new run root")
    manifest.write_text(json.dumps(capture, indent=2)+"\n")
    color = directory / "color"
    color.mkdir(exist_ok=True)
    for i, frame in enumerate(capture["train"]):
        target = color / f"{i:06d}.png"
        if not target.exists():
            with Image.open(frame["path"]) as image:
                image.convert("RGB").resize(tuple(capture["resolution"]), Image.Resampling.BICUBIC).save(target)
    return directory
