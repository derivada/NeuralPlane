# Prepared RGB benchmark adapter (pending runtime validation)

Upstream history: 3dv-casia/NeuralPlane, ef6d8bff01d7e33f642c047139304be4c546ce70.
Upstream remote is retained. The upstream pyproject declares Apache 2.0; the
upstream repository has no top-level LICENSE file. All bundled notices are retained.

Reconstruction accepts only canonical training RGB and supplied metric cameras.
The source reader checks protocol.json for Replica and MuSHRoom Kinect long
captures, official splits and bad-frame flags, RGB dimensions and rigid cameras.
ScanNet++ and Replica use resolution 2; MuSHRoom Kinect uses resolution 1.
COLMAP poses are inverted from world-to-camera to OpenCV camera-to-world, then
converted to OpenGL inside the Nerfstudio parser. No centering, pose optimization,
scale fitting, GT bounds or sensor depth is used. A camera-derived AABB encloses
training camera centers with a fixed one-metre margin. No shared DA3/MoGe data is read.

Condition: neuralplane_native_snu_sam_train_triangulation. SNU ScanNet normals,
SAM ViT-H masks and NetVLAD/LoFTR triangulation use training views only. Native
4000-step training and native planar surface extraction are retained. The native
export's GT-assisted coloring/label projection is bypassed; the scored surface is
the concatenation of native surface pieces. The render model supports test-view
RGB; render_benchmark.py loads held-out cameras/RGB only after reconstruction.
This is a native-cue condition, separate from the older shared-prior benchmark.

Source dependencies are gitlinks, including hLoc at the release-recommended
revision e3342201. The SNU encoder uses the pinned local EfficientNet source;
its complete weights load strictly from the released SNU checkpoint. Detectron2
is needed only for the optional visualization and is not part of this condition.

Before setup or any checks: push this branch, commit its harness gitlink, push the
harness branch and verify repository privacy. scripts/require_published.py enforces
this gate. Then run scripts/setup_primimix_env.sh inside a named tmux session with
PRIMIMIX_EVAL_ROOT pointing to that harness checkout. It creates neuralplane5090
(Python 3.10, torch 2.7.1/cu128, sm_120), builds pinned tiny-cuda-nn, installs
pinned hLoc/SAM and downloads native weights. CUDA_HOME defaults to this host's
~/reconstr/.cuda/cuda-12.8. The environment directory records the full package
manifest, compiler/driver and checkpoint hashes. This setup has not yet been run.

The driver requires DATA, DATASET, SCENES, PY, PRED_ROOT and RESOLUTION. Each scene
writes input hashes, native checkpoints, native planar_mesh.ply, planes.json,
condition.json, test renders and GT comparison images. Completed native stages
are resumable; training refuses to overwrite an existing run directory. A failed
partial training stage currently requires a fresh output condition for retry.
Storage is four bytes per native trained pipeline parameter; derived mesh and
optimizer/buffers are excluded. Smoke validation must precede a frozen sweep.

User override (2026-11-14): "just do it locally 4 now". LOCAL_ONLY_AUTHORIZED=1
permits local committed pins/setup/testing while private repository creation and
pushes are deferred. Reports must retain this local-only provenance.
