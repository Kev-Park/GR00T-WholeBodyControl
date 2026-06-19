# Plan: `parquet_populate.py` — offline motion-token generator

Status: design only, **not yet implemented**. Implement after `eval_parquet_sonic.py`'s
reference-motion bypass is validated to drive a parquet through encoder→decoder→sim
correctly.

## Purpose

`convert_isaac_hdf5_to_lerobot.py` zero-fills `action.motion_token` because computing
tokens requires running `model_encoder.onnx` and the converter intentionally avoids the
runtime dependency. Isaac-GR00T's SONIC VLA fine-tuning reads `action.motion_token`
directly from parquet as a flow-matching supervision target, so the column must be
populated before training. This script does that population pass — same encoder
invocation used by `eval_parquet_sonic.py`, applied row-by-row offline.

## Inputs and outputs

```
Input:  <dataset_root>/data/chunk-*/episode_*.parquet  (action.motion_token zero-filled,
                                                        motion.reference_qpos populated)
Output: same files in place — action.motion_token filled, motion.reference_qpos REMOVED,
        meta/info.json features schema updated to drop the auxiliary column.
```

After this script runs, the dataset is strict SONIC-schema-compliant and trainable.

## Per-row algorithm

```
For each row t in episode parquet:
    1. Window lookup: ref_window = motion.reference_qpos[clip(t + [0,5,...,45], 0, N-1)]
                                                                  shape (10, 36)
    2. Lower-body (MJ order):
         lb_pos = ref_window[:, 7:7+12]                           # (10, 12) MJ order
         lb_vel = np.gradient(lb_pos, 5/50, axis=0)               # 100ms step
         # gear_sonic.joint_names is MUJOCO-grouped (verified via
         # features_sonic_vla.py::_get_joint_group_slices, which requires each joint
         # group to occupy a contiguous range). Lower body = first 12 entries directly,
         # no SONIC-interleaved permutation needed (MUJOCO_TO_ISAACLAB only applies to
         # the UTM decoder's output, not encoder input).
    3. Anchor (current frame, world frame):
         anchor_pos_w     = motion.reference_qpos[t, 0:3]
         anchor_quat_wxyz = motion.reference_qpos[t, 3:7]
         R_robot          = R.from_quat(observation.root_orientation[t])    # executed
         R_anchor         = R.from_quat(anchor_quat_wxyz)                   # intended
         anchor_rot6d     = (R_robot.inv() * R_anchor).as_matrix()[:, :2].flatten("C")
    4. VR passthrough:
         vr_pos      = teleop.vr_3pt_position[t]      (9,)  pelvis-local
         vr_rot6d    = teleop.vr_3pt_orientation[t]   (18,) pelvis-local
    5. Encoder obs:
         enc_obs = build_encoder_obs(
             anchor_pos_world=anchor_pos_w,
             anchor_quat_wxyz=anchor_quat_wxyz,
             anchor_rot6d=anchor_rot6d,
             lower_body_positions_future=lb_pos,
             lower_body_velocities_future=lb_vel,
             vr_3pt_position_anchor_local=vr_pos,
             vr_3pt_rot6d=vr_rot6d,
         )                                            # (1, 1762)
    6. token = encoder_session.run({"obs_dict": enc_obs}).reshape(-1)   # (64,)
    7. Write token → action.motion_token[t]
```

## Post-pass cleanup (per episode)

```
df = df.drop(columns=["motion.reference_qpos"])
df.to_parquet(<path>)   # overwrite in place
```

Then once per dataset, update `<dataset_root>/meta/info.json`:
- Remove `"motion.reference_qpos"` from the `features` dict
- This restores the strict SONIC schema as defined in
  `gear_sonic.data.features_sonic_vla.get_features_sonic_vla()`

## Dependencies

- `onnxruntime` for `model_encoder.onnx`
- `gear_sonic.utils.data_collection.transforms.quat_to_rot6d` (already used by converter)
- A shared `build_encoder_obs` implementation. Two viable hosts:
  - Copy from `WBCBenchmark/Training/scripts/reinforcement_learning/rsl_rl/vla_sonic/planner_to_utm.py`
  - Or refactor it into `gear_sonic/data/encoder_obs_builder.py` (cleaner long-term — both
    WBCBenchmark eval and this script import from one place)

## CLI

```
python -m gear_sonic.scripts.parquet_populate \
    --dataset-root <path> \
    --encoder-onnx <path to model_encoder.onnx> \
    [--dry-run]    # validate without writing
    [--keep-reference]   # don't drop motion.reference_qpos (for diagnostics)
```

## Validation strategy

Before committing to overwrite, run with `--dry-run --keep-reference` and produce
side-by-side `action.motion_token_from_reference` vs `action.motion_token_from_planner`
(the latter computed by also running the planner ONNX on the parquet's `teleop.planner_*`
columns). The L2 distance in token space tells you exactly how much the
gradient-derived planner-command path was losing.

## When to implement

Sequence:
1. ✅ Capture reference motion in `collect_pick_cam.py` → HDF5
2. ✅ Converter writes `motion.reference_qpos` parquet column
3. ✅ Eval reads the column and bypasses the planner
4. **→ Validate eval walks correctly with the reference-motion bypass**
5. Then build this script and tokenize the dataset
6. Then hand off to Isaac-GR00T VLA fine-tuning
