"""Offline converter: producer Isaac Lab HDF5 (Part A v2) -> LeRobot v2.1 SONIC VLA dataset.

Reads robomimic-style HDF5 rollouts written by the TrajGen ``collect_pick_cam.py``
producer (``schema_version=2``, see Part A v2 of the cross-repo plan) and emits
a LeRobot v2.1 dataset whose feature schema matches
``gear_sonic.data.features_sonic_vla.get_features_sonic_vla`` exactly.

Run under ``.venv_data_collection`` (provisioned via
``bash install_scripts/install_data_collection.sh``).

Several feature slots that have no Isaac-Lab analog are zero-filled
(see ``ZERO_FILLED_FIELDS`` and the dataset's ``meta/info.json`` script_config).

``action.motion_token`` is the SONIC VLA's flow-matching supervision target. Pass
``--encoder-onnx <model_encoder.onnx>`` to POPULATE it by running the encoder on the
recorded reference motion (G1 mode, mirroring ``eval_parquet_sonic.py``'s validated
reference-bypass path); without that flag it is zero-filled (old behavior) and a VLA
trained on the dataset will learn to emit null tokens.

Converter v2 (current): planner_{movement,speed,facing,height,mode} are
derived per-frame from ``root_pos_w`` / ``root_quat_w`` so the VLA learns
real locomotion commands. v1 hardcoded them to stationary-task constants.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_sonic_vla import (
    EGO_VIEW_HEIGHT,
    EGO_VIEW_WIDTH,
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
)
from gear_sonic.utils.data_collection.transforms import (
    compute_projected_gravity,
    quat_to_rot6d,
)


SCHEMA_VERSION = 2

# Default encoder ONNX location — the canonical SONIC release path other repo scripts
# use (download_from_hf.py writes here; WBCBenchmark eval points at the same file). Lets
# the converter populate tokens without an explicit --encoder-onnx. Resolved relative to
# the repo root: <GR00T-WholeBodyControl>/gear_sonic_deploy/policy/release/model_encoder.onnx
# (this file lives at <repo>/gear_sonic/scripts/, so parents[2] is the repo root).
_DEFAULT_ENCODER_ONNX = (
    Path(__file__).resolve().parents[2]
    / "gear_sonic_deploy" / "policy" / "release" / "model_encoder.onnx"
)
TELEOP_GROUP_PATH = "data/demo_0/teleop"
OBS_GROUP_PATH = "data/demo_0/obs"
DATA_GROUP_PATH = "data"
DEMO_GROUP_PATH = "data/demo_0"

# Locked-in zero-fill set (see plan Part C: "Zero-fill locked in").
ZERO_FILLED_FIELDS = [
    "action.motion_token",
    "teleop.smpl_joints",
    "teleop.smpl_pose",
    "teleop.body_quat_w",
    "teleop.target_body_orientation",
    "observation.cpp_rotation_offset",
    "teleop.smpl_frame_index",
]

# Local-frame offsets applied by gear_sonic for the VR 3-point representation.
# Mirrors gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer.G1_KEY_FRAME_OFFSETS.
LW_LOCAL_OFFSET = np.array([0.18, -0.025, 0.0], dtype=np.float64)
RW_LOCAL_OFFSET = np.array([0.18, 0.025, 0.0], dtype=np.float64)
TORSO_LOCAL_OFFSET = np.array([0.0, 0.0, 0.35], dtype=np.float64)


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _mat_to_quat_wxyz(rot_mat: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> wxyz quaternion (scalar-first)."""
    quat_xyzw = R.from_matrix(rot_mat).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def _apply_local_offset(T: np.ndarray, offset: np.ndarray) -> np.ndarray:
    """Translate by an offset expressed in the local frame of T (4x4 SE3)."""
    return T[:3, 3] + T[:3, :3] @ offset


def _build_joint_permutation(
    isaac_joint_names: list[str],
    gear_sonic_joint_names: list[str],
) -> tuple[np.ndarray, list[str]]:
    """Return ``perm`` such that ``q_gs = q_il[perm]`` for joints that exist in
    both name lists, plus the list of gear_sonic joint names that are NOT
    present in Isaac Lab (these will be zero-filled).

    ``perm[i]`` is the Isaac-Lab index for ``gear_sonic_joint_names[i]``, or -1
    when the name is missing on the Isaac side.
    """
    isaac_index = {name: i for i, name in enumerate(isaac_joint_names)}
    perm = np.full(len(gear_sonic_joint_names), -1, dtype=np.int64)
    missing: list[str] = []
    for i, name in enumerate(gear_sonic_joint_names):
        idx = isaac_index.get(name)
        if idx is None:
            missing.append(name)
        else:
            perm[i] = idx
    return perm, missing


def _permute_with_zero_fill(
    q_il_batch: np.ndarray, perm: np.ndarray
) -> np.ndarray:
    """Apply a name-based permutation; missing entries (perm == -1) become 0.0."""
    n_frames = q_il_batch.shape[0]
    n_out = perm.shape[0]
    out = np.zeros((n_frames, n_out), dtype=np.float64)
    valid = perm >= 0
    out[:, valid] = q_il_batch[:, perm[valid]]
    return out


def _resize_ego_view(frame: np.ndarray) -> np.ndarray:
    """Downscale the producer ego view -> 480x640 dataset feature.

    The kitchen collection env renders the ego camera at 960x1280 (4:3), so this is a uniform
    2x INTER_AREA shrink (crisp, no aspect distortion). NOTE: a 16:9 source (e.g. 720x1280)
    would be squished non-uniformly into 4:3 here — blurring/horizontally distorting frames —
    so the producer camera must be 4:3 to match the EGO_VIEW (640x480) aspect.
    """
    if frame.shape[:2] == (EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH):
        return frame
    return cv2.resize(
        frame,
        (EGO_VIEW_WIDTH, EGO_VIEW_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


def _read_v2_hdf5(path: Path) -> dict[str, Any]:
    """Slurp every field the converter needs out of one rollout HDF5."""
    with h5py.File(path, "r") as f:
        if DATA_GROUP_PATH not in f:
            raise RuntimeError(
                f"{path}: missing top-level 'data' group. Not a Part A v2 HDF5."
            )
        if TELEOP_GROUP_PATH not in f:
            raise RuntimeError(
                f"{path}: missing '{TELEOP_GROUP_PATH}' group."
            )

        teleop_grp = f[TELEOP_GROUP_PATH]
        schema = int(teleop_grp.attrs.get("schema_version", 0))
        if schema != SCHEMA_VERSION:
            raise RuntimeError(
                f"{path}: teleop schema_version={schema}, expected {SCHEMA_VERSION}. "
                "Regenerate this file with the Part A v2 producer."
            )

        env_args_attr = f[DATA_GROUP_PATH].attrs.get("env_args")
        if env_args_attr is None:
            raise RuntimeError(
                f"{path}: data/@env_args is missing — not a Part A v2 HDF5."
            )
        env_args = json.loads(_decode(env_args_attr))

        obs_grp = f[OBS_GROUP_PATH]
        joint_pos = obs_grp["robot0_joint_pos"][...].astype(np.float64)
        root_pos_w = obs_grp["robot0_root_pos_w"][...].astype(np.float64)
        root_quat_w = obs_grp["robot0_root_quat_w"][...].astype(np.float64)
        ego_view = obs_grp["ego_view_image"][...]
        object_pos_w = (
            obs_grp["object_pos"][...].astype(np.float64)
            if "object_pos" in obs_grp else None
        )
        object_quat_w = (
            obs_grp["object_quat"][...].astype(np.float64)
            if "object_quat" in obs_grp else None
        )

        left_wrist = teleop_grp["left_wrist"][...].astype(np.float64)
        right_wrist = teleop_grp["right_wrist"][...].astype(np.float64)
        torso_pose = teleop_grp["torso_pose"][...].astype(np.float64)
        timestamps = teleop_grp["timestamps"][...].astype(np.float64)

        if "finger_joints" not in teleop_grp:
            raise RuntimeError(
                f"{path}: missing teleop/finger_joints group — converter requires fingers."
            )
        fj_grp = teleop_grp["finger_joints"]
        finger_left = fj_grp["left"][...].astype(np.float64)
        finger_right = fj_grp["right"][...].astype(np.float64)
        finger_left_names = [_decode(n) for n in fj_grp["left_finger_joint_names"][...]]
        finger_right_names = [_decode(n) for n in fj_grp["right_finger_joint_names"][...]]

        step_dt = float(teleop_grp.attrs.get("step_dt", 0.0))

        # Optional reference-motion fields (motion-tracking tasks only). Used to populate
        # the auxiliary `motion.reference_qpos` parquet column, which lets eval and
        # offline tokenization bypass planner_sonic.onnx and feed the encoder's
        # lower-body lookahead from the kinematic intent the WBC was actually tracking.
        ref_root_pos_w = (
            obs_grp["ref_root_pos_w"][...].astype(np.float64)
            if "ref_root_pos_w" in obs_grp else None
        )
        ref_root_quat_w = (
            obs_grp["ref_root_quat_w"][...].astype(np.float64)
            if "ref_root_quat_w" in obs_grp else None
        )
        ref_dof_pos = (
            obs_grp["ref_dof_pos"][...].astype(np.float64)
            if "ref_dof_pos" in obs_grp else None
        )

    return {
        "env_args": env_args,
        "joint_pos": joint_pos,
        "root_pos_w": root_pos_w,
        "root_quat_w": root_quat_w,
        "ego_view": ego_view,
        "left_wrist": left_wrist,
        "right_wrist": right_wrist,
        "torso_pose": torso_pose,
        "timestamps": timestamps,
        "finger_left": finger_left,
        "finger_right": finger_right,
        "finger_left_names": finger_left_names,
        "finger_right_names": finger_right_names,
        "step_dt": step_dt,
        "object_pos_w": object_pos_w,
        "object_quat_w": object_quat_w,
        "ref_root_pos_w": ref_root_pos_w,
        "ref_root_quat_w": ref_root_quat_w,
        "ref_dof_pos": ref_dof_pos,
    }


def _build_canonical_finger_perm(
    hdf5_names: list[str], canonical_names: list[str], side: str
) -> np.ndarray:
    missing = [n for n in canonical_names if n not in hdf5_names]
    if missing:
        raise RuntimeError(
            f"HDF5 {side} finger joint names are missing canonical entries: "
            f"{missing}\nHDF5 names: {hdf5_names}\nCanonical: {canonical_names}"
        )
    return np.array([hdf5_names.index(n) for n in canonical_names], dtype=np.int64)


def _wrist_joint_slot_indices(
    robot_model, side: str
) -> tuple[int, int, int]:
    """Return ``(roll_idx, pitch_idx, yaw_idx)`` in gear_sonic full-q space."""
    return (
        robot_model.dof_index(f"{side}_wrist_roll_joint"),
        robot_model.dof_index(f"{side}_wrist_pitch_joint"),
        robot_model.dof_index(f"{side}_wrist_yaw_joint"),
    )


# =========================================================================
# Motion-token population (encoder pass).
#
# Mirrors the VALIDATED reference-bypass G1-mode tokenization in
# WBCBenchmark/.../eval_parquet_sonic.py (NOT the older teleop-mode sketch in
# PARQUET_POPULATE_PLAN.md). The encoder (model_encoder.onnx) consumes the 29-joint
# full-body FUTURE trajectory in G1 mode (encoder_mode=0), read from the kinematic
# reference, plus a per-frame anchor orientation, and emits the 64-D motion token
# the SONIC VLA fine-tuning supervises against. Joint-order constants are copied
# verbatim from vla_sonic.action_assembler / eval_parquet_sonic.py so this stays
# byte-compatible with the eval path without a cross-repo import.
# =========================================================================

# "SONIC-IsaacLab order in MuJoCo index" (action_assembler.ISAACLAB_TO_MUJOCO): for
# each IsaacLab/SONIC-interleaved slot, the MuJoCo-grouped index to read from.
_ISAACLAB_TO_MUJOCO = np.array([
    0,  6, 12,  1,  7, 13,  2,  8, 14,  3,  9, 15, 22,  4, 10,
    16, 23,  5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
], dtype=np.int64)

# The 29 body joints (no fingers) within gear_sonic's 43-joint layout
# (eval_parquet_sonic.BODY_INDICES_IN_GEAR_SONIC): legs+waist+left_arm = 0..21,
# right_arm = 29..35; skips left_hand 22-28 and right_hand 36-42.
_BODY_INDICES_IN_GEAR_SONIC = np.concatenate(
    [np.arange(0, 22), np.arange(29, 36)]
).astype(np.int64)
assert _BODY_INDICES_IN_GEAR_SONIC.shape[0] == 29

# G1-mode encoder lookahead: 10 frames at 50 Hz stride-5 = 100 ms apart, 0.9 s horizon
# (matches eval_parquet_sonic.py's default --bypass-stride-hz 50).
_TOKEN_LOOKAHEAD_IDX = np.arange(0, 50, 5, dtype=np.int64)  # [0, 5, ..., 45]
_TOKEN_FRAME_RATE_HZ = 50.0

# Encoder obs layout (1762 dims), mirrors vla_sonic.planner_to_utm.ENCODER_LAYOUT.
_ENC_LAYOUT = [
    ("encoder_mode_4", 4),
    ("motion_joint_positions_10frame_step5", 290),
    ("motion_joint_velocities_10frame_step5", 290),
    ("motion_root_z_position_10frame_step5", 10),
    ("motion_root_z_position", 1),
    ("motion_anchor_orientation", 6),
    ("motion_anchor_orientation_10frame_step5", 60),
    ("motion_joint_positions_lowerbody_10frame_step5", 120),
    ("motion_joint_velocities_lowerbody_10frame_step5", 120),
    ("vr_3point_local_target", 9),
    ("vr_3point_local_orn_target", 12),
    ("smpl_joints_10frame_step1", 720),
    ("smpl_anchor_orientation_10frame_step1", 60),
    ("motion_joint_positions_wrists_10frame_step1", 60),
]
_ENC_SLICES: dict[str, slice] = {}
_cursor = 0
for _name, _dim in _ENC_LAYOUT:
    _ENC_SLICES[_name] = slice(_cursor, _cursor + _dim)
    _cursor += _dim
_ENC_TOTAL_DIM = _cursor
assert _ENC_TOTAL_DIM == 1762, f"encoder layout mismatch: {_ENC_TOTAL_DIM}"


def _quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _build_g1_encoder_obs(
    body_pos_future: np.ndarray,       # (10, 29) IsaacLab-interleaved order
    body_vel_future: np.ndarray,       # (10, 29)
    anchor_rot6d_future: np.ndarray,   # (10, 6) row-major flatten of mat[:, :2]
) -> np.ndarray:
    """Assemble the (1, 1762) G1-mode (encoder_mode=0) encoder input.

    Mirrors vla_sonic.planner_to_utm.build_g1_encoder_obs — only the three G1-mode
    slots are non-zero; the encoder's all-zero mode flag selects the robot-motion
    subnetwork that reads them.
    """
    buf = np.zeros(_ENC_TOTAL_DIM, dtype=np.float32)
    # encoder_mode_4 stays [0, 0, 0, 0] → G1 mode.
    buf[_ENC_SLICES["motion_joint_positions_10frame_step5"]] = body_pos_future.reshape(-1)
    buf[_ENC_SLICES["motion_joint_velocities_10frame_step5"]] = body_vel_future.reshape(-1)
    buf[_ENC_SLICES["motion_anchor_orientation_10frame_step5"]] = anchor_rot6d_future.reshape(-1)
    return buf[None, :]


def _compute_episode_motion_tokens(
    encoder_session,
    enc_input_name: str,
    ref_qpos_all: np.ndarray,             # (N, 7 + n_joints) reference qpos, gear_sonic joint order
    executed_root_quat_wxyz: np.ndarray,  # (N, 4) executed robot root quat per frame, wxyz
    n_frames: int,
) -> np.ndarray:
    """Run the encoder per frame → (n_frames, 64) motion tokens.

    Reproduces eval_parquet_sonic.py's reference-bypass G1-mode path exactly:
      - source joints = motion.reference_qpos[:, 7:] (kinematic intent, gear_sonic order)
      - 50 Hz stride-5 lookahead window (100 ms/frame, 0.9 s horizon)
      - velocities by central difference at native 50 Hz, then windowed
      - 29 body joints (BODY_INDICES_IN_GEAR_SONIC) → IsaacLab order (ISAACLAB_TO_MUJOCO)
      - per-frame anchor rot6d = (R_executed_now)^-1 · R_reference(t+k), row-major
        flatten of the first two matrix columns.
    """
    n_total = ref_qpos_all.shape[0]
    src_pos_full = ref_qpos_all[:, 7:].astype(np.float32)        # (N, n_joints) gear_sonic order
    if src_pos_full.shape[1] <= int(_BODY_INDICES_IN_GEAR_SONIC.max()):
        raise ValueError(
            f"reference_qpos carries {src_pos_full.shape[1]} joints, but the 29-body-joint "
            f"selection needs index up to {int(_BODY_INDICES_IN_GEAR_SONIC.max())} (the full "
            "gear_sonic 43-joint layout incl. fingers). Ensure robot_model.joint_names "
            "includes the hands so ref_qpos_all matches eval_parquet_sonic.py's obs_state."
        )
    src_vel_full = np.gradient(
        src_pos_full, 1.0 / _TOKEN_FRAME_RATE_HZ, axis=0
    ).astype(np.float32)
    ref_root_quat_full = ref_qpos_all[:, 3:7].astype(np.float32)  # (N, 4) wxyz (reference root)

    tokens = np.zeros((n_frames, 64), dtype=np.float64)
    for i in range(n_frames):
        idx = np.clip(i + _TOKEN_LOOKAHEAD_IDX, 0, n_total - 1)          # (10,)
        body_pos_mj = src_pos_full[idx][:, _BODY_INDICES_IN_GEAR_SONIC]  # (10, 29) MuJoCo order
        body_vel_mj = src_vel_full[idx][:, _BODY_INDICES_IN_GEAR_SONIC]
        body_pos_future = body_pos_mj[:, _ISAACLAB_TO_MUJOCO]           # (10, 29) IsaacLab order
        body_vel_future = body_vel_mj[:, _ISAACLAB_TO_MUJOCO]

        R_robot = R.from_quat(_quat_wxyz_to_xyzw(executed_root_quat_wxyz[i]))
        ref_quat_window = ref_root_quat_full[idx]                       # (10, 4)
        anchor_rot6d_future = np.zeros((10, 6), dtype=np.float32)
        for k in range(10):
            R_anchor_k = R.from_quat(_quat_wxyz_to_xyzw(ref_quat_window[k]))
            R_rel = (R_robot.inv() * R_anchor_k).as_matrix().astype(np.float32)
            anchor_rot6d_future[k] = R_rel[:, :2].flatten("C")

        enc_obs = _build_g1_encoder_obs(body_pos_future, body_vel_future, anchor_rot6d_future)
        out = encoder_session.run(None, {enc_input_name: enc_obs})[0]
        tokens[i] = np.asarray(out, dtype=np.float64).reshape(-1)[:64]
    return tokens


def convert_one_rollout(
    hdf5_path: Path,
    exporter: Gr00tDataExporter,
    robot_model,
    body_perm: np.ndarray,
    body_missing: list[str],
    canonical_left_finger_names: list[str],
    canonical_right_finger_names: list[str],
    left_wrist_slots: tuple[int, int, int],
    right_wrist_slots: tuple[int, int, int],
    expected_left_body_name: str,
    expected_right_body_name: str,
    rollout_index: int,
    max_frames: int | None = None,
    encoder_session=None,
    enc_input_name: str | None = None,
) -> int:
    print(f"[rollout {rollout_index}] loading {hdf5_path.name}")
    raw = _read_v2_hdf5(hdf5_path)

    n_frames = raw["joint_pos"].shape[0]
    if max_frames is not None and max_frames > 0:
        n_frames = min(n_frames, max_frames)

    # Sanity: HDF5 wrist body names must match what gear_sonic's pinocchio uses.
    with h5py.File(hdf5_path, "r") as _f:
        teleop_attrs = dict(_f[TELEOP_GROUP_PATH].attrs)
    actual_left = _decode(teleop_attrs.get("left_body_name", ""))
    actual_right = _decode(teleop_attrs.get("right_body_name", ""))
    if actual_left != expected_left_body_name:
        raise RuntimeError(
            f"{hdf5_path}: left_body_name attr {actual_left!r} != expected "
            f"{expected_left_body_name!r}"
        )
    if actual_right != expected_right_body_name:
        raise RuntimeError(
            f"{hdf5_path}: right_body_name attr {actual_right!r} != expected "
            f"{expected_right_body_name!r}"
        )

    # Pre-compute permutations & per-rollout fields.
    left_finger_perm = _build_canonical_finger_perm(
        raw["finger_left_names"], canonical_left_finger_names, "left"
    )
    right_finger_perm = _build_canonical_finger_perm(
        raw["finger_right_names"], canonical_right_finger_names, "right"
    )

    q_gs_all = _permute_with_zero_fill(raw["joint_pos"], body_perm)

    # Build reference-motion → gear_sonic permutation. The motion library typically
    # exposes 27 DoFs (no waist_roll/pitch); use name-based mapping with zero-fill so
    # the resulting 29-DoF vector aligns with observation.state.
    ref_qpos_all: np.ndarray | None = None
    if (
        raw.get("ref_root_pos_w") is not None
        and raw.get("ref_root_quat_w") is not None
        and raw.get("ref_dof_pos") is not None
        and raw["env_args"].get("ref_joint_names")
    ):
        ref_joint_names = list(raw["env_args"]["ref_joint_names"])
        ref_perm, ref_missing = _build_joint_permutation(
            ref_joint_names, robot_model.joint_names
        )
        ref_dof_gs = _permute_with_zero_fill(raw["ref_dof_pos"], ref_perm)  # (T, 29) gear_sonic order
        ref_qpos_all = np.concatenate(
            [
                raw["ref_root_pos_w"].astype(np.float32),
                raw["ref_root_quat_w"].astype(np.float32),
                ref_dof_gs.astype(np.float32),
            ],
            axis=1,
        )  # (T, 36) — [root_pos(3), root_quat_wxyz(4), joints_sonic_order(29)]
        if ref_missing and rollout_index == 0:
            print(
                f"[info] reference motion: {len(ref_missing)} gear_sonic joints absent "
                f"from motion_lib URDF — zero-filled in motion.reference_qpos: {ref_missing}"
            )
    elif rollout_index == 0:
        print(
            "[info] no reference-motion fields in HDF5 — motion.reference_qpos column "
            "will be zero-filled (no planner-bypass available downstream)."
        )

    # Motion-token population (optional; requires --encoder-onnx). Computes the
    # whole episode's tokens up front from the reference trajectory, exactly as
    # eval_parquet_sonic.py's reference-bypass G1 mode does. Falls back to
    # zero-fill (old behavior) when no encoder is provided or no reference exists.
    motion_tokens: np.ndarray | None = None
    if encoder_session is not None:
        if ref_qpos_all is not None:
            motion_tokens = _compute_episode_motion_tokens(
                encoder_session, enc_input_name, ref_qpos_all,
                raw["root_quat_w"], n_frames,
            )
            if rollout_index == 0:
                _tmax = float(np.abs(motion_tokens).max())
                print(f"[token] populated action.motion_token via encoder (G1 mode, "
                      f"reference source): {motion_tokens.shape[0]} frames, |token| max={_tmax:.4f}")
                if _tmax < 1e-6:
                    print("[token][WARN] computed tokens are ~all-zero — check the encoder ONNX "
                          "and that motion.reference_qpos is populated (non-trivial).")
        elif rollout_index == 0:
            print("[token][WARN] --encoder-onnx given but this HDF5 has no reference motion "
                  "(motion.reference_qpos) — action.motion_token stays zero-filled.")

    # Frame zero init base quat for the whole episode.
    init_base_quat = raw["root_quat_w"][0].astype(np.float64).copy()

    # Constant zero-fill arrays we can reuse per frame.
    zero_motion_token = np.zeros(64, dtype=np.float64)
    zero_smpl_joints = np.zeros(72, dtype=np.float32)
    zero_smpl_pose = np.zeros(63, dtype=np.float32)
    identity_body_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    identity_target_rot6d = quat_to_rot6d(identity_body_quat).astype(np.float32)
    identity_cpp_offset = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    zero_smpl_frame_idx = np.array([0], dtype=np.int64)
    zero_delta_heading = np.zeros(1, dtype=np.float64)

    # Per-frame planner commands derived from recorded root motion, matching
    # the kinematic planner ONNX input semantics (see gear_sonic_deploy's
    # localmotion_kplanner.hpp for the mode-speed range table).
    step_dt = raw["step_dt"] if raw["step_dt"] > 0.0 else 1.0 / 50.0
    root_pos = raw["root_pos_w"][:n_frames].astype(np.float64)
    root_quat_wxyz = raw["root_quat_w"][:n_frames].astype(np.float64)

    # 3D root velocity via central differences (world frame, MuJoCo Z-up).
    _vel_all = np.gradient(root_pos, step_dt, axis=0).astype(np.float32)

    # Horizontal speed magnitude (derived from velocity, before normalization).
    # Zeroed below the IDLE threshold (0.1 m/s) so target_vel=0 for IDLE frames —
    # passing a small positive target_vel with mode=IDLE is incoherent.
    _raw_speed = np.linalg.norm(_vel_all[:, :2], axis=1, keepdims=True).astype(np.float32)
    planner_speed_all = np.where(_raw_speed >= 0.1, _raw_speed, 0.0).astype(np.float32)

    # Forward unit vector: world-frame direction the torso faces, derived by
    # rotating [1, 0, 0] (body forward) by the root quaternion.
    root_quat_xyzw = np.stack(
        [
            root_quat_wxyz[:, 1],
            root_quat_wxyz[:, 2],
            root_quat_wxyz[:, 3],
            root_quat_wxyz[:, 0],
        ],
        axis=1,
    )
    planner_facing_all = (
        R.from_quat(root_quat_xyzw).apply(np.array([1.0, 0.0, 0.0])).astype(np.float32)
    )

    # Movement direction = facing direction (body forward, Z zeroed and renormalized).
    # The kinematic planner only generates forward locomotion — it cannot strafe.
    # Deriving movement from root velocity would capture physics drift and produce a
    # direction the planner can't act on.  Using facing ensures both inputs are
    # self-consistent and the planner walks in the direction the body was oriented.
    _moving_mask = _raw_speed.reshape(-1) >= 0.1
    _facing_xy = planner_facing_all[:, :2].copy()
    _facing_xy_norm = np.linalg.norm(_facing_xy, axis=1, keepdims=True)
    _facing_xy_unit = np.where(_facing_xy_norm > 1e-6, _facing_xy / _facing_xy_norm, np.array([[1.0, 0.0]]))
    planner_movement_all = np.zeros((n_frames, 3), dtype=np.float32)
    planner_movement_all[_moving_mask, :2] = _facing_xy_unit[_moving_mask]

    # Height sentinel: -1.0 means "use mode default" for all standard modes.
    # From localmotion_kplanner.hpp: height=-1 for IDLE/SLOW_WALK/WALK/RUN (all pick-task modes).
    planner_height_all = np.full((n_frames, 1), -1.0, dtype=np.float32)

    # Mode from horizontal speed: 0=IDLE, 1=SLOW_WALK, 2=WALK, 3=RUN.
    def _speed_to_mode(s: float) -> int:
        s = abs(float(s))
        if s < 0.1:
            return 0
        if s < 0.8:
            return 1
        if s < 2.5:
            return 2
        return 3

    planner_mode_all = np.array(
        [[_speed_to_mode(float(s))] for s in planner_speed_all.flatten()],
        dtype=np.int32,
    )

    # Object pose arrays — None when the HDF5 predates object state recording.
    _obj_pos_src = raw.get("object_pos_w")
    _obj_quat_src = raw.get("object_quat_w")
    _has_object = _obj_pos_src is not None and _obj_quat_src is not None
    if _has_object:
        object_pos_all  = _obj_pos_src[:n_frames].astype(np.float32)
        object_quat_all = _obj_quat_src[:n_frames].astype(np.float32)
    else:
        object_pos_all  = np.zeros((n_frames, 3), dtype=np.float32)
        object_quat_all = np.tile(np.array([1., 0., 0., 0.], dtype=np.float32), (n_frames, 1))

    stream_mode = np.array([5], dtype=np.int32)

    written = 0
    for i in range(n_frames):
        q_gs = q_gs_all[i]

        lw = raw["left_wrist"][i]
        rw = raw["right_wrist"][i]
        tp = raw["torso_pose"][i]

        lw_pos = lw[:3, 3]
        rw_pos = rw[:3, 3]
        lw_quat_wxyz = _mat_to_quat_wxyz(lw[:3, :3])
        rw_quat_wxyz = _mat_to_quat_wxyz(rw[:3, :3])
        observation_eef_state = np.concatenate(
            [lw_pos, lw_quat_wxyz, rw_pos, rw_quat_wxyz], dtype=np.float64
        )

        root_quat = raw["root_quat_w"][i].astype(np.float64)
        projected_gravity = compute_projected_gravity(root_quat).astype(np.float64)

        vr_3pt_position = np.concatenate(
            [
                _apply_local_offset(lw, LW_LOCAL_OFFSET),
                _apply_local_offset(rw, RW_LOCAL_OFFSET),
                _apply_local_offset(tp, TORSO_LOCAL_OFFSET),
            ],
            dtype=np.float32,
        )
        vr_3pt_orientation = np.concatenate(
            [
                quat_to_rot6d(_mat_to_quat_wxyz(lw[:3, :3]).astype(np.float32)),
                quat_to_rot6d(_mat_to_quat_wxyz(rw[:3, :3]).astype(np.float32)),
                quat_to_rot6d(_mat_to_quat_wxyz(tp[:3, :3]).astype(np.float32)),
            ],
            dtype=np.float32,
        )

        left_finger_canonical = raw["finger_left"][i, left_finger_perm].astype(np.float32)
        right_finger_canonical = raw["finger_right"][i, right_finger_perm].astype(np.float32)

        left_wrist_joints = np.array(
            [q_gs[idx] for idx in left_wrist_slots], dtype=np.float32
        )
        right_wrist_joints = np.array(
            [q_gs[idx] for idx in right_wrist_slots], dtype=np.float32
        )

        ego_image = _resize_ego_view(raw["ego_view"][i])

        frame_data = {
            "observation.images.ego_view": ego_image,
            "observation.state": q_gs,
            "observation.eef_state": observation_eef_state,
            "action.wbc": q_gs.copy(),
            "observation.root_orientation": root_quat,
            "observation.projected_gravity": projected_gravity,
            "observation.cpp_rotation_offset": identity_cpp_offset.copy(),
            "observation.init_base_quat": init_base_quat.copy(),
            "teleop.delta_heading": zero_delta_heading.copy(),
            "action.motion_token": (
                motion_tokens[i] if motion_tokens is not None else zero_motion_token.copy()
            ),
            "teleop.smpl_joints": zero_smpl_joints.copy(),
            "teleop.smpl_pose": zero_smpl_pose.copy(),
            "teleop.body_quat_w": identity_body_quat.copy(),
            "teleop.target_body_orientation": identity_target_rot6d.copy(),
            "teleop.left_hand_joints": left_finger_canonical,
            "teleop.right_hand_joints": right_finger_canonical,
            "teleop.smpl_frame_index": zero_smpl_frame_idx.copy(),
            "teleop.left_wrist_joints": left_wrist_joints,
            "teleop.right_wrist_joints": right_wrist_joints,
            "teleop.stream_mode": stream_mode.copy(),
            "teleop.planner_mode": planner_mode_all[i].copy(),
            "teleop.planner_movement": planner_movement_all[i].copy(),
            "teleop.planner_facing": planner_facing_all[i].copy(),
            "teleop.planner_speed": planner_speed_all[i].copy(),
            "teleop.planner_height": planner_height_all[i].copy(),
            "teleop.root_pos_w": root_pos[i].astype(np.float32),
            "teleop.object_pos_w": object_pos_all[i].copy(),
            "teleop.object_quat_w": object_quat_all[i].copy(),
            "teleop.vr_3pt_position": vr_3pt_position,
            "teleop.vr_3pt_orientation": vr_3pt_orientation,
        }

        # Auxiliary planner-bypass column. Layout: [root_pos(3), root_quat_wxyz(4),
        # joints_in_gear_sonic_order(num_joints)] floats per frame. Eval and the offline
        # motion-token populator read this to feed the encoder's lower-body lookahead
        # without invoking planner_sonic.onnx. The populator deletes this column after
        # writing action.motion_token, so the final dataset preserves SONIC schema.
        if ref_qpos_all is not None:
            frame_data["motion.reference_qpos"] = ref_qpos_all[i]
        else:
            frame_data["motion.reference_qpos"] = np.zeros(
                7 + len(robot_model.joint_names), dtype=np.float32
            )

        exporter.add_frame(frame_data)
        written += 1

    exporter.save_episode()
    print(f"[rollout {rollout_index}] wrote {written} frames -> episode saved")
    if body_missing and rollout_index == 0:
        print(
            f"[info] {len(body_missing)} gear_sonic joints have no Isaac Lab "
            f"counterpart and are zero-filled in observation.state / action.wbc: "
            f"{body_missing}"
        )
    return written


def _list_rollouts(root: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.hdf5" if recursive else "*.hdf5"
    files = sorted(root.glob(pattern))
    if not files:
        raise RuntimeError(f"No .hdf5 files found under {root} (recursive={recursive})")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Part A v2 Isaac Lab rollout HDF5 files to a LeRobot v2.1 "
            "Sonic VLA dataset."
        )
    )
    parser.add_argument(
        "--hdf5-root",
        type=Path,
        required=True,
        help="Directory containing producer rollout .hdf5 files.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Output dataset root directory.",
    )
    parser.add_argument(
        "--task-prompt",
        type=str,
        required=True,
        help='Natural-language task description, e.g. "pick up the mustard bottle".',
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Optional dataset name; defaults to the output-path basename.",
    )
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument(
        "--waist-location",
        choices=["lower_body", "upper_body", "lower_and_upper_body"],
        default="lower_body",
    )
    parser.add_argument(
        "--high-elbow",
        action="store_true",
        help="Use the high elbow default joint pose for the gear_sonic robot model.",
    )
    parser.add_argument(
        "--rollout-limit",
        type=int,
        default=-1,
        help="Process at most this many rollout files. -1 = all.",
    )
    parser.add_argument(
        "--max-frames-per-rollout",
        type=int,
        default=-1,
        help="Process at most this many frames per rollout. -1 = all.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Glob .hdf5 files recursively under --hdf5-root.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Wipe --output-path before writing.",
    )
    parser.add_argument(
        "--encoder-onnx",
        type=Path,
        default=_DEFAULT_ENCODER_ONNX,
        help="Path to model_encoder.onnx used to POPULATE action.motion_token by running "
             "the encoder on the reference motion (G1 mode, mirrors eval_parquet_sonic.py). "
             f"Defaults to the canonical release path ({_DEFAULT_ENCODER_ONNX}), so you "
             "normally don't pass it. If the file is absent (or --zero-fill-tokens is set) "
             "the column is zero-filled and a VLA trained on it will emit null tokens. "
             "Population also needs the HDF5's reference-motion fields "
             "(ref_root_pos_w / ref_root_quat_w / ref_dof_pos).",
    )
    parser.add_argument(
        "--zero-fill-tokens",
        action="store_true",
        help="Force action.motion_token to be zero-filled even if an encoder is available "
             "(skip tokenization). Use only for schema-debugging; not trainable.",
    )
    args = parser.parse_args()

    rollouts = _list_rollouts(args.hdf5_root, args.recursive)
    if args.rollout_limit >= 0:
        rollouts = rollouts[: args.rollout_limit]
    print(f"[info] found {len(rollouts)} rollout file(s)")

    # Encoder for motion-token population. Defaults to the canonical release path, so
    # tokens are populated automatically when model_encoder.onnx is present. Absent file
    # or --zero-fill-tokens → graceful zero-fill (old behavior), never a hard crash.
    encoder_session = None
    enc_input_name: str | None = None
    if args.zero_fill_tokens:
        print("[token] --zero-fill-tokens set → action.motion_token ZERO-FILLED (not trainable).")
    elif args.encoder_onnx is not None and args.encoder_onnx.exists():
        import onnxruntime as ort  # noqa: PLC0415

        _avail = ort.get_available_providers()
        _providers = [
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in _avail
        ] or _avail
        encoder_session = ort.InferenceSession(str(args.encoder_onnx), providers=_providers)
        enc_input_name = encoder_session.get_inputs()[0].name
        print(f"[token] encoder ONNX loaded: {args.encoder_onnx} "
              f"(input '{enc_input_name}', providers={_providers}) — action.motion_token "
              "will be POPULATED via the G1-mode reference-bypass encoder pass.")
    else:
        print(f"[token][WARN] encoder ONNX not found at {args.encoder_onnx} → action.motion_token "
              "will be ZERO-FILLED. A VLA fine-tuned on this dataset will learn to emit null "
              "tokens. Pass --encoder-onnx <model_encoder.onnx> or place it at the default path "
              "to populate real tokens.")

    robot_model = get_g1_robot_model(
        waist_location=args.waist_location,
        high_elbow_pose=args.high_elbow,
    )
    expected_left_body = robot_model.supplemental_info.hand_frame_names["left"]
    expected_right_body = robot_model.supplemental_info.hand_frame_names["right"]
    canonical_left_finger = robot_model.supplemental_info.left_hand_actuated_joints
    canonical_right_finger = robot_model.supplemental_info.right_hand_actuated_joints
    left_wrist_slots = _wrist_joint_slot_indices(robot_model, "left")
    right_wrist_slots = _wrist_joint_slot_indices(robot_model, "right")

    # Use the first rollout's env_args to build the body-joint permutation.
    first_raw = _read_v2_hdf5(rollouts[0])
    isaac_joint_names = list(first_raw["env_args"]["joint_names"])
    body_perm, body_missing = _build_joint_permutation(
        isaac_joint_names, robot_model.joint_names
    )
    print(
        f"[info] gear_sonic joints: {len(robot_model.joint_names)}, "
        f"isaac joints: {len(isaac_joint_names)}, "
        f"missing-on-isaac (zero-filled): {len(body_missing)}"
    )

    features = get_features_sonic_vla(robot_model)
    # Auxiliary column carrying the motion-library kinematic reference per frame.
    # Layout: [root_pos(3), root_quat_wxyz(4), joints_in_gear_sonic_order(num_joints)].
    # num_joints matches the full robot_model joint count (body + fingers) so the slice
    # convention is identical to observation.state — eval reads [7:7+29] for body and
    # ignores the finger tail. NOT part of the official SONIC schema —
    # `parquet_populate.py` deletes this column after consuming it to populate
    # action.motion_token, restoring strict SONIC layout.
    _ref_qpos_num_joints = len(robot_model.joint_names)
    features["motion.reference_qpos"] = {
        "dtype": "float32",
        "shape": (7 + _ref_qpos_num_joints,),
        "names": ["root_x", "root_y", "root_z",
                  "root_qw", "root_qx", "root_qy", "root_qz"]
                 + [f"joint_{i}" for i in range(_ref_qpos_num_joints)],
    }
    modality_config = get_modality_config_sonic_vla(robot_model)

    script_config = {
        "source": "isaac_lab_hdf5_converted",
        "source_schema": "producer HDF5 v2 (Part A v2, robomimic-layout)",
        "converter": "convert_isaac_hdf5_to_lerobot.py",
        "converter_version": 2,
        "planner_fields_source": (
            "derived from root_pos_w / root_quat_w in the source HDF5: "
            "planner_movement = world-frame root velocity via np.gradient; "
            "planner_speed = horizontal speed; planner_facing = R(root_quat) @ [1,0,0]; "
            "planner_height = root_pos_w[:, 2]; "
            "planner_mode via speed_to_mode thresholds from "
            "gear_sonic_deploy/.../localmotion_kplanner.hpp"
        ),
        "producer": first_raw["env_args"].get("producer", "unknown"),
        "producer_version": first_raw["env_args"].get("producer_version", 0),
        "task_name": first_raw["env_args"].get("task_name"),
        "robot_name": first_raw["env_args"].get("robot_name"),
        "robot_usd": first_raw["env_args"].get("robot_usd"),
        "fps": args.fps,
        "waist_location": args.waist_location,
        "high_elbow_pose": bool(args.high_elbow),
        "isaac_joint_names": isaac_joint_names,
        "gear_sonic_joint_names": list(robot_model.joint_names),
        "joints_zero_filled_in_state": body_missing,
        "zero_filled_fields": (
            [f for f in ZERO_FILLED_FIELDS if f != "action.motion_token"]
            if encoder_session is not None else ZERO_FILLED_FIELDS
        ),
        "motion_token_source": (
            "encoder model_encoder.onnx (G1 mode, reference-bypass; mirrors "
            "eval_parquet_sonic.py reference path)"
            if encoder_session is not None else "ZERO-FILLED (no --encoder-onnx)"
        ),
        "action_wbc_source": "robot0_joint_pos via name-based permutation",
        "notes": (
            "VLA fine-tuning primary targets per the SONIC paper are the "
            "teleop-format commands (vr_3pt_position, vr_3pt_orientation, "
            "planner_*, hand_joints). Zero-filled fields are auxiliary/token-"
            "space signals only. action.wbc is sourced from joint_pos because "
            "Isaac Lab collection rollouts do not store a separate WBC command."
        ),
    }

    save_root = args.output_path
    if args.dataset_name and not str(save_root).endswith(args.dataset_name):
        save_root = save_root / args.dataset_name
    save_root.parent.mkdir(parents=True, exist_ok=True)

    exporter = Gr00tDataExporter.create(
        save_root=str(save_root),
        fps=args.fps,
        features=features,
        modality_config=modality_config,
        task=args.task_prompt,
        script_config=script_config,
        robot_type="g1",
        overwrite_existing=args.overwrite_existing,
    )

    total_frames = 0
    max_frames = args.max_frames_per_rollout if args.max_frames_per_rollout > 0 else None
    for i, hdf5_path in enumerate(rollouts):
        try:
            written = convert_one_rollout(
                hdf5_path=hdf5_path,
                exporter=exporter,
                robot_model=robot_model,
                body_perm=body_perm,
                body_missing=body_missing,
                canonical_left_finger_names=canonical_left_finger,
                canonical_right_finger_names=canonical_right_finger,
                left_wrist_slots=left_wrist_slots,
                right_wrist_slots=right_wrist_slots,
                expected_left_body_name=expected_left_body,
                expected_right_body_name=expected_right_body,
                rollout_index=i,
                max_frames=max_frames,
                encoder_session=encoder_session,
                enc_input_name=enc_input_name,
            )
            total_frames += written
        except Exception as exc:  # noqa: BLE001
            print(f"[error] rollout {i} ({hdf5_path}): {exc}", file=sys.stderr)
            raise

    print(
        f"[done] wrote {total_frames} frames across {len(rollouts)} episode(s) -> {save_root}"
    )
    if encoder_session is not None:
        print("[done] action.motion_token POPULATED via the encoder (G1-mode reference "
              "bypass). Validate against eval_parquet_sonic.py before training: a few rows' "
              "tokens should match the eval encoder output for the same frames.")
        _remaining_zero = [f for f in ZERO_FILLED_FIELDS if f != "action.motion_token"]
    else:
        _remaining_zero = ZERO_FILLED_FIELDS
    print(
        f"[done] zero-filled fields: {_remaining_zero}. "
        "Run `process_dataset.py --no-remove-stale-smpl` (or skip post-processing) "
        "to avoid dropping frames with zero SMPL."
    )


if __name__ == "__main__":
    main()
