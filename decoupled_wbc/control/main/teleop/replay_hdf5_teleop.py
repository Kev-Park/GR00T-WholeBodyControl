"""Dry-run replay of a producer HDF5 rollout through the teleop retargeting IK.

Reads ``data/demo_0/teleop/left_wrist`` and ``data/demo_0/teleop/right_wrist``
(pelvis-frame 4x4 matrices) from a producer HDF5 file (Part A v2,
robomimic-style layout, ``schema_version=2``) and injects them directly into
``TeleopRetargetingIK``'s ``set_goal`` API, bypassing the wrist pre-processor
and the streamer abstraction. Per-frame upper-body joint targets are collected
and optionally written to an ``.npy`` file for inspection.

With ``--overlay-fingers``, recorded finger joint angles from
``data/demo_0/teleop/finger_joints`` are used to overwrite the hand slots of
each target, bypassing the hand IK solver. Requires that the HDF5 joint names
are a superset of the consumer's canonical Dex3 joint names (confirmed CASE 1
during planning).

This script is intentionally minimal: no sim env, no ROS, no WBC policy. It
isolates the IK injection path as a first-order correctness check on a new
producer HDF5. Sim playback and WBC-level verification are a follow-up.

Example:
    python -m decoupled_wbc.control.main.teleop.replay_hdf5_teleop \\
        --hdf5-path datasets/debug_teleop/2026-04-13/mustard_bottle__...hdf5 \\
        --overlay-fingers \\
        --save-targets /tmp/targets.npy
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np

from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from decoupled_wbc.control.teleop.solver.hand.instantiation.g1_hand_ik_instantiation import (
    instantiate_g1_hand_ik_solver,
)
from decoupled_wbc.control.teleop.teleop_retargeting_ik import TeleopRetargetingIK

SCHEMA_VERSION = 2
TELEOP_GROUP_PATH = "data/demo_0/teleop"


@dataclass
class HDF5TeleopFrames:
    left_wrist: np.ndarray  # (N, 4, 4) float64, pelvis frame
    right_wrist: np.ndarray  # (N, 4, 4) float64, pelvis frame
    left_body_name: str
    right_body_name: str
    finger_left: Optional[np.ndarray]  # (N, K_L) float64 or None
    finger_right: Optional[np.ndarray]  # (N, K_R) float64 or None
    finger_left_names: Optional[List[str]]
    finger_right_names: Optional[List[str]]
    step_dt: float


def _decode_attr(value: object, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def load_hdf5(path: Path) -> HDF5TeleopFrames:
    with h5py.File(path, "r") as f:
        if TELEOP_GROUP_PATH not in f:
            raise RuntimeError(
                f"{path}: HDF5 has no /{TELEOP_GROUP_PATH} group "
                f"(expected Part A v2 robomimic-style layout). "
                f"If this file was produced with Part A v1, regenerate with the v2 producer."
            )
        g = f[TELEOP_GROUP_PATH]

        schema = int(g.attrs.get("schema_version", 0))
        if schema != SCHEMA_VERSION:
            raise RuntimeError(
                f"{path}: teleop schema_version={schema}, expected {SCHEMA_VERSION}"
            )

        frame_tag = _decode_attr(g.attrs.get("frame"))
        if frame_tag != "pelvis":
            raise RuntimeError(
                f"{path}: teleop frame attr is {frame_tag!r}, expected 'pelvis'"
            )

        left_wrist = g["left_wrist"][...].astype(np.float64)
        right_wrist = g["right_wrist"][...].astype(np.float64)
        if left_wrist.ndim != 3 or left_wrist.shape[1:] != (4, 4):
            raise RuntimeError(
                f"{path}: left_wrist shape {left_wrist.shape}, expected (N,4,4)"
            )
        if right_wrist.shape != left_wrist.shape:
            raise RuntimeError(
                f"{path}: left/right wrist shape mismatch {left_wrist.shape} vs {right_wrist.shape}"
            )

        left_body_name = _decode_attr(
            g.attrs.get("left_body_name"), "left_wrist_yaw_link"
        )
        right_body_name = _decode_attr(
            g.attrs.get("right_body_name"), "right_wrist_yaw_link"
        )

        finger_left: Optional[np.ndarray] = None
        finger_right: Optional[np.ndarray] = None
        finger_left_names: Optional[List[str]] = None
        finger_right_names: Optional[List[str]] = None
        if "finger_joints" in g:
            fj = g["finger_joints"]
            finger_left = fj["left"][...].astype(np.float64)
            finger_right = fj["right"][...].astype(np.float64)
            finger_left_names = [
                _decode_attr(n) for n in fj["left_finger_joint_names"][...]
            ]
            finger_right_names = [
                _decode_attr(n) for n in fj["right_finger_joint_names"][...]
            ]
            if finger_left.shape[0] != left_wrist.shape[0]:
                raise RuntimeError(
                    f"{path}: finger_joints/left has {finger_left.shape[0]} frames, "
                    f"wrist has {left_wrist.shape[0]}"
                )
            if finger_right.shape[0] != left_wrist.shape[0]:
                raise RuntimeError(
                    f"{path}: finger_joints/right has {finger_right.shape[0]} frames, "
                    f"wrist has {left_wrist.shape[0]}"
                )

        step_dt = float(g.attrs.get("step_dt", 0.0))

    return HDF5TeleopFrames(
        left_wrist=left_wrist,
        right_wrist=right_wrist,
        left_body_name=left_body_name,
        right_body_name=right_body_name,
        finger_left=finger_left,
        finger_right=finger_right,
        finger_left_names=finger_left_names,
        finger_right_names=finger_right_names,
        step_dt=step_dt,
    )


def build_finger_permutation(
    hdf5_names: List[str], canonical_names: List[str]
) -> np.ndarray:
    missing = [n for n in canonical_names if n not in hdf5_names]
    if missing:
        raise RuntimeError(
            "HDF5 finger joint names are missing canonical entries: "
            f"{missing}\nHDF5 names: {hdf5_names}\nCanonical names: {canonical_names}"
        )
    return np.array([hdf5_names.index(n) for n in canonical_names], dtype=np.int64)


def build_upper_body_slot_map(
    upper_body_indices: List[int], hand_indices: List[int]
) -> np.ndarray:
    ub = list(upper_body_indices)
    try:
        return np.array([ub.index(i) for i in hand_indices], dtype=np.int64)
    except ValueError as exc:
        raise RuntimeError(
            "Hand actuated joint indices are not a subset of the upper_body group. "
            "Check waist_location / supplemental info."
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run HDF5 teleop replay through the retargeting IK."
    )
    parser.add_argument(
        "--hdf5-path", required=True, type=Path, help="Path to producer rollout HDF5."
    )
    parser.add_argument(
        "--overlay-fingers",
        action="store_true",
        help="Override finger slots of the IK output with recorded finger "
        "joint angles (bypasses the hand IK solver).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Process at most this many frames. -1 = all frames.",
    )
    parser.add_argument(
        "--waist-location",
        choices=["lower_body", "upper_body", "lower_and_upper_body"],
        default="lower_body",
        help="Passed to instantiate_g1_robot_model. Must match how the "
        "collection env configures the waist if you care about name-by-name "
        "upper-body alignment.",
    )
    parser.add_argument(
        "--high-elbow",
        action="store_true",
        help="Use the high elbow default joint pose (matches collection config).",
    )
    parser.add_argument(
        "--save-targets",
        type=Path,
        default=None,
        help="Optional .npy path to write the per-frame target_upper_body_pose "
        "array of shape (N, num_upper_body_dofs).",
    )
    args = parser.parse_args()

    np.set_printoptions(precision=5, suppress=True, linewidth=160)

    data = load_hdf5(args.hdf5_path)
    total_frames = data.left_wrist.shape[0]
    num_frames = total_frames if args.max_frames < 0 else min(total_frames, args.max_frames)
    print(f"[info] loaded {total_frames} frames from {args.hdf5_path}")
    print(f"[info] processing {num_frames} frames (step_dt={data.step_dt:.6f}s)")

    robot_model = instantiate_g1_robot_model(
        waist_location=args.waist_location,
        high_elbow_pose=args.high_elbow,
    )

    expected_left = robot_model.supplemental_info.hand_frame_names["left"]
    expected_right = robot_model.supplemental_info.hand_frame_names["right"]
    if data.left_body_name != expected_left:
        raise RuntimeError(
            f"HDF5 left_body_name {data.left_body_name!r} != consumer hand_frame_names['left'] {expected_left!r}"
        )
    if data.right_body_name != expected_right:
        raise RuntimeError(
            f"HDF5 right_body_name {data.right_body_name!r} != consumer hand_frame_names['right'] {expected_right!r}"
        )

    left_hand_ik_solver, right_hand_ik_solver = instantiate_g1_hand_ik_solver()
    retargeting_ik = TeleopRetargetingIK(
        robot_model=robot_model,
        left_hand_ik_solver=left_hand_ik_solver,
        right_hand_ik_solver=right_hand_ik_solver,
        enable_visualization=False,
        body_active_joint_groups=["upper_body"],
    )

    upper_body_indices = list(robot_model.get_joint_group_indices("upper_body"))
    num_upper = len(upper_body_indices)
    print(f"[info] upper_body dof count: {num_upper}")

    left_perm: Optional[np.ndarray] = None
    right_perm: Optional[np.ndarray] = None
    left_slots: Optional[np.ndarray] = None
    right_slots: Optional[np.ndarray] = None
    if args.overlay_fingers:
        if data.finger_left is None or data.finger_right is None:
            raise RuntimeError(
                "--overlay-fingers was set but HDF5 has no /teleop/finger_joints group"
            )
        canonical_left = robot_model.supplemental_info.left_hand_actuated_joints
        canonical_right = robot_model.supplemental_info.right_hand_actuated_joints
        left_perm = build_finger_permutation(data.finger_left_names, canonical_left)
        right_perm = build_finger_permutation(data.finger_right_names, canonical_right)

        left_hand_q_idx = list(robot_model.get_hand_actuated_joint_indices("left"))
        right_hand_q_idx = list(robot_model.get_hand_actuated_joint_indices("right"))
        left_slots = build_upper_body_slot_map(upper_body_indices, left_hand_q_idx)
        right_slots = build_upper_body_slot_map(upper_body_indices, right_hand_q_idx)
        print(
            f"[info] finger overlay enabled: "
            f"left_slots={left_slots.tolist()} right_slots={right_slots.tolist()}"
        )

    targets = np.zeros((num_frames, num_upper), dtype=np.float64)
    for i in range(num_frames):
        ik_data = {
            "body_data": {
                data.left_body_name: data.left_wrist[i],
                data.right_body_name: data.right_wrist[i],
            },
            "left_hand_data": None,
            "right_hand_data": None,
        }
        retargeting_ik.set_goal(ik_data)
        target_upper = np.asarray(retargeting_ik.get_action(), dtype=np.float64).copy()

        if args.overlay_fingers:
            target_upper[left_slots] = data.finger_left[i, left_perm]
            target_upper[right_slots] = data.finger_right[i, right_perm]

        targets[i] = target_upper

    # Print representative frames. Including mid-trajectory and peak-magnitude
    # frames so endpoint-only behavior (e.g. fingers near zero at start/end of a
    # reach-grasp-release rollout) isn't mistaken for "fingers never moved".
    key_frames = [
        (0, "first"),
        (num_frames // 4, "25%"),
        (num_frames // 2, "50%"),
        (3 * num_frames // 4, "75%"),
        (num_frames - 1, "last"),
    ]
    peak_idx = int(np.argmax(np.linalg.norm(targets, axis=1)))
    if peak_idx not in [f[0] for f in key_frames]:
        key_frames.append((peak_idx, "peak"))
    key_frames.sort(key=lambda f: f[0])
    for idx, tag in key_frames:
        print(f"[frame {idx:04d} {tag:>5s}] target_upper_body_pose={targets[idx]}")

    print(
        f"[stats] targets shape={targets.shape} "
        f"min={targets.min():.4f} max={targets.max():.4f} "
        f"mean={targets.mean():.4f}"
    )
    delta = targets[-1] - targets[0]
    print(
        f"[stats] |target[-1] - target[0]|_inf={np.max(np.abs(delta)):.6f} "
        f"|...|_2={np.linalg.norm(delta):.6f}"
    )

    if args.overlay_fingers:
        left_finger_trace = targets[:, left_slots]
        right_finger_trace = targets[:, right_slots]
        print(
            f"[fingers left ] ptp per joint={left_finger_trace.ptp(axis=0)} "
            f"max|.|={np.max(np.abs(left_finger_trace)):.4f}"
        )
        print(
            f"[fingers right] ptp per joint={right_finger_trace.ptp(axis=0)} "
            f"max|.|={np.max(np.abs(right_finger_trace)):.4f}"
        )

    if args.save_targets is not None:
        args.save_targets.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.save_targets, targets)
        print(f"[done] wrote {args.save_targets}")

    print("[done] dry run finished without error")


if __name__ == "__main__":
    main()
