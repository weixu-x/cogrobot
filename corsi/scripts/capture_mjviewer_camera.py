"""Open a robosuite Corsi scene in free-camera mode and save the final viewer camera parameters."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from corsi.envs.robosuite_corsi import create_env


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-path",
        type=str,
        default="corsi_artifacts/mjviewer_free_camera.json",
    )
    parser.add_argument("--distance", type=float, default=0.75)
    parser.add_argument("--azimuth", type=float, default=180.0)
    parser.add_argument("--elevation", type=float, default=-35.0)
    parser.add_argument("--lookat-x", type=float, default=0.0)
    parser.add_argument("--lookat-y", type=float, default=0.0)
    parser.add_argument("--lookat-z", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renderer_config = {
        "cam_config": {
            "lookat": [args.lookat_x, args.lookat_y, args.lookat_z],
            "distance": args.distance,
            "azimuth": args.azimuth,
            "elevation": args.elevation,
        }
    }

    env = create_env(
        render_mode="online",
        online_render_camera=None,
        control_freq=20,
        renderer_config=renderer_config,
    )

    try:
        env.reset()
        print("MuJoCo viewer opened in free-camera mode.")
        print("Use the mouse / trackpad in the viewer to rotate, pan, and zoom.")
        print("Close the viewer window when the camera looks right.")

        while True:
            if env.viewer is None:
                raise RuntimeError("Viewer wrapper was not initialized")
            env.viewer.update()
            viewer_handle = getattr(env.viewer, "viewer", None)
            if viewer_handle is not None and hasattr(viewer_handle, "is_running"):
                if not viewer_handle.is_running():
                    break
            time.sleep(0.02)

        viewer_handle = getattr(env.viewer, "viewer", None)
        if viewer_handle is None:
            raise RuntimeError("Viewer handle is missing; could not capture camera parameters")

        cam = viewer_handle.cam
        result = {
            "lookat": [round(float(value), 6) for value in cam.lookat],
            "distance": round(float(cam.distance), 6),
            "azimuth": round(float(cam.azimuth), 6),
            "elevation": round(float(cam.elevation), 6),
        }

        output_path = REPO_ROOT / args.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

        print("Saved viewer camera parameters:")
        print(json.dumps(result, indent=2))
        print(f"Saved to {output_path}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
