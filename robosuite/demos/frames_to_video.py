import argparse
import glob
import os

import imageio


def main():
    parser = argparse.ArgumentParser(description="Combine image frames into a video.")
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="/home/wei2025/Developer/cogrobot/robosuite/savevideo/corsi_pointing_frames",
        help="Directory containing frame_XXXXX.png files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/wei2025/Developer/cogrobot/robosuite/savevideo/corsi_pointing.mp4",
        help="Output video path (mp4)",
    )
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    args = parser.parse_args()

    pattern = os.path.join(args.frames_dir, "frame_*.png")
    frame_paths = sorted(glob.glob(pattern))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found with pattern: {pattern}")

    writer = imageio.get_writer(args.output, fps=args.fps)
    try:
        for path in frame_paths:
            writer.append_data(imageio.imread(path))
    finally:
        writer.close()

    print(f"Video saved to {args.output}")


if __name__ == "__main__":
    main()
