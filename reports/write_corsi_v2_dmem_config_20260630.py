"""Write a D_mem-specific config derived from the expanded800 base config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create a D_mem-specific Corsi V2 config.")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--dmem", type=int, required=True)
    parser.add_argument("--output-config", required=True)
    args = parser.parse_args(argv)

    config = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    config["D_mem"] = int(args.dmem)
    config["memory_dim"] = int(args.dmem)
    output = Path(args.output_config)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(str(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
