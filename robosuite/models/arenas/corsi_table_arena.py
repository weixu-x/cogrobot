from typing import List, Optional, Tuple

import numpy as np

from robosuite.models.arenas.arena import Arena

# from robosuite.models.arenas.table_arena import TableArena
from robosuite.utils.mjcf_utils import (
    array_to_string,
    find_elements,
    new_body,
    new_geom,
    string_to_array,
    xml_path_completion
)

class CorsiTableArena(Arena):
    def __init__(
        self,
        fname: str = "arenas/table_arena.xml",
        

        # ✅ 为了兼容 Lift / sampler / check_success：补齐这三个
        table_full_size=(0.8, 0.8, 0.05),
        table_friction=(1.0, 5e-3, 1e-4),
        table_offset=(0, 0, 0.8),

        # corsi grid
        rows: int = 3,
        cols: int = 3,
        dx: float = 0.07,
        dy: float = 0.07,

        # blocks
        block_half_size: Tuple[float, float, float] = (0.02, 0.02, 0.015),
        block_z_offset: float = 0.0,
        block_rgba_list: Optional[List[List[float]]] = None,

        # anchor
        tabletop_site_name: str = "table_top",
        table_body_name: str = "table",

    ):
        # 这些要在 super().__init__ 之前设好，因为 Arena.__init__ 会立刻调用 _postprocess_arena()
        self.table_full_size = table_full_size
        self.table_friction = table_friction
        self.table_offset = np.array(table_offset, dtype=float)

        self.table_body_name = table_body_name

        self.rows = int(rows)
        self.cols = int(cols)
        self.dx = float(dx)
        self.dy = float(dy)

        self.block_half_size = np.array(block_half_size, dtype=float)
        self.block_z_offset = float(block_z_offset)
        self.tabletop_site_name = tabletop_site_name

        if block_rgba_list is None:
            block_rgba_list = [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0, 1.0],
                [1.0, 0.5, 0.0, 1.0],
                [0.6, 0.3, 1.0, 1.0],
                [0.8, 0.8, 0.8, 1.0],
            ]
        self.block_rgba_list = [list(map(float, c)) for c in block_rgba_list]

        self.block_names: List[str] = []
        self.block_positions = np.zeros((0, 3), dtype=float)


        # ✅ 关键：用你自己的 table_arena.xml（包含 table_top=0.4，table body pos=0.4 -> 台面=0.8）
        super().__init__(fname=xml_path_completion(fname))

    def _postprocess_arena(self):
        # 1) 找 table_top site（局部）+ table body（world） -> 计算 table_top_world
        site = find_elements(
            root=self.worldbody,
            tags="site",
            attribs={"name": self.tabletop_site_name},
            return_first=True,
        )
        if site is None or site.get("pos") is None:
            raise ValueError(f"Cannot find tabletop site '{self.tabletop_site_name}' with pos in the arena xml.")

        table_body = find_elements(
            root=self.worldbody,
            tags="body",
            attribs={"name": self.table_body_name},
            return_first=True,
        )
        if table_body is None or table_body.get("pos") is None:
            raise ValueError(f"Cannot find table body '{self.table_body_name}' with pos in arena xml.")

        site_pos_local = string_to_array(site.get("pos"))
        table_body_pos = string_to_array(table_body.get("pos"))
        table_top_world = table_body_pos + site_pos_local
        cx, cy, tz = float(table_top_world[0]), float(table_top_world[1]), float(table_top_world[2])

        # 2) block 的 z：桌面 + 半高 + offset
        z = tz + float(self.block_half_size[2]) + float(self.block_z_offset)

        # 3) 在桌面范围内随机采样 9 个点，保证不重叠
        # 你可以把这些参数写成 __init__ 参数（推荐）
        n = self.rows * self.cols  # 你如果固定9，也可以直接 n=9
        x_range = (-0.18, 0.18)    # 相对 table_top_world 的范围（按你桌面大小调）
        y_range = (-0.18, 0.18)
        margin = 0.01              # block 与 block 的额外安全间隙

        pts_xy = self._sample_nonoverlap_xy(
            n=n,
            center_xy=(cx, cy),
            x_range=x_range,
            y_range=y_range,
            min_dist=2.0 * (max(self.block_half_size[0], self.block_half_size[1]) + margin),
            max_tries=5000,
        )

        # 4) 创建 blocks（静态 body + box geom）
        self.block_names = []
        positions = []

        for i, (x, y) in enumerate(pts_xy):
            name = f"corsi_block_{i}"
            rgba = self.block_rgba_list[i % len(self.block_rgba_list)]
            pos = np.array([x, y, z], dtype=float)

            block_body = new_body(name=name, pos=array_to_string(pos))
            block_geom = new_geom(
                name=f"{name}_geom",
                type="box",
                size=array_to_string(self.block_half_size),
                rgba=array_to_string(np.array(rgba, dtype=float)),
                group="1",
            )
            block_body.append(block_geom)
            self.worldbody.append(block_body)

            self.block_names.append(name)
            positions.append(pos)

        self.block_positions = np.array(positions, dtype=float)


    @staticmethod
    def _sample_nonoverlap_xy(n, center_xy, x_range, y_range, min_dist, max_tries=5000):
        """
        Rejection sampling in 2D: sample n points in a rectangle, ensure pairwise distance >= min_dist.
        center_xy: (cx, cy) world center
        x_range/y_range: relative bounds around center, e.g. (-0.2, 0.2)
        """
        cx, cy = center_xy
        xmin, xmax = cx + x_range[0], cx + x_range[1]
        ymin, ymax = cy + y_range[0], cy + y_range[1]

        pts = []
        tries = 0
        while len(pts) < n and tries < max_tries:
            tries += 1
            x = np.random.uniform(xmin, xmax)
            y = np.random.uniform(ymin, ymax)
            ok = True
            for (px, py) in pts:
                if (x - px) ** 2 + (y - py) ** 2 < (min_dist ** 2):
                    ok = False
                    break
            if ok:
                pts.append((float(x), float(y)))

        if len(pts) < n:
            raise RuntimeError(
                f"Failed to sample {n} non-overlapping points. "
                f"Try increasing x/y range or decreasing block size / margin. "
                f"Got {len(pts)} points after {tries} tries."
            )
        return pts


    # @staticmethod
    # def _make_grid_xy(center_xy, rows, cols, dx, dy):
    #     cx, cy = center_xy
    #     pts = []
    #     for r in range(rows):
    #         for c in range(cols):
    #             x = cx + (c - (cols - 1) / 2.0) * dx
    #             y = cy + (r - (rows - 1) / 2.0) * dy
    #             pts.append((float(x), float(y)))
    #     return pts
