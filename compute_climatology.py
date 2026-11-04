"""Build a month-by-UTC climatology from the declared training years."""
import argparse
import numpy as np
import os
import glob
import datetime


def get_year_files(path, year):
    year_dir = os.path.join(path, f"{year}_npy")
    return sorted(glob.glob(os.path.join(year_dir, "*.npy")))


def compute_climatology(path, train_years, height=180, width=360,
                         frames_per_day=96, save_path=None):
    """
    Average TEC in month / UTC-slot bins using training years only.
    Input chunks must retain the regular Jan-1 UTC chronology and grid order.
    """
    clim_sum = np.zeros((12, frames_per_day, height, width), dtype=np.float64)
    clim_count = np.zeros(12, dtype=np.int64)

    for year in train_years:
        files = get_year_files(path, year)
        if not files:
            raise FileNotFoundError(f"未找到 {year} 年数据")
        arrs = [np.load(f, mmap_mode='r') for f in files]
        total_frames = sum(a.shape[0] for a in arrs)
        n_days = total_frames // frames_per_day
        print(f"{year}: 总帧数={total_frames}, 估计天数={n_days}")

        file_idx, offset = 0, 0
        for day in range(n_days):
            day_frames = []
            needed = frames_per_day
            while needed > 0 and file_idx < len(arrs):
                arr = arrs[file_idx]
                avail = arr.shape[0] - offset
                take = min(needed, avail)
                day_frames.append(np.array(arr[offset:offset + take], dtype=np.float32))
                needed -= take
                offset += take
                if offset >= arr.shape[0]:
                    file_idx += 1
                    offset = 0
            if needed > 0:
                break  # 不满一整天的尾巴丢弃
            day_arr = np.concatenate(day_frames, axis=0)  # [96, H, W]

            month = (datetime.date(year, 1, 1) + datetime.timedelta(days=day)).month - 1
            clim_sum[month] += day_arr.astype(np.float64)
            clim_count[month] += 1

        print(f"  {year} 处理完成")

    climatology = clim_sum / clim_count[:, None, None, None].clip(min=1)
    climatology = climatology.astype(np.float32)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.save(save_path, climatology)
        print(f"已保存气候态基准: {save_path}, shape={climatology.shape}")
    return climatology


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--train-years", nargs="+", type=int,
                        default=[2012, 2014, 2016, 2018, 2020])
    parser.add_argument("--frames-per-day", type=int, default=96)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--width", type=int, default=360)
    parser.add_argument("--output", help="Default: DATA_ROOT/climatology.npy")
    args = parser.parse_args()
    if min(args.frames_per_day, args.height, args.width) <= 0:
        parser.error("frames-per-day, height and width must be positive")
    compute_climatology(
        args.data_root, args.train_years, height=args.height, width=args.width,
        frames_per_day=args.frames_per_day,
        save_path=args.output or os.path.join(args.data_root, "climatology.npy"),
    )
