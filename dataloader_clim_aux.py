import numpy as np
import torch
import os
import glob
import datetime
from torch.utils.data import Dataset
from typing import Tuple, Any
from solar_zenith import compute_qd_sequence, compute_sza_sequence


def _numpy_to_tensor(array):
    array = np.ascontiguousarray(array)
    return torch.frombuffer(memoryview(array), dtype=torch.float32).reshape(array.shape).clone()


class TEC_Dataset(Dataset):
    def __init__(self, path, time_step, predict_step, train=False, validation=False, test=False,
                 test_year=None, cache_dir=None, sample_stride=None, max_samples=None,
                 climatology_path=None, frames_per_day=96, use_aux_sza=False,
                 use_aux_qd=False):
        super().__init__()
        self.path = path
        self.train = train
        self.validation = validation
        self.test = test
        self.test_year = test_year
        self.time_step = time_step
        self.predict_step = predict_step
        self.sample_stride = sample_stride or predict_step
        self.max_samples = max_samples
        self.frames_per_day = frames_per_day
        # Deterministic geometry is concatenated only to the historical input.
        self.use_aux_sza = use_aux_sza
        self.use_aux_qd = use_aux_qd
        self.height = 180
        self.width = 360
        # 年份划分
        self.train_years = [2012, 2014, 2016, 2018, 2020]
        self.val_years = [2013, 2017]
        self.test_years = [2015, 2019]
        self.global_min = 0.0
        self.global_max = self._get_or_compute_global_max()
        print(f"[归一化] min={self.global_min:.4f}, max={self.global_max:.4f}")
        # ========== 气候态基准(可选，严格校验) ==========
        self.climatology = None
        if climatology_path is not None:
            if not os.path.exists(climatology_path):
                raise FileNotFoundError(
                    f"气候态基准文件不存在: {climatology_path}\n"
                    f"请先运行 compute_climatology.py 生成，或检查路径是否正确"
                )
            clim = np.load(climatology_path, mmap_mode='r')
            self.climatology = clim / self.global_max
            print(f"[气候态基准] 已加载(mmap模式): {climatology_path}, shape={self.climatology.shape}")
        if self.use_aux_sza:
            print("[物理输入通道] 已启用太阳天顶角(cos SZA)辅助通道")
        if self.use_aux_qd:
            print("[物理输入通道] 已启用准偶极地磁坐标辅助通道")
        # 元信息索引
        self.year_meta = dict()
        self.total_len = 0
        if self.train:
            self._prepare_meta(self.train_years, "训练集")
        elif self.validation:
            self._prepare_meta(self.val_years, "验证集")
        else:
            years = [self.test_year] if self.test_year else self.test_years
            self._prepare_meta(years, "测试集")
        print("*****数据收集完成*****\n")

    def _get_year_files(self, year):
        year_dir = os.path.join(self.path, f"{year}_npy")
        if not os.path.isdir(year_dir):
            return []
        return sorted(glob.glob(os.path.join(year_dir, "*.npy")))

    def _get_or_compute_global_max(self):
        cache_file = os.path.join(self.path, "global_max.txt")
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                return max(float(f.read().strip()), 1e-8)
        print("计算训练集全局最大值...")
        max_val = 0.0
        for year in self.train_years:
            for f in self._get_year_files(year):
                d = np.load(f)
                max_val = max(max_val, float(np.max(d)))
        max_val = max(max_val, 1e-8)
        try:
            with open(cache_file, 'w') as f:
                f.write(str(max_val))
        except Exception:
            pass
        return max_val

    def _prepare_meta(self, target_years, split_name):
        """只保存元信息，不生成全部样本索引"""
        print(f"\n===== 准备{split_name}元信息 =====")
        seq_len = self.time_step + self.predict_step
        for year in target_years:
            files = self._get_year_files(year)
            if not files:
                raise FileNotFoundError(f"未找到 {year} 年数据")
            year_data_list = [np.load(f, mmap_mode='r') for f in files]
            year_total_frames = sum(arr.shape[0] for arr in year_data_list)
            max_start = year_total_frames - seq_len
            if max_start <= 0:
                continue
            n_sample = (max_start // self.sample_stride) + 1
            if self.max_samples is not None:
                n_sample = min(n_sample, self.max_samples - self.total_len)
                if n_sample <= 0:
                    break
            self.year_meta[year] = {
                "memmap_list": year_data_list,
                "total_frames": year_total_frames,
                "n_sample": n_sample,
                "offset": self.total_len
            }
            self.total_len += n_sample
            print(f"  {split_name} {year}: 元信息记录，样本数 {n_sample} (总帧数 {year_total_frames})")
        print(f"  [{split_name}总计] 元信息记录完成，总样本数: {self.total_len}")

    def _read_frames(self, year, start_frame, num_frames):
        """跨文件读取连续的 num_frames 帧数据【完全沿用原版，不动】"""
        file_list = self.year_meta[year]["memmap_list"]
        frames = []
        needed = num_frames
        curr_frame = start_frame
        file_idx = 0
        accum = 0
        while file_idx < len(file_list) and accum + file_list[file_idx].shape[0] <= curr_frame:
            accum += file_list[file_idx].shape[0]
            file_idx += 1
        offset = curr_frame - accum
        while needed > 0 and file_idx < len(file_list):
            arr = file_list[file_idx]
            available = arr.shape[0] - offset
            take = min(needed, available)
            frames.append(np.array(arr[offset: offset + take], dtype=np.float32))
            needed -= take
            offset = 0
            file_idx += 1
        res = np.concatenate(frames, axis=0)
        res = (res - self.global_min) / (self.global_max - self.global_min)
        return res

    def _get_climatology_seq(self, year, start_frame, num_frames):
        """按 (月份, 当天第几帧) 取出对应窗口长度的气候态基准序列"""
        clim_frames = []
        for i in range(num_frames):
            frame_idx = start_frame + i
            day_of_year = frame_idx // self.frames_per_day
            frame_in_day = frame_idx % self.frames_per_day
            month = (datetime.date(year, 1, 1) + datetime.timedelta(days=day_of_year)).month - 1
            clim_frames.append(self.climatology[month, frame_in_day])
        return np.stack(clim_frames, axis=0).astype(np.float32)

    def __getitem__(self, index) -> Tuple[Any, Any]:
        # 根据全局index现场定位year、该年份内部样本序号
        target_year = None
        in_year_idx = 0
        for year, meta in self.year_meta.items():
            if meta["offset"] <= index < meta["offset"] + meta["n_sample"]:
                target_year = year
                in_year_idx = index - meta["offset"]
                break
        assert target_year is not None, "index越界"
        start_frame = in_year_idx * self.sample_stride
        seq_data = self._read_frames(target_year, start_frame, self.time_step + self.predict_step)
        x = seq_data[: self.time_step]
        y = seq_data[self.time_step:]

        # ===== 【新增】太阳天顶角辅助通道：只拼接到输入x，不拼接到目标y =====
        # (网络仍然只预测TEC这一个通道，天顶角只是提供给编码器的额外先验信息)
        if self.use_aux_sza:
            sza_x = compute_sza_sequence(
                target_year, start_frame, self.time_step,
                height=self.height, width=self.width, frames_per_day=self.frames_per_day
            )  # [T_in, H, W]，值域[-1,1]，与TEC归一化后的[0,1]尺度不同但不需要强行对齐
        qd_x = compute_qd_sequence(
            target_year, start_frame, self.time_step,
            height=self.height, width=self.width, frames_per_day=self.frames_per_day
        ) if self.use_aux_qd else None

        # 开启气候态时返回残差 + 真值气候态背景
        if self.climatology is not None:
            clim_seq = self._get_climatology_seq(
                target_year, start_frame, self.time_step + self.predict_step
            )
            x_clim = clim_seq[: self.time_step]
            y_clim = clim_seq[self.time_step:]
            eps = 1e-6
            scale = (x.mean() + eps) / (x_clim.mean() + eps)
            scale = np.clip(scale, 0.3, 3.0)
            x_clim_scaled = x_clim * scale
            y_clim_scaled = y_clim * scale
            x_resid = x - x_clim_scaled
            y_resid = y - y_clim_scaled

            if self.use_aux_sza or self.use_aux_qd:
                # x_resid: [T_in,H,W] -> 拼接sza后 [T_in, 2, H, W]
                aux = [x_resid]
                if self.use_aux_sza:
                    aux.append(sza_x)
                if self.use_aux_qd:
                    aux.extend([qd_x[:, 0], qd_x[:, 1], qd_x[:, 2]])
                x_stack = np.stack(aux, axis=1).astype(np.float32)
                x_tensor = _numpy_to_tensor(x_stack)
            else:
                x_tensor = _numpy_to_tensor(x_resid.astype(np.float32)).unsqueeze(1)
            y_tensor = _numpy_to_tensor(y_resid.astype(np.float32)).unsqueeze(1)
            y_clim_tensor = _numpy_to_tensor(y_clim_scaled.astype(np.float32)).unsqueeze(1)
            return x_tensor, y_tensor, y_clim_tensor

        # 未开启气候态时走原逻辑
        if self.use_aux_sza or self.use_aux_qd:
            aux = [x]
            if self.use_aux_sza:
                aux.append(sza_x)
            if self.use_aux_qd:
                aux.extend([qd_x[:, 0], qd_x[:, 1], qd_x[:, 2]])
            x_stack = np.stack(aux, axis=1).astype(np.float32)
            x_tensor = _numpy_to_tensor(x_stack)
        else:
            x_tensor = _numpy_to_tensor(x).unsqueeze(1)
        y_tensor = _numpy_to_tensor(y).unsqueeze(1)
        return x_tensor, y_tensor

    def __len__(self):
        return self.total_len
