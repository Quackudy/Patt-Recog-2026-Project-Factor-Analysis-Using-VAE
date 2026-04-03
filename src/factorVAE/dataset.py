import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import pandas as pd
import numpy as np
import bisect
from multiprocessing import shared_memory


class TSDataSampler:
    def __init__(
        self, data: pd.DataFrame, start, end, step_len: int,
        fillna_type: str = "none", dtype=None, flt_data=None
    ):
        self.start       = start
        self.end         = end
        self.step_len    = step_len
        self.fillna_type = fillna_type

        assert data.index.names == ["datetime", "instrument"]
        self.data = data.sort_index()

        # ── 1. Pre-fill and put data_arr in shared memory ─────────────────────
        print("Pre-processing dataset (ffill)...")
        temp_df  = self.data.groupby(level='instrument').ffill().fillna(0)
        raw_arr  = temp_df.to_numpy(dtype=np.float32)
        raw_arr  = np.vstack([raw_arr, np.zeros((1, raw_arr.shape[1]), dtype=np.float32)])
        self.nan_idx = len(raw_arr) - 1          # sentinel row index

        self._data_shm, self.data_arr = _to_shm(raw_arr)
        del raw_arr

        # ── 2. Build index without iterrows ───────────────────────────────────
        positions  = np.arange(len(self.data), dtype=np.float64)
        pos_series = pd.Series(positions, index=self.data.index)
        idx_df     = pos_series.unstack(level='instrument').sort_index().sort_index(axis=1)

        # Save only the two small axes; never store idx_df itself (it's huge)
        self._dates_index = idx_df.index    # DatetimeIndex — small
        self._instr_index = idx_df.columns  # ticker Index  — small

        values        = idx_df.to_numpy(dtype=np.float64)
        date_rows, instr_cols = np.where(~np.isnan(values))
        flat_pos      = values[date_rows, instr_cols].astype(int)
        idx_map_dict  = dict(zip(flat_pos.tolist(),
                                 zip(date_rows.tolist(), instr_cols.tolist())))

        # ── 3. Put idx_arr in shared memory ───────────────────────────────────
        idx_arr_raw          = np.full(values.shape, self.nan_idx, dtype=np.int32)
        valid                = ~np.isnan(values)
        idx_arr_raw[valid]   = values[valid].astype(np.int32)
        del values, idx_df

        self._idx_shm, self.idx_arr = _to_shm(idx_arr_raw)
        del idx_arr_raw

        # ── 4. data_index + optional filter ──────────────────────────────────
        self.data_index = self.data.index

        if flt_data is not None:
            flt             = flt_data.reindex(self.data_index).fillna(False).astype(bool)
            idx_map_dict    = self._flt_idx_map(flt, idx_map_dict)
            self.data_index = self.data_index[flt]

        self.idx_map = self._idx_map2arr(idx_map_dict)

        self.start_idx, self.end_idx = self.data_index.slice_locs(
            start=pd.Timestamp(start), end=pd.Timestamp(end)
        )
        del self.data
    # ── Pickle protocol (workers get shm names, re-attach) ───────────────────

    def __getstate__(self):
        state = self.__dict__.copy()
        state['data_arr'] = _shm_desc(self._data_shm, self.data_arr)
        state['idx_arr']  = _shm_desc(self._idx_shm,  self.idx_arr)
        del state['_data_shm'], state['_idx_shm']
        return state

    def __setstate__(self, state):
        data_desc = state.pop('data_arr')
        idx_desc  = state.pop('idx_arr')
        self.__dict__.update(state)
        self._data_shm, self.data_arr = _from_shm(data_desc)
        self._idx_shm,  self.idx_arr  = _from_shm(idx_desc)

    def cleanup(self):
        for shm in (self._data_shm, self._idx_shm):
            try: shm.close(); shm.unlink()
            except Exception: pass

    def __del__(self):
        for attr in ('_data_shm', '_idx_shm'):
            try: getattr(self, attr).close()
            except Exception: pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _idx_map2arr(idx_map):
        dtype    = np.int32
        sentinel = (np.iinfo(dtype).max, np.iinfo(dtype).max)
        max_idx  = max(idx_map.keys())
        return np.array([idx_map.get(i, sentinel) for i in range(max_idx + 1)],
                        dtype=dtype)

    @staticmethod
    def _flt_idx_map(flt_data, idx_map):
        new_map, idx = {}, 0
        for i, exist in enumerate(flt_data):
            if exist:
                new_map[idx] = idx_map[i]; idx += 1
        return new_map

    def get_index(self):
        return self.data_index[self.start_idx: self.end_idx]

    def _get_row_col(self, idx) -> tuple:
        if isinstance(idx, (int, np.integer)):
            real_idx = self.start_idx + int(idx)
            if self.start_idx <= real_idx < self.end_idx:
                return tuple(self.idx_map[real_idx])
            raise KeyError(f"{real_idx} out of [{self.start_idx}, {self.end_idx})")
        if isinstance(idx, tuple):
            date, inst = idx
            i = bisect.bisect_right(self._dates_index, pd.Timestamp(date)) - 1
            j = bisect.bisect_left(self._instr_index, inst)
            return i, j
        raise NotImplementedError(f"Unsupported index type: {type(idx)}")

    def _get_indices(self, row: int, col: int) -> np.ndarray:
        start_r = max(row - self.step_len + 1, 0)
        indices = self.idx_arr[start_r: row + 1, col].copy()
        if len(indices) < self.step_len:
            pad     = np.full(self.step_len - len(indices), self.nan_idx, dtype=np.int32)
            indices = np.concatenate([pad, indices])
        return indices

    # ── __getitem__ ───────────────────────────────────────────────────────────

    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray)):
            row_cols    = [self._get_row_col(i) for i in idx]
            all_indices = np.stack([self._get_indices(r, c) for r, c in row_cols])
            data        = self.data_arr[all_indices]          # (B, T, F)
            # For actual_indices we return the last timestep's MultiIndex label.
            # Sentinel rows map to the zero-pad row — use the real position (r,c)
            # from idx_arr to get the true data_index entry.
            actual = []
            for r, c in row_cols:
                raw_pos = int(self.idx_arr[r, c])
                if raw_pos == self.nan_idx:
                    actual.append(None)
                else:
                    actual.append(self.data_index[raw_pos])
        else:
            r, c    = self._get_row_col(idx)
            indices = self._get_indices(r, c)
            data    = self.data_arr[indices]                  # (T, F)
            raw_pos = int(self.idx_arr[r, c])
            actual  = self.data_index[raw_pos] if raw_pos != self.nan_idx else None

        return data, actual

    def __len__(self):
        return self.end_idx - self.start_idx


# ── Shared-memory helpers ─────────────────────────────────────────────────────

def _to_shm(arr: np.ndarray):
    shm  = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    view = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    np.copyto(view, arr)
    return shm, view

def _shm_desc(shm, view):
    return (shm.name, view.shape, view.dtype)

def _from_shm(desc):
    name, shape, dtype = desc
    shm  = shared_memory.SharedMemory(name=name, create=False)
    view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shm, view


# ── Dataset / Sampler / DataLoader ────────────────────────────────────────────

class TSDatasetH(Dataset):
    DEFAULT_STEP_LEN = 20

    def __init__(self, data, step_len=DEFAULT_STEP_LEN, **kwargs):
        self.step_len = step_len
        self.data     = data
        self.sampler  = TSDataSampler(data=data, step_len=step_len, **kwargs)

    def __getitem__(self, idx): return self.sampler[idx]
    def __len__(self):          return len(self.sampler)


class DateGroupedBatchSampler(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.data_source     = data_source
        self.shuffle         = shuffle
        self.grouped_indices = self._group()

    def _group(self):
        s   = self.data_source.sampler
        idx = s.data_index[s.start_idx: s.end_idx]
        ser = pd.Series(range(len(idx)), index=idx.get_level_values('datetime'))
        return ser.groupby(level='datetime').apply(list).values

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.grouped_indices)
        yield from self.grouped_indices

    def __len__(self):
        return len(self.grouped_indices)


def custom_collate_fn(batch):
    data, indices = zip(*batch)
    data    = torch.utils.data.dataloader.default_collate(data)
    indices = [i for i in indices]   # list of MultiIndex labels or None
    return data, indices


def init_data_loader(df, step_len, shuffle, start, end,
                     num_workers=0, select_feature=None):
    if select_feature is not None:
        df = df[select_feature]
    dataset = TSDatasetH(df, step_len=step_len, start=start, end=end,
                         fillna_type='none')
    sampler = DateGroupedBatchSampler(dataset, shuffle=shuffle)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=custom_collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )


if __name__ == "__main__":
    df = pd.read_pickle('data/sp500_data.pkl')
    step_len = 20

    data_loader = init_data_loader(df, step_len,
                                   shuffle=False, start='2010-01-01', end='2015-01-01',
                                   select_feature=None)

    for batch, indices in data_loader:
        input_data, labels = batch[:, :, :-1], batch[:, -1, -1].unsqueeze(-1)
        print(input_data.shape, labels.shape)
    print("Done")
