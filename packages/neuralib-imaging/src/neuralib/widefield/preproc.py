import json
from datetime import datetime
from multiprocessing import cpu_count
from pathlib import Path

import cv2
import h5py
import numpy as np
import tifffile
from argclz import AbstractParser, argument
from joblib import Parallel, delayed
from neuralib.util.utils import ensure_dir
from rich.console import Console
from tqdm import tqdm

from .meta import PreprocessMeta

# numba acceleration
try:
    from numba import jit

    _HAS_NUMBA = True
except ImportError:

    def jit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


    _HAS_NUMBA = False

# cupy acceleration
try:
    import cupy as cp

    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False

console = Console()


class PreprocessOptions(AbstractParser):
    DESCRIPTION = 'Preprocessing pipeline for widefield calcium imaging dataset'

    GROUP_IO = 'Data I/O Options'
    GROUP_PROC = 'Processing Options'
    GROUP_ACCEL = 'Acceleration Options'

    # ------- IO -------

    file: Path | None = argument(
        '--file',
        ex_group='input',
        group=GROUP_IO,
        help='single input file'
    )

    directory: Path | None = argument(
        '--directory',
        ex_group='input',
        group=GROUP_IO,
        help='directory for input files'
    )

    suffix_pattern: str = argument(
        '--suffix',
        default='.tif',
        group=GROUP_IO,
        help='suffix for directory'
    )

    _output_dir: Path | None = argument(
        '--output_dir',
        default=None,
        group=GROUP_IO,
        help='output directory for all results (dff.npy, f0.npy, mask.npy, etc.)'
    )

    # ------- Process -------

    motion_correction: bool = argument(
        '--motion_corr',
        group=GROUP_PROC,
        help='do the motion correction'
    )

    max_shift: int = argument(
        '--max_shift',
        default=20,
        group=GROUP_PROC,
        help='maximum allowed shift in pixels for motion correction'
    )

    rotate: float | None = argument(
        '--rotate',
        default=None,
        group=GROUP_PROC,
        help='rotate the all sequences in degree'
    )

    chunk_size: int = argument(
        '--chunk_size',
        default=3000,
        group=GROUP_PROC,
        help='number of frames per chunk for processing'
    )

    window_size: int = argument(
        '--window_size',
        default=100,
        group=GROUP_PROC,
        help='window size for rolling baseline (frames)'
    )

    percentile: int = argument(
        '--percentile',
        default=10,
        group=GROUP_PROC,
        help='percentile for baseline calculation'
    )

    n_jobs: int = argument(
        '--n_jobs',
        default=-1,
        group=GROUP_PROC,
        help='number of parallel jobs for processing (-1 = use all CPUs)'
    )

    force_compute: bool = argument(
        '--force_compute',
        group=GROUP_PROC,
        help='force recomputation even if output files exist (overwrite mode)'
    )

    save_f0: bool = argument(
        '--save_f0',
        group=GROUP_PROC,
        help='save F0 baseline to disk (can be disabled to save storage space)'
    )

    # ------- Acceleration -------

    use_gpu: bool = argument(
        '--use_gpu',
        group=GROUP_ACCEL,
        help='use GPU acceleration with CuPy (requires NVIDIA GPU and cupy package)'
    )

    def __init__(self):
        """Initialize internal state variables."""
        self._tif_files: list[Path] | None = None
        self._reference_frame: np.ndarray | None = None
        self._total_frames: int = 0
        self._frame_shape: tuple[int, int] | None = None
        self._transform_cache_path: Path | None = None

        # F0 computation metadata
        self._f0_stride: int | None = None
        self._f0_n_keyframes: int | None = None
        self._f0_n_jobs_used: int | None = None

    def run(self):
        """Main entry point - orchestrates entire pipeline."""
        console.rule("[bold blue]Widefield Calcium Imaging Preprocessing Pipeline")

        # load
        console.log("[bold cyan][1/4][/] Loading input files...")
        self.load()

        # check if results already exist
        dff_path = self.output_dir / 'dff.npy'
        f0_path = self.output_dir / 'f0.h5'
        transform_path = self.output_dir / 'motion_transforms.h5'

        # Check for existing results (f0 only if save_f0 is enabled)
        if self.save_f0:
            results_exist = dff_path.exists() and f0_path.exists()
        else:
            results_exist = dff_path.exists()
        computation_performed = False

        if results_exist and not self.force_compute:
            console.log("[bold cyan][2/4][/] Found existing results, skipping computation")
            console.log(f"  Using existing files in [green]{self.output_dir}")
            console.log("  [dim](Use --force_compute to recompute)")

            # Load transform cache path if motion correction was done
            if transform_path.exists():
                self._transform_cache_path = transform_path
        else:
            computation_performed = True
            if results_exist:
                console.log("[bold cyan][2/4][/] Force recomputing (--force_compute enabled)...")

            # Compute and save reference frame (with rotation applied if specified)
            console.log("[bold cyan][2/4][/] Computing reference frame...")
            reference_path = self.output_dir / 'reference_frame.tif'
            if reference_path.exists() and not self.force_compute:
                console.log("  Loading existing reference frame...")
                self._reference_frame = tifffile.imread(reference_path)
            else:
                self._reference_frame = self._compute_reference_frame()
                tifffile.imwrite(reference_path, self._reference_frame.astype(np.float32))
                console.log(f"  Reference frame saved to: [green]{reference_path}")

            # motion correction
            if self.motion_correction:
                transform_exists = transform_path.exists()
                if transform_exists and not self.force_compute:
                    console.log("[bold cyan][2.5/4][/] Loading existing motion correction...")
                    self._transform_cache_path = transform_path
                else:
                    console.log("[bold cyan][2.5/4][/] Performing motion correction...")
                    self.compute_motion_transforms()
            else:
                console.log("[bold cyan][2.5/4][/] Skipping motion correction")

            # dff
            console.log("[bold cyan][3/4][/] Calculating ΔF/F with time-varying baseline...")
            self.calculate_dff()

        # Save metadata (only if computation was performed)
        if computation_performed:
            console.log("[bold cyan][4/4][/] Saving metadata...")
            self.save_metadata()
        else:
            console.log("[bold cyan][4/4][/] Metadata unchanged (using existing results)")

        console.print(f'\n[bold green]✓[/] Processing complete! Results in [green]{self.output_dir}')

    @property
    def output_dir(self) -> Path:
        """output directory"""
        if self._output_dir is None:
            if self.file is not None:
                self._output_dir = self.file.parent / 'preprocessed'
            if self.directory is not None:
                self._output_dir = self.directory / 'preprocessed'
        return ensure_dir(self._output_dir)

    def load(self):
        """load TIF files and validate dataset consistency"""
        if self.file is not None:
            self._tif_files = [self.file]
        elif self.directory is not None:
            pattern = f'*{self.suffix_pattern}'
            self._tif_files = sorted(self.directory.glob(pattern))
        else:
            raise ValueError("Must specify --file or --directory")

        if not self._tif_files:
            raise ValueError(f"No TIF files found matching pattern")

        console.log(f"  Found {len(self._tif_files)} TIF file(s)")

        # validate
        with tifffile.TiffFile(self._tif_files[0]) as tif:
            first_page = tif.pages[0]
            self._frame_shape = first_page.shape  # (H, W)

        self._total_frames = 0
        frame_shapes = set()

        for tif_path in tqdm(self._tif_files, desc="  Scanning TIF files"):
            with tifffile.TiffFile(tif_path) as tif:
                n_frames = len(tif.pages)
                self._total_frames += n_frames

                frame_shape = tif.pages[0].shape
                frame_shapes.add(frame_shape)

        if len(frame_shapes) > 1:
            raise ValueError(f"Inconsistent frame shapes found: {frame_shapes}")

        # adjust chunk_size if total_frames is smaller
        if self.chunk_size > self._total_frames:
            original_chunk_size = self.chunk_size
            self.chunk_size = self._total_frames
            console.log(f"  Note: Adjusted chunk_size from {original_chunk_size} to {self.chunk_size} (total frames)")

        console.log(f"  Total frames: [yellow]{self._total_frames:,}")
        console.log(f"  Frame shape: [yellow]{self._frame_shape}")
        console.log(f"  Chunk size: [yellow]{self.chunk_size}")
        if self.rotate is not None:
            console.log(f"  Rotation: [yellow]{self.rotate}°[/] (applied as post-processing)")
        console.log(f"  Output directory: [green]{self.output_dir}")

    def _iterate_chunks(self):
        """Generator that yields chunks of frames from TIF files.
        :return: yield (chunk_data, chunk_start, chunk_end) tuples
        """
        frame_idx = 0
        current_chunk = []
        chunk_start_idx = 0

        for tif_path in self._tif_files:
            with tifffile.TiffFile(tif_path) as tif:
                for page in tif.pages:
                    frame = page.asarray().astype(np.float32)
                    current_chunk.append(frame)
                    frame_idx += 1

                    # Yield chunk when it reaches chunk_size or end of data
                    if len(current_chunk) == self.chunk_size:
                        chunk_data = np.stack(current_chunk, axis=0)
                        yield chunk_data, chunk_start_idx, frame_idx
                        current_chunk = []
                        chunk_start_idx = frame_idx

        # remaining frames
        if current_chunk:
            chunk_data = np.stack(current_chunk, axis=0)
            yield chunk_data, chunk_start_idx, frame_idx

    def _compute_reference_frame(self, sample_size: int = 100) -> np.ndarray:
        """
        Compute reference frame by sampling frames from TIF files.

        :param sample_size: Number of frames to sample from each file
        :return: Mean reference frame
        """
        samples = []
        for tif_path in tqdm(self._tif_files, desc="  Sampling frames"):
            with tifffile.TiffFile(tif_path) as tif:
                n_sample = min(sample_size, len(tif.pages))
                for i in range(n_sample):
                    frame = tif.pages[i].asarray().astype(np.float32)
                    samples.append(frame)

        # Stack samples and compute mean reference frame
        samples_array = np.stack(samples, axis=0)
        reference = np.mean(samples_array, axis=0)
        return reference

    def _compute_chunk_transforms(self, chunk: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Compute motion correction transforms for a chunk of frames.

        Args:
            chunk: (n_frames, H, W) array of frames
            reference: (H, W) reference frame

        Returns:
            transforms: (n_frames, 2, 3) array of transformation matrices
        """
        n_frames = chunk.shape[0]
        transforms = np.zeros((n_frames, 2, 3), dtype=np.float32)
        ref_uint8 = cv2.normalize(reference, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        for i in range(n_frames):
            frame_uint8 = cv2.normalize(chunk[i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            warp_matrix = np.eye(2, 3, dtype=np.float32)

            try:
                _, warp_matrix = cv2.findTransformECC(
                    ref_uint8,
                    frame_uint8,
                    warp_matrix,
                    cv2.MOTION_TRANSLATION,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-6)
                )

                # clip shifts to max_shift
                warp_matrix[0, 2] = np.clip(warp_matrix[0, 2], -self.max_shift, self.max_shift)
                warp_matrix[1, 2] = np.clip(warp_matrix[1, 2], -self.max_shift, self.max_shift)

            except cv2.error:
                # use identity transform if ECC fail
                pass

            transforms[i] = warp_matrix

        return transforms

    def compute_motion_transforms(self):
        """
        Compute motion correction transforms using the pre-computed reference frame.
        The reference frame should already be computed and stored in self._reference_frame.
        """
        if self._reference_frame is None:
            raise RuntimeError("Reference frame must be computed before motion correction")

        console.log("  Computing motion correction transforms...")
        self._transform_cache_path = self.output_dir / 'motion_transforms.h5'

        with h5py.File(self._transform_cache_path, 'w') as f:
            f.create_dataset('reference_frame', data=self._reference_frame, compression='gzip')
            console.log(f"  Reference frame also saved in HDF5")

            h5_chunk_size = min(self.chunk_size, self._total_frames)
            transforms_ds = f.create_dataset(
                'transforms',
                shape=(self._total_frames, 2, 3),
                dtype=np.float32,
                chunks=(h5_chunk_size, 2, 3),
                compression='gzip',
                compression_opts=4
            )

            pbar = tqdm(total=self._total_frames, desc="  Computing transforms")
            for chunk_data, chunk_start, chunk_end in self._iterate_chunks():
                chunk_transforms = self._compute_chunk_transforms(chunk_data, self._reference_frame)
                transforms_ds[chunk_start:chunk_end] = chunk_transforms
                pbar.update(chunk_end - chunk_start)

            pbar.close()

        console.log(f"  Transforms cached to: {self._transform_cache_path}")

    def _apply_transforms(self, frames: np.ndarray, transforms: np.ndarray) -> np.ndarray:
        """
        Apply transformation matrices to frames.

        Args:
            frames: (n_frames, H, W) array
            transforms: (n_frames, 2, 3) array of transformation matrices

        Returns:
            corrected_frames: (n_frames, H, W) array
        """
        n_frames = frames.shape[0]
        corrected = np.zeros_like(frames)

        for i in range(n_frames):
            corrected[i] = cv2.warpAffine(
                frames[i],
                transforms[i],
                (self._frame_shape[1], self._frame_shape[0]),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )

        return corrected

    def _load_frames_to_memmap(self, temp_frames_path: Path) -> np.ndarray:
        """
        Load all frames into memory-mapped array with optional motion correction.

        Args:
            temp_frames_path: Path to temporary memory-mapped array file

        Returns:
            Memory-mapped array of frames (read-only mode)
        """
        # Load transform cache if motion correction was performed
        transform_cache = None
        if self.motion_correction and self._transform_cache_path:
            transform_cache = h5py.File(self._transform_cache_path, 'r')
            console.log(f"  Loaded transform cache")

        # Calculate memory requirements
        frames_memory_gb = (self._total_frames * self._frame_shape[0] * self._frame_shape[1] * 4) / (1024 ** 3)
        console.log(f"  Full dataset size: [yellow]{frames_memory_gb:.1f} GB")

        # Load all frames into memory-mapped file
        console.log(f"  Loading all frames into memory-mapped array (single pass)...")
        frames_mmap = np.lib.format.open_memmap(
            temp_frames_path,
            mode='w+',
            dtype=np.float32,
            shape=(self._total_frames, *self._frame_shape)
        )

        pbar = tqdm(total=self._total_frames, desc="  Loading frames")
        for chunk_data, chunk_start, chunk_end in self._iterate_chunks():
            # Apply motion correction if enabled
            if transform_cache is not None:
                chunk_transforms = transform_cache['transforms'][chunk_start:chunk_end]
                chunk_data = self._apply_transforms(chunk_data, chunk_transforms)

            # CRITICAL: Ensure C-contiguous before writing to memmap to prevent grid artifacts
            # Rotation and transforms may return non-contiguous arrays
            chunk_data = np.ascontiguousarray(chunk_data)
            frames_mmap[chunk_start:chunk_end] = chunk_data
            pbar.update(chunk_end - chunk_start)
        pbar.close()

        # Flush to disk and reload in read-only mode
        del frames_mmap
        frames_mmap = np.load(temp_frames_path, mmap_mode='r')

        if transform_cache is not None:
            transform_cache.close()

        return frames_mmap

    def _compute_f0_keyframes(self, temp_frames_path: Path,
                              keyframe_indices: list[int],
                              half_window: int,
                              use_gpu: bool) -> tuple[np.ndarray, int]:
        """
        Compute F0 baseline at keyframe positions.

        :param temp_frames_path: Path to memory-mapped frames array
        :param keyframe_indices: List of frame indices for keyframes
        :param half_window: Half-width of rolling window
        :param use_gpu: Whether GPU is actually being used
        :return: Tuple of (f0_keyframes array, n_jobs used)

        """
        n_keyframes = len(keyframe_indices)
        n_pixels = self._frame_shape[0] * self._frame_shape[1]

        console.log(
            f"  Computing F0 at [yellow]{n_keyframes}[/] keyframes (out of [yellow]{self._total_frames}[/] total)..."
        )

        # check GPU availability
        if self.use_gpu and not _HAS_CUPY:
            console.log(
                f'  [bold yellow]⚠ WARNING:[/] GPU requested but CuPy not available. Install with: pip install cupy-cuda*x'
            )
            console.log(f'  Falling back to CPU processing')
        if use_gpu:
            console.log(f'  [green]✓[/] Using GPU acceleration with CuPy')
            try:
                # Get actual GPU memory info (free, total)
                free_mem, total_mem = cp.cuda.Device().mem_info
                console.log(
                    f'  GPU memory: [yellow]{free_mem / (1024 ** 3):.1f} GB[/] free / [yellow]{total_mem / (1024 ** 3):.1f} GB[/] total'
                )
            except Exception as e:
                console.log(f'  [bold yellow]⚠ WARNING:[/] GPU memory check failed: {e}')
        elif _HAS_NUMBA:
            console.log(f'  [green]✓[/] Using Numba JIT acceleration for fast percentile computation')
        else:
            console.log(f'  [bold yellow]⚠ WARNING:[/] Numba not available. Install with: pip install numba')

        #
        if use_gpu:
            n_jobs = 1
            console.log(f"  Using sequential processing (GPU parallelizes internally)")
        else:
            n_jobs = self.n_jobs if self.n_jobs > 0 else cpu_count()
            console.log(f"  Using {n_jobs} parallel workers")

        # prepare arguments for parallel processing
        args_list = [
            (ki, keyframe_indices[ki], half_window, str(temp_frames_path), None, self.percentile, use_gpu)
            for ki in range(n_keyframes)
        ]

        if n_jobs > 1:
            console.log(f"  Processing keyframes in parallel...")
            results = []
            with tqdm(total=n_keyframes, desc="  Computing keyframe baselines") as pbar:
                for result in Parallel(n_jobs=n_jobs, return_as='generator')(
                        delayed(_compute_keyframe_f0)(args)
                        for args in args_list
                ):
                    results.append(result)
                    pbar.update(1)
        else:
            console.log(f"  Processing keyframes sequentially...")
            results = [
                _compute_keyframe_f0(args)
                for args in tqdm(args_list, desc="  Computing keyframe baselines")
            ]

        # Collect results into array
        f0_keyframes = np.zeros((n_keyframes, n_pixels), dtype=np.float32)
        for ki, f0_values in results:
            f0_keyframes[ki, :] = f0_values

        return f0_keyframes, n_jobs

    def _interpolate_f0_vectorized(self, chunk_start: int,
                                   chunk_len: int,
                                   f0_keyframes: np.ndarray,
                                   keyframe_indices: list[int],
                                   n_keyframes: int,
                                   use_gpu: bool = False) -> np.ndarray:
        """
        Vectorized F0 interpolation for a chunk of frames.

        :param chunk_start: Starting frame index
        :param chunk_len: Number of frames in chunk
        :param f0_keyframes: F0 values at keyframe positions
        :param keyframe_indices: List of keyframe indices
        :param n_keyframes: Number of keyframes
        :param use_gpu: Whether to use GPU acceleration
        :return: Interpolated F0 values for the chunk
        """
        xp = cp if use_gpu and _HAS_CUPY else np

        global_indices = xp.arange(chunk_start, chunk_start + chunk_len, dtype=xp.int32)
        kf_indices_arr = xp.asarray(keyframe_indices, dtype=xp.int32)

        # find surrounding keyframes for all frames at once
        ki = xp.searchsorted(kf_indices_arr, global_indices, side='right') - 1
        ki = xp.clip(ki, 0, n_keyframes - 2)

        # keyframe indices
        kf1_idx = kf_indices_arr[ki]
        kf2_idx = kf_indices_arr[ki + 1]

        # compute interpolation weights (vectorized)
        denom = kf2_idx - kf1_idx
        alpha = xp.where(denom == 0, 0, (global_indices - kf1_idx) / denom).astype(xp.float32)

        if use_gpu and _HAS_CUPY:
            # Transfer all needed keyframes to GPU once, then process in sub-chunks
            # This avoids boundary artifacts from loading different keyframe subsets
            n_pixels = f0_keyframes.shape[1]

            # Find the full range of keyframes needed for this chunk
            ki_cpu = cp.asnumpy(ki)
            ki_min = int(ki_cpu.min())
            ki_max = int(ki_cpu.max())

            # Transfer all needed keyframes to GPU (includes ki_max + 1 for interpolation)
            f0_needed_cpu = f0_keyframes[ki_min:ki_max + 2]  # +2 to include ki_max+1

            try:
                # do everything on GPU at once
                f0_needed_gpu = cp.asarray(f0_needed_cpu, dtype=cp.float32)
                ki_adjusted = ki - ki_min

                bytes_needed = chunk_len * n_pixels * 4

                # if result > 2GB, process in sub-chunks, else process entire chunk on GPU
                if bytes_needed > 2 * 1024 ** 3:
                    target_bytes = 1 * 1024 ** 3
                    sub_chunk_size = max(100, int(target_bytes / (n_pixels * 4)))
                    result = np.empty((chunk_len, n_pixels), dtype=np.float32)

                    for sub_start in range(0, chunk_len, sub_chunk_size):
                        sub_end = min(sub_start + sub_chunk_size, chunk_len)

                        ki_sub = ki_adjusted[sub_start:sub_end]
                        alpha_sub = alpha[sub_start:sub_end]

                        # Interpolate on GPU using the same keyframe data
                        f0_sub_2d = (1 - alpha_sub[:, None]) * f0_needed_gpu[ki_sub] + \
                                    alpha_sub[:, None] * f0_needed_gpu[ki_sub + 1]

                        f0_sub_cpu = cp.asnumpy(f0_sub_2d)
                        result[sub_start:sub_end] = np.ascontiguousarray(f0_sub_cpu)

                        del f0_sub_2d

                    del f0_needed_gpu, ki_adjusted
                    cp.cuda.Device().synchronize()
                    cp.get_default_memory_pool().free_all_blocks()
                else:
                    f0_chunk_2d = (1 - alpha[:, None]) * f0_needed_gpu[ki_adjusted] + \
                                  alpha[:, None] * f0_needed_gpu[ki_adjusted + 1]
                    result = np.ascontiguousarray(cp.asnumpy(f0_chunk_2d))

                    del f0_needed_gpu, ki_adjusted, f0_chunk_2d
                    cp.cuda.Device().synchronize()
                    cp.get_default_memory_pool().free_all_blocks()

            except cp.cuda.memory.OutOfMemoryError:
                console.log(f"  [yellow]GPU out of memory for interpolation, falling back to CPU")
                f0_chunk_2d = (1 - alpha[:, None]) * f0_keyframes[ki] + alpha[:, None] * f0_keyframes[ki + 1]
                result = np.ascontiguousarray(f0_chunk_2d)
        else:
            f0_chunk_2d = (1 - alpha[:, None]) * f0_keyframes[ki] + alpha[:, None] * f0_keyframes[ki + 1]
            result = np.ascontiguousarray(f0_chunk_2d)

        return result

    def _interpolate_and_compute_dff(self, frames_mmap: np.ndarray,
                                     f0_keyframes: np.ndarray,
                                     keyframe_indices: list[int],
                                     dff_path: Path,
                                     f0_path: Path):
        """
        Interpolate F0 baseline and compute ΔF/F.

        :param frames_mmap: Memory-mapped array of frames
        :param f0_keyframes: F0 values at keyframe positions
        :param keyframe_indices: List of keyframe indices
        :param dff_path: Output path for dF/F array
        :param f0_path: Output path for F0 array (if save_f0 enabled)
        """
        console.log(f"  Computing ΔF/F with interpolated baseline...")

        n_keyframes = len(keyframe_indices)
        n_pixels = self._frame_shape[0] * self._frame_shape[1]

        use_gpu = self.use_gpu and _HAS_CUPY
        if use_gpu:
            console.log(f"  Using GPU-accelerated vectorized interpolation")
        else:
            console.log(f"  Using CPU vectorized interpolation")

        dff_output = np.lib.format.open_memmap(
            dff_path,
            mode='w+',
            dtype=np.float32,
            shape=(self._total_frames, *self._frame_shape)
        )

        # optional save as compression
        if self.save_f0:
            f0_file = h5py.File(f0_path, 'w')
            f0_dataset = f0_file.create_dataset(
                'f0',
                shape=(self._total_frames, *self._frame_shape),
                dtype=np.float32,
                chunks=(min(1000, self._total_frames), *self._frame_shape),
                compression='gzip',
                compression_opts=4
            )
        else:
            f0_file = None
            f0_dataset = None

        # process in chunks for memory efficiency
        chunk_size = self.chunk_size
        pbar = tqdm(total=self._total_frames, desc="  Computing dF/F")

        for chunk_start in range(0, self._total_frames, chunk_size):
            chunk_end = min(chunk_start + chunk_size, self._total_frames)
            chunk_len = chunk_end - chunk_start

            chunk_data = frames_mmap[chunk_start:chunk_end]
            chunk_data = np.ascontiguousarray(chunk_data)

            # vectorized interpolation
            f0_chunk_2d = self._interpolate_f0_vectorized(
                chunk_start,
                chunk_len,
                f0_keyframes,
                keyframe_indices,
                n_keyframes,
                use_gpu
            )

            # reshape F0 with explicit C-order to prevent grid artifacts
            f0_chunk_2d = np.ascontiguousarray(f0_chunk_2d)
            f0_chunk = f0_chunk_2d.reshape(chunk_len, *self._frame_shape, order='C')
            epsilon = 1e-10
            dff_chunk = (chunk_data - f0_chunk) / (f0_chunk + epsilon)

            # write
            dff_output[chunk_start:chunk_end] = dff_chunk
            if self.save_f0:
                f0_dataset[chunk_start:chunk_end] = f0_chunk

            pbar.update(chunk_len)

        pbar.close()

        if f0_file is not None:
            f0_file.close()

        del dff_output

    def calculate_dff(self):
        """Calculate ΔF/F with time-varying baseline using rolling window"""
        console.log(f"  Creating output arrays...")
        dff_path = self.output_dir / 'dff.npy'
        f0_path = self.output_dir / 'f0.h5'
        temp_frames_path = self.output_dir / 'temp_frames.npy'

        # Step 1: load frames to memory-mapped array
        frames_mmap = self._load_frames_to_memmap(temp_frames_path)

        # Step 2: compute processing parameters
        win = min(self.window_size, self._total_frames)
        if win != self.window_size:
            console.log(f"  Note: Window size adjusted from {self.window_size} to {win}")

        half_window = win // 2
        stride = max(1, win // 10)

        keyframe_indices = list(range(0, self._total_frames, stride))
        if keyframe_indices[-1] != self._total_frames - 1:
            keyframe_indices.append(self._total_frames - 1)

        # Step 3: compute F0 at keyframes
        use_gpu = True if self.use_gpu and _HAS_CUPY else False
        f0_keyframes, n_jobs = self._compute_f0_keyframes(
            temp_frames_path,
            keyframe_indices,
            half_window,
            use_gpu
        )

        # metadata
        self._f0_stride = stride
        self._f0_n_keyframes = len(keyframe_indices)
        self._f0_n_jobs_used = n_jobs

        # Step 4: interpolate F0 and compute dF/F
        self._interpolate_and_compute_dff(
            frames_mmap,
            f0_keyframes,
            keyframe_indices,
            dff_path,
            f0_path
        )

        # cleanup
        del frames_mmap
        del f0_keyframes
        temp_frames_path.unlink()

        console.log(f"  [green]✓[/] ΔF/F saved to: [green]{dff_path}")
        if self.save_f0:
            console.log(f"  [green]✓[/] F0 baseline saved to: [green]{f0_path}")
        else:
            console.log(f"  [dim]F0 baseline not saved (--save_f0 disabled)")

        # Post-processing: apply rotation
        if self.rotate is not None:
            from wfanalysis.preprocess.util import rotate_sequence
            console.log(f"  [bold cyan]Applying rotation ({self.rotate}°) to dF/F output...")
            rotate_sequence(dff_path, rotate=self.rotate, overwrite=True)
            console.log(f"  [green]✓[/] Rotation applied to dF/F")

            # rotate reference frame
            reference_path = self.output_dir / 'reference_frame.tif'
            if reference_path.exists():
                console.log(f"  Applying rotation to reference frame...")
                ref_img = tifffile.imread(reference_path)
                import cv2
                height, width = ref_img.shape
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, self.rotate, scale=1.0)
                ref_rotated = cv2.warpAffine(ref_img, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR)
                tifffile.imwrite(reference_path, ref_rotated.astype(np.float32))
                console.log(f"  [green]✓[/] Reference frame rotated")

    def save_metadata(self):
        metadata: PreprocessMeta = {
            'timestamp': datetime.now().isoformat(),
            'input_arguments': {
                'input_source': str(self.file) if self.file else str(self.directory),
                'suffix_pattern': self.suffix_pattern,
                'output_dir': str(self.output_dir),
                'motion_correction': self.motion_correction,
                'rotate': self.rotate,
                'chunk_size': self.chunk_size,
                'window_size': self.window_size,
                'percentile': self.percentile,
                'n_jobs': self.n_jobs,
                'max_shift': self.max_shift,
                'force_compute': self.force_compute,
                'save_f0': self.save_f0,
                'use_gpu': self.use_gpu,
            },
            'data_info': {
                'n_tif_files': len(self._tif_files) if self._tif_files else 0,
                'tif_files': [str(f) for f in self._tif_files] if self._tif_files else [],
                'total_frames': self._total_frames,
                'frame_shape': list(self._frame_shape) if self._frame_shape else None,
                'image_height': self._frame_shape[0] if self._frame_shape else None,
                'image_width': self._frame_shape[1] if self._frame_shape else None,
            },
            'processing': {
                'rotation_applied': self.rotate is not None,
                'rotation_degrees': self.rotate,
                'has_numba': _HAS_NUMBA,
                'has_cupy': _HAS_CUPY,
                'gpu_used': self.use_gpu and _HAS_CUPY,
            }
        }

        # f0
        if self._f0_stride is not None:
            f0_metadata = {
                'percentile': self.percentile,
                'stride': self._f0_stride,
                'n_keyframes': self._f0_n_keyframes,
            }
            metadata['f0_baseline'] = f0_metadata
        else:
            metadata['f0_baseline'] = None

        # motion correction metadata
        if self.motion_correction and self._transform_cache_path and self._transform_cache_path.exists():
            motion_metadata = {}
            metadata['motion_correction'] = motion_metadata
        else:
            metadata['motion_correction'] = None

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        console.log(f"  [green]✓[/] Metadata saved to: [green]{metadata_path}")


@jit(nopython=True, parallel=False, cache=True)
def _fast_percentile_2d(data: np.ndarray, percentile: float) -> np.ndarray:
    """
    Fast percentile computation along axis=0 using numba.

    :param data: (n_frames, n_pixels) array
    :param percentile: percentile value (0-100)
    :return: result: (n_pixels,) array of percentile values
    """
    n_frames, n_pixels = data.shape
    result = np.empty(n_pixels, dtype=data.dtype)

    for i in range(n_pixels):
        column = data[:, i].copy()
        column.sort()
        idx = int(percentile / 100.0 * (n_frames - 1))
        result[i] = column[idx]

    return result


def _compute_keyframe_f0(args) -> tuple[int, np.ndarray]:
    ki, frame_idx, half_window, frames_memmap_path, transform_cache_path, percentile, use_gpu = args

    frames_mmap = np.load(frames_memmap_path, mmap_mode='r')
    total_frames = frames_mmap.shape[0]
    n_pixels = frames_mmap.shape[1] * frames_mmap.shape[2]

    start_idx = max(0, frame_idx - half_window)
    end_idx = min(total_frames, frame_idx + half_window + 1)

    window_frames = frames_mmap[start_idx:end_idx]
    window_frames = np.ascontiguousarray(window_frames)

    n_frames_in_window = end_idx - start_idx
    window_data = window_frames.reshape(n_frames_in_window, n_pixels, order='C')

    if use_gpu and _HAS_CUPY:
        try:
            # transfer to GPU, compute percentile, transfer back
            window_data_gpu = cp.asarray(window_data.astype(np.float32))
            f0_values_gpu = cp.percentile(window_data_gpu, percentile, axis=0)
            f0_values = cp.asnumpy(f0_values_gpu).astype(np.float32)

            # ensure all GPU operations complete before cleanup, then clear gpu memory
            cp.cuda.Device().synchronize()
            del window_data_gpu, f0_values_gpu
            cp.get_default_memory_pool().free_all_blocks()

        except cp.cuda.memory.OutOfMemoryError:
            if _HAS_NUMBA:
                f0_values = _fast_percentile_2d(window_data.astype(np.float32), percentile)
            else:
                f0_values = np.percentile(window_data, percentile, axis=0).astype(np.float32)
    elif _HAS_NUMBA:
        f0_values = _fast_percentile_2d(window_data.astype(np.float32), percentile)
    else:
        f0_values = np.percentile(window_data, percentile, axis=0).astype(np.float32)

    return ki, f0_values


if __name__ == '__main__':
    PreprocessOptions().main()
