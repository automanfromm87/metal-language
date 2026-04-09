"""Metal compiler and runtime - compile host code and execute"""

import hashlib
import os
import struct
import subprocess
import tempfile
import numpy as np


# Global compile cache: {code_hash: exe_path}
_compile_cache = {}
# Persistent temp directory (not auto-cleaned)
_cache_dir = None


def _get_cache_dir():
    global _cache_dir
    if _cache_dir is None:
        _cache_dir = tempfile.mkdtemp(prefix="metal_dsl_cache_")
    return _cache_dir


def compile_and_run(metal_code, host_code, pointer_arrays, scalar_values,
                    constexpr_vals, kernel_name, n_runs=1):
    """Compile and execute a Metal kernel

    Data protocol (stdin):
    1. n_buffers uint32_t values: byte size of each buffer
    2. Raw data for each buffer
    3. uint32_t value for each scalar parameter

    When n_runs > 1 (benchmark mode), returns (result_arrays, gpu_times)
    where gpu_times is a list of per-run GPU times in seconds.
    """
    # Check compile cache
    code_hash = hashlib.md5(host_code.encode()).hexdigest()[:16]
    if code_hash in _compile_cache:
        exe_path = _compile_cache[code_hash]
    else:
        cache_dir = _get_cache_dir()
        host_path = os.path.join(cache_dir, f"{kernel_name}_{code_hash}.m")
        exe_path = os.path.join(cache_dir, f"{kernel_name}_{code_hash}")

        with open(host_path, "w") as f:
            f.write(host_code)

        _run_cmd([
            "clang",
            "-framework", "Metal",
            "-framework", "Foundation",
            "-O2",
            "-o", exe_path,
            host_path,
        ], "Host code compilation failed")

        _compile_cache[code_hash] = exe_path

    # Prepare stdin data
    stdin_data = b""

    # 1. Buffer sizes header
    n_buffers = len(pointer_arrays)
    for arr in pointer_arrays:
        stdin_data += struct.pack("<I", arr.nbytes)

    # 2. Buffer data
    for arr in pointer_arrays:
        stdin_data += arr.tobytes()

    # 3. Scalar parameters
    for val in scalar_values:
        stdin_data += struct.pack("<I", int(val))

    # Execute
    result = subprocess.run(
        [exe_path],
        input=stdin_data,
        capture_output=True,
    )

    if result.returncode != 0:
        stderr_text = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"GPU execution failed:\n{stderr_text}")

    # Parse output: all buffers written back in order
    out_bytes = result.stdout
    offset = 0
    result_arrays = []
    for arr in pointer_arrays:
        nbytes = arr.nbytes
        arr_data = out_bytes[offset:offset + nbytes]
        result_arrays.append(
            np.frombuffer(arr_data, dtype=arr.dtype).copy()
        )
        offset += nbytes

    if n_runs > 1:
        # Parse GPU timing data from stderr (n_runs doubles)
        gpu_times = []
        stderr_bytes = result.stderr
        for i in range(n_runs):
            t = struct.unpack("<d", stderr_bytes[i*8:(i+1)*8])[0]
            gpu_times.append(t)
        return result_arrays, gpu_times

    return result_arrays


def _run_cmd(cmd, error_msg):
    """Run a command, raise on failure"""
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"{error_msg}:\n{stderr}")
