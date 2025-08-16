import cupy as cp
from cupy import cuda


def get_device(device_id: int = 0):
    try:
        dev = cuda.Device(device_id)
        dev.use()
        return dev
    except Exception as e:
        raise RuntimeError("CUDA device not available; ensure NVIDIA driver and CuPy are installed.") from e


def gpu_info():
    dev = cuda.Device()
    props = dev.attributes
    name = dev.name.decode() if isinstance(dev.name, bytes) else dev.name
    total_mem = dev.mem_info[1]
    cc_major = props.get(cuda.runtime.cudaDeviceAttr.COMPUTE_CAPABILITY_MAJOR, 0)
    cc_minor = props.get(cuda.runtime.cudaDeviceAttr.COMPUTE_CAPABILITY_MINOR, 0)
    return {
        "name": name,
        "total_mem_gb": total_mem / (1024 ** 3),
        "cc": f"{cc_major}.{cc_minor}",
    }


def as_cupy(x, dtype=cp.float64):
    return cp.asarray(x, dtype=dtype)


def as_numpy(x):
    return cp.asnumpy(x)

