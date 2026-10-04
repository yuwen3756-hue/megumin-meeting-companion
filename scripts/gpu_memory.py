"""Read Windows PDH dedicated GPU memory counters without WMI or drivers."""
import ctypes as C
from ctypes import wintypes as W
import os


class _Number(C.Union):
    _fields_ = [("longValue", W.LONG), ("doubleValue", C.c_double), ("largeValue", C.c_longlong), ("stringValue", C.c_void_p)]


class _Value(C.Structure):
    _fields_ = [("status", W.DWORD), ("number", _Number)]


class _Item(C.Structure):
    _fields_ = [("name", W.LPWSTR), ("value", _Value)]


class GpuMemory:
    def __init__(self):
        self.pdh = C.WinDLL("pdh")
        self.query = C.c_void_p()
        self.pdh.PdhOpenQueryW.argtypes = [W.LPCWSTR, C.c_size_t, C.POINTER(C.c_void_p)]
        self.pdh.PdhAddEnglishCounterW.argtypes = [C.c_void_p, W.LPCWSTR, C.c_size_t, C.POINTER(C.c_void_p)]
        self.pdh.PdhCollectQueryData.argtypes = [C.c_void_p]
        self.pdh.PdhGetFormattedCounterArrayW.argtypes = [C.c_void_p, W.DWORD, C.POINTER(W.DWORD), C.POINTER(W.DWORD), C.c_void_p]
        self.pdh.PdhCloseQuery.argtypes = [C.c_void_p]
        if self.pdh.PdhOpenQueryW(None, 0, C.byref(self.query)):
            raise OSError("PDH query unavailable")
        self.handles = {}
        for name, path in (("adapter", r"\GPU Adapter Memory(*)\Dedicated Usage"),
                           ("process", r"\GPU Process Memory(*)\Dedicated Usage")):
            handle = C.c_void_p()
            if not self.pdh.PdhAddEnglishCounterW(self.query, path, 0, C.byref(handle)):
                self.handles[name] = handle
        if not self.handles:
            self.close()
            raise OSError("Windows GPU memory counters unavailable")

    def sample(self):
        if self.pdh.PdhCollectQueryData(self.query):
            return {}
        result = {}
        for kind, handle in self.handles.items():
            size, count = W.DWORD(), W.DWORD()
            self.pdh.PdhGetFormattedCounterArrayW(handle, 0x200, C.byref(size), C.byref(count), None)
            if not size.value:
                continue
            buffer = C.create_string_buffer(size.value)
            if self.pdh.PdhGetFormattedCounterArrayW(handle, 0x200, C.byref(size), C.byref(count), buffer):
                continue
            entries = C.cast(buffer, C.POINTER(_Item))
            values = {entries[i].name: entries[i].value.number.doubleValue for i in range(count.value)
                      if entries[i].value.status in (0, 1)}
            if kind == "process":
                values = {name: val for name, val in values.items() if f"pid_{os.getpid()}_" in name}
            result[kind + "_dedicated_bytes"] = sum(values.values())
            result[kind + "_instances"] = values
        return result

    def close(self):
        if self.query:
            self.pdh.PdhCloseQuery(self.query)
            self.query = C.c_void_p()


if __name__ == "__main__":
    import json
    sampler = GpuMemory()
    print(json.dumps(sampler.sample(), indent=2))
    sampler.close()
