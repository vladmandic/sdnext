import sys

if sys.platform == "win32":
    import os
    import ctypes
    import ctypes.wintypes

    class hipDeviceProp(ctypes.Structure):
        _fields_ = [
            ('bytes', ctypes.c_byte * 1472) # 1472 in amdhip64_6.dll, shorter in amdhip64_7.dll?
        ]

    class HIP:
        def __init__(self):
            ctypes.windll.kernel32.LoadLibraryA.restype = ctypes.wintypes.HMODULE
            ctypes.windll.kernel32.LoadLibraryA.argtypes = [ctypes.c_char_p]
            self.handle = None
            path = os.environ.get("windir", "C:\\Windows") + "\\System32\\amdhip64_7.dll"
            if not os.path.isfile(path):
                path = os.environ.get("windir", "C:\\Windows") + "\\System32\\amdhip64_6.dll"
            if not os.path.isfile(path):
                path = os.environ.get("windir", "C:\\Windows") + "\\System32\\amdhip64.dll"
            assert os.path.isfile(path)
            self.handle = ctypes.windll.kernel32.LoadLibraryA(path.encode('utf-8'))
            ctypes.windll.kernel32.GetLastError.restype = ctypes.wintypes.DWORD
            ctypes.windll.kernel32.GetLastError.argtypes = []
            assert ctypes.windll.kernel32.GetLastError() == 0
            ctypes.windll.kernel32.GetProcAddress.restype = ctypes.c_void_p
            ctypes.windll.kernel32.GetProcAddress.argtypes = [ctypes.wintypes.HMODULE, ctypes.c_char_p]
            self.hipGetDeviceCount = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_int))(
                ctypes.windll.kernel32.GetProcAddress(self.handle, b"hipGetDeviceCount"))
            self.hipGetDeviceProperties = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(hipDeviceProp), ctypes.c_int)(
                ctypes.windll.kernel32.GetProcAddress(self.handle, b"hipGetDeviceProperties"))

        def __del__(self):
            if self.handle is None:
                return
            # Hopefully this will prevent conflicts with amdhip64_7.dll from ROCm Python packages or HIP SDK
            ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.wintypes.HMODULE]
            ctypes.windll.kernel32.FreeLibrary(self.handle)

        def get_device_count(self):
            count = ctypes.c_int()
            assert self.hipGetDeviceCount(ctypes.byref(count)) == 0
            return count.value

        def get_device_properties(self, device_id):
            prop = hipDeviceProp()
            assert self.hipGetDeviceProperties(ctypes.byref(prop), device_id) == 0
            return prop.bytes

    def get_archs():
        hip = HIP()

        count = hip.get_device_count()
        archs = [None] * count
        for i in range(count):
            prop = hip.get_device_properties(i)[:]

            name = ""
            idx = 0
            while idx < len(prop):
                try:
                    idx = prop.index(0x67, idx) + 1 # 'g'
                except ValueError:
                    break
                if prop[idx] != 0x66: # 'f'
                    continue
                if prop[idx + 1] != 0x78: # 'x'
                    continue

                idx = idx + 2
                while prop[idx] != 0x00:
                    c = prop[idx]
                    idx += 1
                    if (c < 0x30 or c > 0x39) and (c < 0x61 or c > 0x66): # hexadecimal
                        name = ""
                        continue
                    name += chr(c)
                break

            # if name == "", hipDeviceProp does not contain arch name
            if name:
                archs[i] = "gfx" + name

        del hip
        return archs
