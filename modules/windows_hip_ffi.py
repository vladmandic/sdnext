import sys

if sys.platform == "win32":
    import ctypes
    import ctypes.wintypes

    class hipDeviceProp(ctypes.Structure):
        _fields_ = [
            ('__front__', ctypes.c_byte * 396),
            ('gcnArchName', ctypes.c_char * 256),
            ('__rear__', ctypes.c_byte * 820)
        ]

    class HIP:
        def __init__(self):
            ctypes.windll.kernel32.LoadLibraryA.restype = ctypes.wintypes.HMODULE
            ctypes.windll.kernel32.LoadLibraryA.argtypes = [ctypes.c_char_p]
            # amdhip64.dll is a part of AMDGPU drivers
            self.handle = ctypes.windll.kernel32.LoadLibraryA(b"amdhip64.dll")
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
            #ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.wintypes.HMODULE]
            #ctypes.windll.kernel32.FreeLibrary(self.handle)
            # Hopefully it does not make conflicts with amdhip64_7.dll
            pass

        def get_device_count(self):
            count = ctypes.c_int()
            assert self.hipGetDeviceCount(ctypes.byref(count)) == 0
            return count.value

        def get_device_properties(self, device_id):
            prop = hipDeviceProp()
            assert self.hipGetDeviceProperties(ctypes.byref(prop), device_id) == 0
            return prop
