# pylint: disable=redefined-builtin,no-member,protected-access

import os
import torch

from modules import shared, devices

sdnq_version = "0.1.4"

dtype_dict = {
    ### Integers
    "int32": {"min": -2147483648, "max": 2147483647, "num_bits": 32, "sign": 1, "exponent": 0, "mantissa": 31, "target_dtype": torch.int32, "torch_dtype": torch.int32, "storage_dtype": torch.int32, "is_unsigned": False, "is_integer": True, "is_packed": False},
    "int16": {"min": -32768, "max": 32767, "num_bits": 16, "sign": 1, "exponent": 0, "mantissa": 15, "target_dtype": torch.int16, "torch_dtype": torch.int16, "storage_dtype": torch.int16, "is_unsigned": False, "is_integer": True, "is_packed": False},
    "int8": {"min": -128, "max": 127, "num_bits": 8, "sign": 1, "exponent": 0, "mantissa": 7, "target_dtype": torch.int8, "torch_dtype": torch.int8, "storage_dtype": torch.int8, "is_unsigned": False, "is_integer": True, "is_packed": False},
    ### Custom Integers
    "int7": {"min": -64, "max": 63, "num_bits": 7, "sign": 1, "exponent": 0, "mantissa": 6, "target_dtype": "int7", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int6": {"min": -32, "max": 31, "num_bits": 6, "sign": 1, "exponent": 0, "mantissa": 5, "target_dtype": "int6", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int5": {"min": -16, "max": 15, "num_bits": 5, "sign": 1, "exponent": 0, "mantissa": 4, "target_dtype": "int5", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int4": {"min": -8, "max": 7, "num_bits": 4, "sign": 1, "exponent": 0, "mantissa": 3, "target_dtype": "int4", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int3": {"min": -4, "max": 3, "num_bits": 3, "sign": 1, "exponent": 0, "mantissa": 2, "target_dtype": "int3", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    "int2": {"min": -2, "max": 1, "num_bits": 2, "sign": 1, "exponent": 0, "mantissa": 1, "target_dtype": "int2", "torch_dtype": torch.int8, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": True, "is_packed": True},
    ### Unsigned Integers
    "uint32": {"min": 0, "max": 4294967295, "num_bits": 32, "sign": 0, "exponent": 0, "mantissa": 32, "target_dtype": torch.uint32, "torch_dtype": torch.uint32, "storage_dtype": torch.uint32, "is_unsigned": True, "is_integer": True, "is_packed": False},
    "uint16": {"min": 0, "max": 65535, "num_bits": 16, "sign": 0, "exponent": 0, "mantissa": 16, "target_dtype": torch.uint16, "torch_dtype": torch.uint16, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": True, "is_packed": False},
    "uint8": {"min": 0, "max": 255, "num_bits": 8, "sign": 0, "exponent": 0, "mantissa": 8, "target_dtype": torch.uint8, "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": False},
    ### Custom Unsigned Integers
    "uint7": {"min": 0, "max": 127, "num_bits": 7, "sign": 0, "exponent": 0, "mantissa": 7, "target_dtype": "uint7", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint6": {"min": 0, "max": 63, "num_bits": 6, "sign": 0, "exponent": 0, "mantissa": 6, "target_dtype": "uint6", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint5": {"min": 0, "max": 31, "num_bits": 5, "sign": 0, "exponent": 0, "mantissa": 5, "target_dtype": "uint5", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint4": {"min": 0, "max": 15, "num_bits": 4, "sign": 0, "exponent": 0, "mantissa": 4, "target_dtype": "uint4", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint3": {"min": 0, "max": 7, "num_bits": 3, "sign": 0, "exponent": 0, "mantissa": 3, "target_dtype": "uint3", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint2": {"min": 0, "max": 3, "num_bits": 2, "sign": 0, "exponent": 0, "mantissa": 2, "target_dtype": "uint2", "torch_dtype": torch.uint8, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": True, "is_packed": True},
    "uint1": {"min": 0, "max": 1, "num_bits": 1, "sign": 0, "exponent": 0, "mantissa": 1, "target_dtype": torch.bool, "torch_dtype": torch.bool, "storage_dtype": torch.bool, "is_unsigned": True, "is_integer": True, "is_packed": True},
    ### Floats
    "float32": {"min": -3.40282e+38, "max": 3.40282e+38, "num_bits": 32, "sign": 1, "exponent": 8, "mantissa": 23, "target_dtype": torch.float32, "torch_dtype": torch.float32, "storage_dtype": torch.float32, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "bfloat16": {"min": -3.38953e+38, "max": 3.38953e+38, "num_bits": 16, "sign": 1, "exponent": 8, "mantissa": 7, "target_dtype": torch.bfloat16, "torch_dtype": torch.bfloat16, "storage_dtype": torch.bfloat16, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "float16": {"min": -65504.0, "max": 65504.0, "num_bits": 16, "sign": 1, "exponent": 5, "mantissa": 10, "target_dtype": torch.float16, "torch_dtype": torch.float16, "storage_dtype": torch.float16, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "float8_e4m3fn": {"min": -448.0, "max": 448.0, "num_bits": 8, "sign": 1, "exponent": 4, "mantissa": 3, "target_dtype": torch.float8_e4m3fn, "torch_dtype": torch.float8_e4m3fn, "storage_dtype": torch.float8_e4m3fn, "is_unsigned": False, "is_integer": False, "is_packed": False},
    "float8_e5m2": {"min": -57344.0, "max": 57344.0, "num_bits": 8, "sign": 1, "exponent": 5, "mantissa": 2, "target_dtype": torch.float8_e5m2, "torch_dtype": torch.float8_e5m2, "storage_dtype": torch.float8_e5m2, "is_unsigned": False, "is_integer": False, "is_packed": False},
    ### Custom Floats
    "float16_e1m14fn": {"min": -3.9998779296875, "max": 3.9998779296875, "num_bits": 16, "sign": 1, "exponent": 1, "mantissa": 14, "min_normal": 1.00006103515625, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float16_e2m13fn": {"min": -7.99951171875, "max": 7.99951171875, "num_bits": 16, "sign": 1, "exponent": 2, "mantissa": 13, "min_normal": 0.50006103515625, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float16_e3m12fn": {"min": -31.99609375, "max": 31.99609375, "num_bits": 16, "sign": 1, "exponent": 3, "mantissa": 12, "min_normal": 0.125030517578125, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float16_e4m11fn": {"min": -511.875, "max": 511.875, "num_bits": 16, "sign": 1, "exponent": 4, "mantissa": 11, "min_normal": 0.007816314697265625, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": False, "is_integer": False, "is_packed": True},
    # float16_e5m10 is native in PyTorch
    "float8_e1m6fn": {"min": -3.96875, "max": 3.96875, "num_bits": 8, "sign": 1, "exponent": 1, "mantissa": 6, "min_normal": 1.015625, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float8_e2m5fn": {"min": -7.875, "max": 7.875, "num_bits": 8, "sign": 1, "exponent": 2, "mantissa": 5, "min_normal": 0.515625, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float8_e3m4fn": {"min": -31.0, "max": 31.0, "num_bits": 8, "sign": 1, "exponent": 3, "mantissa": 4, "min_normal": 0.1328125, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    # float8_e4m3fn is native in PyTorch
    # float8_e5m2fn is native in PyTorch
    "float7_e1m5fn": {"min": -3.9375, "max": 3.9375, "num_bits": 7, "sign": 1, "exponent": 1, "mantissa": 5, "min_normal": 1.03125, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float7_e2m4fn": {"min": -7.75, "max": 7.75, "num_bits": 7, "sign": 1, "exponent": 2, "mantissa": 4, "min_normal": 0.53125, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float7_e3m3fn": {"min": -30.0, "max": 30.0, "num_bits": 7, "sign": 1, "exponent": 3, "mantissa": 3, "min_normal": 0.140625, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float7_e4m2fn": {"min": -448.0, "max": 448.0, "num_bits": 7, "sign": 1, "exponent": 4, "mantissa": 2, "min_normal": 0.009765625, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float7_e5m1fn": {"min": -98304.0, "max": 98304.0, "num_bits": 7, "sign": 1, "exponent": 5, "mantissa": 1, "min_normal": 4.57763671875e-05, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    #
    "float6_e1m4fn": {"min": -3.875, "max": 3.875, "num_bits": 6, "sign": 1, "exponent": 1, "mantissa": 4, "min_normal": 1.0625, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float6_e2m3fn": {"min": -7.5, "max": 7.5, "num_bits": 6, "sign": 1, "exponent": 2, "mantissa": 3, "min_normal": 0.5625, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float6_e3m2fn": {"min": -28.0, "max": 28.0, "num_bits": 6, "sign": 1, "exponent": 3, "mantissa": 2, "min_normal": 0.15625, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float6_e4m1fn": {"min": -384.0, "max": 384.0, "num_bits": 6, "sign": 1, "exponent": 4, "mantissa": 1, "min_normal": 0.01171875, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float6_e5m0fn": {"min": -65536.0, "max": 65536.0, "num_bits": 6, "sign": 1, "exponent": 5, "mantissa": 0, "min_normal": 6.103515625e-05, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    #
    "float5_e1m3fn": {"min": -3.75, "max": 3.75, "num_bits": 5, "sign": 1, "exponent": 1, "mantissa": 3, "min_normal": 1.125, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float5_e2m2fn": {"min": -7.0, "max": 7.0, "num_bits": 5, "sign": 1, "exponent": 2, "mantissa": 2, "min_normal": 0.625, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float5_e3m1fn": {"min": -24.0, "max": 24.0, "num_bits": 5, "sign": 1, "exponent": 3, "mantissa": 1, "min_normal": 0.1875, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float5_e4m0fn": {"min": -256.0, "max": 256.0, "num_bits": 5, "sign": 1, "exponent": 4, "mantissa": 0, "min_normal": 0.015625, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    #
    "float4_e1m2fn": {"min": -3.5, "max": 3.5, "num_bits": 4, "sign": 1, "exponent": 1, "mantissa": 2, "min_normal": 1.25, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float4_e2m1fn": {"min": -6.0, "max": 6.0, "num_bits": 4, "sign": 1, "exponent": 2, "mantissa": 1, "min_normal": 0.75, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float4_e3m0fn": {"min": -16.0, "max": 16.0, "num_bits": 4, "sign": 1, "exponent": 3, "mantissa": 0, "min_normal": 0.25, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    #
    "float3_e1m1fn": {"min": -3.0, "max": 3.0, "num_bits": 3, "sign": 1, "exponent": 1, "mantissa": 1, "min_normal": 1.5, "target_dtype": "fp3", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    "float3_e2m0fn": {"min": -4.0, "max": 4.0, "num_bits": 3, "sign": 1, "exponent": 2, "mantissa": 0, "min_normal": 1.0, "target_dtype": "fp3", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    #
    "float2_e1m0fn": {"min": -2.0, "max": 2.0, "num_bits": 2, "sign": 1, "exponent": 1, "mantissa": 0, "min_normal": 2.0, "target_dtype": "fp2", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": False, "is_integer": False, "is_packed": True},
    ### Custom Usigned Floats
    "float16_e1m15fnu": {"min": 0, "max": 3.99993896484375, "num_bits": 16, "sign": 0, "exponent": 1, "mantissa": 15, "min_normal": 1.000030517578125, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float16_e2m14fnu": {"min": 0, "max": 7.999755859375, "num_bits": 16, "sign": 0, "exponent": 2, "mantissa": 14, "min_normal": 0.500030517578125, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float16_e3m13fnu": {"min": 0, "max": 31.998046875, "num_bits": 16, "sign": 0, "exponent": 3, "mantissa": 13, "min_normal": 0.1250152587890625, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float16_e4m12fnu": {"min": 0, "max": 511.9375, "num_bits": 16, "sign": 0, "exponent": 4, "mantissa": 12, "min_normal": 0.007814407348632812, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float16_e5m11fnu": {"min": 0, "max": 131040.0, "num_bits": 16, "sign": 0, "exponent": 5, "mantissa": 11, "min_normal": 3.053247928619385e-05, "target_dtype": torch.float16, "torch_dtype": torch.float32, "storage_dtype": torch.uint16, "is_unsigned": True, "is_integer": False, "is_packed": True},
    #
    "float8_e1m7fnu": {"min": 0, "max": 3.984375, "num_bits": 8, "sign": 0, "exponent": 1, "mantissa": 7, "min_normal": 1.0078125, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float8_e2m6fnu": {"min": 0, "max": 7.9375, "num_bits": 8, "sign": 0, "exponent": 2, "mantissa": 6, "min_normal": 0.5078125, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float8_e3m5fnu": {"min": 0, "max": 31.5, "num_bits": 8, "sign": 0, "exponent": 3, "mantissa": 5, "min_normal": 0.12890625, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float8_e4m4fnu": {"min": 0, "max": 496.0, "num_bits": 8, "sign": 0, "exponent": 4, "mantissa": 4, "min_normal": 0.00830078125, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float8_e5m3fnu": {"min": 0, "max": 122880.0, "num_bits": 8, "sign": 0, "exponent": 5, "mantissa": 3, "min_normal": 3.4332275390625e-05, "target_dtype": "fp8", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    #
    "float7_e1m6fnu": {"min": 0, "max": 3.96875, "num_bits": 7, "sign": 0, "exponent": 1, "mantissa": 6, "min_normal": 1.015625, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float7_e2m5fnu": {"min": 0, "max": 7.875, "num_bits": 7, "sign": 0, "exponent": 2, "mantissa": 5, "min_normal": 0.515625, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float7_e3m4fnu": {"min": 0, "max": 31.0, "num_bits": 7, "sign": 0, "exponent": 3, "mantissa": 4, "min_normal": 0.1328125, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float7_e4m3fnu": {"min": 0, "max": 480.0, "num_bits": 7, "sign": 0, "exponent": 4, "mantissa": 3, "min_normal": 0.0087890625, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float7_e5m2fnu": {"min": 0, "max": 114688.0, "num_bits": 7, "sign": 0, "exponent": 5, "mantissa": 2, "min_normal": 3.814697265625e-05, "target_dtype": "fp7", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    #
    "float6_e1m5fnu": {"min": 0, "max": 3.9375, "num_bits": 6, "sign": 0, "exponent": 1, "mantissa": 5, "min_normal": 1.03125, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float6_e2m4fnu": {"min": 0, "max": 7.75, "num_bits": 6, "sign": 0, "exponent": 2, "mantissa": 4, "min_normal": 0.53125, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float6_e3m3fnu": {"min": 0, "max": 30.0, "num_bits": 6, "sign": 0, "exponent": 3, "mantissa": 3, "min_normal": 0.140625, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float6_e4m2fnu": {"min": 0, "max": 448.0, "num_bits": 6, "sign": 0, "exponent": 4, "mantissa": 2, "min_normal": 0.009765625, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float6_e5m1fnu": {"min": 0, "max": 98304.0, "num_bits": 6, "sign": 0, "exponent": 5, "mantissa": 1, "min_normal": 4.57763671875e-05, "target_dtype": "fp6", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    #
    "float5_e1m4fnu": {"min": 0, "max": 3.875, "num_bits": 5, "sign": 0, "exponent": 1, "mantissa": 4, "min_normal": 1.0625, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float5_e2m3fnu": {"min": 0, "max": 7.5, "num_bits": 5, "sign": 0, "exponent": 2, "mantissa": 3, "min_normal": 0.5625, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float5_e3m2fnu": {"min": 0, "max": 28.0, "num_bits": 5, "sign": 0, "exponent": 3, "mantissa": 2, "min_normal": 0.15625, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float5_e4m1fnu": {"min": 0, "max": 384.0, "num_bits": 5, "sign": 0, "exponent": 4, "mantissa": 1, "min_normal": 0.01171875, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float5_e5m0fnu": {"min": 0, "max": 65536.0, "num_bits": 5, "sign": 0, "exponent": 5, "mantissa": 0, "min_normal": 6.103515625e-05, "target_dtype": "fp5", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    #
    "float4_e1m3fnu": {"min": 0, "max": 3.75, "num_bits": 4, "sign": 0, "exponent": 1, "mantissa": 3, "min_normal": 1.125, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float4_e2m2fnu": {"min": 0, "max": 7.0, "num_bits": 4, "sign": 0, "exponent": 2, "mantissa": 2, "min_normal": 0.625, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float4_e3m1fnu": {"min": 0, "max": 24.0, "num_bits": 4, "sign": 0, "exponent": 3, "mantissa": 1, "min_normal": 0.1875, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float4_e4m0fnu": {"min": 0, "max": 256.0, "num_bits": 4, "sign": 0, "exponent": 4, "mantissa": 0, "min_normal": 0.015625, "target_dtype": "fp4", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    #
    "float3_e1m2fnu": {"min": 0, "max": 3.5, "num_bits": 3, "sign": 0, "exponent": 1, "mantissa": 2, "min_normal": 1.25, "target_dtype": "fp3", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float3_e2m1fnu": {"min": 0, "max": 6.0, "num_bits": 3, "sign": 0, "exponent": 2, "mantissa": 1, "min_normal": 0.75, "target_dtype": "fp3", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float3_e3m0fnu": {"min": 0, "max": 16.0, "num_bits": 3, "sign": 0, "exponent": 3, "mantissa": 0, "min_normal": 0.25, "target_dtype": "fp3", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    #
    "float2_e1m1fnu": {"min": 0, "max": 3.0, "num_bits": 2, "sign": 0, "exponent": 1, "mantissa": 1, "min_normal": 1.5, "target_dtype": "fp2", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    "float2_e2m0fnu": {"min": 0, "max": 4.0, "num_bits": 2, "sign": 0, "exponent": 2, "mantissa": 0, "min_normal": 1.0, "target_dtype": "fp2", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
    #
    "float1_e1m0fnu": {"min": 0, "max": 2.0, "num_bits": 1, "sign": 0, "exponent": 1, "mantissa": 0, "min_normal": 2.0, "target_dtype": "fp1", "torch_dtype": torch.float32, "storage_dtype": torch.uint8, "is_unsigned": True, "is_integer": False, "is_packed": True},
}

dtype_dict["fp32"] = dtype_dict["float32"]
dtype_dict["bf16"] = dtype_dict["bfloat16"]
dtype_dict["fp16"] = dtype_dict["float16"]
dtype_dict["fp8"] = dtype_dict["float8_e4m3fn"]
dtype_dict["fp7"] = dtype_dict["float7_e3m3fn"]
dtype_dict["fp6"] = dtype_dict["float6_e3m2fn"]
dtype_dict["fp5"] = dtype_dict["float5_e2m2fn"]
dtype_dict["fp4"] = dtype_dict["float4_e2m1fn"]
dtype_dict["fp3"] = dtype_dict["float3_e1m1fn"]
dtype_dict["fp2"] = dtype_dict["float2_e1m0fn"]
dtype_dict["fp1"] = dtype_dict["float1_e1m0fnu"]
dtype_dict["bool"] = dtype_dict["uint1"]
dtype_dict["int1"] = dtype_dict["uint1"]

torch_dtype_dict = {
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint32: "uint32",
    torch.uint16: "uint16",
    torch.uint8: "uint8",
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float8_e4m3fn: "float8_e4m3fn",
    torch.float8_e5m2: "float8_e5m2",
}

if hasattr(torch, "float8_e4m3fnuz"):
    dtype_dict["float8_e4m3fnuz"] = {"min": -240.0, "max": 240.0, "num_bits": 8, "sign": 1, "exponent": 4, "mantissa": 3, "target_dtype": "fp8", "torch_dtype": torch.float8_e4m3fnuz, "storage_dtype": torch.float8_e4m3fnuz, "is_unsigned": False, "is_integer": False, "is_packed": False}
    torch_dtype_dict[torch.float8_e4m3fnuz] = "float8_e4m3fnuz"
if hasattr(torch, "float8_e5m2fnuz"):
    dtype_dict["float8_e5m2fnuz"] = {"min": -57344.0, "max": 57344.0, "num_bits": 8, "sign": 1, "exponent": 5, "mantissa": 2, "target_dtype": "fp8", "torch_dtype": torch.float8_e5m2fnuz, "storage_dtype": torch.float8_e5m2fnuz, "is_unsigned": False, "is_integer": False, "is_packed": False}
    torch_dtype_dict[torch.float8_e5m2fnuz] = "float8_e5m2fnuz"

linear_types = {"Linear", "SDNQLinear"}
conv_types = {"Conv1d", "Conv2d", "Conv3d", "SDNQConv1d", "SDNQConv2d", "SDNQConv3d"}
conv_transpose_types = {"ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", "SDNQConvTranspose1d", "SDNQConvTranspose2d", "SDNQConvTranspose3d"}
allowed_types = set.union(linear_types, conv_types, conv_transpose_types)

accepted_weight_dtypes = set(dtype_dict.keys())
accepted_matmul_dtypes = {"int8", "fp8", "fp16", "float8_e4m3fn", "float16"}

weights_dtype_order = [
    "uint1", "float1_e1m0fnu",
    "int2", "float2_e1m0fn",
    "uint2", "float2_e1m1fnu", "float2_e2m0fnu",
    "int3", "float3_e1m1fn", "float3_e2m0fn",
    "uint3", "float3_e1m2fnu", "float3_e2m1fnu", "float3_e3m0fnu",
    "int4", "float4_e1m2fn", "float4_e2m1fn", "float4_e3m0fn",
    "uint4", "float4_e1m3fnu", "float4_e2m2fnu", "float4_e3m1fnu", "float4_e4m0fnu",
    "int5", "float5_e1m3fn", "float5_e2m2fn", "float5_e3m1fn", "float5_e4m0fn",
    "uint5", "float5_e1m4fnu", "float5_e2m3fnu", "float5_e3m2fnu", "float5_e4m1fnu", "float5_e5m0fnu",
    "int6", "float6_e1m4fn", "float6_e2m3fn", "float6_e3m2fn", "float6_e4m1fn", "float6_e5m0fn",
    "uint6", "float6_e1m5fnu", "float6_e2m4fnu", "float6_e3m3fnu", "float6_e4m2fnu", "float6_e5m1fnu",
    "int7", "float7_e1m5fn", "float7_e2m4fn", "float7_e3m3fn", "float7_e4m2fn", "float7_e5m1fn",
    "uint7", "float7_e1m6fnu", "float7_e2m5fnu", "float7_e3m4fnu", "float7_e4m3fnu", "float7_e5m2fnu",
    "int8", "float8_e4m3fn", "float8_e5m2", "float8_e1m6fn", "float8_e2m5fn", "float8_e3m4fn",
    "uint8", "float8_e1m7fnu", "float8_e2m6fnu", "float8_e3m5fnu", "float8_e4m4fnu", "float8_e5m3fnu",
]
weights_dtype_order_fp32 = weights_dtype_order + [
    "int16", "float16", "float16_e1m14fn", "float16_e2m13fn", "float16_e3m12fn", "float16_e4m11fn",
    "uint16", "float16_e1m15fnu", "float16_e2m14fnu", "float16_e3m13fnu", "float16_e4m12fnu", "float16_e5m11fnu",
]

is_rdna2 = bool(devices.backend == "rocm" and devices.get_hip_agent().gfx_version < 0x1100)
use_torch_compile = shared.opts.sdnq_dequantize_compile # this setting requires a full restart of the webui to apply

def check_torch_compile(): # dynamo can be disabled after startup
    return use_torch_compile and not torch._dynamo.config.disable # pylint: disable=protected-access


if os.environ.get("SDNQ_USE_TENSORWISE_FP8_MM", None) is None:
    # row-wise FP8 only exist on H100 hardware, sdnq will use software row-wise with tensorwise hardware with this setting
    use_tensorwise_fp8_matmul = bool(devices.backend != "cuda" or (devices.backend == "cuda" and torch.cuda.get_device_capability(devices.device) < (9,0)))
else:
    use_tensorwise_fp8_matmul = os.environ.get("SDNQ_USE_TENSORWISE_FP8_MM", "0").lower() not in {"0", "false", "no"}

if os.environ.get("SDNQ_USE_CONTIGUOUS_MM", None) is None:
    use_contiguous_mm = bool(is_rdna2 or devices.backend in {"ipex", "mps", "cpu", "openvino", "zluda"})
else:
    use_contiguous_mm = bool(os.environ.get("SDNQ_USE_CONTIGUOUS_MM", "0").lower() not in {"0", "false", "no"})

if os.environ.get("SDNQ_USE_TRITON_MM", None) is None:
    use_triton_mm = bool(is_rdna2 or devices.backend == "zluda")
else:
    use_triton_mm = bool(os.environ.get("SDNQ_USE_TRITON_MM", "0").lower() not in {"0", "false", "no"})


if use_triton_mm:
    try:
        from .triton_mm import int_mm
        int_mm_func = int_mm
    except Exception:
        int_mm_func = torch._int_mm
else:
    int_mm_func = torch._int_mm


def fp_mm_torch(x: torch.Tensor, y: torch.Tensor) -> torch.FloatTensor:
    return torch.mm(x,y, out_dtype=torch.float32)

fp_mm_func = None
if os.environ.get("SDNQ_USE_TRITON_MM", "1").lower() not in {"0", "false", "no"}:
    try:
        from .triton_mm import fp_mm
        fp_mm_func = fp_mm
    except Exception:
        fp_mm_func = None

if fp_mm_func is None:
    fp_mm_func = fp_mm_torch


if use_torch_compile:
    torch._dynamo.config.cache_size_limit = max(8192, getattr(torch._dynamo.config, "cache_size_limit", 0))
    torch._dynamo.config.accumulated_recompile_limit = max(8192, getattr(torch._dynamo.config, "accumulated_recompile_limit", 0))
    def compile_func(fn, **kwargs):
        if kwargs.get("fullgraph", None) is None:
            kwargs["fullgraph"] = True
        if kwargs.get("dynamic", None) is None:
            kwargs["dynamic"] = False
        return torch.compile(fn, **kwargs)
else:
    def compile_func(fn, **kwargs): # pylint: disable=unused-argument
        return fn


common_skip_keys = (
    ".time_embed",
    ".context_embedder",
    ".condition_embedder",
    ".x_embedder",
    ".t_embedder",
    ".y_embedder",
    ".emb_in",
    ".txt_in",
    ".img_in",
    ".vid_in",
    ".proj_out",
    ".norm_out",
    ".emb_out",
    ".txt_out",
    ".img_out",
    ".vid_out",
    ".final_layer",
    "multi_modal_projector",
    "time_text_embed",
    "patch_embedding",
    "patch_embed",
    "patch_emb",
    "lm_head",
    "wte",
)

module_skip_keys_dict = {
    "FluxTransformer2DModel": [
        ["single_transformer_blocks.0.norm.linear.weight", "time_text_embed", "time_embed", "context_embedder", "x_embedder", ".proj_out", "norm_out"],
        {}
    ],
    "Flux2Transformer2DModel": [
        ["double_stream_modulation_img", "double_stream_modulation_txt", "single_stream_modulation", "time_guidance_embed", "context_embedder", "x_embedder", ".proj_out", "norm_out"],
        {}
    ],
    "ChromaTransformer2DModel": [
        ["distilled_guidance_layer", "time_text_embed", "context_embedder", "x_embedder", ".proj_out", "norm_out"],
        {}
    ],
    "QwenImageTransformer2DModel": [
        ["transformer_blocks.0.img_mod.1.weight", "time_text_embed", "txt_in", "img_in", "proj_out", "norm_out"],
        {}
    ],
    "WanTransformer3DModel": [
        ["scale_shift_table", "patch_embedding", "condition_embedder", "proj_out", "norm_out"],
        {}
    ],
    "LongCatVideoTransformer3DModel": [
        ["blocks.0.adaLN_modulation.1.weight", "x_embedder", "t_embedder", "y_embedder", "final_layer"],
        {}
    ],
    "LTX2VideoTransformer3DModel": [
        [
            "audio_time_embed", "time_embed", "audio_caption_projection", "caption_projection", "proj_in", "audio_proj_in", "proj_out", "audio_proj_out",
            "av_cross_attn_audio_scale_shift", "av_cross_attn_audio_v2a_gate", "av_cross_attn_video_a2v_gate", "av_cross_attn_video_scale_shift",
        ],
        {}
    ],
    "Lumina2Transformer2DModel": [
        ["layers.0.norm1.linear.weight", "time_caption_embed", "x_embedder", "norm_out"],
        {}
    ],
    "ZImageTransformer2DModel": [
        ["layers.0.adaLN_modulation.0.weight", "t_embedder", "cap_embedder", "siglip_embedder", "all_x_embedder", "all_final_layer"],
        {}
    ],
    "GlmImageTransformer2DModel": [
        ["transformer_blocks.0.norm1.linear.weight", "image_projector", "glyph_projector", "prior_projector", "time_condition_embed", "norm_out", "proj_out"],
        {}
    ],
    "GlmImageForConditionalGeneration": [
        ["lm_head", "patch_embed", "embeddings", "embed_tokens", "vqmodel"],
        {}
    ],
    "HunyuanImage3ForCausalMM": [
        ["lm_head", "patch_embed", "time_embed", "time_embed_2", "final_layer", "wte", "ln_f", "timestep_emb", "vae", "vision_aligner", "head", "post_layernorm", "embeddings"],
        {}
    ],
    "Emu3ForCausalLM": [
        ["lm_head", "vq_model", "tokenizer"],
        {}
    ],
    "Gemma3nForCausalLM": [
        ["lm_head", "correction_coefs", "prediction_coefs", "embedding_projection"],
        {}
    ],
    "MoondreamModel": [
        ["lm_head", "region", "wte", "post_ln", "proj_mlp", "patch_emb", "pos_emb"],
        {}
    ],
    "NaDiT": [
        [".emb_in", ".txt_in", ".vid_in", ".emb_scale", ".vid_out", ".vid_out_norm", ".vid_out_ada"],
        {}
    ],
}

module_skip_keys_dict["LongCatImageTransformer2DModel"] = module_skip_keys_dict["FluxTransformer2DModel"]
module_skip_keys_dict["ChronoEditTransformer3DModel"] = module_skip_keys_dict["WanTransformer3DModel"]
module_skip_keys_dict["Gemma3nForConditionalGeneration"] = module_skip_keys_dict["Gemma3nForCausalLM"]
module_skip_keys_dict["HfMoondream"] = module_skip_keys_dict["MoondreamModel"]
module_skip_keys_dict["NaDiTUpscaler"] = module_skip_keys_dict["NaDiT"]
