

def mb_to_byte(mb: int) -> int:
    return mb * 1024*1024

unknown_cache_size = 2
cache_size_dict = {
    0x0000: mb_to_byte(unknown_cache_size),
    0xE212: mb_to_byte(4), # Arc Pro B50 / Xe2
    0xE211: mb_to_byte(18), # Arc Pro B60 / Xe2
    0xE20B: mb_to_byte(18), # Arc B580 / Xe2
    0xE20C: mb_to_byte(18), # Arc B570 / Xe2
    0x64A0: mb_to_byte(4), # Arc 130V Mobile / Arc 140V Mobile / Lunar Lake / Xe2
    0x6420: mb_to_byte(unknown_cache_size*2), #  (?) (EU: 64/56) / Lunar Lake / Xe2
    0x64B0: mb_to_byte(unknown_cache_size), # (?) (EU: 32) / Lunar Lake / Xe2
    0x7D51: mb_to_byte(4), # Arc 130T Mobile / Arc 140T Mobile / Arrow Lake-H / Xe-LPG
    0x7D67: mb_to_byte(unknown_cache_size*2), # (?) (EU: 64/48/32) / Arrow Lake-S / Xe-LPG
    0x7D41: mb_to_byte(unknown_cache_size*2), # (?) (EU: 64) / Arrow Lake-U / Xe-LPG
    0x7DD5: mb_to_byte(unknown_cache_size*2), # (?) (EU: 128/112) / Meteor Lake / Xe-LPG
    0x7D45: mb_to_byte(unknown_cache_size), # (?) (EU: 64/48) / Meteor Lake / Xe-LPG
    0x7D40: mb_to_byte(unknown_cache_size), # (?) (EU: 64/48) / Meteor Lake / Xe-LPG
    0x7D55: mb_to_byte(unknown_cache_size*2), # (?) (EU: 128/112) / Meteor Lake / Xe-LPG
    0x0BD5: mb_to_byte(408), # Max 1550 / Xe-HPC
    0x0BDA: mb_to_byte(204) , # Max 1100 / Xe-HPC
    0x56C0: mb_to_byte(16), # Flex 170 / Xe-HPG
    0x56C1: mb_to_byte(4), # Flex 140 / Xe-HPG
    0x5690: mb_to_byte(16), # Arc A770M / Xe-HPG
    0x5691: mb_to_byte(12), # Arc A730M / Xe-HPG
    0x5696: mb_to_byte(8), # Arc A570M / Xe-HPG
    0x5692: mb_to_byte(8), # Arc A550M / Xe-HPG
    0x5697: mb_to_byte(8), # Arc A530M / Xe-HPG
    0x5693: mb_to_byte(4), # Arc A370M / Xe-HPG
    0x5694: mb_to_byte(4), # Arc A350M / Xe-HPG
    0x56A0: mb_to_byte(16), # Arc A770 / Xe-HPG
    0x56A1: mb_to_byte(16), # Arc A750 / Xe-HPG
    0x56A2: mb_to_byte(8), # Arc A580 / Xe-HPG
    0x56A5: mb_to_byte(4), # Arc A380 / Xe-HPG
    0x56A6: mb_to_byte(4), # Arc A310 / Xe-HPG
    0x56B3: mb_to_byte(12), # Arc Pro A60 / Xe-HPG
    0x56B2: mb_to_byte(8), # Arc Pro A60M / Xe-HPG
    0x56B1: mb_to_byte(4), # Arc Pro A40/A50 / Xe-HPG
    0x56B0: mb_to_byte(4), # Arc Pro A30M / Xe-HPG
    0x56BA: mb_to_byte(unknown_cache_size*2), # Arc A380E / Xe-HPG
    0x56BC: mb_to_byte(unknown_cache_size*2), # Arc A370E / Xe-HPG
    0x56BD: mb_to_byte(unknown_cache_size*2), # Arc A350E / Xe-HPG
    0x56BB: mb_to_byte(unknown_cache_size*2), # Arc A310E / Xe-HPG
}
