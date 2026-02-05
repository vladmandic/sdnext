# pylint: disable=superfluous-parens

import cv2
import numpy as np

MASK_2 = 0x00FF00
MASK_13 = 0xFF00FF

Ymask = 0x00FF0000
Umask = 0x0000FF00
Vmask = 0x000000FF

trY = 0x00300000
trU = 0x00000700
trV = 0x00000006


def RGBtoYUV(c):
    r = (c & 0xFF0000) >> 16
    g = (c & 0x00FF00) >> 8
    b = (c & 0x0000FF)
    ret = (int(0.299*r + 0.587*g + 0.114*b) << 16) + (int((-0.169*r - 0.331*g + 0.5*b) + 128) << 8) + (int((0.5*r - 0.419*g - 0.081*b) + 128))
    return ret


def Diff(w1, w2):
    # Mask against RGBMASK to discard the alpha channel
    YUV1 = RGBtoYUV(w1)
    YUV2 = RGBtoYUV(w2)
    return ((abs((YUV1 & Ymask) - (YUV2 & Ymask)) > trY ) or ( abs((YUV1 & Umask) - (YUV2 & Umask)) > trU ) or ( abs((YUV1 & Vmask) - (YUV2 & Vmask)) > trV ))


def Interp1( pc, c1, c2, dest ):
    # pc = (c1*3+c2) >> 2
    if c1 == c2:
        dest[pc] = c1
        return
    dest[pc] = ((((c1 & MASK_2) * 3 + (c2 & MASK_2)) >> 2) & MASK_2) + ((((c1 & MASK_13) * 3 + (c2 & MASK_13)) >> 2) & MASK_13)
    dest[pc] |= (c1 & 0xFF000000)


def Interp2( pc, c1, c2, c3, dest ):
    # pc = (c1*2+c2+c3) >> 2
    dest[pc] = (((((c1 & MASK_2) << 1) + (c2 & MASK_2) + (c3 & MASK_2)) >> 2) & MASK_2) + (((((c1 & MASK_13) << 1) + (c2 & MASK_13) + (c3 & MASK_13)) >> 2) & MASK_13)
    dest[pc] |= (c1 & 0xFF000000)


def Interp3( pc, c1, c2, dest ):
    # pc = (c1*7+c2)/8
    if c1 == c2:
        dest[pc] = c1
        return
    dest[pc] = ((((c1 & MASK_2) * 7 + (c2 & MASK_2)) >> 3) & MASK_2) + ((((c1 & MASK_13) * 7 + (c2 & MASK_13)) >> 3) & MASK_13)
    dest[pc] |= (c1 & 0xFF000000)


def Interp4( pc, c1, c2, c3, dest ):
    # pc = (c1*2+(c2+c3)*7)/16
    dest[pc] = (((((c1 & MASK_2) << 1) + (c2 & MASK_2) * 7 + (c3 & MASK_2) * 7) >> 4) & MASK_2) + (((((c1 & MASK_13) << 1) + (c2 & MASK_13) * 7 + (c3 & MASK_13) * 7) >> 4) & MASK_13)
    dest[pc] |= (c1 & 0xFF000000)


def Interp5( pc, c1, c2, dest ):
    # pc = (c1+c2) >> 1
    if c1 == c2:
        dest[pc] = c1
        return
    dest[pc] = ((((c1 & MASK_2) + (c2 & MASK_2)) >> 1) & MASK_2) + ((((c1 & MASK_13) + (c2 & MASK_13)) >> 1) & MASK_13)
    dest[pc] |= (c1 & 0xFF000000)


def Interp6( pc, c1, c2, c3, dest ):
    # pc = (c1*5+c2*2+c3)/8
    dest[pc] = ((((c1 & MASK_2) * 5 + ((c2 & MASK_2) << 1) + (c3 & MASK_2)) >> 3) & MASK_2) + ((((c1 & MASK_13) * 5 + ((c2 & MASK_13) << 1) + (c3 & MASK_13)) >> 3) & MASK_13)
    dest[pc] |= (c1 & 0xFF000000)


def Interp7( pc, c1, c2, c3, dest ):
    # pc = (c1*6+c2+c3)/8
    dest[pc] = ((((c1 & MASK_2) * 6 + (c2 & MASK_2) + (c3 & MASK_2)) >> 3) & MASK_2) + ((((c1 & MASK_13) * 6 + (c2 & MASK_13) + (c3 & MASK_13)) >> 3) & MASK_13)
    dest[pc] |= (c1 & 0xFF000000)


def Interp8( pc, c1, c2, dest ):
    # pc = (c1*5+c2*3)/8
    if c1 == c2:
        dest[pc] = c1
        return
    dest[pc] = ((((c1 & MASK_2) * 5 + (c2 & MASK_2) * 3) >> 3) & MASK_2) + ((((c1 & MASK_13) * 5 + (c2 & MASK_13) * 3) >> 3) & MASK_13)
    dest[pc] |= (c1 & 0xFF000000)


def Interp9( pc, c1, c2, c3, dest ):
    # pc = (c1*2+(c2+c3)*3)/8
    dest[pc] = (((((c1 & MASK_2) << 1) + (c2 & MASK_2) * 3 + (c3 & MASK_2) * 3) >> 3) & MASK_2) + (((((c1 & MASK_13) << 1) + (c2 & MASK_13) * 3 + (c3 & MASK_13) * 3) >> 3) & MASK_13)
    dest[pc] |= (c1 & 0xFF000000)


def Interp10( pc, c1, c2, c3, dest ):
    # pc = (c1*14+c2+c3)/16
    dest[pc] = ((((c1 & MASK_2) * 14 + (c2 & MASK_2) + (c3 & MASK_2)) >> 4) & MASK_2) + ((((c1 & MASK_13) * 14 + (c2 & MASK_13) + (c3 & MASK_13)) >> 4) & MASK_13)
    dest[pc] |= (c1 & 0xFF000000)


def hqx( img: cv2.Mat, scale_factor: int ) -> cv2.Mat:
    # We can only scale with a factor of 2, 3 or 4
    if scale_factor not in [2, 3, 4]:
        return img

    height, width = img.shape[0], img.shape[1]

    # pack RGB colors into integers
    src = []
    dest = [None] * (height * width * scale_factor * scale_factor)
    for row in range(height):
        for col in range(width):
            b, g, r = img[row][col]
            src.append((r << 16) + (g << 8) + b)

    if scale_factor == 2:
        hq2x( width, height, src, dest )
    elif scale_factor == 3:
        hq3x( width, height, src, dest )
    elif scale_factor == 4:
        hq4x( width, height, src, dest )

    scaled_img = np.zeros((height * scale_factor, width * scale_factor, 3))
    cnt = 0
    for row in range(scaled_img.shape[0]):
        for col in range(scaled_img.shape[1]):
            pc = dest[cnt]
            r, g, b = (pc & 0x00FF0000) >> 16, (pc & 0x0000FF00) >> 8, (pc & 0x000000FF)
            scaled_img[row][col] = [b, g, r]
            cnt += 1

    return scaled_img


def hq2x( width: int, height: int, src, dest ) -> None: # noqa:C901
    w = [None] * 10
    dpL = width << 1
    dp = 0
    sp = 0

    #   +----+----+----+
    #   |    |    |    |
    #   | w1 | w2 | w3 |
    #   +----+----+----+
    #   |    |    |    |
    #   | w4 | w5 | w6 |
    #   +----+----+----+
    #   |    |    |    |
    #   | w7 | w8 | w9 |
    #   +----+----+----+

    for j in range(height):
        prevline = -width if j > 0 else 0
        nextline = width if j < height - 1 else 0

        for i in range(width):
            w[2] = src[sp + prevline]
            w[5] = src[sp]
            w[8] = src[sp + nextline]

            if i > 0:
                w[1] = src[sp + prevline - 1]
                w[4] = src[sp - 1]
                w[7] = src[sp + nextline - 1]
            else:
                w[1] = w[2]
                w[4] = w[5]
                w[7] = w[8]

            if i < width - 1:
                w[3] = src[sp + prevline + 1]
                w[6] = src[sp + 1]
                w[9] = src[sp + nextline + 1]
            else:
                w[3] = w[2]
                w[6] = w[5]
                w[9] = w[8]

            pattern = 0
            flag = 1

            YUV1 = RGBtoYUV(w[5])

            for k in range(1, 10):
                if k == 5:
                    continue

                if w[k] != w[5]:
                    YUV2 = RGBtoYUV(w[k])
                    if ( ( abs((YUV1 & Ymask) - (YUV2 & Ymask)) > trY ) or ( abs((YUV1 & Umask) - (YUV2 & Umask)) > trU ) or ( abs((YUV1 & Vmask) - (YUV2 & Vmask)) > trV ) ):
                        pattern |= flag
                flag <<= 1

            match pattern:
                case 0|1|4|32|128|5|132|160|33|129|36|133|164|161|37|165:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 2|34|130|162:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 16|17|48|49:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 64|65|68|69:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 8|12|136|140:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 3|35|131|163:
                    Interp1(dp, w[5], w[4], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 6|38|134|166:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 20|21|52|53:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 144|145|176|177:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 192|193|196|197:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 96|97|100|101:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 40|44|168|172:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 9|13|137|141:
                    Interp1(dp, w[5], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 18|50:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 80|81:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 72|76:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 10|138:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 66:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 24:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 7|39|135:
                    Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 148|149|180:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 224|228|225:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 41|169|45:
                    Interp1(dp, w[5], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 22|54:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 208|209:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 104|108:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 11|139:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 19|51:
                    if (Diff(w[2], w[6])):
                        Interp1(dp, w[5], w[4], dest)
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp6(dp, w[5], w[2], w[4], dest)
                        Interp9(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 146|178:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                        Interp1(dp+dpL+1, w[5], w[8], dest)
                    else:
                        Interp9(dp+1, w[5], w[2], w[6], dest)
                        Interp6(dp+dpL+1, w[5], w[6], w[8], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                case 84|85:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp6(dp+1, w[5], w[6], w[2], dest)
                        Interp9(dp+dpL+1, w[5], w[6], w[8], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                case 112|113:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL, w[5], w[4], dest)
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp6(dp+dpL, w[5], w[8], w[4], dest)
                        Interp9(dp+dpL+1, w[5], w[6], w[8], dest)
                case 200|204:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                        Interp1(dp+dpL+1, w[5], w[6], dest)
                    else:
                        Interp9(dp+dpL, w[5], w[8], w[4], dest)
                        Interp6(dp+dpL+1, w[5], w[8], w[6], dest)
                case 73|77:
                    if (Diff(w[8], w[4])):
                        Interp1(dp, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp6(dp, w[5], w[4], w[2], dest)
                        Interp9(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 42|170:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                        Interp1(dp+dpL, w[5], w[8], dest)
                    else:
                        Interp9(dp, w[5], w[4], w[2], dest)
                        Interp6(dp+dpL, w[5], w[4], w[8], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 14|142:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                        Interp1(dp+1, w[5], w[6], dest)
                    else:
                        Interp9(dp, w[5], w[4], w[2], dest)
                        Interp6(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 67:
                    Interp1(dp, w[5], w[4], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 70:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 28:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 152:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 194:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 98:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 56:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 25:
                    Interp1(dp, w[5], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 26|31:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 82|214:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 88|248:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 74|107:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 27:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[3], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 86:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[9], dest)
                case 216:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp1(dp+dpL, w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 106:
                    Interp1(dp, w[5], w[4], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 30:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 210:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    Interp1(dp+1, w[5], w[3], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 120:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[9], dest)
                case 75:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp1(dp+dpL, w[5], w[7], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 29:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 198:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 184:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 99:
                    Interp1(dp, w[5], w[4], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 57:
                    Interp1(dp, w[5], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 71:
                    Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 156:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 226:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 60:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 195:
                    Interp1(dp, w[5], w[4], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 102:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 153:
                    Interp1(dp, w[5], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 58:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp7(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp7(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 83:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp7(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp7(dp+dpL+1, w[5], w[6], w[8], dest)
                case 92:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp7(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp7(dp+dpL+1, w[5], w[6], w[8], dest)
                case 202:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp7(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp7(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 78:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp7(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp7(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 154:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp7(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp7(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 114:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp7(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp7(dp+dpL+1, w[5], w[6], w[8], dest)
                case 89:
                    Interp1(dp, w[5], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp7(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp7(dp+dpL+1, w[5], w[6], w[8], dest)
                case 90:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp7(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp7(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp7(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp7(dp+dpL+1, w[5], w[6], w[8], dest)
                case 55|23:
                    if (Diff(w[2], w[6])):
                        Interp1(dp, w[5], w[4], dest)
                        dest[dp+1] = w[5]
                    else:
                        Interp6(dp, w[5], w[2], w[4], dest)
                        Interp9(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 182|150:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        Interp1(dp+dpL+1, w[5], w[8], dest)
                    else:
                        Interp9(dp+1, w[5], w[2], w[6], dest)
                        Interp6(dp+dpL+1, w[5], w[6], w[8], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                case 213|212:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+1, w[5], w[2], dest)
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp6(dp+1, w[5], w[6], w[2], dest)
                        Interp9(dp+dpL+1, w[5], w[6], w[8], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                case 241|240:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp6(dp+dpL, w[5], w[8], w[4], dest)
                        Interp9(dp+dpL+1, w[5], w[6], w[8], dest)
                case 236|232:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        Interp1(dp+dpL+1, w[5], w[6], dest)
                    else:
                        Interp9(dp+dpL, w[5], w[8], w[4], dest)
                        Interp6(dp+dpL+1, w[5], w[8], w[6], dest)
                case 109|105:
                    if (Diff(w[8], w[4])):
                        Interp1(dp, w[5], w[2], dest)
                        dest[dp+dpL] = w[5]
                    else:
                        Interp6(dp, w[5], w[4], w[2], dest)
                        Interp9(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 171|43:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        Interp1(dp+dpL, w[5], w[8], dest)
                    else:
                        Interp9(dp, w[5], w[4], w[2], dest)
                        Interp6(dp+dpL, w[5], w[4], w[8], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 143|15:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        Interp1(dp+1, w[5], w[6], dest)
                    else:
                        Interp9(dp, w[5], w[4], w[2], dest)
                        Interp6(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 124:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[9], dest)
                case 203:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp1(dp+dpL, w[5], w[7], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 62:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 211:
                    Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[3], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 118:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[9], dest)
                case 217:
                    Interp1(dp, w[5], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp1(dp+dpL, w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 110:
                    Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 155:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[3], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 188:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 185:
                    Interp1(dp, w[5], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 61:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 157:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 103:
                    Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 227:
                    Interp1(dp, w[5], w[4], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 230:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 199:
                    Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 220:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp7(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 158:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp7(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 234:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp7(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 242:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp7(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 59:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp7(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 121:
                    Interp1(dp, w[5], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp7(dp+dpL+1, w[5], w[6], w[8], dest)
                case 87:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp7(dp+dpL+1, w[5], w[6], w[8], dest)
                case 79:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp7(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 122:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp7(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp7(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp7(dp+dpL+1, w[5], w[6], w[8], dest)
                case 94:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp7(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp7(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp7(dp+dpL+1, w[5], w[6], w[8], dest)
                case 218:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp7(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp7(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp7(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 91:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp7(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp7(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp7(dp+dpL+1, w[5], w[6], w[8], dest)
                case 229:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 167:
                    Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 173:
                    Interp1(dp, w[5], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 181:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 186:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp7(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp7(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 115:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp7(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp7(dp+dpL+1, w[5], w[6], w[8], dest)
                case 93:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp7(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp7(dp+dpL+1, w[5], w[6], w[8], dest)
                case 206:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp7(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp7(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 205|201:
                    Interp1(dp, w[5], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+dpL, w[5], w[7], dest)
                    else:
                        Interp7(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 174|46:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[4], dest)
                    else:
                        Interp7(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 179|147:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+1, w[5], w[3], dest)
                    else:
                        Interp7(dp+1, w[5], w[2], w[6], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 117|116:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                    else:
                        Interp7(dp+dpL+1, w[5], w[6], w[8], dest)
                case 189:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 231:
                    Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 126:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[9], dest)
                case 219:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 125:
                    if (Diff(w[8], w[4])):
                        Interp1(dp, w[5], w[2], dest)
                        dest[dp+dpL] = w[5]
                    else:
                        Interp6(dp, w[5], w[4], w[2], dest)
                        Interp9(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+dpL+1, w[5], w[9], dest)
                case 221:
                    Interp1(dp, w[5], w[2], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+1, w[5], w[2], dest)
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp6(dp+1, w[5], w[6], w[2], dest)
                        Interp9(dp+dpL+1, w[5], w[6], w[8], dest)
                    Interp1(dp+dpL, w[5], w[7], dest)
                case 207:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        Interp1(dp+1, w[5], w[6], dest)
                    else:
                        Interp9(dp, w[5], w[4], w[2], dest)
                        Interp6(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[7], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 238:
                    Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        Interp1(dp+dpL+1, w[5], w[6], dest)
                    else:
                        Interp9(dp+dpL, w[5], w[8], w[4], dest)
                        Interp6(dp+dpL+1, w[5], w[8], w[6], dest)
                case 190:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        Interp1(dp+dpL+1, w[5], w[8], dest)
                    else:
                        Interp9(dp+1, w[5], w[2], w[6], dest)
                        Interp6(dp+dpL+1, w[5], w[6], w[8], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                case 187:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        Interp1(dp+dpL, w[5], w[8], dest)
                    else:
                        Interp9(dp, w[5], w[4], w[2], dest)
                        Interp6(dp+dpL, w[5], w[4], w[8], dest)
                    Interp1(dp+1, w[5], w[3], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 243:
                    Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[3], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp6(dp+dpL, w[5], w[8], w[4], dest)
                        Interp9(dp+dpL+1, w[5], w[6], w[8], dest)
                case 119:
                    if (Diff(w[2], w[6])):
                        Interp1(dp, w[5], w[4], dest)
                        dest[dp+1] = w[5]
                    else:
                        Interp6(dp, w[5], w[2], w[4], dest)
                        Interp9(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[9], dest)
                case 237|233:
                    Interp1(dp, w[5], w[2], dest)
                    Interp2(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp1(dp+dpL, w[5], w[7], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 175|47:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 183|151:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp1(dp+1, w[5], w[3], dest)
                    Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 245|244:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                case 250:
                    Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[3], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 123:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[3], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[9], dest)
                case 95:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[7], dest)
                    Interp1(dp+dpL+1, w[5], w[9], dest)
                case 222:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 252:
                    Interp2(dp, w[5], w[1], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                case 249:
                    Interp1(dp, w[5], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[2], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp1(dp+dpL, w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 235:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    Interp2(dp+1, w[5], w[3], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp1(dp+dpL, w[5], w[7], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 111:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[6], dest)
                case 63:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp2(dp+dpL+1, w[5], w[9], w[8], dest)
                case 159:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp1(dp+1, w[5], w[3], dest)
                    Interp2(dp+dpL, w[5], w[7], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 215:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp1(dp+1, w[5], w[3], dest)
                    Interp2(dp+dpL, w[5], w[7], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 246:
                    Interp2(dp, w[5], w[1], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                case 254:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                case 253:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp1(dp+dpL, w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                case 251:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[3], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp1(dp+dpL, w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 239:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp1(dp, w[5], w[4], dest)
                    Interp1(dp+1, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp1(dp+dpL, w[5], w[7], dest)
                    Interp1(dp+dpL+1, w[5], w[6], dest)
                case 127:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp2(dp+1, w[5], w[2], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp+dpL, w[5], w[8], w[4], dest)
                    Interp1(dp+dpL+1, w[5], w[9], dest)
                case 191:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp1(dp+1, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[8], dest)
                    Interp1(dp+dpL+1, w[5], w[8], dest)
                case 223:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp1(dp+1, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp2(dp+dpL+1, w[5], w[6], w[8], dest)
                case 247:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp1(dp+1, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp1(dp+dpL+1, w[5], w[9], dest)
                case 255:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                    else:
                        Interp1(dp+1, w[5], w[3], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                    else:
                        Interp1(dp+dpL, w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp1(dp+dpL+1, w[5], w[9], dest)
            sp += 1
            dp += 2
        dp += dpL


def hq3x( width: int, height: int, src, dest ) -> None: # noqa:C901
    w = [None] * 10
    dpL = width * 3
    dp = 0
    sp = 0

    #   +----+----+----+
    #   |	|	|	|
    #   | w1 | w2 | w3 |
    #   +----+----+----+
    #   |	|	|	|
    #   | w4 | w5 | w6 |
    #   +----+----+----+
    #   |	|	|	|
    #   | w7 | w8 | w9 |
    #   +----+----+----+

    for j in range(height):
        prevline = -width if j > 0 else 0
        nextline = width if j < height - 1 else 0

        for i in range(width):
            w[2] = src[sp + prevline]
            w[5] = src[sp]
            w[8] = src[sp + nextline]

            if i > 0:
                w[1] = src[sp + prevline - 1]
                w[4] = src[sp - 1]
                w[7] = src[sp + nextline - 1]
            else:
                w[1] = w[2]
                w[4] = w[5]
                w[7] = w[8]

            if i < width - 1:
                w[3] = src[sp + prevline + 1]
                w[6] = src[sp + 1]
                w[9] = src[sp + nextline + 1]
            else:
                w[3] = w[2]
                w[6] = w[5]
                w[9] = w[8]

            pattern = 0
            flag = 1

            YUV1 = RGBtoYUV(w[5])

            for k in range(1, 10):
                if k == 5:
                    continue
                if w[k] != w[5]:
                    YUV2 = RGBtoYUV(w[k])
                    if ( ( abs((YUV1 & Ymask) - (YUV2 & Ymask)) > trY ) or ( abs((YUV1 & Umask) - (YUV2 & Umask)) > trU ) or ( abs((YUV1 & Vmask) - (YUV2 & Vmask)) > trV ) ):
                        pattern |= flag
                flag <<= 1

            match pattern:
                case 0|1|4|32|128|5|132|160|33|129|36|133|164|161|37|165:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 2|34|130|162:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 16|17|48|49:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 64|65|68|69:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 8|12|136|140:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 3|35|131|163:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 6|38|134|166:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 20|21|52|53:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 144|145|176|177:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 192|193|196|197:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 96|97|100|101:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 40|44|168|172:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 9|13|137|141:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 18|50:
                    Interp1(dp, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        Interp1(dp+2, w[5], w[3], dest)
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 80|81:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 72|76:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 10|138:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 66:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 24:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 7|39|135:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 148|149|180:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 224|228|225:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 41|169|45:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 22|54:
                    Interp1(dp, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 208|209:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 104|108:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 11|139:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 19|51:
                    if (Diff(w[2], w[6])):
                        Interp1(dp, w[5], w[4], dest)
                        dest[dp+1] = w[5]
                        Interp1(dp+2, w[5], w[3], dest)
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        Interp1(dp+1, w[2], w[5], dest)
                        Interp5(dp+2, w[2], w[6], dest)
                        Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 146|178:
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        Interp1(dp+2, w[5], w[3], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                    else:
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp5(dp+2, w[2], w[6], dest)
                        Interp1(dp+dpL+2, w[6], w[5], dest)
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                case 84|85:
                    if (Diff(w[6], w[8])):
                        Interp1(dp+2, w[5], w[2], dest)
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        Interp1(dp+dpL+2, w[6], w[5], dest)
                        Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp5(dp+(dpL << 1)+2, w[6], w[8], dest)
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                case 112|113:
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp1(dp+dpL+2, w[5], w[6], dest)
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL << 1)+1, w[8], w[5], dest)
                        Interp5(dp+(dpL << 1)+2, w[6], w[8], dest)
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                case 200|204:
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                    else:
                        Interp1(dp+dpL, w[5], w[4], dest)
                        Interp5(dp+(dpL << 1), w[8], w[4], dest)
                        Interp1(dp+(dpL << 1)+1, w[8], w[5], dest)
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                case 73|77:
                    if (Diff(w[8], w[4])):
                        Interp1(dp, w[5], w[2], dest)
                        dest[dp+dpL] = w[5]
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        Interp1(dp+dpL, w[4], w[5], dest)
                        Interp5(dp+(dpL << 1), w[8], w[4], dest)
                        Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 42|170:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                        Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    else:
                        Interp5(dp, w[4], w[2], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[4], w[5], dest)
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 14|142:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                        dest[dp+1] = w[5]
                        Interp1(dp+2, w[5], w[6], dest)
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[4], w[2], dest)
                        Interp1(dp+1, w[2], w[5], dest)
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 67:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 70:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 28:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 152:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 194:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 98:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 56:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 25:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 26|31:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 82|214:
                    Interp1(dp, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 88|248:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 74|107:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 27:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 86:
                    Interp1(dp, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 216:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 106:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 30:
                    Interp1(dp, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 210:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 120:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 75:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 29:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 198:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 184:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 99:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 57:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 71:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 156:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 226:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 60:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 195:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 102:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 153:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 58:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 83:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 92:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 202:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 78:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 154:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 114:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 89:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 90:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 55|23:
                    if (Diff(w[2], w[6])):
                        Interp1(dp, w[5], w[4], dest)
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        Interp1(dp+1, w[2], w[5], dest)
                        Interp5(dp+2, w[2], w[6], dest)
                        Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 182|150:
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                    else:
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp5(dp+2, w[2], w[6], dest)
                        Interp1(dp+dpL+2, w[6], w[5], dest)
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                case 213|212:
                    if (Diff(w[6], w[8])):
                        Interp1(dp+2, w[5], w[2], dest)
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        Interp1(dp+dpL+2, w[6], w[5], dest)
                        Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp5(dp+(dpL << 1)+2, w[6], w[8], dest)
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                case 241|240:
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp1(dp+dpL+2, w[5], w[6], dest)
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL << 1)+1, w[8], w[5], dest)
                        Interp5(dp+(dpL << 1)+2, w[6], w[8], dest)
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                case 236|232:
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                    else:
                        Interp1(dp+dpL, w[5], w[4], dest)
                        Interp5(dp+(dpL << 1), w[8], w[4], dest)
                        Interp1(dp+(dpL << 1)+1, w[8], w[5], dest)
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                case 109|105:
                    if (Diff(w[8], w[4])):
                        Interp1(dp, w[5], w[2], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        Interp1(dp+dpL, w[4], w[5], dest)
                        Interp5(dp+(dpL << 1), w[8], w[4], dest)
                        Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 171|43:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                        Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    else:
                        Interp5(dp, w[4], w[2], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[4], w[5], dest)
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 143|15:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        Interp1(dp+2, w[5], w[6], dest)
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[4], w[2], dest)
                        Interp1(dp+1, w[2], w[5], dest)
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 124:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 203:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 62:
                    Interp1(dp, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 211:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 118:
                    Interp1(dp, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 217:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 110:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 155:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 188:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 185:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 61:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 157:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 103:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 227:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 230:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 199:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 220:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 158:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 234:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 242:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 59:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 121:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 87:
                    Interp1(dp, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 79:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[6], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 122:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    if (Diff(w[6], w[8])):
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 94:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 218:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 91:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 229:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 167:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 173:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 181:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 186:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 115:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 93:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 206:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 205|201:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 174|46:
                    if (Diff(w[4], w[2])):
                        Interp1(dp, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 179|147:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 117|116:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 189:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 231:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 126:
                    Interp1(dp, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 219:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 125:
                    if (Diff(w[8], w[4])):
                        Interp1(dp, w[5], w[2], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        Interp1(dp+dpL, w[4], w[5], dest)
                        Interp5(dp+(dpL << 1), w[8], w[4], dest)
                        Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 221:
                    if (Diff(w[6], w[8])):
                        Interp1(dp+2, w[5], w[2], dest)
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        Interp1(dp+dpL+2, w[6], w[5], dest)
                        Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp5(dp+(dpL << 1)+2, w[6], w[8], dest)
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                case 207:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        Interp1(dp+2, w[5], w[6], dest)
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[4], w[2], dest)
                        Interp1(dp+1, w[2], w[5], dest)
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 238:
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                    else:
                        Interp1(dp+dpL, w[5], w[4], dest)
                        Interp5(dp+(dpL << 1), w[8], w[4], dest)
                        Interp1(dp+(dpL << 1)+1, w[8], w[5], dest)
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                case 190:
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                    else:
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp5(dp+2, w[2], w[6], dest)
                        Interp1(dp+dpL+2, w[6], w[5], dest)
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                case 187:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                        Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    else:
                        Interp5(dp, w[4], w[2], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[4], w[5], dest)
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 243:
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp1(dp+dpL+2, w[5], w[6], dest)
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL << 1)+1, w[8], w[5], dest)
                        Interp5(dp+(dpL << 1)+2, w[6], w[8], dest)
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                case 119:
                    if (Diff(w[2], w[6])):
                        Interp1(dp, w[5], w[4], dest)
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        Interp1(dp+1, w[2], w[5], dest)
                        Interp5(dp+2, w[2], w[6], dest)
                        Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 237|233:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp2(dp+2, w[5], w[2], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 175|47:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 183|151:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 245|244:
                    Interp2(dp, w[5], w[4], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 250:
                    Interp1(dp, w[5], w[1], dest)
                    dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 123:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 95:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 222:
                    Interp1(dp, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 252:
                    Interp1(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 249:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 235:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 111:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 63:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 159:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 215:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 246:
                    Interp1(dp, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 254:
                    Interp1(dp, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 253:
                    Interp1(dp, w[5], w[2], dest)
                    Interp1(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[2], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 251:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+dpL] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp3(dp+dpL, w[5], w[4], dest)
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+dpL+2] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 239:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    Interp1(dp+2, w[5], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp1(dp+dpL+2, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp1(dp+(dpL << 1)+2, w[5], w[6], dest)
                case 127:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp4(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                    else:
                        Interp4(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[9], dest)
                case 191:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp1(dp+(dpL << 1)+2, w[5], w[8], dest)
                case 223:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp4(dp, w[5], w[4], w[2], dest)
                        Interp3(dp+dpL, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                        dest[dp+dpL+2] = w[5]
                    else:
                        Interp3(dp+1, w[5], w[2], dest)
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        Interp3(dp+dpL+2, w[5], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                        Interp4(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 247:
                    Interp1(dp, w[5], w[4], dest)
                    dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[4], dest)
                    dest[dp+dpL+1] = w[5]
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[4], dest)
                    dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                case 255:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[4], w[2], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                    else:
                        Interp2(dp+2, w[5], w[2], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                    else:
                        Interp2(dp+(dpL << 1), w[5], w[8], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+2] = w[5]
                    else:
                        Interp2(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
            sp += 1
            dp += 3
        dp += (dpL << 1)


def hq4x( width: int, height: int, src, dest ) -> None: # noqa:C901
    w = [None] * 10
    dpL = width << 2
    dp = 0
    sp = 0

    #   +----+----+----+
    #   |    |    |    |
    #   | w1 | w2 | w3 |
    #   +----+----+----+
    #   |    |    |    |
    #   | w4 | w5 | w6 |
    #   +----+----+----+
    #   |    |    |    |
    #   | w7 | w8 | w9 |
    #   +----+----+----+

    for j in range(height):
        prevline = -width if j > 0 else 0
        nextline = width if j < height - 1 else 0

        for i in range(width):
            w[2] = src[sp + prevline]
            w[5] = src[sp]
            w[8] = src[sp + nextline]

            if i > 0:
                w[1] = src[sp + prevline - 1]
                w[4] = src[sp - 1]
                w[7] = src[sp + nextline - 1]
            else:
                w[1] = w[2]
                w[4] = w[5]
                w[7] = w[8]

            if i < width - 1:
                w[3] = src[sp + prevline + 1]
                w[6] = src[sp + 1]
                w[9] = src[sp + nextline + 1]
            else:
                w[3] = w[2]
                w[6] = w[5]
                w[9] = w[8]

            pattern = 0
            flag = 1

            YUV1 = RGBtoYUV(w[5])

            for k in range(1, 10):
                if k == 5:
                    continue

                if w[k] != w[5]:
                    YUV2 = RGBtoYUV(w[k])
                    if ( ( abs((YUV1 & Ymask) - (YUV2 & Ymask)) > trY ) or ( abs((YUV1 & Umask) - (YUV2 & Umask)) > trU ) or ( abs((YUV1 & Vmask) - (YUV2 & Vmask)) > trV ) ):
                        pattern |= flag
                flag <<= 1

            match pattern:
                case 0|1|4|32|128|5|132|160|33|129|36|133|164|161|37|165:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 2|34|130|162:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 16|17|48|49:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 64|65|68|69:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 8|12|136|140:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 3|35|131|163:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 6|38|134|166:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 20|21|52|53:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 144|145|176|177:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 192|193|196|197:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 96|97|100|101:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 40|44|168|172:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 9|13|137|141:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 18|50:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 80|81:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 72|76:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 10|138:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                        dest[dp+dpL+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 66:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 24:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 7|39|135:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 148|149|180:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 224|228|225:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 41|169|45:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 22|54:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    dest[dp+dpL+2] = w[5]
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 208|209:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 104|108:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 11|139:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 19|51:
                    if (Diff(w[2], w[6])):
                        Interp8(dp, w[5], w[4], dest)
                        Interp3(dp+1, w[5], w[4], dest)
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp, w[5], w[2], dest)
                        Interp1(dp+1, w[2], w[5], dest)
                        Interp8(dp+2, w[2], w[6], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                        Interp2(dp+dpL+3, w[6], w[5], w[2], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 146|178:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                        Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                    else:
                        Interp2(dp+2, w[2], w[5], w[6], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                        Interp8(dp+dpL+3, w[6], w[2], dest)
                        Interp1(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp1(dp+(dpL * 3)+3, w[5], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                case 84|85:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    if (Diff(w[6], w[8])):
                        Interp8(dp+3, w[5], w[2], dest)
                        Interp3(dp+dpL+3, w[5], w[2], dest)
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        Interp1(dp+3, w[5], w[6], dest)
                        Interp1(dp+dpL+3, w[6], w[5], dest)
                        Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                        Interp8(dp+(dpL << 1)+3, w[6], w[8], dest)
                        Interp2(dp+(dpL * 3)+2, w[8], w[5], w[6], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 112|113:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3), w[5], w[4], dest)
                        Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                        Interp2(dp+(dpL << 1)+3, w[6], w[5], w[8], dest)
                        Interp1(dp+(dpL * 3), w[5], w[8], dest)
                        Interp1(dp+(dpL * 3)+1, w[8], w[5], dest)
                        Interp8(dp+(dpL * 3)+2, w[8], w[6], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                case 200|204:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                        Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[4], w[5], w[8], dest)
                        Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp8(dp+(dpL * 3)+1, w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp1(dp+(dpL * 3)+3, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                case 73|77:
                    if (Diff(w[8], w[4])):
                        Interp8(dp, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[2], dest)
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp, w[5], w[4], dest)
                        Interp1(dp+dpL, w[4], w[5], dest)
                        Interp8(dp+(dpL << 1), w[4], w[8], dest)
                        Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp2(dp+(dpL * 3)+1, w[8], w[5], w[4], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 42|170:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                        Interp3(dp+(dpL << 1), w[5], w[8], dest)
                        Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp2(dp+1, w[2], w[5], w[4], dest)
                        Interp8(dp+dpL, w[4], w[2], dest)
                        Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                        Interp1(dp+(dpL << 1), w[4], w[5], dest)
                        Interp1(dp+(dpL * 3), w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 14|142:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp3(dp+2, w[5], w[6], dest)
                        Interp8(dp+3, w[5], w[6], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp8(dp+1, w[2], w[4], dest)
                        Interp1(dp+2, w[2], w[5], dest)
                        Interp1(dp+3, w[5], w[2], dest)
                        Interp2(dp+dpL, w[4], w[5], w[2], dest)
                        Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 67:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 70:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 28:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 152:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 194:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 98:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 56:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 25:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 26|31:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 82|214:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    dest[dp+dpL+2] = w[5]
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 88|248:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                case 74|107:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 27:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 86:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    dest[dp+dpL+2] = w[5]
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 216:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 106:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 30:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 210:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 120:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 75:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 29:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 198:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 184:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 99:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 57:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 71:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 156:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 226:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 60:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 195:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 102:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 153:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 58:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp+2, w[5], w[2], dest)
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+dpL+3, w[5], w[6], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 83:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp+2, w[5], w[2], dest)
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+dpL+3, w[5], w[6], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp1(dp+(dpL << 1)+3, w[5], w[6], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[8], dest)
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 92:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[8], dest)
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp1(dp+(dpL << 1)+3, w[5], w[6], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[8], dest)
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 202:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 78:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 154:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp+2, w[5], w[2], dest)
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+dpL+3, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 114:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp+2, w[5], w[2], dest)
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+dpL+3, w[5], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp1(dp+(dpL << 1)+3, w[5], w[6], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[8], dest)
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                case 89:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[8], dest)
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp1(dp+(dpL << 1)+3, w[5], w[6], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[8], dest)
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 90:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp+2, w[5], w[2], dest)
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+dpL+3, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[8], dest)
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp1(dp+(dpL << 1)+3, w[5], w[6], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[8], dest)
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 55|23:
                    if (Diff(w[2], w[6])):
                        Interp8(dp, w[5], w[4], dest)
                        Interp3(dp+1, w[5], w[4], dest)
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+2] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp1(dp, w[5], w[2], dest)
                        Interp1(dp+1, w[2], w[5], dest)
                        Interp8(dp+2, w[2], w[6], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                        Interp2(dp+dpL+3, w[6], w[5], w[2], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 182|150:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+2] = w[5]
                        dest[dp+dpL+3] = w[5]
                        Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                    else:
                        Interp2(dp+2, w[2], w[5], w[6], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                        Interp8(dp+dpL+3, w[6], w[2], dest)
                        Interp1(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp1(dp+(dpL * 3)+3, w[5], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                case 213|212:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    if (Diff(w[6], w[8])):
                        Interp8(dp+3, w[5], w[2], dest)
                        Interp3(dp+dpL+3, w[5], w[2], dest)
                        dest[dp+(dpL << 1)+2] = w[5]
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp1(dp+3, w[5], w[6], dest)
                        Interp1(dp+dpL+3, w[6], w[5], dest)
                        Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                        Interp8(dp+(dpL << 1)+3, w[6], w[8], dest)
                        Interp2(dp+(dpL * 3)+2, w[8], w[5], w[6], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 241|240:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+2] = w[5]
                        dest[dp+(dpL << 1)+3] = w[5]
                        Interp8(dp+(dpL * 3), w[5], w[4], dest)
                        Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                        Interp2(dp+(dpL << 1)+3, w[6], w[5], w[8], dest)
                        Interp1(dp+(dpL * 3), w[5], w[8], dest)
                        Interp1(dp+(dpL * 3)+1, w[8], w[5], dest)
                        Interp8(dp+(dpL * 3)+2, w[8], w[6], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                case 236|232:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                        Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[4], w[5], w[8], dest)
                        Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp8(dp+(dpL * 3)+1, w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp1(dp+(dpL * 3)+3, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                case 109|105:
                    if (Diff(w[8], w[4])):
                        Interp8(dp, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[2], dest)
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp1(dp, w[5], w[4], dest)
                        Interp1(dp+dpL, w[4], w[5], dest)
                        Interp8(dp+(dpL << 1), w[4], w[8], dest)
                        Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp2(dp+(dpL * 3)+1, w[8], w[5], w[4], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 171|43:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                        Interp3(dp+(dpL << 1), w[5], w[8], dest)
                        Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp2(dp+1, w[2], w[5], w[4], dest)
                        Interp8(dp+dpL, w[4], w[2], dest)
                        Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                        Interp1(dp+(dpL << 1), w[4], w[5], dest)
                        Interp1(dp+(dpL * 3), w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 143|15:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        Interp3(dp+2, w[5], w[6], dest)
                        Interp8(dp+3, w[5], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp8(dp+1, w[2], w[4], dest)
                        Interp1(dp+2, w[2], w[5], dest)
                        Interp1(dp+3, w[5], w[2], dest)
                        Interp2(dp+dpL, w[4], w[5], w[2], dest)
                        Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 124:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 203:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 62:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    dest[dp+dpL+2] = w[5]
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 211:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 118:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    dest[dp+dpL+2] = w[5]
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 217:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 110:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 155:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 188:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 185:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 61:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 157:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 103:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 227:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 230:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 199:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 220:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[8], dest)
                        dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                case 158:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                        dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 234:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 242:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp+2, w[5], w[2], dest)
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+dpL+3, w[5], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                case 59:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp+2, w[5], w[2], dest)
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+dpL+3, w[5], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 121:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp1(dp+(dpL << 1)+3, w[5], w[6], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[8], dest)
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 87:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    dest[dp+dpL+2] = w[5]
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp1(dp+(dpL << 1)+3, w[5], w[6], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[8], dest)
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 79:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 122:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp+2, w[5], w[2], dest)
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+dpL+3, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp1(dp+(dpL << 1)+3, w[5], w[6], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[8], dest)
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 94:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                        dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[8], dest)
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp1(dp+(dpL << 1)+3, w[5], w[6], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[8], dest)
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 218:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp+2, w[5], w[2], dest)
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+dpL+3, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[8], dest)
                        dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                case 91:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp+2, w[5], w[2], dest)
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+dpL+3, w[5], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[8], dest)
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp1(dp+(dpL << 1)+3, w[5], w[6], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[8], dest)
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 229:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 167:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 173:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 181:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 186:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp+2, w[5], w[2], dest)
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+dpL+3, w[5], w[6], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 115:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp+2, w[5], w[2], dest)
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+dpL+3, w[5], w[6], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp1(dp+(dpL << 1)+3, w[5], w[6], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[8], dest)
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                case 93:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[8], dest)
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp1(dp+(dpL << 1)+3, w[5], w[6], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[8], dest)
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 206:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 205|201:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    if (Diff(w[8], w[4])):
                        Interp1(dp+(dpL << 1), w[5], w[7], dest)
                        Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                        Interp8(dp+(dpL * 3), w[5], w[7], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    else:
                        Interp1(dp+(dpL << 1), w[5], w[4], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 174|46:
                    if (Diff(w[4], w[2])):
                        Interp8(dp, w[5], w[1], dest)
                        Interp1(dp+1, w[5], w[1], dest)
                        Interp1(dp+dpL, w[5], w[1], dest)
                        Interp3(dp+dpL+1, w[5], w[1], dest)
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        Interp1(dp+1, w[5], w[2], dest)
                        Interp1(dp+dpL, w[5], w[4], dest)
                        dest[dp+dpL+1] = w[5]
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 179|147:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    if (Diff(w[2], w[6])):
                        Interp1(dp+2, w[5], w[3], dest)
                        Interp8(dp+3, w[5], w[3], dest)
                        Interp3(dp+dpL+2, w[5], w[3], dest)
                        Interp1(dp+dpL+3, w[5], w[3], dest)
                    else:
                        Interp1(dp+2, w[5], w[2], dest)
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+2] = w[5]
                        Interp1(dp+dpL+3, w[5], w[6], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 117|116:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                        Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                    else:
                        dest[dp+(dpL << 1)+2] = w[5]
                        Interp1(dp+(dpL << 1)+3, w[5], w[6], dest)
                        Interp1(dp+(dpL * 3)+2, w[5], w[8], dest)
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                case 189:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 231:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 126:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 219:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 125:
                    if (Diff(w[8], w[4])):
                        Interp8(dp, w[5], w[2], dest)
                        Interp3(dp+dpL, w[5], w[2], dest)
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp1(dp, w[5], w[4], dest)
                        Interp1(dp+dpL, w[4], w[5], dest)
                        Interp8(dp+(dpL << 1), w[4], w[8], dest)
                        Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp2(dp+(dpL * 3)+1, w[8], w[5], w[4], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 221:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    if (Diff(w[6], w[8])):
                        Interp8(dp+3, w[5], w[2], dest)
                        Interp3(dp+dpL+3, w[5], w[2], dest)
                        dest[dp+(dpL << 1)+2] = w[5]
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp1(dp+3, w[5], w[6], dest)
                        Interp1(dp+dpL+3, w[6], w[5], dest)
                        Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                        Interp8(dp+(dpL << 1)+3, w[6], w[8], dest)
                        Interp2(dp+(dpL * 3)+2, w[8], w[5], w[6], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 207:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        Interp3(dp+2, w[5], w[6], dest)
                        Interp8(dp+3, w[5], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp8(dp+1, w[2], w[4], dest)
                        Interp1(dp+2, w[2], w[5], dest)
                        Interp1(dp+3, w[5], w[2], dest)
                        Interp2(dp+dpL, w[4], w[5], w[2], dest)
                        Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 238:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                        Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                    else:
                        Interp2(dp+(dpL << 1), w[4], w[5], w[8], dest)
                        Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp8(dp+(dpL * 3)+1, w[8], w[4], dest)
                        Interp1(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp1(dp+(dpL * 3)+3, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                case 190:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+2] = w[5]
                        dest[dp+dpL+3] = w[5]
                        Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                        Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                    else:
                        Interp2(dp+2, w[2], w[5], w[6], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                        Interp8(dp+dpL+3, w[6], w[2], dest)
                        Interp1(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp1(dp+(dpL * 3)+3, w[5], w[6], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                case 187:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                        Interp3(dp+(dpL << 1), w[5], w[8], dest)
                        Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp2(dp+1, w[2], w[5], w[4], dest)
                        Interp8(dp+dpL, w[4], w[2], dest)
                        Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                        Interp1(dp+(dpL << 1), w[4], w[5], dest)
                        Interp1(dp+(dpL * 3), w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 243:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+2] = w[5]
                        dest[dp+(dpL << 1)+3] = w[5]
                        Interp8(dp+(dpL * 3), w[5], w[4], dest)
                        Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                        Interp2(dp+(dpL << 1)+3, w[6], w[5], w[8], dest)
                        Interp1(dp+(dpL * 3), w[5], w[8], dest)
                        Interp1(dp+(dpL * 3)+1, w[8], w[5], dest)
                        Interp8(dp+(dpL * 3)+2, w[8], w[6], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                case 119:
                    if (Diff(w[2], w[6])):
                        Interp8(dp, w[5], w[4], dest)
                        Interp3(dp+1, w[5], w[4], dest)
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+2] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp1(dp, w[5], w[2], dest)
                        Interp1(dp+1, w[2], w[5], dest)
                        Interp8(dp+2, w[2], w[6], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                        Interp2(dp+dpL+3, w[6], w[5], w[2], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 237|233:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[6], dest)
                    Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp7(dp+dpL+2, w[5], w[6], w[2], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[2], dest)
                    dest[dp+(dpL << 1)] = w[5]
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL * 3)] = w[5]
                    else:
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        dest[dp+(dpL * 3)+1] = w[5]
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 175|47:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        dest[dp+1] = w[5]
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp7(dp+(dpL << 1)+2, w[5], w[6], w[8], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[6], dest)
                    Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 183|151:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    dest[dp+2] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+3] = w[5]
                    else:
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    dest[dp+dpL+2] = w[5]
                    dest[dp+dpL+3] = w[5]
                    Interp6(dp+(dpL << 1), w[5], w[4], w[8], dest)
                    Interp7(dp+(dpL << 1)+1, w[5], w[4], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[4], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 245|244:
                    Interp2(dp, w[5], w[2], w[4], dest)
                    Interp6(dp+1, w[5], w[2], w[4], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp6(dp+dpL, w[5], w[4], w[2], dest)
                    Interp7(dp+dpL+1, w[5], w[4], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    dest[dp+(dpL << 1)+3] = w[5]
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    dest[dp+(dpL * 3)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 250:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                case 123:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 95:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 222:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    dest[dp+dpL+2] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 252:
                    Interp8(dp, w[5], w[1], dest)
                    Interp6(dp+1, w[5], w[2], w[1], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 249:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp6(dp+2, w[5], w[2], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    dest[dp+(dpL << 1)] = w[5]
                    dest[dp+(dpL << 1)+1] = w[5]
                    dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL * 3)] = w[5]
                    else:
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        dest[dp+(dpL * 3)+1] = w[5]
                case 235:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp6(dp+dpL+3, w[5], w[6], w[3], dest)
                    dest[dp+(dpL << 1)] = w[5]
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL * 3)] = w[5]
                    else:
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        dest[dp+(dpL * 3)+1] = w[5]
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 111:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        dest[dp+1] = w[5]
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp6(dp+(dpL << 1)+3, w[5], w[6], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 63:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp6(dp+(dpL * 3)+2, w[5], w[8], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 159:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                        dest[dp+2] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+3] = w[5]
                    else:
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                        dest[dp+dpL+3] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp6(dp+(dpL * 3)+1, w[5], w[8], w[7], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 215:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    dest[dp+2] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+3] = w[5]
                    else:
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    dest[dp+dpL+2] = w[5]
                    dest[dp+dpL+3] = w[5]
                    Interp6(dp+(dpL << 1), w[5], w[4], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 246:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                    Interp6(dp+dpL, w[5], w[4], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    dest[dp+dpL+2] = w[5]
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    dest[dp+(dpL << 1)+3] = w[5]
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    dest[dp+(dpL * 3)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 254:
                    Interp8(dp, w[5], w[1], dest)
                    Interp1(dp+1, w[5], w[1], dest)
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                    Interp1(dp+dpL, w[5], w[1], dest)
                    Interp3(dp+dpL+1, w[5], w[1], dest)
                    dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 253:
                    Interp8(dp, w[5], w[2], dest)
                    Interp8(dp+1, w[5], w[2], dest)
                    Interp8(dp+2, w[5], w[2], dest)
                    Interp8(dp+3, w[5], w[2], dest)
                    Interp3(dp+dpL, w[5], w[2], dest)
                    Interp3(dp+dpL+1, w[5], w[2], dest)
                    Interp3(dp+dpL+2, w[5], w[2], dest)
                    Interp3(dp+dpL+3, w[5], w[2], dest)
                    dest[dp+(dpL << 1)] = w[5]
                    dest[dp+(dpL << 1)+1] = w[5]
                    dest[dp+(dpL << 1)+2] = w[5]
                    dest[dp+(dpL << 1)+3] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL * 3)] = w[5]
                    else:
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        dest[dp+(dpL * 3)+1] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 251:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                    Interp1(dp+2, w[5], w[3], dest)
                    Interp8(dp+3, w[5], w[3], dest)
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[3], dest)
                    Interp1(dp+dpL+3, w[5], w[3], dest)
                    dest[dp+(dpL << 1)] = w[5]
                    dest[dp+(dpL << 1)+1] = w[5]
                    dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL * 3)] = w[5]
                    else:
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        dest[dp+(dpL * 3)+1] = w[5]
                case 239:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        dest[dp+1] = w[5]
                    Interp3(dp+2, w[5], w[6], dest)
                    Interp8(dp+3, w[5], w[6], dest)
                    dest[dp+dpL] = w[5]
                    dest[dp+dpL+1] = w[5]
                    Interp3(dp+dpL+2, w[5], w[6], dest)
                    Interp8(dp+dpL+3, w[5], w[6], dest)
                    dest[dp+(dpL << 1)] = w[5]
                    dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL << 1)+3, w[5], w[6], dest)
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL * 3)] = w[5]
                    else:
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        dest[dp+(dpL * 3)+1] = w[5]
                    Interp3(dp+(dpL * 3)+2, w[5], w[6], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[6], dest)
                case 127:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        dest[dp+1] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+2] = w[5]
                        dest[dp+3] = w[5]
                        dest[dp+dpL+3] = w[5]
                    else:
                        Interp5(dp+2, w[2], w[5], dest)
                        Interp5(dp+3, w[2], w[6], dest)
                        Interp5(dp+dpL+3, w[6], w[5], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL * 3)] = w[5]
                        dest[dp+(dpL * 3)+1] = w[5]
                    else:
                        Interp5(dp+(dpL << 1), w[4], w[5], dest)
                        Interp5(dp+(dpL * 3), w[8], w[4], dest)
                        Interp5(dp+(dpL * 3)+1, w[8], w[5], dest)
                        dest[dp+(dpL << 1)+1] = w[5]
                    Interp3(dp+(dpL << 1)+2, w[5], w[9], dest)
                    Interp1(dp+(dpL << 1)+3, w[5], w[9], dest)
                    Interp1(dp+(dpL * 3)+2, w[5], w[9], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[9], dest)
                case 191:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+3] = w[5]
                    else:
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                        dest[dp+dpL+3] = w[5]
                    Interp3(dp+(dpL << 1), w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+2, w[5], w[8], dest)
                    Interp3(dp+(dpL << 1)+3, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3), w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+1, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+2, w[5], w[8], dest)
                    Interp8(dp+(dpL * 3)+3, w[5], w[8], dest)
                case 223:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                        dest[dp+1] = w[5]
                        dest[dp+dpL] = w[5]
                    else:
                        Interp5(dp, w[2], w[4], dest)
                        Interp5(dp+1, w[2], w[5], dest)
                        Interp5(dp+dpL, w[4], w[5], dest)
                        dest[dp+2] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+3] = w[5]
                    else:
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                        dest[dp+dpL+3] = w[5]
                    Interp1(dp+(dpL << 1), w[5], w[7], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[7], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL << 1)+3] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp5(dp+(dpL << 1)+3, w[6], w[5], dest)
                        Interp5(dp+(dpL * 3)+2, w[8], w[5], dest)
                        Interp5(dp+(dpL * 3)+3, w[8], w[6], dest)
                    Interp8(dp+(dpL * 3), w[5], w[7], dest)
                    Interp1(dp+(dpL * 3)+1, w[5], w[7], dest)
                case 247:
                    Interp8(dp, w[5], w[4], dest)
                    Interp3(dp+1, w[5], w[4], dest)
                    dest[dp+2] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+3] = w[5]
                    else:
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                    Interp8(dp+dpL, w[5], w[4], dest)
                    Interp3(dp+dpL+1, w[5], w[4], dest)
                    dest[dp+dpL+2] = w[5]
                    dest[dp+dpL+3] = w[5]
                    Interp8(dp+(dpL << 1), w[5], w[4], dest)
                    Interp3(dp+(dpL << 1)+1, w[5], w[4], dest)
                    dest[dp+(dpL << 1)+2] = w[5]
                    dest[dp+(dpL << 1)+3] = w[5]
                    Interp8(dp+(dpL * 3), w[5], w[4], dest)
                    Interp3(dp+(dpL * 3)+1, w[5], w[4], dest)
                    dest[dp+(dpL * 3)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
                case 255:
                    if (Diff(w[4], w[2])):
                        dest[dp] = w[5]
                    else:
                        Interp2(dp, w[5], w[2], w[4], dest)
                        dest[dp+1] = w[5]
                        dest[dp+2] = w[5]
                    if (Diff(w[2], w[6])):
                        dest[dp+3] = w[5]
                    else:
                        Interp2(dp+3, w[5], w[2], w[6], dest)
                        dest[dp+dpL] = w[5]
                        dest[dp+dpL+1] = w[5]
                        dest[dp+dpL+2] = w[5]
                        dest[dp+dpL+3] = w[5]
                        dest[dp+(dpL << 1)] = w[5]
                        dest[dp+(dpL << 1)+1] = w[5]
                        dest[dp+(dpL << 1)+2] = w[5]
                        dest[dp+(dpL << 1)+3] = w[5]
                    if (Diff(w[8], w[4])):
                        dest[dp+(dpL * 3)] = w[5]
                    else:
                        Interp2(dp+(dpL * 3), w[5], w[8], w[4], dest)
                        dest[dp+(dpL * 3)+1] = w[5]
                        dest[dp+(dpL * 3)+2] = w[5]
                    if (Diff(w[6], w[8])):
                        dest[dp+(dpL * 3)+3] = w[5]
                    else:
                        Interp2(dp+(dpL * 3)+3, w[5], w[8], w[6], dest)
            sp += 1
            dp += 4
        dp += (dpL * 3)


def main():
    try:
        import sys
        file_path = sys.argv[1]
        out_path = sys.argv[2]
        scale_factor = int(sys.argv[3])
    except Exception:
        print('usage: python hqx.py <file_path: str> <out_path: str> <scale_factor: int>')
        exit(-1)

    img = cv2.imread(file_path)
    cv2.imwrite(out_path, hqx(img, scale_factor))


if __name__ == '__main__':
    main()
