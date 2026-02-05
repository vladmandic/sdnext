'''
This is the python implementation of icbi.m

Author: gyf
Begin: 2019-1-16
'''
import numpy as np
import cv2


def icbi(IM,ZK = 1,SZ = 8,PF = 1,ST = 20,TM = 100,TC = 50,SC = 1,TS = 100,AL = 1,BT = -1,GM = 5):
    '''

    :param IM: Source image
    :param ZK: Power of zoom factor (default:1)
    :param SZ: Number of image bits per layer (default:8)
    :param PF: Potential to be minimized (default:1)
    :param ST: Maximum number of iterations (default:20)
    :param TM: Maximum edge step (default:100)
    :param TC: Edge continuity threshold (deafult:50).
    :param SC: Stopping criterion: 1 = change under threshold, 0 = ST iterations (default:1).
    :param TS: Threshold on image change for stopping iterations (default:100).
    :param AL: Weight for Curvature Continuity energy (default:1.0).
    :param BT: Weight for Curvature enhancement energy (default:-1.0).
    :param GM: Weight for Isophote smoothing energy (default:5.0).
    :return: EI: Enlarged image
    '''
    H = IM.shape[0]
    W = IM.shape[1]
    if ZK < 1:
        EI = cv2.resize(IM,(H*(2**ZK),W*(2**ZK)))

    #check image type
    IDIM = np.ndim(IM)
    if IDIM == 3:
        CL = IM.shape[2] #number of colors

    elif IDIM == 2:
        IM = np.reshape(IM,(H,W,1))
        CL = 1
    else:
        print('Unrecognized image type, please use RGB or grayscale images')
        return 0


    #calculate final size
    fm = H * (2**ZK) - (2**ZK - 1)
    fn = W * (2**ZK) - (2**ZK - 1)

    #initialize output image
    if SZ>32:
        EI  = np.zeros([fm,fn,CL],dtype= np.uint64)

    elif SZ>16:
        EI = np.zeros([fm,fn,CL],dtype= np.uint32)

    elif SZ>8:
        EI = np.zeros([fm,fn,CL],dtype= np.uint16)

    else:
        EI = np.zeros([fm,fn,CL],dtype= np.uint8)

    #each image color
    IMG = IM.copy()
    for CID in range(CL):
        IMG = IM[:,:,CID]
        #The image is enlarged by scaling factor 2**ZK-1 at each cycle
        for _ZF in range(ZK):

            #size of enlarged image
            mm = 2*H - 1
            nn = 2*W - 1

            #initialize expanded and support matrix
            IMGEXP = np.zeros([mm,nn])
            D1 = np.zeros([mm,nn])
            D2 = np.zeros([mm,nn])
            D3 = np.zeros([mm,nn])
            C1 = np.zeros([mm,nn])
            C2 = np.zeros([mm,nn])

            #copy low resolution grid on high resolution grid
            IMGEXP[::2,::2] = IMG

            #interpolation at borders (average value of 2 neighbors)
            for i in range(1,mm-1,2):
                #left col
                IMGEXP[i,0] = (IMGEXP[i-1,0]+IMGEXP[i+1,0])/2
                #right col
                IMGEXP[i,nn-1] = (IMGEXP[i-1,nn-1]+IMGEXP[i+1,nn-1])/2

            for i in range(1,nn,2):
                #top row
                IMGEXP[0,i] = (IMGEXP[0,i-1] + IMGEXP[0,i+1])/2
                #bottom row
                IMGEXP[mm-1,i] = (IMGEXP[mm-1,i-1]+IMGEXP[mm-1,i+1])/2

            #Calculate interpolated points in two steps
            #s = 0 calculates on diagonal directions
            #s = 1 calculates on vertical and horizontal directions
            for s in range(2):
                #FCBI (Fast Curvature Based Interpolation)
                for i in range(1,mm-s,2-s):
                    for j in range(1+(s*(1-np.mod(i+1,2))),nn-s,2):
                        v1 = np.abs(IMGEXP[i-1,j-1+s]-IMGEXP[i+1,j+1-s])
                        v2 = np.abs(IMGEXP[i+1-s,j-1]-IMGEXP[i-1+s,j+1])
                        p1 = (IMGEXP[i-1,j-1+s]+IMGEXP[i+1,j+1-s])/2
                        p2 = (IMGEXP[i+1-s,j-1]+IMGEXP[i-1+s,j+1])/2
                        if (v1<TM) and (v2<TM) and (i>2-s) and i<mm-4-s and j>2-s and j<nn-4-s and (np.abs(p1-p2)<TM):
                            if np.abs( IMGEXP[i-1-s,j-3+2*s] + IMGEXP[i-3+s,j-1+2*s] + IMGEXP[i+1+s,j+3-2*s] +IMGEXP[i+3-s,j+1-2*s] + 2*p2-6*p1)> np.abs( IMGEXP[i-3+2*s,j+1+s] + IMGEXP[i-1+2*s,j+3-s] + IMGEXP[i+3-2*s,j-1-s] +IMGEXP[i+1-2*s,j-3+s] + 2*p1-6*p2):
                                IMGEXP[i,j] = p1

                            else:
                                IMGEXP[i,j] = p2

                        else:
                            if v1<v2:
                                IMGEXP[i,j] = p1
                            else:
                                IMGEXP[i,j] = p2

                step = 4.0/(1+s)

                #iterative refinement
                for g in range(ST):
                    diff = 0

                    if g<ST/4 -1:
                        step = 1
                    elif g<ST/2 -1:
                        step = 2
                    elif g<3*ST/4 -1:
                        step = 2

                    #computation of derivatives:
                    for i in range(3-2*s,mm-3+s):
                        for j in range(3-2*s+(1-s)*np.mod(i+1,2),nn-3+s,2-s):
                            C1[i,j] = (IMGEXP[i-1+s,j-1] - IMGEXP[i+1-s,j+1])/2
                            C2[i,j] = (IMGEXP[i+1-2*s,j-1+s] - IMGEXP[i-1+2*s,j+1-s])/2
                            D1[i,j] = IMGEXP[i-1+s,j-1] + IMGEXP[i+1-s,j+1] - 2*IMGEXP[i,j]
                            D2[i,j] = IMGEXP[i+1,j-1+s] + IMGEXP[i-1,j+1-s] - 2*IMGEXP[i,j]
                            D3[i,j] = (IMGEXP[i-s,j-2+s] - IMGEXP[i-2+s,j+s] + IMGEXP[i+s,j+2-s] - IMGEXP[i+2-s,j-s])/2


                    for i in range(5-3*s,mm-5+3*s,2-s):
                        for j in range(5+s*(np.mod(i+1,2)-2),nn-5+3*s,2):
                            c_1 = 1
                            c_2 = 1
                            c_3 = 1
                            c_4 = 1
                            if np.abs(IMGEXP[i+1-s,j+1] - IMGEXP[i,j])>TC:
                                c_1 = 0

                            if np.abs(IMGEXP[i-1+s,j-1] - IMGEXP[i,j])>TC:
                                c_2 = 0

                            if np.abs(IMGEXP[i+1,j-1+s] - IMGEXP[i,j])>TC:
                                c_3 = 0

                            if np.abs(IMGEXP[i-1,j+1-s] - IMGEXP[i,j])>TC:
                                c_4 = 0


                            EN1 = c_1*np.abs(D1[i,j] - D1[i+1-s,j+1]) + c_2*np.abs(D1[i,j] - D1[i-1+s,j-1])
                            EN2 = c_3*np.abs(D1[i,j] - D1[i+1,j-1+s]) + c_4*np.abs(D1[i,j] - D1[i-1,j+1-s])
                            EN3 = c_1*np.abs(D2[i,j] - D2[i+1-s,j+1]) + c_2*np.abs(D2[i,j] - D2[i-1+s,j-1])
                            EN4 = c_3*np.abs(D2[i,j] - D2[i+1,j-1+s]) + c_4*np.abs(D2[i,j] - D2[i-1,j+1-s])
                            EN5 = np.abs(IMGEXP[i-2+2*s,j-2] + IMGEXP[i+2-2*s,j+2] - 2*IMGEXP[i,j])
                            EN6 = np.abs(IMGEXP[i+2,j-2+2*s] + IMGEXP[i-2,j+2-2*s] - 2*IMGEXP[i,j])

                            EA1 = c_1*np.abs(D1[i,j] - D1[i+1-s,j+1] - 3*step) + c_2*np.abs(D1[i,j] - D1[i-1+s,j-1] - 3*step)
                            EA2 = c_3*np.abs(D1[i,j] - D1[i+1,j-1+s] - 3*step) + c_4*np.abs(D1[i,j] - D1[i-1,j+1-s] - 3*step)
                            EA3 = c_1*np.abs(D2[i,j] - D2[i+1-s,j+1] - 3*step) + c_2*np.abs(D2[i,j] - D2[i-1+s,j-1] - 3*step)
                            EA4 = c_3*np.abs(D2[i,j] - D2[i+1,j-1+s] - 3*step) + c_4*np.abs(D2[i,j] - D2[i-1,j+1-s] - 3*step)
                            EA5 = np.abs(IMGEXP[i-2+2*s,j-2] + IMGEXP[i+2-2*s,j+2] - 2*IMGEXP[i,j] - 2*step)
                            EA6 = np.abs(IMGEXP[i+2,j-2+2*s] + IMGEXP[i-2,j+2-2*s] - 2*IMGEXP[i,j] - 2*step)

                            ES1 = c_1*np.abs(D1[i,j] - D1[i+1-s,j+1] + 3*step) + c_2*np.abs(D1[i,j] - D1[i-1+s,j-1] + 3*step)
                            ES2 = c_3*np.abs(D1[i,j] - D1[i+1,j-1+s] + 3*step) + c_4*np.abs(D1[i,j] - D1[i-1,j+1-s] + 3*step)
                            ES3 = c_1*np.abs(D2[i,j] - D2[i+1-s,j+1] + 3*step) + c_2*np.abs(D2[i,j] - D2[i-1+s,j-1] + 3*step)
                            ES4 = c_3*np.abs(D2[i,j] - D2[i+1,j-1+s] + 3*step) + c_4*np.abs(D2[i,j] - D2[i-1,j+1-s] + 3*step)
                            ES5 = np.abs(IMGEXP[i-2+2*s,j-2] + IMGEXP[i+2-2*s,j+2] - 2*IMGEXP[i,j] + 2*step)
                            ES6 = np.abs(IMGEXP[i+2,j-2+2*s] + IMGEXP[i-2,j+2-2*s] - 2*IMGEXP[i,j] + 2*step)

                            EISO = (C1[i,j]*C1[i,j]*D2[i,j] - 2*C1[i,j]*C2[i,j]*D3[i,j] + C2[i,j]*C2[i,j]*D1[i,j])/(C1[i,j]*C1[i,j]+C2[i,j]*C2[i,j])

                            if np.abs(EISO) < 0.2:
                                EISO = 0

                            if PF==1:
                                EN = AL*(EN1 + EN2 + EN3 + EN4) + BT*(EN5 + EN6)
                                EA = AL*(EA1 + EA2 + EA3 + EA4) + BT*(EA5 + EA6)
                                ES = AL*(ES1 + ES2 + ES3 + ES4) + BT*(ES5 + ES6)

                            elif PF==2:
                                EN = AL*(EN1 + EN2 + EN3 + EN4)
                                EA = AL*(EA1 + EA2 + EA3 + EA4) - GM*np.sign(EISO)
                                ES = AL*(ES1 + ES2 + ES3 + ES4) - GM*np.sign(EISO)

                            else:
                                EN = AL*(EN1 + EN2 + EN3 + EN4) + BT*(EN5 + EN6)
                                EA = AL*(EA1 + EA2 + EA3 + EA4) + BT*(EA5 + EA6) - GM*np.sign(EISO)
                                ES = AL*(ES1 + ES2 + ES3 + ES4) + BT*(ES5 + ES6) + GM*np.sign(EISO)

                            if (EN>EA) and (ES>EA):
                                IMGEXP[i,j] = IMGEXP[i,j] + step
                                diff = diff + step

                            elif (EN>ES) and (EA>ES):
                                IMGEXP[i,j] = IMGEXP[i,j] - step
                                diff = diff + step

                    if (SC==1) and (diff<TS):
                        break

            #assign the expanded image to the current image
            IMG = IMGEXP

        EI[:,:,CID] = np.round(IMG)

    #back to 2D array if gray
    if CL ==1:
        EI = np.reshape(EI,(fm,fn))

    return EI
