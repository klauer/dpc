import numpy as np
import matplotlib.pyplot as plt

#a = lt.load_tiff(36025, 256, 256, 512, 512, 2)

def load(filename, nx_prb=256, ny_prb=256, x_raw=512, y_raw=512, threshold=0):
    diff_array = np.zeros((nx_prb, ny_prb))

    if 1:
        with open(filename, 'r') as f:
            np.fromfile(f, dtype='int32', count=2)
            tmp = np.fromfile(f, dtype='int16', count=x_raw * y_raw)

    else:
        tmp = np.arange(x_raw * y_raw)

    tmp.resize(y_raw, x_raw)
    #tmp = np.fliplr(np.transpose(tmp * 1.))

    tmp[np.where(tmp < threshold)] = 0.

    #t = np.zeros((516, 516))

    t = np.zeros((x_raw+4,y_raw+4))
    t[0:x_raw/2-1,0:y_raw/2-1] = tmp[0:x_raw/2-1,0:y_raw/2-1]
    t[x_raw/2+5:x_raw+4,0:y_raw/2-1] = tmp[x_raw/2+1:x_raw,0:y_raw/2-1]
    t[0:x_raw/2-1,y_raw/2+5:y_raw+4] = tmp[0:x_raw/2-1,y_raw/2+1:y_raw]
    t[x_raw/2+5:x_raw+4,y_raw/2+5:y_raw+4] = tmp[x_raw/2+1:x_raw,y_raw/2+1:y_raw]
    
    for i in range(y_raw):
        t[x_raw/2-1:x_raw/2+2,i] = tmp[x_raw/2-1,i] / 3.
        t[x_raw/2+2:x_raw/2+5,i] = tmp[x_raw/2,i] / 3.
    
    for i in range(x_raw):
        t[i,y_raw/2-1:y_raw/2+2] = tmp[i,y_raw/2-1] / 3.
        t[i,y_raw/2+2:y_raw/2+5] = tmp[i,y_raw/2] / 3.
    
    t[x_raw/2-1:x_raw/2+2,y_raw/2-1:y_raw/2+2] = tmp[x_raw/2-1,y_raw/2-1] / 9.
    t[x_raw/2-1:x_raw/2+2,y_raw/2+2:y_raw/2+5] = tmp[x_raw/2-1,y_raw/2] / 9.
    t[x_raw/2+2:x_raw/2+5,y_raw/2-1:y_raw/2+2] = tmp[x_raw/2,y_raw/2-1] / 9.
    t[x_raw/2+2:x_raw/2+5,y_raw/2+2:y_raw/2+5] = tmp[x_raw/2,y_raw/2] / 9.

    # t[141:147, 110:117] = 0.

    # t2 = t[105:105 + 256, 395 - 256:395]
    # diff_array[:, :] = np.sqrt(t2[:, :])

    if 0:
        plt.close('all')
        plt.figure()
        plt.imshow(np.log(diff_array[:, :] + 0.001))

    #t[209, 264] = 0
    #diff_array[106, 125] = 0.
    return t



def orig(file_name,nx_prb, ny_prb, x_raw=512, y_raw=512, threshold=0):
    diff_array = np.zeros((nx_prb, ny_prb))
    tmp = np.arange(x_raw * y_raw)

    tmp.resize(y_raw,x_raw)

    tmp = np.fliplr(np.transpose(tmp * 1.))

    index = np.where(tmp < threshold)

    tmp[index] = 0.



    t = np.zeros((516,516))

    t[:255,:255] = tmp[:255,:255].copy()

    t[:255,516-255:] = tmp[:255,512-255:].copy()

    t[516-255:,:255] = tmp[512-255:,:255].copy()

    t[516-255:,516-255:] = tmp[512-255:,512-255:].copy()



    t[:255,255] = tmp[:255,255] / 3.

    t[:255,256] = tmp[:255,255] / 3.

    t[:255,257] = tmp[:255,255] / 3.

    t[:255,258] = tmp[:255,256] / 3.

    t[:255,259] = tmp[:255,256] / 3.

    t[:255,260] = tmp[:255,256] / 3.



    t[516-255:,255] = tmp[512-255:,255] / 3.

    t[516-255:,256] = tmp[512-255:,255] / 3.

    t[516-255:,257] = tmp[512-255:,255] / 3.

    t[516-255:,258] = tmp[512-255:,256] / 3.

    t[516-255:,259] = tmp[512-255:,256] / 3.

    t[516-255:,260] = tmp[512-255:,256] / 3.



    t[255,:255] = tmp[255,:255] / 3.

    t[256,:255] = tmp[255,:255] / 3.

    t[257,:255] = tmp[255,:255] / 3.

    t[258,:255] = tmp[255,:255] / 3.

    t[259,:255] = tmp[255,:255] / 3.

    t[260,:255] = tmp[255,:255] / 3.



    t[255,516-255:] = tmp[255,512-255:] / 3.

    t[256,516-255:] = tmp[255,512-255:] / 3.

    t[257,516-255:] = tmp[255,512-255:] / 3.

    t[258,516-255:] = tmp[255,512-255:] / 3.

    t[259,516-255:] = tmp[255,512-255:] / 3.

    t[260,516-255:] = tmp[255,512-255:] / 3.



    for i in range(255,258):

        for j in range(255,258):

            t[i,j] = tmp[255,255] / 9.



    for i in range(258,261):

        for j in range(255,258):

            t[i,j] = tmp[256,255] / 9.



    for i in range(255,258):

        for j in range(258,261):

            t[i,j] = tmp[255,256] / 9.



    for i in range(258,261):

        for j in range(258,261):

            t[i,j] = tmp[256,256] / 9.



    t[141:147,110:117] = 0.





    t2 = t[105:105+256,395-256:395]

    diff_array[:,:] = np.sqrt(t2[:,:])



    plt.close('all')

    plt.figure()

    plt.imshow(np.log(diff_array[:,:]+0.001))



    diff_array[106,125] = 0.
    return diff_array

if 0:
    plt.figure(0)
    old = orig('', 256, 256)
    plt.figure(1)
    new = load_tiff('', 256, 256)
    print(sum(new - old))
    plt.show()
