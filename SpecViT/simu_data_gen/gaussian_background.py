import numpy as np
import random
# ------------------------------------------------
#  This function generates Dax images for motion-blur NN analysis training
# --------------------------------------------------------------------------
#  Input:
#  Dmap: Diffusivity in um2/s unit. 
#  dim: Frame size
#  FrameLength: Length of simulation frame
#  Particle_density: [Mean sigma] of normal distribution of particle per frame
#  sim_t: random-walk step interval in ms
#  exposure_t: exposure time in unit of ms
#  pl : pixel length in um
#  bg: [mean sd] of Gaussian distribution
#  Photon_burst: photon count per ms
#  sig: PSF sigma assuming gaussian
#  gain_factor: gain factor of imaging setup
# --------------------------------------------------------------------------
def Trainingset_gaussian_background(Dmap,FrameLength,exposure_t,sim_t,
                                    photon_burst,bg,pl,sl,sig,gain_factor,centroid,sig_spec):
    
    dim_spec = [64,16]
    dim_pos  = [16,16]
    sim_step = int(exposure_t/sim_t)
    daxmovie = np.zeros((16,80,FrameLength)) # 64,16

    x,y = np.meshgrid(range(73),range(16))
    pos_x,pos_y = np.meshgrid(range(16),range(16))

    prob = np.random.rand()

    for f in range(FrameLength):
        frame = np.random.normal(bg[0], bg[1], (16, 73))
        frame_pos = np.random.normal(bg[0], bg[1], (16, 16))

        ## spectral channel
        tmp_x = np.random.normal(0, 1)
        tmp_y = np.random.normal(0, 1)
        tmp = np.array([tmp_x,tmp_y])
        spec_pos = (np.array(dim_spec)/2 + tmp)*pl # 32 8
        spec_pos[0] = spec_pos[0] + SpecPixMapping(centroid)*pl # spectral centroid

        int_pos = (np.array(dim_pos)/2 + tmp)*pl

        for _ in range(sim_step):
            ang = 2*np.pi*np.random.randn()
            step = 2*np.sqrt(Dmap*sim_t/1000)*np.array([np.cos(ang),-np.sin(ang)])

            ##################specatral channel######################
            spec_pos = spec_pos + step
            if spec_pos[0] < 0 or spec_pos[0] > 73*pl:
                spec_pos[0] = spec_pos[0] - 2*step[0]
            if spec_pos[1] < 0 or spec_pos[1] > 16*pl:          
                spec_pos[1] = spec_pos[1] - 2*step[1]

            psf_x = np.exp(-np.power((x - spec_pos[0]/pl - 0.5),2)/(2*np.power(sig_spec/sl,2)))
            psf_y = np.exp(-np.power((y - spec_pos[1]/pl - 0.5),2)/(2*np.power(sig/pl,2)))
            amp = sim_t*photon_burst*gain_factor/(np.sqrt(2*np.pi*(np.power(sig/pl,2)+np.power(sig_spec/sl,2))))
            
            if np.any(frame < 0):
               frame[np.where(frame < 0)]= np.random.normal(bg[0], bg[1])

            frame = frame + amp * psf_x * psf_y

            ##################position channel######################
            int_pos = int_pos + step
            if int_pos[0] < 0 or int_pos[0] > dim_pos[0]*pl:
                int_pos[0] = int_pos[0] - 2*step[0]
            if int_pos[1] < 0 or int_pos[1] > dim_pos[1]*pl:
                int_pos[1] = int_pos[1] - 2*step[1]

            psf_x = np.exp(-np.power((pos_x - int_pos[0]/pl - 0.5),2)/(2*np.power(sig/pl,2)))
            psf_y = np.exp(-np.power((pos_y - int_pos[1]/pl - 0.5),2)/(2*np.power(sig/pl,2)))
            amp = 0.5*sim_t*photon_burst*gain_factor/(2*np.pi*(np.power(sig/pl,2)))
            
            if np.any(frame_pos < 0):
               frame_pos[np.where(frame_pos < 0)]= np.random.normal(bg[0], bg[1])

            frame_pos = frame_pos + amp * psf_x * psf_y

        frame = np.array(frame, dtype=np.uint16)
        frame_filter = specFilter(frame, bg, centroid, prob)
        norm_frame_filter = (frame_filter - frame_filter.min()) / \
                            (frame_filter.max() - frame_filter.min())

        frame_pos = np.array(frame_pos, dtype=np.uint16)
        norm_frame_pos = (frame_pos - frame_pos.min()) / \
                            (frame_pos.max() - frame_pos.min())

        daxmovie[:,:,f] = np.concatenate((norm_frame_pos, norm_frame_filter),axis=1)

    return daxmovie

def PixSpecMapping(pix):
    spec = 0.02215*pix**2+3.077*pix+591.6
    # spec = 0.001384*pix**2+0.7694*pix+591.6
    return spec

def SpecPixMapping(spec):
    pix = -0.0009465*spec**2+1.455*spec-529.4
    # pix = -0.003786*spec**2+5.821*spec-2118
    return pix


def specFilter(spec_img,bg,centroid,prob):
    spec_img_filter = np.random.normal(bg[0], bg[1], (16,64))
    transRateR488 = np.array([0.99,0.7,0.3,0.1,0.1,0.01])
    transRateL488 = np.array([0.01,0.1,0.1,0.3,0.7,0.99])
    
    transRateR561 = np.array([0.99,0.9,0.7,0.3,0.1,0.01])
    transRateL561 = np.array([0.01,0.1,0.3,0.7,0.9,0.99])

    transRateR640 = np.array([0.99,0.9,0.7,0.3,0.1,0.01])
    transRateL640 = np.array([0.01,0.1,0.3,0.7,0.9,0.99])

    probability = 0.1
    if prob <= probability:
        spec_img_filter[:,0:6] = (1 - transRateL488) * spec_img_filter[:,0:6] + transRateL488 * spec_img[:,0:6]
        spec_img_filter[:,6:18] = spec_img[:,6:18]
        spec_img_filter[:,18:24] = (1 - transRateR488) * spec_img_filter[:,18:24] + transRateR488 * spec_img[:,18:24]

        spec_img_filter[:,24:30] = (1 - transRateL561) * spec_img_filter[:,31:37] + transRateL561 * spec_img[:,31:37]
        spec_img_filter[:,30:38] = spec_img[:,37:45]
        spec_img_filter[:,38:44] = (1 - transRateR561) * spec_img_filter[:,38:44] + transRateR561 * spec_img[:,45:51]

        spec_img_filter[:,44:50] = (1 - transRateL640) * spec_img_filter[:,44:50] + transRateL640 * spec_img[:,53:59]
        spec_img_filter[:,50:64] = spec_img[:,59:73]
    else:
        if centroid >= PixSpecMapping(-34) and centroid <= PixSpecMapping(-11):
            spec_img_filter[:,0:6] = (1 - transRateL488) * spec_img_filter[:,0:6] + transRateL488 * spec_img[:,0:6]
            spec_img_filter[:,6:18] = spec_img[:,6:18]
            spec_img_filter[:,18:24] = (1 - transRateR488) * spec_img_filter[:,18:24] + transRateR488 * spec_img[:,18:24]

        elif centroid > PixSpecMapping(-11) and centroid <= PixSpecMapping(16):
            spec_img_filter[:,24:30] = (1 - transRateL561) * spec_img_filter[:,31:37] + transRateL561 * spec_img[:,31:37]
            spec_img_filter[:,30:38] = spec_img[:,37:45]
            spec_img_filter[:,38:44] = (1 - transRateR561) * spec_img_filter[:,38:44] + transRateR561 * spec_img[:,45:51]

        elif centroid > PixSpecMapping(16) and centroid <= PixSpecMapping(38):
            spec_img_filter[:,44:50] = (1 - transRateL640) * spec_img_filter[:,44:50] + transRateL640 * spec_img[:,53:59]
            spec_img_filter[:,50:64] = spec_img[:,59:73]

    return spec_img_filter