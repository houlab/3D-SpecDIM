import numpy as np
from gaussian_background import Trainingset_gaussian_background
import tifffile
import matplotlib.pyplot as plt
import os 
import csv
import random
from tqdm import tqdm
# position位置 (4,4)->(11,11)

saveDir         = "/data/shah/projects/spms_track/simu_data/test/"
exposure_t      = 20               # simulation exposure time in ms unit
fileNum         = 200
N_per_D         = 20              # simulation movie length per diffusivity
sim_t           = 1/100*exposure_t            # simulation time interval in ms
pixel_size      = 237.92               # pixel length in nm
N_per_S         = 720
gain_factor     = 20                # gain factor of imaging setup
sl              = 2.849             # spectral  length in nm/pixel

daxmovie = np.zeros(shape=(16,80,N_per_S*fileNum))
csv_data = {'photons':np.zeros(N_per_S*fileNum), 'background':np.zeros(N_per_S*fileNum),
            'diffusivity':np.zeros(N_per_S*fileNum), 'centroid':np.zeros(N_per_S*fileNum)}

for i in tqdm(range(fileNum),desc="Processing"):
    centroid = round(random.uniform(510.00,740.00),2)

    for j in range(N_per_S // N_per_D):
        photon_burst = random.uniform(100, 300)
        background = [random.uniform(500,1100), random.uniform(70,150)]
        diffusivity = random.uniform(0,0.5)
        sig_spec = round(random.uniform(10.000,20.000),3)
        sig = round(np.random.normal(0.2240746,0.05),3)  # PSF size in um

        DaxNameHead = f'{round(1000/exposure_t)}Hz_centroid{centroid:.2f}' # not include .dax in the string
        DaxNameHead = DaxNameHead.replace('.','_')
        count = 0

        tempdax = Trainingset_gaussian_background(diffusivity, N_per_D, exposure_t, sim_t, 
                                                  photon_burst, background, pixel_size/1000, sl, 
                                                  sig, gain_factor, centroid, sig_spec)

        ii = i*N_per_S+j*N_per_D
        jj = i*N_per_S+(j+1)*N_per_D

        daxmovie[:,:,ii:jj] = tempdax
        csv_data['photons'][ii:jj] = photon_burst
        csv_data['background'][ii:jj] = background[0]
        csv_data['diffusivity'][ii:jj] = diffusivity
        csv_data['centroid'][ii:jj] = round(centroid,2)

    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    img = daxmovie[:,:,i*N_per_S:(i+1)*N_per_S].transpose((2,0,1))
    tifffile.imwrite(saveDir+DaxNameHead+'.tiff', img)

# save csv label file
with open(saveDir+'labels.csv','w') as csvfile:
    writer = csv.writer(csvfile)
    # write header
    csv_header = csv_data.keys()
    writer.writerow(csv_header)
    # write data
    csv_info = csv_data.values()
    csv_info = list(map(list, zip(*list(csv_info))))
    writer.writerows(csv_info)

