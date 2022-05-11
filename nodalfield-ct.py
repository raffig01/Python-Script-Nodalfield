import nibabel as nib
import numpy as np
import math
import os.path
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

""" 

This script creates a nodes array nx3 (n: number of nodes) of nodal coordinates
and appends columns for 4 variables. The variables are the hounsfield unit,
broad radiation value (total), focused radiation value (total), and tumour
value.

The variable values are obtained from a corresponding image by quering the pixel
value at each nodal coordinate. These are stored into an array (size n) and
appended to the main solution array (nx8 once completed) which is then outputed
(only variable values are output, not coordinates).

> Hounsfield is determined from the CT image. Each pixel of the CT images is
  imported in a 3D array. Then using the inverse affine matrix we can convert
  world coordinates to pixel coordinates to extract the HU.

> Radiation field is obtained from separate image files in a similar way to
  Hounsfield. There are two such images corresponding to phase 1 (broad) and
  phase 2 (focused) radiotherapy sessions. The first radiation field is the
  total value for broad radiation and the second is the total radiation values
  for the focused radiotherapy.

> Cancer field is zero everywhere for now.
"""

### Specify case details
patient='CASE01'
patientlung='right_lung'

# right: 01, 06, 08
# left: 05, 07, 09

basepath='/mnt/c/Users/raffi/Desktop/Diploma Thesis/Lungs - Radiation/'+patient+'/'
### Required files (CT images, Mesh)
nifti_ct_file = basepath+'NIFTI/'+patient+'_ct.nii'
nifti_rd_files = [basepath+'NIFTI/'+patient+'_phase_I.nii',basepath+'NIFTI/'+patient+'_phase_II.nii']
nifti_tm_file = basepath+'NIFTI/'+patient+'_ct0_tumour.nii'
msh_file = basepath+'Segmentation_'+patientlung+' (meshmixer).msh'

### Case name (used in output files)
name=patient+'_'+patientlung
### Specify output
out_dir='/mnt/c/Users/raffi/Desktop/' #/home/schlang/pool/'
warnings=0

### Obtain nodes coordinates from msh file
f_msh = open(msh_file,'r')
lines_msh = f_msh.readlines()
f_msh.close()

nodes_size = int(lines_msh[4])
nodes=np.zeros((nodes_size,3))
for line in lines_msh[5:nodes_size+5]:
    coord = np.fromstring(line, dtype=float, sep=' ')
    nodes[int(coord[0])-1,0]=coord[1]
    nodes[int(coord[0])-1,1]=coord[2]
    nodes[int(coord[0])-1,2]=coord[3]

if nodes_size == nodes[:,0].size:
    print("Imported ", nodes[:,0].size, " nodes.")
else:
    print("ERROR: Number of imported nodes does not match nodes specified in msh file")

#### HOUNSFIELD ###

# Load ct images file
ct_img = nib.load(nifti_ct_file)

# Convert the voxel orientation to RAS
ct_img = nib.as_closest_canonical(ct_img)
print('[CT image] Voxel orientation is '+str(nib.aff2axcodes(ct_img.affine)))
print('[CT image] Voxel size', ct_img.header.get_zooms())

### Obtain inverse affine matrix
ct_invaff_mat=np.linalg.inv(ct_img.affine)

# Get data from ct to a numpy array
ct_na = ct_img.get_fdata()
print('[CT image] Dimensions', ct_img.shape)
# Plot a slice (for testing purposes)
#slice=ct_na[:,:,66]
#plt.imshow(slice.T, cmap='gray', origin='lower')
#plt.show()

### Populate Hounsfield value for each node
hf=np.zeros((nodes_size,1))
for idx, node in enumerate(nodes):
    v_pos = ct_invaff_mat.dot(np.append(node[:3],1))
    hf[idx] = ct_na[tuple(v_pos[:3].astype(int))]
    if hf[idx] < -999.99: # BOCOC images all have minHU=-1024
        print('WARNING: Hounsfield value for node '+str(idx)+' is '+str(hf[idx])+' (below -1000). Set to -999')
        hf[idx]=-999
        warnings+=1
    #print(nodes[0,:],v_pos[:3].astype(int))

print('[CT image] Maximum value:', np.max(hf))
print('[CT image] Minimum value:', np.min(hf))
nodes=np.append(nodes,hf,axis=1)

### RADIATION ###

img=0
# Load radiation dose images file
for nifti_rd_file in nifti_rd_files:

    rd_img = nib.load(nifti_rd_file)

    # Convert the voxel orientation to RAS
    rd_img = nib.as_closest_canonical(rd_img)

    print('[RD image '+str(img)+'] Voxel orientation is '+str(nib.aff2axcodes(rd_img.affine)))
    print('[RD image '+str(img)+'] Voxel size', rd_img.header.get_zooms())
    rd_aff_mat = rd_img.affine
    # IMPORTANT: When converting radiation dose dicom to nifty, slice thickness
    # is sometimes not transferred. We set it to 3 mm which is used in BOCOC
    # images but this is not applicaple everywhere
    if rd_aff_mat[2,2] == 1:
        rd_aff_mat[2,2] = 3
        
    rd_invaff_mat=np.linalg.inv(rd_aff_mat)

    # Get data from ct to a numpy array
    rd_na = rd_img.get_fdata()
    print('[RD array '+str(img)+'] Maximum value:', np.max(rd_na))
    print('[RD image '+str(img)+'] Dimensions', rd_img.shape)

    rd=np.zeros((nodes_size,1), dtype=np.double)
    for idx, node in enumerate(nodes):
        v_pos = rd_invaff_mat.dot(np.append(node[:3],1))
        rd[idx] = rd_na[tuple(v_pos[:3].astype(int))]
        # When converting RT DICOM image to nifty using dcm2niix, it was found
        # that it required scalling of the pixel values in order to match Slicer
        # data (scalling was different for some images). 
        if img == 0:
            rd[idx] /= 198.467765570619
        elif img == 1:
            rd[idx] /= 694.485345995242 # 198.795960642154 # for p08
        if rd[idx] < 0:
            print('WARNING: Radiation value for node '+str(idx)+' is '+str(rd[idx])+' (negative). Set to 0')
            rd[idx]=0
            warnings+=1
    print('[RD image '+str(img)+'] Maximum value:', np.max(rd))
    print('[RD image '+str(img)+'] Minimum value:', np.min(rd))
    nodes=np.append(nodes,rd,axis=1)
    img+=1

# Code for creating mock radiation field (not used)
ARTIFICIAL_RA=False
if ARTIFICIAL_RA:
    #### Set position of mean depending on nodes coordinates range
    x_min=np.min(nodes[:,0])
    y_min=np.min(nodes[:,1])
    z_min=np.min(nodes[:,2])
    x_max=np.max(nodes[:,0])
    y_max=np.max(nodes[:,1])
    z_max=np.max(nodes[:,2])
    
    x_var=(x_max-x_min)
    y_var=(y_max-y_min)
    z_var=(z_max-z_min)
    x_mean=x_var/2 + x_min
    y_mean=y_var/4 + y_min
    z_mean=z_var/3 + z_min
    #print(x_max,y_max,z_max)

    x_var_coeff,y_var_coeff,z_var_coeff=1,0.5,0.2
    
    rd=np.zeros((nodes_size,1))
    for index, node in enumerate(nodes):
        rd[index]=math.exp(-((node[0]-x_mean)/(x_var*x_var_coeff))**2) * math.exp(-((node[1]-y_mean)/(y_var*y_var_coeff))**2) * math.exp(-((node[2]-z_mean)/(z_var*z_var_coeff))**2)
        rd[index]*=70

### TUMOUR ####
tm=np.zeros((nodes_size,1))

if os.path.isfile(nifti_tm_file):
    # Load ct images file
    tm_img = nib.load(nifti_tm_file)

    # Convert the voxel orientation to RAS
    tm_img = nib.as_closest_canonical(tm_img)
    print('[TM image] Voxel orientation is '+str(nib.aff2axcodes(tm_img.affine)))
    print('[TM image] Voxel size', tm_img.header.get_zooms())

    ### Obtain inverse affine matrix
    tm_invaff_mat=np.linalg.inv(tm_img.affine)

    # Get data from tm to a numpy array
    tm_na = tm_img.get_fdata()
    print('[TM image] Dimensions', tm_img.shape)

    ### Populate tumour value for each node
    for idx, node in enumerate(nodes):
        v_pos = tm_invaff_mat.dot(np.append(node[:3],1))
        tm[idx] = tm_na[tuple(v_pos[:3].astype(int))]

else:
    print("WARNING: Tumour segmentation file not found! Output is all zeros")
    warnings+=1

#print('[TM image] Maximum value:', np.max(tm))
#print('[TM image] Minimum value:', np.min(tm))
nodes=np.append(nodes,tm,axis=1)

### Output to file
nodal_field_file=out_dir+name+'-nodal_field.dat'
f_out = open(nodal_field_file, 'w')
f_out.write('4\n\n')
f_out.write('v02 a01 a02 v00\n\n')
for node in nodes:
    f_out.write(' '.join(map(str,node[3:]))+'\n')
    # output node coordinates as well
    #f_out.write(' '.join(map(str,node))+'\n')
f_out.close()

print("The nodal field has been successfully output at", nodal_field_file)
if warnings: print('There have been '+str(warnings)+' WARNINGS')
