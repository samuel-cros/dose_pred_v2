#
import numpy as np
import math

###############################################################################
# PTV Coverage (DXX)
###############################################################################
# - plan: 3D radiation plan
# - tumor_segmentation_gy: 3D volume where 0->void, Gy->tumor
# - coverage_value: 1 or 2 digit number corresponding to the coverage %
# returns average coverage on XX% of the radiation sent to the tumor
def ptv_coverage(plan, tumor_segmentation_gy, coverage_value):
    
    ## Get non-zero indices from tumor_seg
    non_zero = tumor_segmentation_gy.nonzero()
    
    ## For each voxel i in the plan where the tumor is
    # Compute value_i / prescribed_i
    per_voxel_coverage = plan[non_zero] / tumor_segmentation_gy[non_zero]
    # Descending sort
    per_voxel_coverage[::-1].sort()
        
    # Compute how many voxels to take
    coverage_index = \
        math.floor(len(per_voxel_coverage) * coverage_value / 100) - 1
    # Average the corresponding number of voxels    
    average_coverage = \
        np.sum(per_voxel_coverage[:coverage_index]) / (coverage_index + 1)
        
    # Return average coverage on coverage_value% of the tumor
    return average_coverage

###############################################################################
# Structure max, mean dose (Dmax, Dmean)
###############################################################################
# - plan: 3D radiation plan
# - structure_segmentation_bin: 3D volume where 0->void, 1->tumor
# returns plan max, mean dose for given structure
def structure_m_dose(plan, structure_segmentation_bin):
    
    ## Get non-zero indices from tumor_seg
    non_zero = structure_segmentation_bin.nonzero()
    
    # Return the max, mean dose
    return np.max(plan[non_zero]), np.mean(plan[non_zero])

###############################################################################
# Homogeneity 1 (Homogeneity)
###############################################################################
# - plan: 3D radiation plan
# - tumor_segmentation_gy: 3D volume where 0->void, Gy->tumor
# returns homogeneity 1 = (D2 - D98) / D50
def homogeneity_1(plan, tumor_segmentation_gy):
    
    ## Get non-zero indices from tumor_seg
    non_zero = tumor_segmentation_gy.nonzero()
    
    ## For each voxel i in the plan where the tumor is
    # Compute value_i / prescribed_i
    per_voxel_coverage = plan[non_zero] / tumor_segmentation_gy[non_zero]
    # Descending sort
    per_voxel_coverage[::-1].sort()
    
    # Compute how many voxels to take
    # 2%
    index_D2 = math.floor(len(per_voxel_coverage) * 2 / 100) - 1
    # 98%
    index_D98 = math.floor(len(per_voxel_coverage) * 98 / 100) - 1
    # 50%
    index_D50 = math.floor(len(per_voxel_coverage) * 50 / 100) - 1
    
    # Average the corresponding number of voxels  
    D2 = np.sum(per_voxel_coverage[:index_D2]) / (index_D2 + 1)
    D98 = np.sum(per_voxel_coverage[:index_D98]) / (index_D98 + 1)
    D50 = np.sum(per_voxel_coverage[:index_D50]) / (index_D50 + 1)
    
    # Return homogeneity 1
    return (D2 - D98) / D50

###############################################################################
# Homogeneity 2 (Dose homogeneity index DHI)
###############################################################################
# - plan: 3D radiation plan
# - tumor_segmentation_gy: 3D volume where 0->void, Gy->tumor
# returns homogeneity 2 = (D95 / D5)
def homogeneity_2(plan, tumor_segmentation_gy):
    
    ## Get non-zero indices from tumor_seg
    non_zero = tumor_segmentation_gy.nonzero()
    
    ## For each voxel i in the plan where the tumor is
    # Compute value_i / prescribed_i
    per_voxel_coverage = plan[non_zero] / tumor_segmentation_gy[non_zero]
    # Descending sort
    per_voxel_coverage[::-1].sort()
    
    # Compute how many voxels to take
    # 5%
    index_D5 = math.floor(len(per_voxel_coverage) * 5 / 100) - 1
    # 95%
    index_D95 = math.floor(len(per_voxel_coverage) * 98 / 100) - 1
    
    # Average the corresponding number of voxels  
    D5 = np.sum(per_voxel_coverage[:index_D5]) / (index_D5 + 1)
    D95 = np.sum(per_voxel_coverage[:index_D95]) / (index_D95 + 1)
    
    # Return homogeneity 2
    return (D95 / D5)

###############################################################################
# Homogeneity 3 (Homogeneity index HI)
###############################################################################
# - plan: 3D radiation plan
# - tumor_segmentation_bin: 3D volume where 0->void, 1->tumor
# - prescribed_dose: prescribed dose for the target volume
# returns homogeneity 3 = (Dmax / prescribed_dose)
def homogeneity_3(plan, tumor_segmentation_bin, prescribed_dose):
    
    ## Get non-zero indices from tumor_seg
    non_zero = tumor_segmentation_bin.nonzero()
    
    ## Get max value from plan on the tumor
    Dmax = np.max(plan[non_zero])
    
    # Return homogeneity 3
    return (Dmax / prescribed_dose)
    
###############################################################################
# van't Riet conformation number
###############################################################################
# - plan: 3D radiation plan
# - tumor_segmentation_gy: 3D volume where 0->void, Gy->tumor
# - prescribed_dose: prescribed dose for the target volume
# returns van't Riet conformation number
# (TV_100p_iso * TV_100p_iso) / (TV * V_100p_iso))
def vant_riet(plan, tumor_segmentation_gy, prescribed_dose):
    
    # Compute number of elements of the tumor receiving a dose equal or 
    # greater than the prescribed dose
    TV_100p_iso = len(np.where(tumor_segmentation_gy >= prescribed_dose)[0])
    
    # Compute number of elements of the tumor
    TV = len(tumor_segmentation_gy.nonzero()[0])
    
    # Compute number of elements in the plan receiving a dose equal or 
    # greater than the prescribed dose
    V_100p_iso = len(np.where(plan >= prescribed_dose)[0])
    
    # Return van't Riet conformation number
    return (TV_100p_iso * TV_100p_iso) / (TV * max(1, V_100p_iso))
    
    
###############################################################################
# Dose spillage
###############################################################################
# - plan: 3D radiation plan
# - tumor_segmentation_bin: 3D volume where 0->void, 1->tumor
# - prescribed_dose: prescribed dose for the target volume
# returns the dose spillage (V_50p_iso / TV)
def dose_spillage(plan, tumor_segmentation_bin, prescribed_dose):
    
    # Compute number of elements in the plan receiving a dose equal or
    # greater than 50% of the prescribed dose
    V_50p_iso = \
        len(np.where(plan >= 0.5 * prescribed_dose)[0])
    
    # Compute number of elements of the tumor
    TV = len(tumor_segmentation_bin.nonzero()[0])
    
    # Return dose spillage (V_50p_iso / TV)
    return (V_50p_iso / TV)


###############################################################################
# Conformity index CI
###############################################################################
# - plan: 3D radiation plan
# - tumor_segmentation_gy: 3D volume where 0->void, Gy->tumor
# - prescribed_dose: prescribed dose for the target volume
# returns Conformity index TV_PIV / TV
def conformity_index(plan, tumor_segmentation_gy, prescribed_dose):
    
    ## Get non-zero indices from tumor_seg
    non_zero = tumor_segmentation_gy.nonzero()
    
    # Compute number of elements of the tumor volume covered by the prescribed
    # dose in the plan
    TV_PIV = len(np.where(plan[non_zero] >= prescribed_dose)[0])
    
    # Compute number of elements of the tumor
    TV = len(non_zero[0])
    
    # Return conformity index (TV_PIV / TV)
    return (TV_PIV / TV)

###############################################################################
# Paddick's conformity index PCI
###############################################################################
# - plan: 3D radiation plan
# - tumor_segmentation_gy: 3D volume where 0->void, Gy->tumor
# - prescribed_dose: prescribed dose for the target volume
# returns paddick's conformity index (TV_PIV * TV_PIV) / (TV * PIV)
def paddick_conformity_index(plan, tumor_segmentation_gy, prescribed_dose):
    
    ## Get non-zero indices from tumor_seg
    non_zero = tumor_segmentation_gy.nonzero()
    
    # Compute number of elements of the tumor volume covered by the prescribed
    # dose in the plan
    TV_PIV = len(np.where(plan[non_zero] >= prescribed_dose)[0])
    
    # Compute number of elements of the tumor
    TV = len(non_zero[0])
    
    # Compute number of elements of the tumor that should receive the
    # prescribed dose
    PIV = len(np.where(tumor_segmentation_gy == prescribed_dose)[0])
    
    # Return Paddick's conformity index (TV_PIV * TV_PIV) / (TV * PIV)
    return (TV_PIV * TV_PIV) / (TV * PIV)
    
###############################################################################
# Gradient index GI
###############################################################################
# - plan: 3D radiation plan
# - prescribed_dose: prescribed dose for the target volume
# returns the dose spillage (V_50p_iso / V_100p_iso)
def gradient_index(plan, prescribed_dose):
    
    # Compute number of elements in the plan receiving a dose equal or
    # greater than 50% of the prescribed dose
    V_50p_iso = \
        len(np.where(plan >= 0.5 * prescribed_dose)[0])
    
    # Compute number of elements in the plan receiving a dose equal or
    # greater than 100% of the prescribed dose
    V_100p_iso = \
        len(np.where(plan >= prescribed_dose)[0])
        
    # Return Gradient index (V_50p_iso / V_100p_iso)
    return (V_50p_iso / max(1, V_100p_iso))

###############################################################################
# Dose volume histogram
###############################################################################
# - plan: 3D radiation plan
# - prescribed_dose: prescribed dose for the target volume
# returns the dose volume histogram TODO
    
    
