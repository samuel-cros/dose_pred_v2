#
import numpy as np
import math

from keras.losses import mean_squared_error as mse
from keras.losses import mean_absolute_error as mae

### Evaluation metrics
# Daily = for day-to-day comparison of results between networks
# Visual = to be visualized with box plots and others

# --------------------------------------------------------------------------- #
# Daily
# Visually, 1 Line per architecture, 1 Column per metric, Mean + STD, t-test p-values
# --------------------------------------------------------------------------- #

###############################################################################
# PTV Coverage (DXX) - Absolute percent error (between predicted and ref normalized with 
# the prescribed dose) ~ 0 to 5%
###############################################################################
def ptv_coverage_absolute_percent_error(plan, ref_plan, tumor_segmentation_bin, coverage_value, prescribed_dose):
    
    ## Get non-zero indices from tumor_seg
    non_zero = tumor_segmentation_bin.nonzero()
    
    ## For each voxel i in the plan where the tumor is
    # Compute abs(value_i - ref_i) / prescribed_i
    per_voxel_coverage = (abs(plan[non_zero] - ref_plan[non_zero]) / prescribed_dose) * 100
    # Descending sort
    per_voxel_coverage[::-1].sort()
        
    # Compute how many voxels to take
    coverage_index = \
        math.floor(len(per_voxel_coverage) * coverage_value / 100)
    # Average the corresponding number of voxels    
    average_coverage = \
        np.sum(per_voxel_coverage[:coverage_index]) / coverage_index
        
    # Return average coverage on coverage_value% of the tumor
    return average_coverage

###############################################################################
# Max dose error whole plan against prescribed (Dmax)
###############################################################################
# - prescribed_dose
# - predicted_plan
# - reference_plan
def dmax_absolute_error(prescribed_dose, predicted_plan, reference_plan):
    
    return (abs(np.max(predicted_plan) - np.max(reference_plan)) / prescribed_dose) * 100

###############################################################################
# Homogeneity 1 (Homogeneity index HI) ~ 0
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
    index_D2 = math.floor(len(per_voxel_coverage) * 2 / 100)
    # 98%
    index_D98 = math.floor(len(per_voxel_coverage) * 98 / 100)
    # 50%
    index_D50 = math.floor(len(per_voxel_coverage) * 50 / 100)
    
    # Average the corresponding number of voxels  
    D2 = np.sum(per_voxel_coverage[:index_D2]) / index_D2
    D98 = np.sum(per_voxel_coverage[:index_D98]) / index_D98
    D50 = np.sum(per_voxel_coverage[:index_D50]) / index_D50
    
    # Return homogeneity 1
    return (D2 - D98) / D50

###############################################################################
# Homogeneity 2 (Dose homogeneity index DHI) ~ 1
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
    index_D5 = math.floor(len(per_voxel_coverage) * 5 / 100)
    # 95%
    index_D95 = math.floor(len(per_voxel_coverage) * 95 / 100)
    
    # Average the corresponding number of voxels  
    D5 = np.sum(per_voxel_coverage[:index_D5]) / index_D5
    D95 = np.sum(per_voxel_coverage[:index_D95]) / index_D95
    
    # Return homogeneity 2
    return (D95 / D5)

# --------------------------------------------------------------------------- #
# Daily - more subject to variations, could be worse when results are good
# --------------------------------------------------------------------------- #

###############################################################################
# PTV Coverage (DXX) ~ 1
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
        math.floor(len(per_voxel_coverage) * coverage_value / 100)
    # Average the corresponding number of voxels    
    average_coverage = \
        np.sum(per_voxel_coverage[:coverage_index]) / coverage_index
        
    # Return average coverage on coverage_value% of the tumor
    return average_coverage

###############################################################################
# Max dose error against prescribed (whole plan) (Dmax)
###############################################################################
# - prescribed_dose
# - predicted_plan
# returns 1 - max(predicted_plan) / prescribed_dose
def max_dose_error_vs_prescribed(prescribed_dose, predicted_plan):
    
    return np.max(predicted_plan) / prescribed_dose

###############################################################################
# Conformity index CI ~ 1
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
# van't Riet conformation number ~ <=1, good if 0.60 or higher
###############################################################################
# - plan: 3D radiation plan
# - tumor_segmentation_gy: 3D volume where 0->void, Gy->tumor
# - prescribed_dose: prescribed dose for the target volume
# returns van't Riet conformation number
# (TV_100p_iso * TV_100p_iso) / (TV * V_100p_iso))
def vant_riet(plan, tumor_segmentation_gy, prescribed_dose):
    
    ## Get non-zero indices from tumor_seg
    non_zero = tumor_segmentation_gy.nonzero()
    
    # Compute number of elements of the tumor receiving a dose equal or 
    # greater than the prescribed dose
    TV_100p_iso = len(np.where(plan[non_zero] >= prescribed_dose)[0])
    
    # Compute number of elements of the tumor
    TV = len(tumor_segmentation_gy.nonzero()[0])
    
    # Compute number of elements in the plan receiving a dose equal or 
    # greater than the prescribed dose
    V_100p_iso = len(np.where(plan >= prescribed_dose)[0])
    
    # Return van't Riet conformation number
    return (TV_100p_iso * TV_100p_iso) / (TV * max(1, V_100p_iso))

###############################################################################
# Dose spillage (R50) ~ 
# - CHUM - should explode
# - OpenKBP - should be better
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

# --------------------------------------------------------------------------- #
# Visual - Excel > Insert > Chart > Statistical > Box and Whiskers
# --------------------------------------------------------------------------- #

###############################################################################
# Max dose error per structure against prescribed (Dmax_pe per structure)
###############################################################################
# - prescribed_dose
# - predicted_plan
# - reference_plan
# - structure_segmentation
# returns 1 - max(predicted_plan) / prescribed_dose
def dmax_structure_absolute_error(prescribed_dose, predicted_plan, reference_plan, structure_segmentation):
    
    # Compute non-zero values
    non_zero = structure_segmentation.nonzero()
    
    return (abs(np.max(predicted_plan[non_zero]) - np.max(reference_plan[non_zero])) / prescribed_dose) * 100

###############################################################################
# Mean dose error per structure against prescribed (Dmean_pe per structure)
###############################################################################
# - prescribed_dose
# - predicted_plan
# - reference_plan
# - structure_segmentation
# returns 1 - max(predicted_plan) / prescribed_dose
def dmean_structure_absolute_error(prescribed_dose, predicted_plan, reference_plan, structure_segmentation):
    
    # Compute non-zero values
    non_zero = structure_segmentation.nonzero()
    
    return (abs(np.mean(predicted_plan[non_zero]) - np.mean(reference_plan[non_zero])) / prescribed_dose) * 100

# --------------------------------------------------------------------------- #
# Visual - TODO
# --------------------------------------------------------------------------- #

###############################################################################
# Dose volume histogram
###############################################################################
# - plan: 3D radiation plan
# - prescribed_dose: prescribed dose for the target volume
# returns the dose volume histogram TODO

# --------------------------------------------------------------------------- #
# Bonus
# --------------------------------------------------------------------------- #

###############################################################################
# MAE on structure (Dose Score when computed on body)
###############################################################################
# - reference_plan
# - predicted_plan
# - structure_segmentation
# returns mse between the reference plan and the prediction for a given structure
def mae_on_structure(reference_plan, predicted_plan, structure_segmentation):
    
    ## Get non-zero indices from the structure segmentation
    non_zero = structure_segmentation.nonzero()
    
    ## Compute MSE between the reference and the prediction
    return mae(reference_plan[non_zero], predicted_plan[non_zero])

###############################################################################
# MSE on structure
###############################################################################
# - reference_plan
# - predicted_plan
# - structure_segmentation
# returns mse between the reference plan and the prediction for a given structure
def mse_on_structure(reference_plan, predicted_plan, structure_segmentation):
    
    ## Get non-zero indices from the structure segmentation
    non_zero = structure_segmentation.nonzero()
    
    ## Compute MSE between the reference and the prediction
    return mse(reference_plan[non_zero], predicted_plan[non_zero])

###############################################################################
# Max dose error against reference (whole plan)
###############################################################################
# - reference_plan
# - predicted_plan
# returns 1 - max(predicted_plan) / max(reference_plan)
def max_dose_error_vs_reference(reference_plan, predicted_plan):
    
    return np.max(predicted_plan) / np.max(reference_plan)

###############################################################################
# Mean dose error against reference (whole plan)
###############################################################################
# - reference_plan
# - predicted_plan
# returns 1 - mean(predicted_plan) / mean(reference_plan)
def mean_dose_error_vs_reference(reference_plan, predicted_plan):
    
    return np.mean(predicted_plan) / np.mean(reference_plan)

###############################################################################
# PTV Coverage (DXX) - Percent error (between predicted and ref normalized with 
# the prescribed dose ~ -5 to 5%
###############################################################################
# - plan: 3D radiation plan
# - ref_plan
# - tumor_segmentation_gy: 3D volume where 0->void, Gy->tumor
# - coverage_value: 1 or 2 digit number corresponding to the coverage %
# returns average coverage on XX% of the radiation sent to the tumor
def ptv_coverage_percent_error(plan, ref_plan, tumor_segmentation_gy, coverage_value, prescribed_dose):
    
    ## Get non-zero indices from tumor_seg
    non_zero = tumor_segmentation_gy.nonzero()
    
    ## For each voxel i in the plan where the tumor is
    # Compute (value_i - ref_i) / prescribed_i
    per_voxel_coverage = ((plan[non_zero] - ref_plan[non_zero]) / tumor_segmentation_gy[non_zero]) * 100
    # Descending sort
    per_voxel_coverage[::-1].sort()
        
    # Compute how many voxels to take
    coverage_index = \
        math.floor(len(per_voxel_coverage) * coverage_value / 100)
    # Average the corresponding number of voxels    
    average_coverage = \
        np.sum(per_voxel_coverage[:coverage_index]) / coverage_index
        
    # Return average coverage on coverage_value% of the tumor
    return average_coverage

# 'error' = [(D_true - D_predict) / D_prescribed] * 100, Dmax, Dmean

# --------------------------------------------------------------------------- #
# Old
# --------------------------------------------------------------------------- #

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


    
    
