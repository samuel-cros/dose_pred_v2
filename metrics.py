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
# - prediction
# - reference
# - mask
# - coverage value
# - prescribed value
# returns DXX percent error (|DXX_ref - DXX_pred| / prescribed_value)
def ptv_coverage_absolute_percent_error(prediction, reference, mask, coverage_value, prescribed_value):

    # Mask along the given entry
    masked_pred = prediction.flatten()[(mask.flatten()).astype(bool)]
    masked_ref = reference.flatten()[(mask.flatten()).astype(bool)]
    
    # Return DXX percent error (|DXX_ref - DXX_pred| / prescribed_value)
    return (np.abs(np.percentile(masked_ref, 100-coverage_value) - \
            np.percentile(masked_pred, 100-coverage_value)) / prescribed_value) * 100

###############################################################################
# Max dose error whole plan against prescribed (Dmax)
###############################################################################
# - prescribed_value
# - predicted_plan
# - reference_plan
def dmax_absolute_error(prescribed_value, prediction, reference):
    
    return (abs(np.max(prediction) - np.max(reference)) / prescribed_value) * 100

###############################################################################
# Homogeneity 1 (Homogeneity index HI) ~ 0
###############################################################################
# - prediction
# - mask
# returns homogeneity 1 = (D2 - D98) / D50
def homogeneity_1(prediction, mask):
    
    # Mask along the given entry
    masked_pred = prediction.flatten()[(mask.flatten()).astype(bool)]
    
    # Compute coverages
    D2 = np.percentile(masked_pred, 98)
    D98 = np.percentile(masked_pred, 2)
    D50 = np.percentile(masked_pred, 50)
    
    # Return homogeneity 1
    return (D2 - D98) / D50

###############################################################################
# Homogeneity 2 (Dose homogeneity index DHI) ~ 1
###############################################################################
# - prediction
# - mask
# returns homogeneity 2 = (D95 / D5)
def homogeneity_2(prediction, mask):
    
    # Mask along the given entry
    masked_pred = prediction.flatten()[(mask.flatten()).astype(bool)]
    
    # Average the corresponding number of voxels  
    D5 = np.percentile(masked_pred, 95)
    D95 = np.percentile(masked_pred, 5)
    
    # Return homogeneity 2
    return (D95 / D5)

# --------------------------------------------------------------------------- #
# Daily - more subject to variations, could be worse when results are good
# --------------------------------------------------------------------------- #

###############################################################################
# PTV Coverage (DXX) ~ 1
###############################################################################
# - prediction
# - mask
# - coverage_value: 1 or 2 digit number corresponding to the coverage %
# - prescribed_value
# returns coverage on XX% of the radiation sent to the tumor
def ptv_coverage(prediction, mask, coverage_value, prescribed_value):
    
    # Mask along the given entry
    masked_pred = prediction.flatten()[(mask.flatten()).astype(bool)]
    
    # Compute coverages
    return np.percentile(masked_pred, 100-coverage_value) / prescribed_value

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

###############################################################################
# Dose score
###############################################################################
# - prediction
# - reference
# - possible dose mask (OpenKBP), body channel (CHUM)
# returns the dose score (~MAE but normalized over number of non-zero voxels in mask)
def dose_score(prediction, reference, mask):
    
    # From OpenKBP
    # dose_score_vec[idx] = np.sum(np.abs(reference_dose - new_dose)) / np.sum(self.possible_dose_mask)
    
    # From my understanding of OpenKBP evaluation codes:
    # - reference and prediction are taken as full volume
    # - mask shows voxels with possible non-zero values for a dose (which
    #   can be approximated by the body channel in the CHUM data)
    # As a result, we don't compute a true average since reference and 
    # prediction don't have the same number of entries as the denominator
    # formed with mask.
    return np.sum(np.abs(reference - prediction)) / np.sum(np.sum(mask))

###############################################################################
# DVH score
###############################################################################
# - prediction L*W*H
# - reference L*W*H
# - oar_channels L*W*H*channels
# - tv_channels L*W*H*channels
# - voxel size (OpenKBP), 1 (CHUM)
# returns the dvh score
def dvh_score(prediction, reference, oar_channels, tv_channels, voxel_size):
    
    # Setup dvh_lists
    DVH_list_pred = []
    DVH_list_ref = []
    
    # For each OAR
    for oar_channel_number in range(oar_channels.shape[-1]):
        
        # Only non empty channels
        if oar_channels[:, :, :, oar_channel_number].any():
        
            # Setup
            oar_mask = (oar_channels[:, :, :, oar_channel_number].flatten()).astype(bool)
            oar_prediction = prediction.flatten()[oar_mask]
            oar_reference = reference.flatten()[oar_mask]
            oar_size = len(oar_prediction)
            
            # Percentile on fractional volume in 0.1cc
            voxels_in_tenth_of_cc = np.maximum(1, np.round(100/np.prod(voxel_size)))
            fractional_volume_to_evaluate = 100 - np.minimum(0.99, voxels_in_tenth_of_cc/oar_size) * 100
            DVH_list_pred.append(np.percentile(oar_prediction, fractional_volume_to_evaluate))
            DVH_list_ref.append(np.percentile(oar_reference, fractional_volume_to_evaluate))
            
            # Mean dose
            DVH_list_pred.append(oar_prediction.mean())
            DVH_list_ref.append(oar_reference.mean())
        
    # For each TV
    for tv_channel_number in range(tv_channels.shape[-1]):
        
        # Only non empty channels
        if tv_channels[:, :, :, tv_channel_number].any():
        
            # Setup
            tv_mask = (tv_channels[:, :, :, tv_channel_number].flatten()).astype(bool)
            tv_prediction = prediction.flatten()[tv_mask]
            tv_reference = reference.flatten()[tv_mask]
            
            # D99, 1st percentile
            DVH_list_pred.append(np.percentile(tv_prediction, 1))
            DVH_list_ref.append(np.percentile(tv_reference, 1))
            
            # D95, 5th percentile
            DVH_list_pred.append(np.percentile(tv_prediction, 5))
            DVH_list_ref.append(np.percentile(tv_reference, 5))
            
            # D1, 99th percentile
            DVH_list_pred.append(np.percentile(tv_prediction, 99))
            DVH_list_ref.append(np.percentile(tv_reference, 99))
        
        
    # From OpenKBP
    # dvh_score = np.nanmean(np.abs(self.reference_dose_metric_df - self.new_dose_metric_df).values)
    return np.mean(np.abs(np.array(DVH_list_ref) - np.array(DVH_list_pred)))
    
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


    
    
