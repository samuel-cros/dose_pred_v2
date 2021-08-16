## Remark #TODO
# Things could be cleaner by putting all of this in a class and initialize
# the min and max values with the corresponding dataset only once in __init__.

# CHUM Variables
ct_min_value_CHUM = -1000.0
ct_max_value_CHUM = 3071.0

rd_min_value_CHUM = 0.0
rd_max_value_CHUM = 74.92199 
# 74.92199 "less than 5"; 72.96721 "less than 3"; 77.64046 "all"

# OpenKBP Variables
ct_min_value_OpenKBP = None #TODO
ct_max_value_OpenKBP= None #TODO

rd_min_value_OpenKBP = 0.0
rd_max_value_OpenKBP= 79.998

# Takes an input value/array/.. and applies min/max normalization
def standardize_ct(value, dataset):
    
    if dataset == 'CHUM':
        ct_min_value, ct_max_value = ct_min_value_CHUM, ct_max_value_CHUM
    elif dataset == 'OpenKBP':
        ct_min_value, ct_max_value = ct_min_value_OpenKBP, ct_max_value_OpenKBP
    else:
        raise ValueError("Unknown dataset. Handled datasets are CHUM and OpenKBP")
        
    value -= ct_min_value
    value /= (ct_max_value - ct_min_value)
    return value

# Take an input between [a,b] and map it to [c,d]
def map_intervals(value, a, b, c, d):
   
    return c + ((d-c)/(b-a))*(value-a)

# Unapply min-max norm on rd value
def unstandardize_rd(value, dataset):
    
    if dataset == 'CHUM':
        rd_min_value, rd_max_value = rd_min_value_CHUM, rd_max_value_CHUM
    elif dataset == 'OpenKBP':
        rd_min_value, rd_max_value = rd_min_value_OpenKBP, rd_max_value_OpenKBP
    else:
        raise ValueError("Unknown dataset. Handled datasets are CHUM and OpenKBP")
    
    value[value < 0] = 0.0
    value *= (rd_max_value - rd_min_value)
    value += rd_min_value
    return value

# Apply min-max norm on rd value
def standardize_rd(value, dataset):
    
    if dataset == 'CHUM':
        rd_min_value, rd_max_value = rd_min_value_CHUM, rd_max_value_CHUM
    elif dataset == 'OpenKBP':
        rd_min_value, rd_max_value = rd_min_value_OpenKBP, rd_max_value_OpenKBP
    else:
        raise ValueError("Unknown dataset. Handled datasets are CHUM and OpenKBP")
    
    value[value < 0] = 0.0
    value -= rd_min_value
    value /= (rd_max_value - rd_min_value)
    return value

# Unapply min-max norm on rd tensor
def unstandardize_rd_tensor(value, dataset):
    
    if dataset == 'CHUM':
        rd_min_value, rd_max_value = rd_min_value_CHUM, rd_max_value_CHUM
    elif dataset == 'OpenKBP':
        rd_min_value, rd_max_value = rd_min_value_OpenKBP, rd_max_value_OpenKBP
    else:
        raise ValueError("Unknown dataset. Handled datasets are CHUM and OpenKBP")
    
    value *= (rd_max_value - rd_min_value)
    value += rd_min_value
    return value

# Apply min-max norm on rd tensor
def standardize_rd_tensor(tensor, dataset):
    
    if dataset == 'CHUM':
        rd_min_value, rd_max_value = rd_min_value_CHUM, rd_max_value_CHUM
    elif dataset == 'OpenKBP':
        rd_min_value, rd_max_value = rd_min_value_OpenKBP, rd_max_value_OpenKBP
    else:
        raise ValueError("Unknown dataset. Handled datasets are CHUM and OpenKBP")
    
    tensor -= rd_min_value
    tensor /= (rd_max_value - rd_min_value)
    return tensor
