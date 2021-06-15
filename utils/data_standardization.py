# Variables
min_value = -1000.0
max_value = 3071.0

rd_min_value = 0.0
rd_max_value = 74.92199 
# 74.92199 "less than 5"; 72.96721 "less than 3"; 77.64046 "all"

# Takes an input value/array/.. and applies min/max normalization
def standardize_ct(value):
    value -= min_value
    value /= (max_value - min_value)
    return value

# Take an input between [a,b] and map it to [c,d]
def map_intervals(value, a, b, c, d):
    return c + ((d-c)/(b-a))*(value-a)

# Unapply min-max norm on rd value
def unstandardize_rd(value):
    value *= (rd_max_value - rd_min_value)
    value += rd_min_value
    return value

# Apply min-max norm on rd value
def standardize_rd(value):
    value[value < 0] = 0.0
    value -= rd_min_value
    value /= (rd_max_value - rd_min_value)
    return value
