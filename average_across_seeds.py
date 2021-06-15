# 
import csv
import sys
import ast

# Open the final csv
average_across_seeds_csv = open('res_v2/average_across_seeds_csv.csv', 'w', newline='')

fields = ['ID', 'PTV coverage', 'Structure max dose', 'Structure mean dose', 
            'Homogeneity', "van't Riet conformation number", 'Dose spillage', 
            'Conformity index', "Paddick's conformity index", 'Gradient index']
average_across_seeds_writer = csv.DictWriter(average_across_seeds_csv, fieldnames=fields)
average_across_seeds_writer.writeheader()
new_average_row = {}
new_average_row['PTV coverage'] = {}
new_average_row['Structure max dose'] = {}
new_average_row['Structure mean dose'] = {}
new_average_row['Homogeneity'] = {}
count_struct_m_dose = {}

# Setup
folder_name = 'baseline' # resp 'attention'
use_att = '' # resp '_att'
paths = []

# Collect paths to each csv
for i in range(1, 4):
    paths.append('res_v2/' + folder_name + '_seed' + str(i) + \
                    '/dr_0.0_o_adam_lr_0.001_e_200_loss_mse' + \
                    use_att + '/results_validation/predicted_volumes_generate_predictions/metrics_pred.csv')

count_seed = 0
# For each csv
for path in paths:

    # Open
    metrics_pred_csv = open(path, 'r', newline='')
    metrics_pred_reader = csv.DictReader(metrics_pred_csv)

    # Read the average line
    for row in metrics_pred_reader:
        average_row = row

    # Convert back into dict of dicts
    to_convert = [1, 2, 3, 4]
    for id in to_convert:
        average_row[fields[id]] = ast.literal_eval(average_row[fields[id]])
    average_row['ID'] = 'Average for seed ' + str(count_seed)

    '''
    print(isinstance(average_row['PTV coverage'], dict))
    print(type(average_row['PTV coverage']))
    sys.exit()
    '''

    # Write it
    average_across_seeds_writer.writerow(average_row)
    
    # Sum it
    for field in average_row:
        # Pass if it's ID
        if field == "ID":
            pass
        # Special treatment for structure max and mean dose since there is a chance
        # to be lacking the structure
        elif field == "Structure max dose" or field == "Structure mean dose":
            for subfield in average_row[field]:
                try:
                    new_average_row[field][subfield] += average_row[field][subfield]
                    count_struct_m_dose[subfield] += 1
                except KeyError:
                    new_average_row[field][subfield] = average_row[field][subfield]
                    count_struct_m_dose[subfield] = 1
        elif isinstance(average_row[field], dict):
            for subfield in average_row[field]:
                try:
                    new_average_row[field][subfield] += average_row[field][subfield]
                except KeyError:
                    new_average_row[field][subfield] = average_row[field][subfield]
        else:
            try:
                new_average_row[field] += float(average_row[field])
            except KeyError:
                new_average_row[field] = float(average_row[field])

    # Incr
    count_seed += 1

# Average across seeds
for field in new_average_row:
    # Special treatment for structure max and mean dose since there is a chance
    # to be lacking the structure
    if field == "Structure max dose" or field == "Structure mean dose":
        for subfield in new_average_row[field]:
            new_average_row[field][subfield] /= count_struct_m_dose[subfield]
    elif isinstance(new_average_row[field], dict):
        for subfield in new_average_row[field]:
            new_average_row[field][subfield] /= count_seed
    else:
        new_average_row[field] /= count_seed


new_average_row['ID'] = folder_name + ' average across seeds'
average_across_seeds_writer.writerow(new_average_row)

###############################################################################
###############################################################################

average_across_seeds_writer.writerow({})

new_average_row = {}
new_average_row['PTV coverage'] = {}
new_average_row['Structure max dose'] = {}
new_average_row['Structure mean dose'] = {}
new_average_row['Homogeneity'] = {}
count_struct_m_dose = {}

# Setup
folder_name = 'attention' # resp 'attention'
use_att = '_att' # resp '_att'
paths = []

# Collect paths to each csv
for i in range(1, 4):
    paths.append('res_v2/' + folder_name + '_seed' + str(i) + \
                    '/dr_0.0_o_adam_lr_0.001_e_200_loss_mse' + \
                    use_att + '/results_validation/predicted_volumes_generate_predictions/metrics_pred.csv')

count_seed = 0
# For each csv
for path in paths:

    # Open
    metrics_pred_csv = open(path, 'r', newline='')
    metrics_pred_reader = csv.DictReader(metrics_pred_csv)

    # Read the average line
    for row in metrics_pred_reader:
        average_row = row

    # Convert back into dict of dicts
    to_convert = [1, 2, 3, 4]
    for id in to_convert:
        average_row[fields[id]] = ast.literal_eval(average_row[fields[id]])
    average_row['ID'] = 'Average for seed ' + str(count_seed)

    '''
    print(isinstance(average_row['PTV coverage'], dict))
    print(type(average_row['PTV coverage']))
    sys.exit()
    '''

    # Write it
    average_across_seeds_writer.writerow(average_row)
    
    # Sum it
    for field in average_row:
        # Pass if it's ID
        if field == "ID":
            pass
        # Special treatment for structure max and mean dose since there is a chance
        # to be lacking the structure
        elif field == "Structure max dose" or field == "Structure mean dose":
            for subfield in average_row[field]:
                try:
                    new_average_row[field][subfield] += average_row[field][subfield]
                    count_struct_m_dose[subfield] += 1
                except KeyError:
                    new_average_row[field][subfield] = average_row[field][subfield]
                    count_struct_m_dose[subfield] = 1
        elif isinstance(average_row[field], dict):
            for subfield in average_row[field]:
                try:
                    new_average_row[field][subfield] += average_row[field][subfield]
                except KeyError:
                    new_average_row[field][subfield] = average_row[field][subfield]
        else:
            try:
                new_average_row[field] += float(average_row[field])
            except KeyError:
                new_average_row[field] = float(average_row[field])

    # Incr
    count_seed += 1

# Average across seeds
for field in new_average_row:
    # Special treatment for structure max and mean dose since there is a chance
    # to be lacking the structure
    if field == "Structure max dose" or field == "Structure mean dose":
        for subfield in new_average_row[field]:
            new_average_row[field][subfield] /= count_struct_m_dose[subfield]
    elif isinstance(new_average_row[field], dict):
        for subfield in new_average_row[field]:
            new_average_row[field][subfield] /= count_seed
    else:
        new_average_row[field] /= count_seed


new_average_row['ID'] = folder_name + ' average across seeds'
average_across_seeds_writer.writerow(new_average_row)