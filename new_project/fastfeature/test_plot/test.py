import pyupset as pyu
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_truepositives(ground_truth, detected):
    true_positives = []
    for x in range(detected.shape[0]):
        for y in range(detected.shape[1]):
            if ground_truth[x,y] == True and detected[x,y] == True:
                true_positives.append(str(x) + ',' + str(y))

    df = pd.DataFrame.from_dict({'TP': true_positives})
    return df

def generate_truepositives_2_file(ground_truth, detected, name):
    true_positives = []
    for x in range(detected.shape[0]):
        for y in range(detected.shape[1]):
            if ground_truth[x,y] == True and detected[x,y] == True:
                true_positives.append(str(x) + ',' + str(y))

    f = open("/tmp/" + name, "w+")
    for t in true_positives:
        f.write(t + '\n')
    f.close()
    return set(true_positives)

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3



#create dict
true_positives_dict = {}


#Blackoak
#load detected matrices

ed2 = np.load('/home/felix/phd/overlap/Address/ED2/ed2_results.npy')

nadeef = np.load('/home/felix/phd/overlap/Address/NADEEF/matrix_detected.npy')
katara = np.load('/home/felix/phd/overlap/Address/KATARA/matrix_detected.npy')

'''
gaussian = np.load('/home/felix/phd/overlap/Address/dBoost/Gaussian/dboost_gausian.npy')
histogram = np.load('/home/felix/phd/overlap/Address/dBoost/Histogram/dboost_hist.npy')
mixture = np.load('/home/felix/phd/overlap/Address/dBoost/Mixture/dboost_mixture.npy')
'''

outlier = np.load('/home/felix/phd/overlap/Address/dBoost/Gaussian_Sample/dboost_gausian_BlackOakUppercase_gausian0.88_stat_0.5.npy')

active_clean = np.load('/home/felix/phd/overlap/Address/ActiveClean/ed2_results.npy')
boost_clean = np.load('/home/felix/phd/overlap/Address/BoostClean/ed2_results.npy')

metadata_driven_error_detection = np.load('/home/felix/phd/overlap/Address/Metadata-Driven Error Detection/larysa_samples_150_run_0.npy')

#load data
ground_truth = np.load('/home/felix/phd/overlap/Address/GroundTruth/address_ground_truth.npy')


'''
nadeef = np.load('/home/felix/phd/overlap/Hospital/NADEEF/matrix_detected_hospital_nadeef.npy')
katara = np.load('/home/felix/phd/overlap/Hospital/KATARA/matrix_detected.npy')
histogram = np.load('/home/felix/phd/overlap/Hospital/dBoost/Histogram/dboost_hist.npy')

ground_truth = np.load('/home/felix/phd/overlap/Hospital/Groundtruth/ground_truth.npy')

print(ground_truth.shape)
'''

tp = {}
tp['ED2'] = generate_truepositives_2_file(ground_truth, ed2, 'ED2')

tp['ActiveClean'] = generate_truepositives_2_file(ground_truth, active_clean, 'ActiveClean')
tp['BoostClean'] = generate_truepositives_2_file(ground_truth, boost_clean, 'BoostClean')

tp['MDED'] = generate_truepositives_2_file(ground_truth, metadata_driven_error_detection, 'MDED')

tp['Outlier Detection'] = generate_truepositives_2_file(ground_truth, outlier, 'Outlier Detection')

tp['KATARA'] = generate_truepositives_2_file(ground_truth, katara, 'KATARA')

tp['NADEEF'] = generate_truepositives_2_file(ground_truth, nadeef, 'NADEEF')


#tp['gaussian'] = generate_truepositives_2_file(ground_truth, gaussian, 'gaussian')
#tp['hist'] = generate_truepositives_2_file(ground_truth, histogram, 'histogram')
#tp['mix'] = generate_truepositives_2_file(ground_truth, mixture, 'mixture')





c_str = ''
a_str = ''
for a_key, a_value in tp.items():
    a_str += a_key + ","
    for b_key, b_value in tp.items():
        print(a_key + " tp of " + b_key + " tp: " + str(len(a_value.intersection(b_value)) / float(len(b_value))))
        c_str += "{:0.2f}".format(round(len(a_value.intersection(b_value)) / float(len(b_value)),2)) + ","
        #c_str += str(len(a_value.intersection(b_value))) + ","
    c_str += '\n'
print(a_str)
print(c_str)




'''
#store true positives in dict
true_positives_dict['ED2'] = generate_truepositives(ground_truth, ed2)

true_positives_dict['NADEEF'] = generate_truepositives(ground_truth, nadeef)

true_positives_dict['KATARA'] = generate_truepositives(ground_truth, katara)

true_positives_dict['Gaussian'] = generate_truepositives(ground_truth, gaussian)
true_positives_dict['Histogram'] = generate_truepositives(ground_truth, histogram)
true_positives_dict['Mixture'] = generate_truepositives(ground_truth, mixture)

true_positives_dict['ActiveClean'] = generate_truepositives(ground_truth, active_clean)
true_positives_dict['BoostClean'] = generate_truepositives(ground_truth, boost_clean)

pyu.plot(true_positives_dict, sort_by='degree', inters_size_bounds=(2500, np.inf))
plt.show()
'''
