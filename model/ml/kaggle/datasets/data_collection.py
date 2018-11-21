

def get_data():
    #csv path -> target column id
    data_collection = []


    #kaggle data
    '''
    data_collection.append(("/home/felix/.kaggle/datasets/aakash2016/big-mart-sales-dataset/Train_UWu5bXk.csv", 10)) # 10= Outlet_Type, Regression: 11= Item_Outlet_Sales
    data_collection.append(("/home/felix/.kaggle/datasets/aariyan101/bank-notes/bank_note_data.csv", 4))
    data_collection.append(("/home/felix/.kaggle/datasets/abcsds/pokemon/Pokemon.csv", 12)) # easy but unbalanced
    data_collection.append(("/home/felix/.kaggle/datasets/ajay1735/hmeq-data/hmeq.csv", 0))
    data_collection.append(("/home/felix/.kaggle/datasets/akashkr/student-datafest-2018/train_HK6lq50.csv", 15))
    data_collection.append(("/home/felix/.kaggle/datasets/akhilsaichinthala/classifications/Dataset for Classification.csv", 1)) #attrition in a jobdata_
    data_collection.append(("/home/felix/.kaggle/datasets/amanajmera1/framingham-heart-study-dataset/framingham.csv", 15))
    '''

    #OpenML / ExploreKit data
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/dataset_53_heart-statlog_heart.csv", 13))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/dataset_27_colic_horse.csv", 22))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/phpAmSP4g_cancer.csv", 30))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/phpOJxGL9_indianliver.csv", 10))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/dataset_29_credit-a_credit.csv", 15))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/dataset_37_diabetes_diabetes.csv", 8))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/dataset_31_credit-g_german_credit.csv", 20))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/dataset_23_cmc_contraceptive.csv", 9))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/phpkIxskf_bank_data.csv", 16))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/vehicleNorm.csv", 100))
    data_collection.append(("/home/felix/datasets/ExploreKit/csv/phpn1jVwe_mammography.csv", 6))

    return data_collection