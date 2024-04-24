import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def resolve_label_disagreements(data, extra_features, group1_features, group2_features, group3_features):
    """
    Resolve labeling disagreements in a DataFrame according to JustRAIGS challenge guidance. 
    Assigns final labels to features based on the availability and agreement of labels from three graders. 
    If the 3rd grader's labels are available, they are used as final labels.
    If not available, labels from the 1st and 2nd graders are compared; 
    if they agree, the agreed label is used, otherwise, NaN is assigned.
    """
    for idx in range(len(data)):
        # If G3 label available, choose those as final values
        if not pd.isna(data.iloc[idx]['Label G3']):
            for feature, g3_feature in zip(extra_features, group3_features):
                data.at[idx, feature] = data.at[idx, g3_feature]
        else:
            for feature, g1_feature, g2_feature in zip(extra_features, group1_features, group2_features):
                # If G1 & G2 agree on feature label, choose this agreement as final label
                if data.at[idx, g1_feature] == data.at[idx, g2_feature]:
                    data.at[idx, feature] = data.at[idx, g1_feature]
                 # If G1 & G2 don't agree, assign NaN
                else:
                    data.at[idx, feature] = np.nan
    return data

def fill_nan_with_most_frequent(data, features):
    """Fill NaN values with the most frequent value for each feature."""
    for feature in features:
        most_frequent_value = data[feature].mode()[0]
        data[feature].fillna(most_frequent_value, inplace=True)
        print(f"{feature}: Most frequent value used for NaN replacement is {most_frequent_value}")
    return data

def main():
    # Load data
    df_temp = pd.read_csv('./JustRAIGS_Train_labels.csv', sep=';')

    # Filter and relabel the dataset
    rg_instances = df_temp[df_temp['Final Label'] == 'RG']
    rg_instances['Final Label'] = 1  # Setting RG class label to 1
    rg_instances.reset_index(drop=True, inplace=True)
    
    nrg_instances = df_temp[df_temp['Final Label'] == 'NRG']
    nrg_instances['Final Label'] = 0  # Setting NRG class label to 0
    nrg_instances.reset_index(drop=True, inplace=True)
    
    # Define the extra features for different groups
    extra_features = ['ANRS', 'ANRI', 'RNFLDS', 'RNFLDI', 'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']
    extra_features_g1 = ['G1 ' + feature for feature in extra_features]
    extra_features_g2 = ['G2 ' + feature for feature in extra_features]
    extra_features_g3 = ['G3 ' + feature for feature in extra_features]

    # Resolve labeling disagreements
    rg_instances = resolve_label_disagreements(rg_instances, extra_features, extra_features_g1, extra_features_g2, extra_features_g3)

    # Fill NaN values with the most frequent value
    rg_instances = fill_nan_with_most_frequent(rg_instances, extra_features)

    # Split the dataset for 10 features classification
    train_rg, test_rg = train_test_split(rg_instances[[
    'Eye ID', 'Final Label', 'ANRS', 'ANRI', 'RNFLDS', 'RNFLDI',
    'BCLVS', 'BCLVI', 'NVT', 'DH', 'LD', 'LC']], test_size=0.1, random_state=42, shuffle=True)
    
    # Split the data into training and testing datasets for glaucoma classification. 
    # The test set is balanced, containing 10% of all referable glaucoma cases 
    # along with an equal number of non-referable glaucoma cases.
    train_nrg, test_nrg = train_test_split(nrg_instances[[
    'Eye ID', 'Final Label']], test_size=len(test_rg), random_state=42, shuffle=True)
    
    # Merge RG and NRG for glaucoma classification
    train_glaucoma = pd.concat([train_rg[['Eye ID', 'Final Label']], train_nrg], ignore_index=True)
    test_glaucoma = pd.concat([test_rg[['Eye ID', 'Final Label']], test_nrg], ignore_index=True)

    # Shuffle the data
    train_rg = train_rg.sample(frac=1).reset_index(drop=True)
    test_rg = test_rg.sample(frac=1).reset_index(drop=True)
    train_glaucoma = train_glaucoma.sample(frac=1).reset_index(drop=True)
    test_glaucoma = test_glaucoma.sample(frac=1).reset_index(drop=True)

    # Save to CSV
    train_rg.to_csv('./10_features_no_mask_train.csv', index=False)
    test_rg.to_csv('./10_features_no_mask_test.csv', index=False)
    train_glaucoma.to_csv('./glaucoma_no_mask_train.csv', index=False)
    test_glaucoma.to_csv('./glaucoma_no_mask_test.csv', index=False)

if __name__ == "__main__":
    main()