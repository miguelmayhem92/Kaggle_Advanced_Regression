import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import scipy as sp
from sklearn.metrics import mean_absolute_error, mean_squared_log_error

#########################################################################################
########################## data cleaning functions #####################################
#########################################################################################

dict_to_replace = {'Neighborhood': {'Crawfor':'CollgCr','NWAmes':'CollgCr','Gilbert':'CollgCr','ClearCr':'CollgCr','Blmngtn':'CollgCr','Veenker':'CollgCr' , 'BrkSide':'Edwards', 'NPkVill':'Edwards','BrDale':'Edwards',
                                   'SWISU':'Edwards','Blueste':'Edwards','NoRidge':'NridgHt','StoneBr':'NridgHt',  'Somerst':'Timber', 'NAmes':'Mitchel',
                                   'MeadowV':'Edwards','IDOTRR':'Edwards','Sawyer':'Edwards'},
                   'MasVnrType':{'BrkCmn':'None'},
                   'Fireplaces':{2:1,3:1},
                   'GarageCars':{4:2},
                   'TotRmsAbvGrd':{14:6},
                   'OverallQual':{1:2},
                   'TotRmsAbvGrd':{11:10,12:10},
                   'SaleCondition':{'Alloca':'Normal','Family':'Normal','AdjLand':'Abnorml'},
                   'BsmtExposure':{'Mn':'Av'},
                   'BsmtFinType1':{'BLQ':'LwQ','Rec':'LwQ','ALQ':'Unf'}}
dict_to_replace_2 = {'Neighborhood':{'Edwards':1,'OldTown':2,'Mitchel':3,'SawyerW':4,'CollgCr':5,'Timber':6,'NridgHt':7},
                    'MasVnrType':{np.nan: 'Stone'},
                    'BsmtQual':{np.nan: 'Fa'},
                    'BsmtExposure':{np.nan: 'Null'},
                    'BsmtFinType1':{np.nan: 'Null'}
                    }

replacer_categorical = {"Neighborhood": dict_to_replace['Neighborhood'],
                               "MasVnrType": dict_to_replace['MasVnrType'] ,
                               "GarageCars": dict_to_replace['GarageCars'] ,
                               "SaleCondition": dict_to_replace['SaleCondition'] ,
                               "BsmtExposure": dict_to_replace['BsmtExposure'] ,
                               "BsmtFinType1": dict_to_replace['BsmtFinType1'] ,
                               "Neighborhood": dict_to_replace['Neighborhood'] ,
                               "TotRmsAbvGrd": dict_to_replace['TotRmsAbvGrd'] ,
                               "OverallQual": dict_to_replace['OverallQual'] ,
                              "Fireplaces": dict_to_replace['Fireplaces'] }

replacer_categorical_2 = {"Neighborhood": dict_to_replace_2['Neighborhood'],
                         "BsmtQual": dict_to_replace_2['BsmtQual'],
                         "BsmtExposure": dict_to_replace_2['BsmtExposure'],
                          "BsmtFinType1": dict_to_replace_2['BsmtFinType1'],
                         "MasVnrType": dict_to_replace_2['MasVnrType']}

additional_numerical = ['OverallQual','TotRmsAbvGrd','GarageCars','FullBath','Neighborhood']
additional_categorical = ['GenQual']

def convert_categoricals(data,replacer_categorical,replacer_categorical_2):
    data = data.copy()
    data = data.replace(replacer_categorical)
    data = data.replace(replacer_categorical_2)
    data['GenQual'] = np.where((data.ExterQual == 'Ex') & (data.BsmtQual == 'Ex') & (data.KitchenQual == 'Ex'), 'Sup',
                            np.where((data.ExterQual == 'TA') & (data.BsmtQual == 'TA') & (data.KitchenQual == 'TA'), 'TA',
                            np.where((data.ExterQual == 'Gd') | (data.BsmtQual == 'Gd') | (data.KitchenQual == 'Gd'), 'Gd', 'Fa'
                                    )))
    return data


def control_outlier_numerical(data):
    data['BsmtFinSF1'] = np.where( (data.BsmtFinSF1 > 2000) & (data.BsmtUnfSF > 0) & (data.BsmtUnfSF < 142) ,1800,
                        np.where( (data.BsmtFinSF1 > 2000) & (data.Neighborhood == 1) , 600,
                        np.where( (data.BsmtFinSF1 > 2000) & (data.SaleCondition == 'Abnorml'), 1800,
                        np.where( (data.BsmtFinSF1 > 2000) & (data.BsmtExposure == 'Av'), 1800,
                        np.where( (data.BsmtFinSF1 > 2000) , 1800, data.BsmtFinSF1)
                        ))))
    data['TotalBsmtSF'] = np.where( (data.TotalBsmtSF > 2900) & (data.MasVnrType == 'None'), 2050,
                          np.where( (data.TotalBsmtSF > 2900) & (data.MasVnrType == 'BrkFace'), 1100,
                          np.where( (data.TotalBsmtSF > 2900) & (data.SaleCondition == 'Partial'), 1000,
                          np.where( (data.TotalBsmtSF > 2900) & (data.Neighborhood == 1), 1100,
                          np.where( (data.TotalBsmtSF > 2900) &  (data.OpenPorchSF > 112) & (data.OpenPorchSF < 547), 1100,
                          np.where( (data.TotalBsmtSF > 2900) &  (data.OpenPorchSF > 112) & (data.OpenPorchSF < 547), 1100,
                          np.where( (data.TotalBsmtSF > 2900) &  (data.OpenPorchSF == 0) , 1500,
                          np.where( (data.TotalBsmtSF > 2900) &  (data.OpenPorchSF > 63) & (data.OpenPorchSF < 112), 2050, data.BsmtFinSF1
                                  ))))))))
    data['GrLivArea'] = np.where( (data.GrLivArea > 4000) & (data.SaleCondition == 'Partial'), 1100,
                        np.where( (data.GrLivArea > 4000) & (data.GenQual == 'Fa'), 2000,
                        np.where( (data.GrLivArea > 3000) & (data.GrLivArea < 3450)  & (data.GenQual == 'Gd'),data.GrLivArea - 500 ,
                        np.where( (data.GrLivArea > 4200) & (data.GenQual == 'Gd'), 3200 ,
                        np.where( (data.GrLivArea > 4500) & (data.GenQual == 'Sup'), 2000 ,
                        np.where( (data.GrLivArea > 4000) & (data.GrLivArea < 4500) & (data.GenQual == 'Sup'), 3200 ,data.GrLivArea
                                ))))))
    data['TotRmsAbvGrd'] = np.where( (data.TotRmsAbvGrd > 11) & (data.MasVnrType == 'None'), 10,
                            np.where( (data.TotRmsAbvGrd > 11) & (data.Fireplaces == 0), 10,
                            np.where( (data.TotRmsAbvGrd > 11) & (data.SaleCondition == 'Abnorml'), 9,data.TotRmsAbvGrd
                                   )))
    return data

def convert_TotalBsmtSF(TotalBsmtSF, BsmtFinSF1, BsmtUnfSF):
    art_total = BsmtFinSF1 + BsmtUnfSF
    if art_total != 0:
        result_rate = (art_total - BsmtUnfSF)/art_total
    elif art_total == 0:
        result_rate = 0
    elif BsmtFinSF1 == 0 and TotalBsmtSF == 0:
        result_rate = 0
    return result_rate


def split_category_in_categorical(data):
    data['MasVnrType'] = np.where((data.MasVnrType == 'BrkFace') & (data.year_remod > 22), 'BrkFaceN1',
                                  np.where((data.MasVnrType == 'BrkFace') & (data.year_remod <= 22), 'BrkFaceN0',
                                           data.MasVnrType))

    data['MasVnrType'] = np.where((data.MasVnrType == 'Stone') & (data.year_remod > 8), 'Stone1',
                                  np.where((data.MasVnrType == 'Stone') & (data.year_remod <= 8), 'Stone0',
                                           data.MasVnrType))

    data['Fireplaces'] = np.where((data.Fireplaces == 1) & (data.year_age > 23), 2,
                                  np.where((data.Fireplaces == 1) & (data.year_age <= 23), 1,
                                           data.Fireplaces))  ########

    data['BsmtFinType1'] = np.where((data.BsmtFinType1 == 'GLQ') & (data.year_remod > 20), 'GLQ1',
                                    np.where((data.BsmtFinType1 == 'GLQ') & (data.year_remod <= 20), 'GLQ0',
                                             data.BsmtFinType1))

    return data


def years_vars_and_extra_numericasl(data, additional_numerical):
    data['year_age'] = data['YrSold'] - data['YearBuilt']
    data['year_remod'] = data['YrSold'] - data['YearRemodAdd']

    for column in additional_numerical:
        data[column] = data[column].astype('int', copy=False)

    return data

def mask_feature(feature):
    if feature == 0:
        return 0
    else:
        return 1

def feature_transformation_1(data, dict_to_replace, additional_numerical, replacer_categorical, replacer_categorical_2,
                             additional_numerical_list, numericals_to_mask):
    data = convert_categoricals(data, replacer_categorical, replacer_categorical_2)

    data = years_vars_and_extra_numericasl(data, additional_numerical_list)

    data['RateBsmt'] = data.apply(lambda x: convert_TotalBsmtSF(x['TotalBsmtSF'], x['BsmtFinSF1'], x['BsmtUnfSF']),
                                  axis=1)
    
    for to_mask in numericals_to_mask:
        label = f'Mask{to_mask}'
        data[label] = data.apply(lambda x: mask_feature(x[to_mask]), axis=1)

    data = control_outlier_numerical(data)
    data = split_category_in_categorical(data)

    return data

#########################################################################################
########################## data ML functions #############################################
#########################################################################################

selected_interaction_terms = ['SaleCondition_Abnorml_OverallQual_ord1','SaleCondition_Abnorml_OverallQual_ord2','SaleCondition_Abnorml_OverallQual_ord3','SaleCondition_Normal_OverallQual_ord1',
                              'SaleCondition_Normal_OverallQual_ord2','SaleCondition_Normal_OverallQual_ord3','SaleCondition_Partial_OverallQual_ord1','BsmtExposure_Av_GrLivArea_ord1','BsmtExposure_Av_GrLivArea_ord2',
 'BsmtExposure_Av_GrLivArea_ord3','BsmtExposure_Gd_GrLivArea_ord1','BsmtExposure_No_GrLivArea_ord1','BsmtExposure_Null_GrLivArea_ord1','BsmtExposure_Null_GrLivArea_ord2','BsmtExposure_Null_GrLivArea_ord3',
 'BsmtExposure_Av_OverallQual_ord1','BsmtExposure_Av_OverallQual_ord2','BsmtExposure_Av_OverallQual_ord3','BsmtExposure_Gd_OverallQual_ord1','BsmtExposure_Gd_OverallQual_ord2','BsmtExposure_No_OverallQual_ord1',
 'BsmtExposure_No_OverallQual_ord2','BsmtExposure_Null_OverallQual_ord1','BsmtExposure_Null_OverallQual_ord2','BsmtExposure_Null_OverallQual_ord3','BsmtExposure_Av_Neighborhood_ord1','BsmtExposure_Av_Neighborhood_ord2',
 'BsmtExposure_Av_Neighborhood_ord3','BsmtExposure_Gd_Neighborhood_ord1','BsmtExposure_Gd_Neighborhood_ord2','BsmtExposure_Gd_Neighborhood_ord3','BsmtExposure_No_Neighborhood_ord1','BsmtExposure_No_Neighborhood_ord2',
 'BsmtExposure_Null_Neighborhood_ord1','BsmtExposure_Null_Neighborhood_ord2','BsmtFinType1_GLQ0_GrLivArea_ord1','BsmtFinType1_GLQ0_GrLivArea_ord2','BsmtFinType1_GLQ1_GrLivArea_ord1','BsmtFinType1_GLQ1_GrLivArea_ord2',
 'BsmtFinType1_LwQ_GrLivArea_ord1','BsmtFinType1_LwQ_GrLivArea_ord2','BsmtFinType1_LwQ_GrLivArea_ord3','BsmtFinType1_Null_GrLivArea_ord1','BsmtFinType1_Unf_GrLivArea_ord1','BsmtFinType1_Unf_GrLivArea_ord2','GenQual_Fa_GrLivArea_ord1',
 'GenQual_Fa_GrLivArea_ord2','GenQual_Gd_GrLivArea_ord1','GenQual_Gd_GrLivArea_ord2','GenQual_Gd_GrLivArea_ord3','GenQual_Sup_GrLivArea_ord1',
 'GenQual_Sup_GrLivArea_ord2','GenQual_Sup_GrLivArea_ord3','GenQual_TA_GrLivArea_ord1','GenQual_TA_GrLivArea_ord2','GenQual_TA_GrLivArea_ord3']

selected_categorical_features  = ['GenQual', 'BsmtExposure', 'SaleCondition', 'BsmtFinType1']

selected_categorical_dummies_features = ['BsmtExposure_Av','BsmtExposure_Gd','BsmtFinType1_GLQ0','GenQual_Gd','BsmtFinType1_Unf',
 'GenQual_TA','SaleCondition_Normal','BsmtExposure_Null','SaleCondition_Abnorml','BsmtFinType1_LwQ',
 'BsmtExposure_No','SaleCondition_Partial','GenQual_Fa','BsmtFinType1_GLQ1','GenQual_Sup','BsmtFinType1_Null']

selected_numerical_features = ['GrLivArea', 'Neighborhood', 'OverallQual']

def columniser(data,target,numericals, categoricals):
    data = data[categoricals + numericals + [target]]
    return data

def scaler(dataset):
    scaler = StandardScaler()
    scaler.fit(dataset)

    dataset_scaled = scaler.transform(dataset)

    return dataset_scaled, scaler

def prep_interactin_terms(dataset, WholeCategoricals, WholeNumericals, SelectedDummies, dataReferencial, scale = False,
                          scalers_exist = None,lambdas_exist = None, base_columns = None ):
    ## getting the dummies
    dataset = dataset.copy()
    dummies = pd.get_dummies(dataset[WholeCategoricals])
    new_dataset = pd.concat([dummies,dataset[WholeNumericals + ['SalePrice']]],axis = 1)

    missingcolumns = [x for x in SelectedDummies if x not in new_dataset.columns]
    for col in missingcolumns:
        new_dataset[col] = 0
        
    ## getting the interaction terms
    interaction_terms_list = list()
    for i in range(len(dataReferencial)):
        rows_var = dataReferencial.iloc[i,::]
        numerical_var,categorical_var, category_x, order_x = rows_var.NumercialVar,rows_var.CategoricalVar,rows_var.Category,rows_var.Order

        for ords in range(1,order_x+1):
            interaction_term = f'{categorical_var}_{category_x}_{numerical_var}_ord{ords}'
            categorical = f'{categorical_var}_{category_x}'
            new_dataset[interaction_term] = new_dataset[categorical] * new_dataset[numerical_var] ** ords
            interaction_terms_list.append(interaction_term)
    
    
    ## getting the numericals and categoricals

    cat_columns_computed = SelectedDummies
    cat_columns_computed = list(set(cat_columns_computed))
    num_columns_computed = WholeNumericals + interaction_terms_list
    num_columns_computed = list(set(num_columns_computed))
    
    ###Result dataset with no scaling:
    dataset_treated = new_dataset[num_columns_computed + cat_columns_computed + ['SalePrice']].reset_index(drop = True)
    
    ### scaling
    if scale == True:
        
        data_toscale = dataset_treated.copy()
        dataset = data_toscale[num_columns_computed + ['SalePrice']]
        
        ## Scaling from 0 to 1
        data_scaled, scalerx = scaler(dataset = dataset)
        data_scaled = pd.DataFrame(data = data_scaled, columns = num_columns_computed + ['SalePrice']).reset_index(drop = True)

        dataset_treated = pd.concat([data_scaled[num_columns_computed], dataset_treated[cat_columns_computed ] , data_scaled['SalePrice'] ], axis = 1)
            
        return dataset_treated, scalerx, num_columns_computed, cat_columns_computed, interaction_terms_list
    
    if scale == False:
        ## creqting not existing feqtures 
        whole_base_columns = base_columns[0] + base_columns[1]
        missingcolumns = [x for x in whole_base_columns if x not in dataset_treated.columns ]
        for col in missingcolumns:
            dataset_treated[col] = 0
            
        numericals_list = base_columns[0]
        categoricals_list = base_columns[1]
        data_numericals = dataset_treated[numericals_list + ['SalePrice']]
        data_categoricals = dataset_treated[categoricals_list]
        
        data_numericals = scalers_exist.transform(data_numericals)
        data_numericals = pd.DataFrame(data_numericals, columns = numericals_list + ['SalePrice'])
       
        dataset_reconverted = pd.concat([data_numericals,data_categoricals ], axis = 1)
        dataset_reconverted = dataset_reconverted[numericals_list + list(categoricals_list) + ['SalePrice'] ]
        
        if missingcolumns:
            for col in missingcolumns:
                dataset_reconverted[col] = 0
                
        return dataset_reconverted
    
def prob(x,mean , sd):
    s = sp.stats.norm(mean, sd).pdf(x)
    return round(s,4)

def try_augmentation(data, target, numericals, categoricals, seed, n ):
    data = data.copy()
    meany, stdy = np.mean(data[target]),np.std(data[target])
    data['vector'] = data.apply(lambda x: prob(x[target], meany, stdy), axis=1)
    data_sample = data.sample(n = n, weights = 'vector', random_state = seed)
    data_sample = data_sample.reset_index(drop= True)
    
    ## adding noise to numericals
    for i in data_sample.index:
        q75, q25 = np.percentile(data_sample[target], [75 ,25])
        iqr = q75 - q25
        rangex = iqr * 0.1
        noise = np.random.uniform(0,rangex,1)[0]
        data_sample.loc[i,target] = data_sample.loc[i,target] + noise
    data_sample = data_sample.drop(columns = 'vector')
    return data_sample

def traindata_totrain(data, augmentation, augment = False):
    if augment:
        data_result = pd.concat([data,augmentation],axis = 0, ignore_index = True)
        #data_result = [selected_features + ['SalePrice']]
    else:
        data_result = data
    return data_result


def inverse_scaling(X,Y,scalers, numericals_list, categoricals_list):
    data_numericals = X[numericals_list]
    data_numericals['SalePrice'] = Y
    data_categoricals = X[categoricals_list]
    
    data_numericals = scalers.inverse_transform(data_numericals)
    data_numericals = pd.DataFrame(data_numericals, columns = numericals_list + ['SalePrice'])

    dataset_reconverted = pd.concat([data_numericals,data_categoricals ], axis = 1, ignore_index = False)
    dataset_reconverted = dataset_reconverted[numericals_list + list(categoricals_list) + ['SalePrice'] ].reset_index(drop = True)
    return dataset_reconverted

def dataframe_to_plot(data1,data2):
    df_plot = data1[['SalePrice']]
    df_plot['y_pred'] = data2[['SalePrice']]
    df_plot['y_pred'] = abs(df_plot['y_pred'])
    df_plot['error'] = df_plot['y_pred'] - df_plot['SalePrice']
    return df_plot

def plot_error_distrib(data):
    fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(23,6))

    sns.scatterplot(ax = axs[0] ,data = df_plot, x = df_plot.index, y = 'error')
    sns.scatterplot(ax = axs[1] ,data = df_plot, y = df_plot.SalePrice, x = 'error')
    sns.histplot(ax = axs[2] , data=df_plot, x="y_pred")
    fig.show()
    
def correct_extreme_values(array):
    med = np.median(array)
    array_result = np.where(array > 845000, med, array)
    return array_result

def gets_metrics(dataplot,typex):
    train_msle = mean_squared_log_error(dataplot.SalePrice,dataplot.y_pred)
    train_mae = mean_absolute_error(dataplot.SalePrice,dataplot.y_pred)
    train_mets = {typex: {'MSLE':train_msle,'MAE':train_mae}}
    return train_mets