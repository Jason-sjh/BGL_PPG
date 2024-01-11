import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import os
from datetime import datetime
import seaborn as sns

def clarke_error_grid(ref_values, pred_values, title_string, folder_path = 'result/clarke_error_grid_plot/'):
    #Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(pred_values)), "Unequal number of values (reference : {}) (prediction : {}).".format(len(ref_values), len(pred_values))
    #Checks to see if the values are within the normal physiological range, otherwise it gives a warning
    if max(ref_values) > 400 or max(pred_values) > 400:
        print ("Input Warning: the maximum reference value {} or the maximum prediction value {} exceeds the normal physiological range of glucose (<400 mg/dl).".format(max(ref_values), max(pred_values)))
    if min(ref_values) < 0 or min(pred_values) < 0:
        print ("Input Warning: the minimum reference value {} or the minimum prediction value {} is less than 0 mg/dl.".format(min(ref_values),  min(pred_values)))

    #Clear plot
    plt.clf()

    #Set up plot
    plt.figure(figsize=(8, 8))

    plt.scatter(ref_values, pred_values, marker='o', color='black', s=8)
    plt.title(title_string + " Clarke Error Grid")
    plt.xlabel("Reference Concentration (mg/dl)")
    plt.ylabel("Prediction Concentration (mg/dl)")
    plt.xticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.yticks([0, 50, 100, 150, 200, 250, 300, 350, 400])
    plt.gca().set_facecolor('white')

    #Set axes lengths
    plt.gca().set_xlim([0, 400])
    plt.gca().set_ylim([0, 400])
    plt.gca().set_aspect((400)/(400))

    #Plot zone lines
    plt.plot([0,400], [0,400], ':', c='black')                      
    plt.plot([0, 175/3], [70, 70], '-', c='black')
    #plt.plot([175/3, 320], [70, 400], '-', c='black')
    plt.plot([175/3, 400/1.2], [70, 400], '-', c='black')           
    plt.plot([70, 70], [84, 400],'-', c='black')
    plt.plot([0, 70], [180, 180], '-', c='black')
    plt.plot([70, 290],[180, 400],'-', c='black')
    # plt.plot([70, 70], [0, 175/3], '-', c='black')
    plt.plot([70, 70], [0, 56], '-', c='black')                     
    # plt.plot([70, 400],[175/3, 320],'-', c='black')
    plt.plot([70, 400], [56, 320],'-', c='black')
    plt.plot([180, 180], [0, 70], '-', c='black')
    plt.plot([180, 400], [70, 70], '-', c='black')
    plt.plot([240, 240], [70, 180],'-', c='black')
    plt.plot([240, 400], [180, 180], '-', c='black')
    plt.plot([130, 180], [0, 70], '-', c='black')


    plt.text(30, 15, "A", fontsize=15)
    plt.text(370, 260, "B", fontsize=15)
    plt.text(280, 370, "B", fontsize=15)
    plt.text(160, 370, "C", fontsize=15)
    plt.text(160, 15, "C", fontsize=15)
    plt.text(30, 140, "D", fontsize=15)
    plt.text(370, 120, "D", fontsize=15)
    plt.text(30, 370, "E", fontsize=15)
    plt.text(370, 15, "E", fontsize=15)


    zone = [0] * 5
    for i in range(len(ref_values)):
        if (ref_values[i] <= 70 and pred_values[i] <= 70) or (pred_values[i] <= 1.2*ref_values[i] and pred_values[i] >= 0.8*ref_values[i]):
            zone[0] += 1    #Zone A

        elif (ref_values[i] >= 180 and pred_values[i] <= 70) or (ref_values[i] <= 70 and pred_values[i] >= 180):
            zone[4] += 1    #Zone E

        elif ((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[i] + 110) or ((ref_values[i] >= 130 and ref_values[i] <= 180) and (pred_values[i] <= (7/5)*ref_values[i] - 182)):
            zone[2] += 1    #Zone C
        elif (ref_values[i] >= 240 and (pred_values[i] >= 70 and pred_values[i] <= 180)) or (ref_values[i] <= 175/3 and pred_values[i] <= 180 and pred_values[i] >= 70) or ((ref_values[i] >= 175/3 and ref_values[i] <= 70) and pred_values[i] >= (6/5)*ref_values[i]):
            zone[3] += 1    #Zone D
        else:
            zone[1] += 1    #Zone B

    zone_counts = count_clarke_error_grid_zones(ref_values, pred_values)
    plt.text(200, -50, zone_counts, ha='center')
    
    if not os.path.exists(folder_path):
        print(f" Make Folder {folder_path}")
        os.makedirs(folder_path)        
  
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")       
    plt.savefig(  f"{folder_path}{title_string}_{current_time}.png")      

    
    
def count_clarke_error_grid_zones(ref_values, pred_values):
    zone_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0, 'E': 0}

    for ref, pred in zip(ref_values, pred_values):
        if (ref <= 70 and pred <= 70) or (0.8 * ref <= pred <= 1.2 * ref):
            zone_counts['A'] += 1
        elif (ref >= 180 and pred <= 70) or (ref <= 70 and pred >= 180):
            zone_counts['E'] += 1
        elif ((70 <= ref <= 290 and pred >= ref + 110) or
              (130 <= ref <= 180 and pred <= (7/5)*ref - 182)):
            zone_counts['C'] += 1
        elif ((ref >= 240 and (70 <= pred <= 180)) or
              (ref <= 175/3 and (70 <= pred <= 180)) or
              (175/3 <= ref <= 70 and pred >= (6/5)*ref)):
            zone_counts['D'] += 1        
        else:
            zone_counts['B'] += 1 
    return zone_counts


def corr_plot(feature_data, label_data,folder_path = './result/EDA'):

    full_arr = np.hstack((feature_data, label_data.reshape(-1,1)) )
    correlation_matrix = np.corrcoef(full_arr, rowvar=False)

    plt.figure(figsize=(12, 10))    
    new_xticklabels = ['SE_mean', 'SE_var', 'SE_iqr', 'SE_skew', 'KTE_mean', 'KTE_var', 'KTE_iqr', 'KTE_skew', 'LogE_mean', 'LogE_var', 'LogE_iqr','LogE_skew', 'BGL']
    # new_xticklabels = ['SE_mean', 'SE_var', 'SE_iqr', 'SE_skew', 'KTE_mean', 'KTE_var', 'KTE_iqr', 'KTE_skew', 'LogE_mean', 'LogE_var', 'LogE_iqr','LogE_skew','AR_ppg_1','AR_ppg_2','AR_ppg_3', 'AR_kte_1','AR_kte_2','AR_kte_3', 'AR_LogE_1', 'AR_LogE_2','AR_LogE_3', 'BGL']
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                xticklabels=new_xticklabels,
                yticklabels=new_xticklabels)

    plt.title('Correlation heatmap')
    if not os.path.exists(folder_path):
        print(f" Make Folder {folder_path}")
        os.makedirs(folder_path)          
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")           
    plt.savefig( f"{folder_path}/corr_heatmap_{current_time}.png")      
    
def sns_pair_plot(feature_data, label_data,folder_path = './result/EDA'):
    full_arr = np.hstack((feature_data, label_data.reshape(-1,1)) )

    new_xticklabels = ['SE_mean', 'SE_var', 'SE_iqr', 'SE_skew', 'KTE_mean', 'KTE_var', 'KTE_iqr', 'KTE_skew', 'LogE_mean', 'LogE_var', 'LogE_iqr','LogE_skew', 'BGL']
    # new_xticklabels = ['SE_mean', 'SE_var', 'SE_iqr', 'SE_skew', 'KTE_mean', 'KTE_var', 'KTE_iqr', 'KTE_skew', 'LogE_mean', 'LogE_var', 'LogE_iqr','LogE_skew','AR_ppg_1','AR_ppg_2','AR_ppg_3', 'AR_kte_1','AR_kte_2','AR_kte_3', 'AR_LogE_1', 'AR_LogE_2','AR_LogE_3', 'BGL']

    df_full = pd.DataFrame(full_arr)
    df_full.columns = new_xticklabels
    sam = df_full.sample(100)


    plt.figure(figsize=(15, 15))  
    plt.title('Hist and scatter plot')  
    sns.pairplot(sam)   
    
    if not os.path.exists(folder_path):
        print(f" Make Folder {folder_path}")
        os.makedirs(folder_path)          
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")           
    plt.savefig( f"{folder_path}/HistAndScatter_{current_time}.png")      
