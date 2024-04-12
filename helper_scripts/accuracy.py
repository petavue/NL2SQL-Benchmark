import pandas as pd
import shutil
import subprocess
import os
import copy
import argparse
import pathlib
import json


dataset = './bird_equal_split_360.csv'
tssq_path = '' #add the DAMO-CONVAI/BIRD path which has the evauation script eg:- /Users/username/petavue/DAMO-ConvAI
start_path = './'
dest = tssq_path
command = f'python -u {tssq_path}/bird/llm/src/evaluation.py --db_root_path {tssq_path}/bird/llm/data/dev_databases/ --predicted_sql_path {tssq_path}/bird/llm/evaldata/ --data_mode dev --ground_truth_path {tssq_path}/bird/llm/evaldata/ --mode_gt gt --mode_predict gpt --diff_json_path {tssq_path}/bird/llm/evaldata/dev.json --meta_time_out 30.0'
#command.split()  #uncomment this on windows


dff=pd.read_csv(dataset)


def bucket(dataset,pred,parent_directory):
    def bucketing_gold(dataset):
        simple = []
        moderate = []
        challenging = []
        data = pd.read_csv(dataset)
        for index, row in data.iterrows():
            if row["difficulty"] == "simple":
                simple += [index]
            elif row["difficulty"] == "moderate":
                moderate += [index]
            else:
                challenging += [index]
        return(simple, moderate, challenging, data)

    def CsvSorting_gold(s, m, c, d):
        simple_rows = d.iloc[s]
        moderate_rows = d.iloc[m]        
        challenging_rows = d.iloc[c]
        si = int(len(simple_rows)) 
        s1 = simple_rows.iloc[0:20]
        s2 = moderate_rows.iloc[0:20]
        s3 = challenging_rows.iloc[0:20]
        df  = pd.concat([s1, s2, s3], ignore_index=True)
        subfolder_name = "60_size"
        subfolder_path = folder_path(subfolder_name,parent_directory)
        goldfile(subfolder_path, df)
        dev_data = df.to_json(orient='records', indent=4)
        filename = 'dev.json'
        filepath = os.path.join(subfolder_path,filename)
        with open(filepath, 'w') as json_file:
            json_file.write(dev_data)
        m1 = simple_rows.iloc[20:60]
        m2 = moderate_rows.iloc[20:60]
        m3 = challenging_rows.iloc[20:60]
        df1  = pd.concat([m1, m2, m3], ignore_index=True)    
        dev_data = df1.to_json(orient='records', indent=4)
        subfolder_name = "120_size"
        subfolder_path = folder_path(subfolder_name,parent_directory)
        goldfile(subfolder_path, df1)
        filename = 'dev.json'
        filepath = os.path.join(subfolder_path,filename)  
        with open(filepath, 'w') as json_file:
            json_file.write(dev_data)         
        c1 = simple_rows.iloc[60:120]
        c2 = moderate_rows.iloc[60:120]
        c3 = challenging_rows.iloc[60:120]
        df2  = pd.concat([c1, c2, c3], ignore_index=True)    
        dev_data = df2.to_json(orient='records', indent=4)
        subfolder_name = "180_size"
        subfolder_path =folder_path(subfolder_name,parent_directory)
        goldfile(subfolder_path, df2)
        filename = 'dev.json'
        filepath = os.path.join(subfolder_path,filename)    
        with open(filepath, 'w') as json_file:
            json_file.write(dev_data) 
        
    def CsvSorting_json(s, m, c, d):
        simple_rows = d.iloc[s]
        moderate_rows = d.iloc[m]        
        challenging_rows = d.iloc[c]
        si = int(len(simple_rows)) 
        s1 = simple_rows.iloc[0:20]
        s2 = moderate_rows.iloc[0:20]
        s3 = challenging_rows.iloc[0:20]
        df  = pd.concat([s1, s2, s3], ignore_index=True)
        json_data = {str(index): query for index, query in enumerate(df['value'])}
        subfolder_name = "60_size"
        subfolder_path = folder_path(subfolder_name,parent_directory)
        file_path = os.path.join(subfolder_path, 'predict_dev.json')
        with open(file_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=4)
        m1 = simple_rows.iloc[20:60]
        m2 = moderate_rows.iloc[20:60]
        m3 = challenging_rows.iloc[20:60]
        df1  = pd.concat([m1, m2, m3], ignore_index=True)
        json_data1 = {str(index): query for index, query in enumerate(df1['value'])}
        subfolder_name = "120_size"
        subfolder_path = folder_path(subfolder_name,parent_directory)
        file_path = os.path.join(subfolder_path, 'predict_dev.json')
        with open(file_path, 'w') as json_file:
            json.dump(json_data1, json_file, indent=4)
        c1 = simple_rows.iloc[60:120]
        c2 = moderate_rows.iloc[60:120]
        c3 = challenging_rows.iloc[60:120]
        df2  = pd.concat([c1, c2, c3], ignore_index=True)
        json_data2 = {str(index): query for index, query in enumerate(df2['value'])}
        subfolder_name = "180_size"
        subfolder_path = folder_path(subfolder_name,parent_directory)
        file_path = os.path.join(subfolder_path, 'predict_dev.json')
        with open(file_path, 'w') as json_file:
            json.dump(json_data2, json_file, indent=4)
            
    def goldfile(subfolder_path, df):
        sql_file = os.path.join(subfolder_path, "dev_gold.sql")
        with open(sql_file, 'w') as f:
            for val1,val2 in zip(df['sql_query'],df['db_id']):
                f.write(str(val1) + '\t' + str(val2) + '\n')

    def folder_path(subfolder_name,parent_directory):
        subfolder_path = os.path.join(parent_directory, subfolder_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path) 
        return subfolder_path   

    def read_json_file(filepath):
        with open(filepath, 'r') as f:
            cache = f.read()
            data = eval(cache)
        df = pd.DataFrame(list(data.items()), columns=['key', 'value'])
        return df
        
    s , m , c , d= bucketing_gold(dataset)
    filepath = pred
    CsvSorting_gold(s, m, c, d)
    df = read_json_file(filepath)
    CsvSorting_json(s, m, c, df)

def bucket_accuracy(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item).replace(os.sep, '/')
        folder_name = os.path.basename(os.path.dirname(item_path))
        if os.path.isdir(item_path):
            bucket_accuracy(item_path)   
        elif item == 'metrics.csv':
            predicted_path = os.path.join(directory, 'predict_dev.json')
            bucket(dataset,predicted_path,directory)
bucket_accuracy('./')

all_data = {}
template_data = {"0_Instructions":{"60_Inferences":[],"120_Inferences":[],"180_Inferences":[],"360_Inferences":[]},"5_Instructions":{"60_Inferences":[],"120_Inferences":[],"180_Inferences":[],"360_Inferences":[]},"7_Instructions":{"60_Inferences":[],"120_Inferences":[],"180_Inferences":[],"360_Inferences":[]},"9_Instructions":{"60_Inferences":[],"120_Inferences":[],"180_Inferences":[],"360_Inferences":[]},"11_Instructions":{"60_Inferences":[],"120_Inferences":[],"180_Inferences":[],"360_Inferences":[]}}

data = {'environment': [],'model': [],'instruction': [], 'hardness': [],'shot':[],'dataset_size': [],'metric':[],'value':[]}
new_df = pd.DataFrame(data)

def copy_to_destination(gold_path, predicted_path,dev_path,env,model_name,inst,dataset):
    if not model_name in all_data.keys():
        all_data[model_name] = copy.deepcopy(template_data)
    shutil.copy(gold_path, f'{dest}/bird/llm/evaldata')
    shutil.copy(predicted_path, dest+"/bird/llm/evaldata")
    shutil.copy(dev_path, dest+"/bird/llm/evaldata")

    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:  
        print("Command executed successfully!")
    else:
        print("Command failed!")
    print("Output:", result.stdout)
    lines = result.stdout.splitlines()
    words_list = []
    for line in lines:
        words = line.split()
        words_list.append(words)
    instruction=inst.split('_')[0]
    for i in range(4):
        new_row_values = [env,model_name,instruction,words_list[1][i],0,words_list[2][-1],'accuracy',words_list[4][i+1]]
        new_df.loc[len(new_df)] = new_row_values

def find_metric_files(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item).replace(os.sep, '/')
        folder_name = os.path.basename(os.path.dirname(item_path))
        if os.path.isdir(item_path):
            find_metric_files(item_path)   
        elif item == 'dev_gold.sql':
            gold_path = item_path
            predicted_path = os.path.join(directory, 'predict_dev.json')
            dev_path = os.path.join(directory, 'dev.json')
            if os.path.exists(predicted_path and dev_path):
                if folder_name=='360_Inferences':
                    print(directory)
                    print(f"Evaluating {directory.split('/')[1]} {directory.split('/')[-3]}  {directory.split('/')[-2]}  {directory.split('/')[-1]} ")
                    copy_to_destination(gold_path, predicted_path,dev_path,directory.split('/')[1].strip(),directory.split('/')[-3].strip(),directory.split('/')[-2].strip(),directory.split('/')[-1].strip())
                else:
                    print(directory)
                    print(f"Evaluating {directory.split('/')[1]} {directory.split('/')[-4]}  {directory.split('/')[-3]}  {directory.split('/')[-2]} {directory.split('/')[-1]}")
                    copy_to_destination(gold_path, predicted_path,dev_path,directory.split('/')[1].strip(),directory.split('/')[-4].strip(),directory.split('/')[-3].strip(),directory.split('/')[-1].strip())

find_metric_files(start_path)

new_df.to_csv('Final_acc_self1.csv',index=False)