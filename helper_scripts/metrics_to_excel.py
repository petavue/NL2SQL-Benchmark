import os
import pandas as pd
import datetime
import argparse
import numpy as np

# constants
dataset = './bird_equal_split_360.csv'
directory_to_search = './'
hardness_list = ['simple', 'moderate', 'challenging', 'total']

#cost and token for each model
models_dict = {
        'Llama-2-70b-chat-hf': {'input_cost': 1.0, 'output_cost': 1.0, 'per_tokens': 1000000, 'platform': 'anyscale', 'as_on_date':'1/04/24'},
        'Mistral-7B-Instruct-v0.1': {'input_cost': 0.15, 'output_cost': 0.15, 'per_tokens': 1000000, 'platform': 'anyscale','as_on_date':'1/04/24'},
        'Mixtral-8x7B-Instruct-v0.1': {'input_cost': 0.50, 'output_cost': 0.50, 'per_tokens': 1000000, 'platform': 'anyscale','as_on_date':'1/04/24'},
        'CodeLlama-70b-Instruct-hf': {'input_cost': 1.0, 'output_cost': 1.0, 'per_tokens': 1000000, 'platform': 'anyscale','as_on_date':'1/04/24'},
        'meta.llama2-70b-chat-v1': {'input_cost': 0.00195*1000, 'output_cost': 0.00256*1000, 'per_tokens': 1000000, 'platform': 'amazon-bedrock','as_on_date':'7/3/24'},
        'anthropic.claude-3-sonnet-20240229-v1_0': {'input_cost': 0.00300*1000, 'output_cost': 0.01500*1000, 'per_tokens': 1000000, 'platform': 'amazon-bedrock','as_on_date':'1/04/24'},
        'anthropic.claude-3-haiku-20240307-v1_0': {'input_cost': 0.00025*1000, 'output_cost': 0.00125*1000, 'per_tokens': 1000000, 'platform': 'amazon-bedrock','as_on_date':'1/04/24'},
        'mistral.mixtral-8x7b-instruct-v0_1': {'input_cost': 0.00045*1000, 'output_cost': 0.0007*1000, 'per_tokens': 1000000, 'platform': 'amazon-bedrock','as_on_date':'1/04/24'},
        'mistral.mistral-7b-instruct-v0_2': {'input_cost': 0.00015*1000, 'output_cost': 0.0002*1000, 'per_tokens': 1000000, 'platform': 'amazon-bedrock','as_on_date':'1/04/24'},
        'gpt-4-turbo-preview': {'input_cost': 0.01*1000, 'output_cost': 0.03*1000, 'per_tokens': 1000000, 'platform': 'open-ai','as_on_date':'1/04/24'},
        'gpt-3.5-turbo-16k': {'input_cost': 0.0005*1000, 'output_cost': 0.0015*1000, 'per_tokens': 1000000, 'platform': 'open-ai','as_on_date':'1/04/24'},
        'gemini-1.0-pro-latest': {'input_cost': 0, 'output_cost': 0, 'per_tokens': 1000000, 'platform': 'gemini','as_on_date':'7/3/24'},
        'claude-3-haiku-20240307': {'input_cost': 0.25, 'output_cost': 1.25, 'per_tokens': 1000000, 'platform': 'anthropic','as_on_date':'1/04/24'},
        'claude-3-sonnet-20240229': {'input_cost': 3, 'output_cost': 15, 'per_tokens': 1000000, 'platform': 'anthropic','as_on_date':'1/04/24'},
        'claude-3-opus-20240229': {'input_cost': 15, 'output_cost': 75, 'per_tokens': 1000000, 'platform': 'anthropic','as_on_date':'1/04/24'},
    }

mapping={'anthropic.claude-3-haiku-20240307-v1_0':'claude-3-haiku',
       'anthropic.claude-3-sonnet-20240229-v1_0':'claude-3-sonnet',
       'meta.llama2-70b-chat-v1':'llama2-70b', 'mistral.mistral-7b-instruct-v0_2':'mistral-7b',
       'mistral.mixtral-8x7b-instruct-v0_1':'mixtral-8x7b','claude-3-haiku-20240307':'claude-3-haiku', 
       'claude-3-opus-20240229':'claude-3-opus',
       'claude-3-sonnet-20240229':'claude-3-sonnet','CodeLlama-70b-Instruct-hf':'CodeLlama-70b', 
       'Llama-2-70b-chat-hf':'llama2-70b',
       'Mistral-7B-Instruct-v0.1':'mistral-7b', 'Mixtral-8x7B-Instruct-v0.1':'mixtral-8x7b',
       'gemini-1.0-pro-latest':'gemini',
       'gpt-3.5-turbo-16k':'gpt-3.5', 'gpt-4-turbo-preview':'gpt-4',
       'Mixtral-8x7B-Instruct-v0.1':'mixtral-8x7b','Mistral-7B-Instruct-v0.2':'mistral-7b-v2',
       'Mistral-7B-Instruct-v0.1':'mistral-7b-v1',
       'CodeLlama-70b-Instruct-hf':'CodeLlama-70b','CodeLlama-34b-Instruct-hf':'CodeLlama-34b',
       'dbrx-instruct':'dbrx','sqlcoder-7b-2':'sqlcoder-7b-2','sqlcoder-70b-alpha':'sqlcoder-70b-alpha',
       'WizardCoder-33B-V1.1':'WizardCoder'}

template_data = {'environment': [],'model': [],'instruction': [],'dataset_size': [],'metric': [],'hardness': [],'value': []}
entire_df = pd.DataFrame(template_data)

#Buckting the 360 dataset into 60,120,180 and 360
def bucket(dataset,parent_directory,metric_file):
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
    
    def metrics(s,m,c,d):
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
        df.to_csv(subfolder_path+'/'+'metrics.csv',index=False)
        m1 = simple_rows.iloc[20:60]
        m2 = moderate_rows.iloc[20:60]
        m3 = challenging_rows.iloc[20:60]
        df1  = pd.concat([m1, m2, m3], ignore_index=True)
        subfolder_name = "120_size"
        subfolder_path = folder_path(subfolder_name,parent_directory)
        df1.to_csv(subfolder_path+'/'+'metrics.csv',index=False)
        c1 = simple_rows.iloc[60:120]
        c2 = moderate_rows.iloc[60:120]
        c3 = challenging_rows.iloc[60:120]
        df2  = pd.concat([c1, c2, c3], ignore_index=True)
        subfolder_name = "180_size"
        subfolder_path = folder_path(subfolder_name,parent_directory)
        df2.to_csv(subfolder_path+'/'+'metrics.csv',index=False)

    def folder_path(subfolder_name,parent_directory):
        subfolder_path = os.path.join(parent_directory, subfolder_name)
        if not os.path.exists(subfolder_path):
            os.makedirs(subfolder_path) 
        return subfolder_path   

    s , m , c , d= bucketing_gold(dataset)
    df1=pd.read_csv(metric_file)
    metrics(s,m,c,df1)
    
def make_metric_bucket(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item).replace(os.sep, '/')
            folder_name = os.path.basename(os.path.dirname(item_path))
            if os.path.isdir(item_path):
                make_metric_bucket(item_path) 
            elif item == 'dev.json':
                metric_path = os.path.join(directory, 'metrics.csv')
                bucket(dataset,directory,metric_path)

#calling bucketing script
make_metric_bucket(directory_to_search)

#function to convert min and hr to secs.
def time_str_to_seconds(time_str):
    if time_str=='0':
        return 0
    time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S.%f")
    return (time_obj - datetime.datetime(1900, 1, 1)).total_seconds()


def get_processed_df(complete_dataframe, model_name,environment,inst,dataset_size,separate_all=True):
    #initializing cost,token per inference for anycale , bedrock, gemini, anthropic and openai. 
    if model_name in models_dict:
        input_cost = models_dict[model_name]['input_cost']
        output_cost = models_dict[model_name]['output_cost']
        token_per = models_dict[model_name]['per_tokens']
    
    #template to save the metrics.
    final_dict = {
        "hardness": hardness_list,
        "avg_time": [],
        "total_cost": [],
        "total_output_tokens": [],
        "total_input_tokens": [],
        "throughput": [],
        "PSR": [],
    }
    
    #converting infernece time to secs.
    complete_dataframe['seconds'] = complete_dataframe['response_time'].apply(lambda x: time_str_to_seconds(x))
    #calculating correct output tokens for self-hosted.
    if environment == 'self-hosted':
        complete_dataframe['llm_response_tokens'] = complete_dataframe['llm_response_tokens'] - complete_dataframe['llm_prompt_tokens']
    
    
    #------------------------------
    #code for simple, moderate and challenging.
    
    for hardness in hardness_list:
        if hardness == 'total':
            continue
        curr_hardness_df = complete_dataframe[complete_dataframe['hardness']==hardness] 
        #curr_hardness - input token and output token
        curr_hardness_input_tokens = curr_hardness_df['llm_prompt_tokens'].sum()
        curr_hardness_output_tokens = curr_hardness_df['llm_response_tokens'].sum()
        #filtering data which has error or 'no sql'
        curr_hardness_filtered_df = curr_hardness_df[curr_hardness_df['seconds'] != 0]
        #curr_hardness - average time
        curr_hardness_avg_time = curr_hardness_filtered_df['seconds'].mean()
        #curr_hardness - throughput
        curr_hardness_filtered_df['throughput_per_inference'] = curr_hardness_filtered_df['llm_response_tokens']/curr_hardness_filtered_df['seconds']
        curr_hardness_throughput = curr_hardness_filtered_df['throughput_per_inference'].mean()
        #curr_hardness - cost
        if environment=='self-hosted':
            # (curr_hardness cost = curr_hardness inference seconds / 3600 ) * 10$ per hour
            curr_hardness_cost = (curr_hardness_filtered_df['seconds'].sum())/360
        else:
            curr_hardness_input_token_cost = (curr_hardness_input_tokens*input_cost)/(token_per)
            curr_hardness_output_token_cost = (curr_hardness_output_tokens*output_cost)/(token_per)
            curr_hardness_cost = (curr_hardness_input_token_cost + curr_hardness_output_token_cost)
        #curr_hardness - PSR
        curr_hardness_PSR = (len(curr_hardness_filtered_df)/len(curr_hardness_df))*100
        
        #adding the metrics data to template dict
        final_dict['avg_time'].append(curr_hardness_avg_time) 
        final_dict['total_cost'].append(curr_hardness_cost)
        final_dict['total_output_tokens'].append(curr_hardness_output_tokens)
        final_dict['total_input_tokens'].append(curr_hardness_input_tokens)
        final_dict['throughput'].append(curr_hardness_throughput)
        final_dict['PSR'].append(curr_hardness_PSR)
    
    #------------------------------
    
    
    
    #------------------------------
    #code for total.
    
    #total - input token and output token
    total_input_tokens = complete_dataframe['llm_prompt_tokens'].sum()
    total_output_tokens = complete_dataframe['llm_response_tokens'].sum()
    #filtering data which has error or 'no sql'
    filtered_dataframe = complete_dataframe[complete_dataframe['seconds'] != 0]
    #total - average time
    total_avg_time = filtered_dataframe['seconds'].mean()
    #total - throughput
    filtered_dataframe['throughput_per_inference'] = filtered_dataframe['llm_response_tokens']/filtered_dataframe['seconds']
    total_throughput = filtered_dataframe['throughput_per_inference'].mean()
    #total - cost
    if environment=='self-hosted':
        # (total cost = total inference seconds / 3600 ) * 10$ per hour
        total_cost = (filtered_dataframe['seconds'].sum())/360
    else:
        total_input_token_cost = (total_input_tokens*input_cost)/(token_per)
        total_output_token_cost = (total_output_tokens*output_cost)/(token_per)
        total_cost = (total_input_token_cost + total_output_token_cost)
    
    #total - PSR
    total_PSR = (len(filtered_dataframe)/len(complete_dataframe))*100
    
    #adding the metrics data to template dict
    final_dict['avg_time'].append(total_avg_time) 
    final_dict['total_cost'].append(total_cost)
    final_dict['total_output_tokens'].append(total_output_tokens)
    final_dict['total_input_tokens'].append(total_input_tokens)
    final_dict['throughput'].append(total_throughput)
    final_dict['PSR'].append(total_PSR)
    
    #------------------------------
    
    for i in range(4):
        temp_row=([environment,model_name,inst,dataset_size,'total_cost($)',final_dict['hardness'][i],final_dict['total_cost'][i]])
        entire_df.loc[len(entire_df)] = temp_row
        temp_row=([environment,model_name,inst,dataset_size,'total_output_tokens',final_dict['hardness'][i],final_dict['total_output_tokens'][i]])
        entire_df.loc[len(entire_df)] = temp_row
        temp_row=([environment,model_name,inst,dataset_size,'total_input_tokens',final_dict['hardness'][i],final_dict['total_input_tokens'][i]])
        entire_df.loc[len(entire_df)] = temp_row
        temp_row=([environment,model_name,inst,dataset_size,'throughput(tok/sec)',final_dict['hardness'][i],final_dict['throughput'][i]])
        entire_df.loc[len(entire_df)] = temp_row
        temp_row=([environment,model_name,inst,dataset_size,'PSR',final_dict['hardness'][i],final_dict['PSR'][i]])
        entire_df.loc[len(entire_df)] = temp_row
        temp_row=([environment,model_name,inst,dataset_size,'latency',final_dict['hardness'][i],final_dict['avg_time'][i]])
        entire_df.loc[len(entire_df)] = temp_row 
    
def find_metric_csv(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item).replace(os.sep, '/')
            folder_name = os.path.basename(os.path.dirname(item_path))
            if os.path.isdir(item_path):
                find_metric_csv(item_path)
            elif item == 'metrics.csv':
                df_metric = pd.read_csv(f'{item_path}')
                if folder_name=='360_Inferences':
                    environment= directory.split('/')[1]
                    model_name = item_path.split('/')[-4]
                    inst=directory.split('/')[-2].strip()
                    inst=inst.split('_')[0]
                    dataset_size=directory.split('/')[-1].strip()
                    dataset_size=dataset_size.split('_')[0]
                else:
                    environment= directory.split('/')[1]
                    model_name = item_path.split('/')[-5]
                    inst=directory.split('/')[-3].strip()
                    inst=inst.split('_')[0]
                    dataset_size=directory.split('/')[-1].strip()
                    dataset_size=dataset_size.split('_')[0]
                get_processed_df(df_metric,model_name,environment,inst,dataset_size)
                
                
find_metric_csv(directory_to_search)

entire_df['model'] = entire_df['model'].map(mapping)
entire_df.to_csv('metrics_consolidated_for_all_buckets.csv',index=False)
df_met=pd.read_csv('metrics_consolidated_for_all_buckets.csv')


df_met=pd.read_csv('metrics_consolidated_for_all_buckets.csv')
df_met['value'] = df_met['value'].round(3)
df_met = df_met[(df_met['dataset_size'] == 360) & (df_met['metric'] != 'total_cost($)') & (df_met['metric'] != 'PSR') & (df_met['metric'] != 'latency') ]


# Pivot the DataFrame
pivot_df = df_met.pivot_table(index=['model', 'metric','environment'], columns=['instruction', 'hardness'], values='value')

# Save the DataFrame to an Excel file
pivot_df.T.to_excel("metrics_all_hardness.xlsx")


total_cost_sum = pd.DataFrame(columns=['environment', 'model', 'instruction', 'dataset_size', 'metric',
       'hardness', 'value'])

df_met=pd.read_csv('metrics_consolidated_for_all_buckets.csv')
df_met['value'] = df_met['value'].round(3)

df_met = df_met[(df_met['dataset_size'] == 360) & (df_met['hardness'] == 'total') & (df_met['metric'] != 'throughput(tok/sec)') & (df_met['metric'] != 'PSR') & (df_met['metric'] != 'latency') ]

inst_list = df_met["instruction"].unique()
env_list = df_met["environment"].unique()
metric_list = df_met['metric'].unique()
models = {
    env_list[0]: df_met[df_met["environment"] == env_list[0]]["model"].unique(),
    env_list[1]: df_met[df_met["environment"] == env_list[1]]["model"].unique(),
    env_list[2]: df_met[df_met["environment"] == env_list[2]]["model"].unique(),
    env_list[3]: df_met[df_met["environment"] == env_list[3]]["model"].unique(),
    env_list[4]: df_met[df_met["environment"] == env_list[4]]["model"].unique(),
    env_list[5]: df_met[df_met["environment"] == env_list[5]]["model"].unique(),
}
for env in env_list:
    env_df= df_met[df_met['environment']==env]
    temp_models_lis= models[env]
    for mod in temp_models_lis:
        model_df= env_df[env_df['model']==mod]
        for metric in metric_list:
#             print(model_df)
            metric_df= model_df[model_df['metric']==metric]
            total_metric = metric_df['value'].sum()
            appending_list = [env,mod,'0-11',360,metric,'total',total_metric]
            new_row = pd.DataFrame([appending_list], columns=metric_df.columns)
            total_cost_sum = pd.concat([total_cost_sum, new_row], ignore_index=True)
                      
            
    
# Pivot the DataFrame
pivot_df = total_cost_sum.pivot_table(index=['model', 'metric','environment'], columns=['instruction', 'hardness'], values='value')

# Save the DataFrame to an Excel file
pivot_df.T.to_excel("cost_total_hardness.xlsx")


import pandas as pd

df_met=pd.read_csv('metrics_consolidated_for_all_buckets.csv')
df_met['value'] = df_met['value'].round(3)
df_met = df_met[(df_met['dataset_size'] == 360) & (df_met['hardness'] == 'total') & (df_met['metric'] != 'throughput(tok/sec)') & (df_met['metric'] != 'PSR') & (df_met['metric'] != 'latency') ]

# Pivot the DataFrame
pivot_df = df_met.pivot_table(index=['model', 'metric','environment'], columns=['instruction', 'hardness'], values='value')

# Save the DataFrame to an Excel file
pivot_df.T.to_excel("cost_all_hardness.xlsx")