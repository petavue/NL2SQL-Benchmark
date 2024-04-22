import pandas as pd
import os
import sqlite3
import argparse

def get_schema(db_id, data_folder):
    initial_dir = os.getcwd()
    try:
        if os.path.isdir(os.path.join(data_folder, db_id)):
            os.chdir(os.path.join(data_folder, db_id))
            if os.path.isfile(f'./{db_id}.sqlite'):
                sqlite_file = f'./{db_id}.sqlite'
                conn = sqlite3.connect(sqlite_file)
                cur = conn.cursor()
                cur.execute("SELECT name, sql FROM sqlite_master WHERE type='table';")
                tables = cur.fetchall()
                final_schema = ''
                for name, sql in tables:
                    final_schema += (f"{sql}")
                conn.close()
                return final_schema
            else:
                return "No sqlite file found"
        else:
            return "No database found"
    except Exception as e:
        print(e)
    finally:
        os.chdir(initial_dir)

def generate_subsets(df,dataset_size):
    unique_hardness_values = df['difficulty'].unique()
    max_split=dataset_size//len(unique_hardness_values)
    combined_subset = pd.DataFrame()
    for hardness_value in unique_hardness_values:
        subset_df = df[df['difficulty'] == hardness_value].head(max_split)
        combined_subset = pd.concat([combined_subset, subset_df])
    return combined_subset

def main(json_file,data_folder,dataset_size):
    df = pd.read_json(json_file)
    df['schema'] = df['db_id'].apply(lambda x: get_schema(str(x), data_folder))
    
    df_simple=df[df['difficulty']=='simple']
    df_moderate=df[df['difficulty']=='moderate']
    df_challenging=df[df['difficulty']=='challenging']

    df_simple.to_csv(f"./bird_processed_df_simple.csv")
    df_moderate.to_csv(f"./bird_processed_df_moderate.csv")
    df_challenging.to_csv(f"./bird_processed_df_challenging.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process JSON file to generate CSV with schema information.")
    parser.add_argument("json_file", type=str, help="Path to the JSON file")
    parser.add_argument("--data_folder", type=str, help="Name of the data folder")
    parser.add_argument("--dataset_size", type=int,default=360 ,help="Dataset Size")
    args = parser.parse_args()
    
    main(args.json_file, args.data_folder, args.dataset_size)
    
#python bird_data/bird_data_extraction.py dev_databases/dev.json --data_folder dev_databases/ --dataset_size 360