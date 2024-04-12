import pandas as pd

context_db_avoid_list = ["-- phpMyAdmin SQL Dump"]
hardness_csv_list = [
    "bird_processed_df_simple.csv",
    "bird_processed_df_moderate.csv",
    "bird_processed_df_challenging.csv"
]

df = pd.read_csv('bird_processed_df_1534.csv')
df_simple=df[df['difficulty']=='simple']
df_moderate=df[df['difficulty']=='moderate']
df_challenging=df[df['difficulty']=='challenging']

df_simple.to_csv(f"bird_processed_df_simple.csv")
df_moderate.to_csv(f"bird_processed_df_moderate.csv")
df_challenging.to_csv(f"bird_processed_df_challenging.csv")

df_size = [360]
unique_hardness_value=3

def split_equal_df(n, hardness_df_list,unique_hardness_value):
    split = int(n / unique_hardness_value)
    temp_df = pd.DataFrame()
    temp_df = []
    for i in range(len(hardness_df_list)):
        sample_df=hardness_df_list[i].sample(n=split)
        temp_df.append(sample_df)
    final_df = pd.concat(temp_df, ignore_index=True)
    final_df.rename(
        columns={"Unnamed: 0": "index_in_original", "SQL": "sql_query"}, inplace=True
    )
    correct_order = [
        "index_in_original",
        "difficulty",
        "db_id",
        "sql_query",
        "question",
        "schema",
        "evidence"
    ]
    final_df = final_df[correct_order]
    final_df.to_csv(f"./bird_equal_split_{len(final_df)}.csv")
    return final_df


hardness_df_list = []
for hardness_csv in hardness_csv_list:
    df = pd.read_csv(f"{hardness_csv}")
    mask = []
    for faulty_context in context_db_avoid_list:
        mask.append(df["schema"].str.contains(faulty_context))
    mask = pd.concat(mask, ignore_index=True)
    filtered_df = df[~mask]
    hardness_df_list.append(filtered_df)

split_equal_df(360, hardness_df_list,unique_hardness_value)
