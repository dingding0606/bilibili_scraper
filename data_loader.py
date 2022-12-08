import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import csv

def excitedness(title):
    '''
    Helper method for evaluating how exciting the title of the video is.
    Return: the number of ? and ! in the title.
    '''
    return title.count('!') + title.count('?') + title.count('！') + title.count('？')

def memeness(title):
    '''
    Helper method for evaluating how meme-like the title of the video is.
    Return: the number of empty spaces in the title.
    '''
    return title.count(' ')

def load_bilibili_data(standardize=True):
    '''
    Helper method for loading Bilibili video data
    '''

    all_variables = []

    data = pd.read_csv("chosen_dataset/DATA_DEC6_CSV.csv")

    # copyright: 1 = original videos, 2 = reposted videos
    data['copyright_original'] = data.apply(lambda row: 1 if (row.copyright == 1) else 0, axis=1)

    # is_story: convert from boolean to 0 or 1
    data['is_story'] = data.apply(lambda row: 1 if (row.is_story == True) else 0, axis=1)

    # dim_width & dim_height & dim_rotate:
    # exchange dim_width and dim_height when dim_rotate is 1.
    data['dim_is_horizontal'] = data.apply(lambda row: 1 if (row.dim_width > row.dim_height and row.dim_rotate == 0) or (row.dim_width <= row.dim_height and row.dim_rotate == 1) else 0, axis=1)

    # title:
    data['title_length'] = data.apply(lambda row: len(str(row.title)), axis=1)
    data['title_excitedness'] = data.apply(lambda row: excitedness(str(row.title)), axis=1)
    data['title_memeness'] = data.apply(lambda row: memeness(str(row.title)), axis=1)

    # desc:
    # add in the future.

    # category: convert tid into broader categories in the future

    # total_tag_subscriber_count:
    # calculate the average tag subscriber stats
    data['avg_tag_subscriber_count'] = data.apply(lambda row: float(row.total_tag_subscriber_count) / float(row.num_of_tags), axis=1)
    data['avg_tag_archive_count'] = data.apply(lambda row: float(row.total_tag_archive_count) / float(row.num_of_tags), axis=1)

    # up_sex: Male, Female, Hidden; one hot encoding used here.
    data['up_sex_is_male'] = data.apply(lambda row: 1 if row.up_sex == "男" else 0, axis=1)
    data['up_sex_is_female'] = data.apply(lambda row: 1 if row.up_sex == "女" else 0, axis=1)
    data['up_sex_is_hidden'] = data.apply(lambda row: 1 if row.up_sex == "保密" else 0, axis=1)

    # up_is_official: 0 = official (verified by the platform), -1 = unofficial (not yet verified)
    # converted into an indicator variable where up_is_official = 1 means YES and 0 means NO.
    data['up_is_official'] = data.apply(lambda row: 1 if row.up_is_official == 0 else 0, axis=1)

    relevant_variables = set(["bv", "copyright_original", "dim_is_horizontal",
                                "duration", "is_story", "no_reprint",
                                "hd5", "view", "coin", "danmu", "favorite", "like",
                                "share", "reply", "videos", "num_of_tags",
                                "up_archive_count", "up_follower",
                                "up_like_num", "up_following", "up_is_official",
                                "up_level", "title_length", "title_excitedness",
                                "title_memeness", "avg_tag_subscriber_count",
                                "avg_tag_archive_count", "up_sex_is_male",
                                "up_sex_is_female", "up_sex_is_hidden"])
                                # "up_article_count" is dropped because all are zero in my dataset

    # drop irrelevant features and only keep relevant_variables
    data.drop(columns=set(data.columns) - relevant_variables, inplace=True)

    # Get response values from file
    response = pd.read_csv("chosen_dataset/DATA_THREE_DAY_CSV.csv")
    response.drop(columns=set(response.columns) - set(["bv", "view"]), inplace=True)
    response.rename(columns = {'view':'final_view'}, inplace = True) # rename the column so that easier for merging

    # Merge response into data by matching bv; mode of merge: "inner"
    data = pd.merge(data, response, on="bv")

    data_clean = data.drop(columns=["bv"])

    # Train, Test, Validation Split
    Xmat = data_clean.drop(columns=["final_view"]) #.to_numpy()
    Y = data_clean["final_view"] #.to_numpy()
    Xmat_train_and_val, Xmat_test, Y_train_and_val, Y_test = train_test_split(Xmat, Y, test_size=0.2, random_state=42)
    Xmat_train, Xmat_val, Y_train, Y_val = train_test_split(Xmat_train_and_val, Y_train_and_val, test_size=0.2, random_state=42)

    n, d = Xmat_train.shape

    no_need_to_standardize = ["is_story", "no_reprint", "hd5", "up_is_official",
                                "copyright_original", "dim_is_horizontal",
                                "up_sex_is_male", "up_sex_is_female", "up_sex_is_hidden"]

    # if degree_two_poly_terms:
    #     for i in range(0, len(Xmat_train.columns)):
    #         for j in range(i, len(Xmat_train.columns)):
    #             feature_i = Xmat_train.columns[i]
    #             feature_j = Xmat_train.columns[j]
    #
    #             # add interaction terms
    #             inter_train = pd.DataFrame(Xmat_train[feature_i] * Xmat_train[feature_j], columns=[feature_i + ":" + feature_j])
    #             Xmat_train = pd.concat([Xmat_train, inter_train], axis=1)
                # Xmat_train[feature_i + ":" + feature_j] = Xmat_train[feature_i] * Xmat_train[feature_j]
                # Xmat_train_and_val[feature_i + ":" + feature_j] = Xmat_train_and_val[feature_i] * Xmat_train_and_val[feature_j]
                # Xmat_val[feature_i + ":" + feature_j] = Xmat_val[feature_i] * Xmat_val[feature_j]
                # Xmat_test[feature_i + ":" + feature_j] = Xmat_test[feature_i] * Xmat_test[feature_j]

    if standardize:
        # Standardized the dataset using mean and std from training set
        for (colname, colval) in Xmat_train.items():

            if colname not in no_need_to_standardize:
                mean = colval.mean()
                std = colval.std()

                Xmat_train[colname] = (Xmat_train[colname] - mean) / std
                Xmat_train_and_val[colname] = (Xmat_train_and_val[colname] - mean) / std
                Xmat_val[colname] = (Xmat_val[colname] - mean) / std
                Xmat_test[colname] = (Xmat_test[colname] - mean) / std

    return Xmat_train_and_val, Y_train_and_val, Xmat_train, Xmat_val, Xmat_test, Y_train, Y_val, Y_test


load_bilibili_data()
