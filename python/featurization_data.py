# -*- coding: utf-8 -*-

import os
import pandas as pd
from get_data_names import get_data_names
import sys
sys.path.append("./pyspark")
from get_competition_dates import get_competition_dates


def win_team_1(score_team_1, score_team_2):
    if score_team_1 > score_team_2:
        return 2.0
    elif score_team_1 < score_team_2:
        return 1.0
    else:
        return 0.0


class FeaturizationData:

    features_names = {"matchesGroup_home": "home", "matchesGroup_away": "away",
                      "matchesGroup_neutral": "neutral", "matchesGroup_wins": "wins",
                      "matchesGroup_losses": "losses", "matchesGroup_draws": "draws",
                      "goalsGroup_for": "for", "goalsGroup_against": "against"}

    def __init__(self, year, confederations, path_training, stage=None):
        self.year = year
        self.confederations = confederations
        self.path_training = path_training
        self.stage = stage

        self.dic_data = None
        self.data_union = None
        self.start_date = None
        self.end_date = None

    def __str__(self):
        pass

    def run(self):
        self.loop_all_confederations()
        self.union_all_confederation()
        self.save_training()

    def get_data_union(self):
        return self.data_union

    def set_dates(self):
        if self.stage is not None:
            self.start_date = get_competition_dates(self.year)[self.stage][0]
            self.end_date = get_competition_dates(self.year)[self.stage][1]

    def get_dates(self):
        return self.start_date, self.end_date

    def load_start_data(self, confederation):
        path = "./data/{0}/{1}_World_Cup_{2}_qualifying_start.tsv".format(confederation, self.year, confederation)
        data = pd.read_csv(path, sep='\t', header=None, names=get_data_names("qualifying_start"))
        for key, value in self.features_names.iteritems():
            data[value] = data.apply(lambda row: row[key] / float(row["matchesGroup_total"]), axis=1)
        data = data.rename(columns={'teamGroup_team': 'team'})
        return data[["team"]+self.features_names.values()]

    def load_results_data(self, confederation):
        path = "./data/{0}/{1}_World_Cup_{2}_qualifying_results.tsv".format(confederation, self.year, confederation)
        data = pd.read_csv(path, sep="\t", names=get_data_names("qualifying_results"), header=None)
        data["label"] = data.apply(lambda row: win_team_1(int(row["score_team_1"]), int(row["score_team_2"])), axis=1)
        data["diff_points"] = data.apply(lambda row: float(row["score_team_1"]) - float(row["score_team_2"]), axis=1)
        data["new_date"] = data.apply(lambda row: (str(row["year"]) + "/" +
                                                   str(row["month"]).zfill(2) + "/" +
                                                   str(row["date"]).zfill(2)),
                                      axis=1)
        data = data[["team_1", "team_2", "label", "diff_points", "new_date"]].rename(columns={'new_date': 'date'})
        if (self.start_date is not None) and (self.end_date is not None):
            data = data[(data.date >= self.start_date) & (data.date <= self.end_date)]
        return data

    def compute_data_confederation(self, confederation):
        df_qualifying_results = self.load_results_data(confederation)
        df_qualifying_start = self.load_start_data(confederation)

        df_qualifying_results = (df_qualifying_results
                                 .merge(df_qualifying_start, left_on=["team_1"], right_on=["team"])
                                 .rename(columns={feature: feature+"_1" for feature in self.features_names.values()})
                                 .drop(["team"], axis=1)
                                 .merge(df_qualifying_start, left_on=["team_2"], right_on=["team"])
                                 .rename(columns={feature: feature+"_2" for feature in self.features_names.values()})
                                 .drop(["team"], axis=1))
        for feature in self.features_names.values():
            df_qualifying_results[feature] = df_qualifying_results.apply(lambda x: x[feature+"_1"] - x[feature+"_2"],
                                                                         axis=1)

        df_qualifying_results["matches"] = df_qualifying_results.apply(
            lambda x: str(x["team_1"]) + "/" + str(x["team_2"]) + "_" + str(x["date"]), axis=1)

        return df_qualifying_results[["matches", "label", "diff_points"] + self.features_names.values()]

    def loop_all_confederations(self):
        self.dic_data = {confederation: self.compute_data_confederation(confederation)
                         for confederation in self.confederations}

    def union_all_confederation(self):
        self.data_union = pd.concat(self.dic_data.values())

    def save_training(self):
        if not os.path.isdir(self.path_training):
            os.makedirs(self.path_training)
        if os.path.isfile(os.path.join(self.path_training, str(self.year)+'.csv')):
            os.remove(os.path.join(self.path_training, str(self.year)+'.csv'))
        self.data_union.to_csv(os.path.join(self.path_training, str(self.year)+'.csv'), sep=',', header=True, index=False)


if __name__ == "__main__":
    confederations = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "playoffs", "UEFA", "WCP"]

    for year in ["2018", "2014", "2010", "2006"]:
        print("Year: {0}".format(year))
        featurization_data = FeaturizationData(year, confederations, "./test/sklearn/training")
        featurization_data.run()
