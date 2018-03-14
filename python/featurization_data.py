
# Pyspark libraries
import numpy as np
# Python libraries
import os
from pyspark.ml.feature import StringIndexer
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType, BooleanType
# My libraries
from get_data_schema import get_data_schema
from get_spark_session import get_spark_session


class FeaturizationData:
    def __init__(self, spark_session, year, list_confederation, path_training, path_string_indexer, stage=None,
                 list_date=None):
        self.spark = spark_session
        self.year = year
        self.confederations = list_confederation
        self.path_training = path_training
        self.path_string_indexer = path_string_indexer
        self.stage = stage
        self.list_date = list_date

        self.dic_data = {}
        self.data_union = None
        self.start_date = None
        self.end_date = None

    def __str__(self):
        s = "FeaturizationData class:\n"
        s += "List of confederations: {0} \n".format(self.confederations)
        s += "Spark Session: {0}".format(self.spark)
        return s

    def run(self):
        self.get_dates()
        self.loop_all_confederations()
        self.union_all_confederation()
        self.data_indexer()
        self.save_data()

    def get_dates(self):
        if self.list_date is not None:
            self.start_date = self.list_date[0]
            self.end_date = self.list_date[1]

    def get_data_union(self):
        return self.data_union

    def get_confederation(self):
        return self.confederations

    def save_data(self):
        return (self.data_union
                .coalesce(1)
                .write.mode('overwrite')
                .parquet(os.path.join(self.path_training, self.year)))

    def load_start_data(self, confederation):
        udf_get_percentage_game = udf(lambda x, y: x / y, FloatType())

        def convert_string_to_float(x):
            x_replace_minus = x.replace(u'\u2212', '-')
            if x_replace_minus == '-':
                return np.nan
            else:
                return float(x_replace_minus)

        udf_convert_string_to_float = udf(lambda x: convert_string_to_float(x), FloatType())
        udf_create_features = udf(lambda s, t, u, v, w, x, y, z: Vectors.dense([s, t, u, v, w, x, y, z]), VectorUDT())

        names_start_to_convert = get_data_schema("qualifying_start").names
        names_start_to_convert.remove("teamGroup_team")
        path = "./data/{0}/{1}_World_Cup_{2}_qualifying_start.tsv".format(confederation, self.year, confederation)
        return (self.spark.read.csv(path, sep="\t", schema=get_data_schema("qualifying_start"), header=False)
        .select([udf_convert_string_to_float(col(name)).alias(name) for name in names_start_to_convert] +
                ["teamGroup_team"])
        .withColumn("features", udf_create_features(
                                        udf_get_percentage_game(col("matchesGroup_home"), col("matchesGroup_total")),
                                        udf_get_percentage_game(col("matchesGroup_away"), col("matchesGroup_total")),
                                        udf_get_percentage_game(col("matchesGroup_neutral"), col("matchesGroup_total")),
                                        udf_get_percentage_game(col("matchesGroup_wins"), col("matchesGroup_total")),
                                        udf_get_percentage_game(col("matchesGroup_losses"), col("matchesGroup_total")),
                                        udf_get_percentage_game(col("matchesGroup_draws"), col("matchesGroup_total")),
                                        udf_get_percentage_game(col("goalsGroup_for"), col("matchesGroup_total")),
                                        udf_get_percentage_game(col("goalsGroup_against"), col("matchesGroup_total"))))
        .withColumnRenamed("teamGroup_team", "team")
        .select("team", "features"))

    def load_results_data(self, confederation):
        def win_team_1(score_team_1, score_team_2):
            if score_team_1 > score_team_2:
                return 2.0
            elif score_team_1 < score_team_2:
                return 1.0
            else:
                return 0.0

        udf_win_team_1 = udf(lambda team_1, team_2: win_team_1(team_1, team_2), FloatType())

        def convert_string_to_float(x):
            x_replace_minus = x.replace(u'\u2212', '-')
            if x_replace_minus == '-':
                return np.nan
            else:
                return float(x_replace_minus)

        udf_convert_string_to_float = udf(lambda x: convert_string_to_float(x), FloatType())

        udf_get_date = udf(lambda date, month, year:  str(year) + "/" + str(month) + "/" + str(date), StringType())

        names_results_to_convert = get_data_schema("qualifying_results").names
        names_results_to_remove = ["year", "month", "date",  "team_1", "team_2", "score_team_1", "score_team_2",
                                   "tournament", "country_played"]
        for name in names_results_to_remove: names_results_to_convert.remove(name)
        path = "./data/{0}/{1}_World_Cup_{2}_qualifying_results.tsv".format(confederation, self.year, confederation)
        data = self.spark.read.csv(path, sep="\t", schema=get_data_schema("qualifying_results"), header=False)\
                              .select([udf_convert_string_to_float(col(name)).alias(name)
                                       for name in names_results_to_convert] + names_results_to_remove)\
                              .withColumn("label", udf_win_team_1(col("score_team_1"), col("score_team_2")))\
                              .withColumn("new_date", udf_get_date(col("date"), col("month"), col("year")))\
                              .select(col("team_1"), col("team_2"), col("label"), col("new_date").alias("date"))

        if (self.start_date is not None) and (self.end_date is not None):
            def filter_date(date, start_date, end_date):
                if (date >= start_date) and (date <= end_date):
                    return True
                else:
                    return False
            start_date, end_date = self.start_date, self.end_date
            udf_filter_date = udf(lambda date: filter_date(date, start_date, end_date), BooleanType())
            return data.filter(udf_filter_date(col("date")))
        else:
            return data  

    def get_data_confederation(self, confederation):
        udf_diff_features = udf(lambda features_1, features_2: features_1 - features_2, VectorUDT())
        udf_team1_team2 = udf(lambda team_1, team_2, date: team_1 + "/" + team_2 + "_" + date, StringType())
        df_qualifying_results = self.load_results_data(confederation)
        df_qualifying_start = self.load_start_data(confederation)
        return (df_qualifying_results
                .join(df_qualifying_start, df_qualifying_results.team_1 == df_qualifying_start.team)
                .withColumnRenamed("features", "features_1").drop("team")
                .join(df_qualifying_start, df_qualifying_results.team_2 == df_qualifying_start.team)
                .withColumnRenamed("features", "features_2").drop("team")
                .withColumn("features", udf_diff_features(col("features_1"), col("features_2")))
                .select(col("label"), col("features"), udf_team1_team2(col("team_1"), col("team_2"),
                                                                       col("date")).alias("matches")))

    def loop_all_confederations(self):
        for confederation in self.confederations:
            self.dic_data[confederation] = self.get_data_confederation(confederation)

    def union_all_confederation(self):
        schema = StructType([
            StructField("label", FloatType(), True),
            StructField("features", VectorUDT(), True),
            StructField("matches", StringType(), True)])

        data_union_0 = self.spark.createDataFrame(self.spark.sparkContext.emptyRDD(), schema)

        for tp in self.dic_data.iteritems():
            data_union_0 = data_union_0.union(tp[1])
        data_union_0.count()
        self.data_union = data_union_0

    def data_indexer(self):
        string_indexer = StringIndexer(inputCol="matches", outputCol="id")
        model = string_indexer.fit(self.data_union)
        self.data_union = model.transform(self.data_union).drop("matches")
        model.write().overwrite().save(os.path.join(self.path_string_indexer, self.year))


if __name__ == "__main__":
    spark = get_spark_session("World_Cup")
    # years = ["2014", "2010", "2006"]
    # confederations = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "playoffs", "UEFA", "WCP"]
    years = ["2018"]
    confederations = ["AFC", "CAF", "CONCACAF", "CONMEBOL", "OFC", "playoffs", "UEFA"]
    for year in years:
        print("Year: {0}".format(year))
        featurization_data = FeaturizationData(spark, year, confederations, "./test/training", "./test/string_indexer")
        featurization_data.run()
