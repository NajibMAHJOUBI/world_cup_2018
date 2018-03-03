
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import col, udf
from pyspark.ml.linalg import Vectors, VectorUDT



def convert_string_to_float(x):
    x_replace_minus = x.replace(u'\u2212', '-')
    if x_replace_minus == '-':
        return np.nan
    else:
        return float(x_replace_minus)

class FeaturizationData:
    def __init__(self, spark_session, list_confederation):
        self.spark = spark_session 
        self.confederations = list_confederation
	self.schema_qualifying_start = StructType([
				    StructField("rankGroup_local", StringType(), True),
				    StructField("rankGroup_global", StringType(), True),
				    StructField("teamGroup_team", StringType(), True),
				    StructField("ratingGroup_rating", StringType(), True),
				    StructField("highestGroup_rank_max", StringType(), True),
				    StructField("highestGroup_rating_max", StringType(), True),
				    StructField("averageGroup_rank_avg", StringType(), True),
				    StructField("averageGroup_rating_avg", StringType(), True),
				    StructField("lowestGroup_rank_min", StringType(), True),
				    StructField("lowestGroup_rating_min", StringType(), True),
				    StructField("change3mGroup_rank_three_month_change", StringType(), True),
				    StructField("change3mGroup_rating_three_month_change", StringType(), True),
				    StructField("change6mGroup_rank_six_month_change", StringType(), True),
				    StructField("change6mGroup_rating_six_month_change", StringType(), True),
				    StructField("change1yGroup_rank_one_year_change", StringType(), True),
				    StructField("change1yGroup_rating_one_year_change", StringType(), True),
				    StructField("change2yGroup_rank_two_year_change", StringType(), True),
				    StructField("change2yGroup_rating_two_year_change", StringType(), True),
				    StructField("change5yGroup_rank_five_year_change", StringType(), True),
				    StructField("change5yGroup_rating_five_year_change", StringType(), True),
				    StructField("change10yGroup_rank_ten_year_change", StringType(), True),
				    StructField("change10yGroup_rating_ten_year_change", StringType(), True),
				    StructField("matchesGroup_total", StringType(), True),
				    StructField("matchesGroup_home", StringType(), True),
				    StructField("matchesGroup_away", StringType(), True),
				    StructField("matchesGroup_neutral", StringType(), True),
				    StructField("matchesGroup_wins", StringType(), True),
				    StructField("matchesGroup_losses", StringType(), True),
				    StructField("matchesGroup_draws", StringType(), True),
				    StructField("goalsGroup_for", StringType(), True),
				    StructField("goalsGroup_against", StringType(), True)
	])
        self.names_start_to_convert = self.schema_qualifying_start.names
        self.names_start_to_convert.remove("teamGroup_team")

	self.schema_qualifying_results = StructType([
	    StructField("year", StringType(), True),
	    StructField("month", StringType(), True),
	    StructField("date", StringType(), True),
	    StructField("team_1", StringType(), True),
	    StructField("team_2", StringType(), True),
	    StructField("score_team_1", IntegerType(), True),
	    StructField("score_team_2", IntegerType(), True),
	    StructField("tournament", StringType(), True),
	    StructField("country_played", StringType(), True),
	    StructField("rating_moved", StringType(), True),
	    StructField("rating_team_1", StringType(), True),
	    StructField("rating_team_2", StringType(), True),
	    StructField("rank_moved_team_1", StringType(), True),
	    StructField("rank_moved_team_2", StringType(), True),
	    StructField("rank_team_1", StringType(), True),
	    StructField("rank_team_2", StringType(), True)
	])

	self.names_results_to_convert = self.schema_qualifying_results.names
	self.names_results_to_remove = ["date",  "team_1", "team_2", "score_team_1", "score_team_2", "tournament", "country_played"]
	for name in self.names_results_to_remove: self.names_results_to_convert.remove(name)

    def __str__(self):
        s = "List of confederations: {0} \n".format(self.confederations)
        s += "Spark Session: {0}".format(self.spark)
        return s

    def run(self): 
        self.loop_all_confederations()
        self.union_all_confederation()

    def get_data_union(self):
        return self.data_union

    def get_qualifying_start_data(self, confederation):
        udf_get_percentage_game = udf(lambda x, y: x / y, FloatType())

	def convert_string_to_float(x):
	    x_replace_minus = x.replace(u'\u2212', '-')
	    if x_replace_minus == '-':
		return np.nan
	    else:
		return float(x_replace_minus)
        udf_convert_string_to_float = udf(lambda x: convert_string_to_float(x), FloatType())

        udf_create_features = udf(lambda s,t,u,v,w,x,y,z: Vectors.dense([s,t,u,v,w,x,y,z]), VectorUDT())

        path = "./data/{0}/2014_World_Cup_{1}_qualifying_start.tsv".format(confederation, confederation)
        return self.spark.read.csv(path, sep="\t", 
                                      schema=self.schema_qualifying_start, header=False)\
                                 .select([udf_convert_string_to_float(col(name)).alias(name) for name in self.names_start_to_convert] + ["teamGroup_team"])\
                                 .withColumn("features", udf_create_features(
                                             udf_get_percentage_game(col("matchesGroup_home"), col("matchesGroup_total")), 
                                             udf_get_percentage_game(col("matchesGroup_away"), col("matchesGroup_total")),
                                             udf_get_percentage_game(col("matchesGroup_neutral"), col("matchesGroup_total")),
                                             udf_get_percentage_game(col("matchesGroup_wins"), col("matchesGroup_total")), 
                                             udf_get_percentage_game(col("matchesGroup_losses"), col("matchesGroup_total")), 
                                             udf_get_percentage_game(col("matchesGroup_draws"), col("matchesGroup_total")),
                                             udf_get_percentage_game(col("goalsGroup_for"), col("matchesGroup_total")),  
                                             udf_get_percentage_game(col("goalsGroup_against"), col("matchesGroup_total"))))\
                                  .withColumnRenamed("teamGroup_team", "team")\
                                  .select("team", "features")

    def get_qualifying_results_data(self, confederation):
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
        path = "./data/{0}/2014_World_Cup_{1}_qualifying_results.tsv".format(confederation, confederation)
        return self.spark.read.csv(path, sep="\t", schema=self.schema_qualifying_results, header=False)\
                              .select([udf_convert_string_to_float(col(name)).alias(name) for name in self.names_results_to_convert] + self.names_results_to_remove)\
                              .select("team_1", "team_2", "score_team_1", "score_team_2")\
                              .withColumn("label", udf_win_team_1(col("score_team_1"), col("score_team_2")))\
                              .select("team_1", "team_2", "label")


    def get_data_confederation(self, confederation):
        udf_diff_features = udf(lambda features_1, features_2: features_1 - features_2, VectorUDT())
        df_qualifying_results = self.get_qualifying_results_data(confederation)
        df_qualifying_start = self.get_qualifying_start_data(confederation)
        return df_qualifying_results.join(df_qualifying_start, df_qualifying_results.team_1 == df_qualifying_start.team)\
                                 .withColumnRenamed("features", "features_1").drop("team")\
                                 .join(df_qualifying_start, df_qualifying_results.team_2 == df_qualifying_start.team)\
                                 .withColumnRenamed("features", "features_2").drop("team")\
                                 .withColumn("features", udf_diff_features(col("features_1"), col("features_2")))\
                                 .select("label", "features")


    def loop_all_confederations(self):
        self.dic_data = {}
        for confederation in self.confederations:
            self.dic_data[confederation] = self.get_data_confederation(confederation)

    def union_all_confederation(self):
        schema = StructType([
	    StructField("label", FloatType(), True),
	    StructField("features", VectorUDT(), True)])

        self.data_union = self.spark.createDataFrame(self.spark.sparkContext.emptyRDD(), schema)

        for tp in self.dic_data.iteritems(): self.data_union = self.data_union.union(tp[1])
        self.data_union.count()
        


