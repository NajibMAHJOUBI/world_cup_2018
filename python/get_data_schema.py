# -*- coding: utf-8 -*-
"""
 Get the schema for the dataste used in the world cup prediction project 
"""

from pyspark.sql.types import StructType, StructField, StringType, FloatType, IntegerType

def get_data_schema(data_schema):
    if data_schema == "qualifying_start":
        return StructType([
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
		  StructField("goalsGroup_against", StringType(), True)])
    elif data_schema == "qualifying_results":
        return StructType([
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

