from pyspark import SparkContext
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler 
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
sc =SparkContext()

def load_data():
    spark=SparkSession.builder.appName('forest').getOrCreate()      
    forest=spark.read.csv("hdfs://namenode//covtype.csv",header=True,inferSchema=True)
    forest_df=forest.withColumn("Cover_Type",col("Cover_Type")-1)# label start from 0
    type0_df=forest_df.filter(forest_df.Cover_Type==0).limit(20000)
    type1_df=forest_df.filter(forest_df.Cover_Type==1).limit(20000)
    type2_df=forest_df.filter(forest_df.Cover_Type==2).limit(20000)
    type3_df=forest_df.filter(forest_df.Cover_Type==3)
    type4_df=forest_df.filter(forest_df.Cover_Type==4)
    type5_df=forest_df.filter(forest_df.Cover_Type==5)
    type6_df=forest_df.filter(forest_df.Cover_Type==6)
    print(type0_df.count()," ",type1_df.count()," ",
          type2_df.count()," ",type3_df.count()," ",
          type4_df.count()," ",type5_df.count()," ",type6_df.count())
    return type0_df, type1_df, type2_df, type3_df, type4_df, type5_df, type6_df

def load_training_and_validation_data(type0_df, type1_df, type2_df, type3_df, type4_df, type5_df, type6_df):
    new_df_1=type0_df.union(type1_df).union(type2_df).cache()
    new_df_2=new_df_1.union(type3_df).union(type4_df).cache()
    new_df_3=new_df_2.union(type5_df).cache()
    new_df_4=new_df_3.union(type6_df).cache()
    print("total dataset number: ",new_df_4.count())
    # disrupt pyspark.sql.functions.rand generate[0.0, 1.0] double random number
    new_df_5 = new_df_4.withColumn('rand', rand(seed=42))
    # order by rander numbers
    new_df_6 = new_df_5.orderBy("rand")
    # drop the column with rand numbers
    new_df = new_df_6.drop("rand")

    divided_forest_rdd=new_df.rdd.randomSplit([0.7,0.3])# this will not change the order of data, need to use above method
    training_forest_data=divided_forest_rdd[0].toDF().cache()
    validation_forest_data=divided_forest_rdd[1].toDF().cache()
    training_rows_number=training_forest_data.count()
    validation_rows_number=validation_forest_data.count()
    print("training_rows_number: ",training_rows_number," validation_rows_number:",validation_rows_number)
    return training_forest_data, validation_forest_data

def load_data_ten_fold_cross_validation(feature_cols,type0_df, type1_df, type2_df, type3_df, type4_df, type5_df, type6_df):
    new_df_1=type0_df.union(type1_df).union(type2_df).cache()
    new_df_2=new_df_1.union(type3_df).union(type4_df).cache()
    new_df_3=new_df_2.union(type5_df).cache()
    new_df_4=new_df_3.union(type6_df).cache()
    print("total dataset number: ",new_df_4.count())
    # disrupt pyspark.sql.functions.rand generate[0.0, 1.0] double random number
    new_df_5 = new_df_4.withColumn('rand', rand(seed=42))
    # order by rander numbers
    new_df_6 = new_df_5.orderBy("rand")
    # drop the column with rand numbers
    new_df_7 = new_df_6.drop("rand")
    
    assembler=VectorAssembler(inputCols=feature_cols,outputCol="features")
    new_df=assembler.transform(new_df_7).select(col("features"),col("Cover_Type").alias("label"))
    
    divided_forest_rdd=new_df.rdd.randomSplit([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1])# this will not change the order of data, need to use above method
    divided_forest_df_list=[]
    for i in divided_forest_rdd:
        divided_forest_df_list.append(i.toDF())
    print(len(divided_forest_df_list))
    return divided_forest_df_list

def get_empty_df(my_schema):
    spark=SparkSession.builder.appName('empty').getOrCreate()  
    empty_df = spark.sparkContext.parallelize([]).toDF(my_schema)
    return empty_df

def transform_data(training_forest_data, validation_forest_data,feature_cols):
    assembler=VectorAssembler(inputCols=feature_cols,outputCol="features")
    training_data_final=assembler.transform(training_forest_data).select(col("features"),col("Cover_Type").alias("label"))
    print("final training data:")
    training_data_final.show(truncate=False, n=15)
    validation_data_final=assembler.transform(validation_forest_data).select(col("features"),col("Cover_Type").alias("label"))
    print("final validation data:")
    validation_data_final.show(truncate=False, n=15)
    return training_data_final,validation_data_final


def get_features_title():
    feature_cols=["Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Wilderness_Area1",
    "Wilderness_Area2",
    "Wilderness_Area3",
    "Wilderness_Area4",
    "Soil_Type1",
    "Soil_Type2",
    "Soil_Type3",
    "Soil_Type4",
    "Soil_Type5",
    "Soil_Type6",
    "Soil_Type7",
    "Soil_Type8",
    "Soil_Type9",
    "Soil_Type10",
    "Soil_Type11",
    "Soil_Type12",
    "Soil_Type13",
    "Soil_Type14",
    "Soil_Type15",
    "Soil_Type16",
    "Soil_Type17",
    "Soil_Type18",
    "Soil_Type19",
    "Soil_Type20",
    "Soil_Type21",
    "Soil_Type22",
    "Soil_Type23",
    "Soil_Type24",
    "Soil_Type25",
    "Soil_Type26",
    "Soil_Type27",
    "Soil_Type28",
    "Soil_Type29",
    "Soil_Type30",
    "Soil_Type31",
    "Soil_Type32",
    "Soil_Type33",
    "Soil_Type34",
    "Soil_Type35",
    "Soil_Type36",
    "Soil_Type37",
    "Soil_Type38",
    "Soil_Type39",
    "Soil_Type40"]
    return feature_cols