#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
import sys
import numpy as np
import pandas as pd


# In[4]:


from pyspark.sql import SparkSession


# In[5]:


from pyspark.sql.types import IntegerType, DoubleType


# In[6]:


from pyspark.sql.functions import col, desc
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


# In[7]:


spark = SparkSession.builder.appName("modtrain").getOrCreate()


# In[8]:


spark


# In[11]:


print("Reading data from {}...".format("TrainingDataset.csv"))
training = spark.read.format("csv").load("TrainingDataset.csv", header=True, sep=";")


# In[12]:


training = training.toDF("fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar", "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density", "pH", "sulphates", "alcohol", "label")


# In[13]:


training = training \
        .withColumn("fixed_acidity", col("fixed_acidity").cast(DoubleType())) \
        .withColumn("volatile_acidity", col("volatile_acidity").cast(DoubleType())) \
        .withColumn("citric_acid", col("citric_acid").cast(DoubleType())) \
        .withColumn("residual_sugar", col("residual_sugar").cast(DoubleType())) \
        .withColumn("chlorides", col("chlorides").cast(DoubleType())) \
        .withColumn("free_sulfur_dioxide", col("free_sulfur_dioxide").cast(IntegerType())) \
        .withColumn("total_sulfur_dioxide", col("total_sulfur_dioxide").cast(IntegerType())) \
        .withColumn("density", col("density").cast(DoubleType())) \
        .withColumn("pH", col("pH").cast(DoubleType())) \
        .withColumn("sulphates", col("sulphates").cast(DoubleType())) \
        .withColumn("alcohol", col("alcohol").cast(DoubleType())) \
        .withColumn("label", col("label").cast(IntegerType()))


# In[14]:


features = training.columns


# In[15]:


features = features[:-1]


# In[16]:


va = VectorAssembler(inputCols=features, outputCol="features")


# In[17]:


va_df = va.transform(training)


# In[18]:


va_df = va_df.select(["features", "label"])


# In[19]:


training = va_df


# In[20]:


layers = [11, 8, 8, 8, 8, 10]


# In[21]:


from pyspark.ml.classification import DecisionTreeClassifier


# In[22]:


dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth =2)
dtModel = dt.fit(training)


# In[23]:


print("Saving file to {}...".format("dsp.csv"))


# In[24]:


dtModel.write().overwrite().save("dsp.csv")


# In[ ]:




