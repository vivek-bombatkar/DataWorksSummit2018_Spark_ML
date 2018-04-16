// Databricks notebook source
// MAGIC %md # RandomForestClassification and spark pipeline

// COMMAND ----------

// Source URL
val fileURL = "https://www.dropbox.com/s/dwbe77flspe480g/mushrooms_ix.csv?raw=1"
// drop file to ensure idempotency
val result = dbutils.fs.rm("/tmp/mushrooms_ix.csv")
println(result)
// read data from Source URL
dbutils.fs.put("/tmp/mushrooms_ix.csv", scala.io.Source.fromURL(fileURL).mkString)

// COMMAND ----------

import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions._
import org.apache.spark.sql.hive
import org.apache.spark.ml.{Pipeline, PipelineStage, PipelineModel, Transformer}
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.{MulticlassClassificationEvaluator}
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer, VectorAssembler, PCA, VectorSlicer}
import org.apache.spark.ml.linalg.{Vector, DenseVector}
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator, CrossValidatorModel}

// COMMAND ----------

// load csv data as a DataFrame
var df = spark.read.format("csv")
           .option("header", "true")
           .option("inferSchema", "true")
           .load("dbfs:/tmp/mushrooms_ix.csv")

// cache DataFrame In-Memory
df.cache()
println(df.count())

df.printSchema()

// COMMAND ----------

// Count Distinct for all columns (hard-coded)
display(
  df.agg(
    countDistinct("class").alias("class"),
    countDistinct("cap-shape").alias("cap-shape"),
    countDistinct("cap-color").alias("cap-color"),
    countDistinct("bruises").alias("bruises"),
    countDistinct("gill-color").alias("gill-color"),
    countDistinct("veil-type").alias("veil-type"),
    countDistinct("ring-number").alias("ring-number"),
    countDistinct("spore-color").alias("spore-color")
  )
)

// COMMAND ----------

display(
  df.groupBy("bruises").count()
)

// COMMAND ----------

// Null-Value substitution in column 'bruises'
// org.apache.spark.sql.DataFrameNaFunctions
df = df.na.fill(Map("bruises" -> "no"))
df.select("bruises").distinct().show()

// COMMAND ----------

// drop rows containing columns with Null-Values
println(df.count())
df = df.na.drop()
df.count()

// COMMAND ----------

// drop veil-type
df.select("veil-type").distinct.show
df = df.drop("veil-type")

// COMMAND ----------

// Separate label "class" and the features
var label = "class"
var features = for (col <- df.columns if (col != label)) yield col

// Indexer for label
var labelIndexer = new StringIndexer()
                         .setInputCol(label)
                         .setOutputCol("i_"+label)

// Indexers for the feature columns
var featureIndexers = Array[StringIndexer]()
for (f <- features)
    featureIndexers = featureIndexers :+ new StringIndexer()
                             .setInputCol(f)
                             .setOutputCol("i_"+f)
                             .setHandleInvalid("skip")

// Generate a feature vector as standard parameter for the Spark-supplied ML algorithms
var featureColumns = featureIndexers.map(f => f.getOutputCol)
var assembler = new VectorAssembler()
                      .setInputCols(featureColumns)
                      .setOutputCol("features")

// Automatically identify categorical features
// setMaxCategories: features with more than 12 distinct values are treated as continuous
var catVectorIndexer = new VectorIndexer()
                             .setInputCol(assembler.getOutputCol)
                             .setOutputCol("catFeatures")
                             .setMaxCategories(12)

// COMMAND ----------

// Create a Rondom Forest Classifier-Object
var rfClassifier = new RandomForestClassifier()
                         .setLabelCol(labelIndexer.getOutputCol)
                         .setFeaturesCol(catVectorIndexer.getOutputCol)
                         .setPredictionCol("predictedIndex")

// Mapping of indexes for the label to the original values
var labels = labelIndexer.fit(df).labels

// Back conversion of values for the predicted label
var labelConverter = new IndexToString()
                           .setInputCol(rfClassifier.getPredictionCol)
                           .setOutputCol("predictedLabel")
                           .setLabels(labels)

// COMMAND ----------

// Creation of a Machine Learning Pipleine for Mushroom classification
var pipeline = new Pipeline().setStages(
                                Array(labelIndexer) ++
                                featureIndexers :+
                                assembler :+
                                catVectorIndexer :+
                                rfClassifier :+
                                labelConverter)

// Split of original dataset into a training- and test data set
// 70% training- and 30% test data
var Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))

// Creation of a PipelineModel based on the training data set
var model = pipeline.fit(trainingData)

// Application of the created PipelineModel on the test data set 
var predictions = model.transform(testData)

// COMMAND ----------

// Show erroneous classified test data
display(
  predictions.select("class", "predictedLabel", "catfeatures", "cap-shape", "cap-color", "bruises")
             .where("class <> predictedLabel")
)

// COMMAND ----------

// Evaluation of the PipelineModel
var evaluator = new MulticlassClassificationEvaluator()
                      .setLabelCol(labelIndexer.getOutputCol)
                      .setPredictionCol(rfClassifier.getPredictionCol)
                      .setMetricName("accuracy")

var accuracy = evaluator.evaluate(predictions)

println(f"Accuracy is $accuracy%.3f")

// COMMAND ----------

// Accuracy of PipelineModel
display(
predictions.withColumn("classified", when(predictions("class") =!= predictions("predictedLabel"), "wrong").otherwise("right")).groupBy("classified").count()
)

// COMMAND ----------

// register predictions DataFrame as TempView to be referenced with Spark SQL
predictions.createOrReplaceTempView("predictions_tab")

// COMMAND ----------

// MAGIC %sql
// MAGIC select 
// MAGIC   case when class = 'poisonous' and predictedLabel = 'edible' then 'deadly' else 'not deadly' end as impact,
// MAGIC   count(*) cnt
// MAGIC from predictions_tab
// MAGIC group by 
// MAGIC   case when class = 'poisonous' and predictedLabel = 'edible' then 'deadly' else 'not deadly' end

// COMMAND ----------


