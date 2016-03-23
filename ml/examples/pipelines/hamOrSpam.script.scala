/**
  * Launch following commands:
  *    export MASTER="local-cluster[3,2,4096]"
  *   bin/sparkling-shell -i examples/scripts/mlconf_2015_hamSpam.script.script.scala
  *
  * When running using spark shell or using scala rest API:
  *    SQLContext is available as sqlContext
  *     - if you want to use sqlContext implicitly, you have to redefine it like: implicit val sqlContext = sqlContext,
  *      but better is to use it like this: implicit val sqlContext = SQLContext.getOrCreate(sc)
  *    SparkContext is available as sc
  */

import org.apache.spark.ml.feature._
import org.apache.spark.ml.h2o.H2ODeepLearning
import org.apache.spark.ml.h2o.feature.{DatasetSplitter, ColRemover}
import org.apache.spark.ml.{Pipeline, PipelineModel}

import org.apache.spark.sql.types.{StringType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.SparkFiles
import water.app.SparkContextSupport
import org.apache.spark.rdd.RDD


val smsDataFileName = "smsData.txt"
val smsDataFilePath = "examples/smalldata/"+smsDataFileName

// Register files to SparkContext
SparkContextSupport.addFiles(sc,smsDataFilePath)


def load(dataFile: String)(implicit sqlContext: SQLContext): DataFrame = {
  val smsSchema = StructType(Array(
    StructField("label", StringType, nullable = false),
    StructField("text", StringType, nullable = false)))
  val rowRDD = sc.textFile(SparkFiles.get(dataFile)).map(_.split("\t")).filter(r => !r(0).isEmpty).map(p => Row(p(0),p(1)))
  sqlContext.createDataFrame(rowRDD, smsSchema)
}

// Create SQL support
implicit val sqlContext = SQLContext.getOrCreate(sc)
import sqlContext.implicits._

// Start H2O services
import org.apache.spark.h2o._
implicit val h2oContext = H2OContext.getOrCreate(sc)
import h2oContext._

/**
 * Define the pipeline stages
 */
val tokenizer = new RegexTokenizer()
  .setInputCol("text")
  .setOutputCol("words")
  .setMinTokenLength(3)
  .setGaps(false)
  .setPattern("[a-zA-Z]")

val stopWordsRemover = new StopWordsRemover()
  .setInputCol(tokenizer.getOutputCol)
  .setOutputCol("filtered")
  .setStopWords(Array("the", "a", "", "in", "on", "at", "as", "not", "for"))
  .setCaseSensitive(false)

val hashingTF = new HashingTF()
  .setNumFeatures(1 << 10)
  .setInputCol(tokenizer.getOutputCol)
  .setOutputCol("wordToIndex")

val idf = new IDF()
  .setMinDocFreq(4)
  .setInputCol(hashingTF.getOutputCol)
  .setOutputCol("tf_idf")

val colRemover = new ColRemover()
  .setKeep(true)
  .setColumns(Array[String]("label","tf_idf"))

val splitter = new DatasetSplitter()
  .setKeys(Array[String]("train.hex", "valid.hex"))
  .setRatios(Array[Double](0.8))
  .setTrainKey("train.hex") // specify which key represents the training dataset, because the dataset with this key
                            // is passed further in the pipeline

val dl = new H2ODeepLearning()
    .setEpochs(10)
    .setL1(0.001)
    .setL2(0.0)
    .setHidden(Array[Int](200, 200))
    .setValidKey("valid.hex") // specify dataset which acts as testing Frame
    .setResponseColumn("label")

// Create the pipeline by defining all the stages
val pipeline = new Pipeline().setStages(Array(tokenizer, stopWordsRemover, hashingTF, idf,colRemover,splitter, dl))

// Train the pipeline model and pass additional extra parameters such validation frame
val data = load("smsData.txt")
val model = pipeline.fit(data)


/*
 * Make predictions on unlabeled data
 * Spam detector
 */
def isSpam(smsText: String,
           model: PipelineModel,
           h2oContext: H2OContext,
           hamThreshold: Double = 0.5):Boolean = {
  import h2oContext._
  val smsTextSchema = StructType(Array(StructField("text", StringType, nullable = false)))
  val smsTextRowRDD = sc.parallelize(Seq("Hello my name is earl")).map(Row(_))
  val smsTextDF = sqlContext.createDataFrame(smsTextRowRDD, smsTextSchema)
  val prediction: H2OFrame = model.transform(smsTextDF)
  prediction.vecs()(1).at(0) < hamThreshold
}

println(isSpam("Michal, h2oworld party tonight in MV?", model, h2oContext))
println(isSpam("We tried to contact you re your reply to our offer of a Video Handset? 750 anytime any networks mins? UNLIMITED TEXT?", model, h2oContext))
