package org.apache.spark.ml.h2o.feature

import hex.FrameSplitter
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.h2o._
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.sql.types.StructType
import water.Key
import water.fvec.{Frame, H2OFrame}

/**
  * Simple Transformer which splits the input dataset into multiple ones
  */
class DatasetSplitter(implicit h2oContext: H2OContext, sqlContext: SQLContext) extends Transformer with DatasetSplitterParams{

  private def split(df: H2OFrame, keys: Seq[String], ratios: Seq[Double]): Array[Frame] = {
    val ks = keys.map(Key.make[Frame](_)).toArray
    val splitter = new FrameSplitter(df, ratios.toArray, ks, null)
    water.H2O.submitTask(splitter)
    // return results
    splitter.getResult
  }


  override def transform(dataset: DataFrame): DataFrame = {
    require($(keys).nonEmpty, "Keys can not be empty")

    // when trainKey is not in the list of keys, the splitter choose first key as key for training dataset
    val trainKeyFinal = if(!$(keys).contains($(trainKey)) || $(trainKey).isEmpty){
      $(keys)(0)
    }else{
      $(trainKey)
    }
    import h2oContext._
    // we further don't work with splits, it's up to user to work with them using the specified keys
    split(dataset,$(keys),$(ratios))
    val train:H2OFrame = Key.make[Frame](trainKeyFinal).get()
    h2oContext.asDataFrame(train)
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  /** @group setParam */
  def setRatios(value: Array[Double]) = set(ratios, value)

  /** @group setParam */
  def setKeys(value: Array[String]) = set(keys, value)

  /** @group setParam
    *
    *  When specified trainKey is not in the list of keys, the splitter passes the first split as train dataset
    * */
  def setTrainKey(value: String) = set(trainKey, value)

  override val uid: String = "h2o_frame_splitter"

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    schema
  }
}

trait DatasetSplitterParams extends Params {

  /**
    * By default it is set to Array(1.0) which does not split the dataset at all
    */
  final val ratios = new Param[Array[Double]](this, "ratios", "Determines in which ratios split the dataset")

  setDefault(ratios->Array[Double](1.0))

  /** @group getParam */
  def getRatios: Array[Double] = $(ratios)

  /**
    * By default it is set to Array("train.hex")
    */
  final val keys = new Param[Array[String]](this, "keys", "Sets the keys for split frames")

  setDefault(keys->Array[String]("train.hex"))

  /** @group getParam */
  def getKeys: Array[String] = $(keys)

  /**
    * By default it is set to "train.hex"
    */
  final val trainKey = new Param[String](this, "trainKey", "Specify which key from keys specify the training dataset")

  setDefault(trainKey->"train.hex")

  /** @group getParam */
  def getTrainKey: String = $(trainKey)

}
