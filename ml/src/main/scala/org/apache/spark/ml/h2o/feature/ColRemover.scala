package org.apache.spark.ml.h2o.feature

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.{StructType, StructField}

/**
  * Simple Transformer which removes specified columns from the dataset
  */
class ColRemover extends Transformer with ColRemoverParams{

  override def transform(dataset: DataFrame): DataFrame = {
    val columnsToRemove = if($(keep)){
      dataset.columns.filter(col => !$(columns).contains(col))
    }else{
      $(columns)
    }
    columnsToRemove.foreach{
      col => dataset.drop(col)
    }
    dataset
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  /** @group setParam */
  def setKeep(value: Boolean) = set(keep, value)

  /** @group setParam */
  def setColumns(value: Array[String]) = set(columns, value)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    val columnsToLeft = if($(keep)){
      $(columns)
    }else{
      schema.fieldNames.filter(col => !$(columns).contains(col))
    }
    StructType(columnsToLeft.map{
     col =>  StructField(col,schema(col).dataType,schema(col).nullable,schema(col).metadata)
    })
  }

  override val uid: String = "h2o_col_remover"
}

trait ColRemoverParams extends Params {
  /**
    * By default it is set to false which means removing specified columns
    */
  final val keep = new BooleanParam(this, "keep", "Determines if the column specified in the 'columns' parameter should be kept or removed")

  setDefault(keep->false)

  /** @group getParam */
  def getKeep: Boolean = $(keep)

  /**
    * By default it is empty array which means no columns are removed
    */
  final val columns = new Param[Array[String]](this, "columns", "List of columns to be kept or removed")

  setDefault(columns->Array[String]())

  /** @group getParam */
  def getColumns: Array[String] = $(columns)
}
