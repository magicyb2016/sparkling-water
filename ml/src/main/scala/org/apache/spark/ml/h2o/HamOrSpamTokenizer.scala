package org.apache.spark.ml.h2o

import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.sql.types.{StringType, ArrayType, StructField, StructType}

/**
  * Ham or Spam Tokenizer
  */
class HamOrSpamTokenizer(implicit val sqlContext: SQLContext) extends Transformer{
  override def transform(dataset: DataFrame): DataFrame = {
    import sqlContext.implicits._
    val ignoredWords = Seq("the", "a", "", "in", "on", "at", "as", "not", "for")
    val ignoredChars = Seq(',', ':', ';', '/', '<', '>', '"', '.', '(', ')', '?', '-', '\'','!','0', '1')

    val texts = dataset.map( r=> {
      var smsText = r(0).asInstanceOf[String].toLowerCase
      for( c <- ignoredChars) {
        smsText = smsText.replace(c, ' ')
      }

      val words =smsText.split(" ").filter(w => !ignoredWords.contains(w) && w.length>2).distinct

      words.toSeq
    })
    texts.toDS().toDF()
  }

  override def copy(extra: ParamMap): Transformer = defaultCopy(extra)

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = {
    StructType(schema.fields ++ Seq(StructField("words",ArrayType(StringType,containsNull = false))))
  }

  override val uid: String = "HamOrSpamTokenizer"
}
