/*
* Licensed to the Apache Software Foundation (ASF) under one or more
* contributor license agreements.  See the NOTICE file distributed with
* this work for additional information regarding copyright ownership.
* The ASF licenses this file to You under the Apache License, Version 2.0
* (the "License"); you may not use this file except in compliance with
* the License.  You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package org.apache.spark.ml.h2o

import hex.deeplearning.{DeepLearning, DeepLearningModel}
import DeepLearningModel.DeepLearningParameters
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.h2o.{H2OFrame, H2OContext}
import org.apache.spark.ml.param.shared.{HasOutputCol, HasInputCol}
import org.apache.spark.ml.param._
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.{Estimator, PredictionModel}
import org.apache.spark.mllib
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.StructType
import water.Key
import water.fvec.Frame

/**
 * Deep learning ML component.
 */
class H2ODeepLearningModel(model: DeepLearningModel)
  extends PredictionModel[mllib.linalg.Vector, H2ODeepLearningModel] {

  override protected def predict(features: mllib.linalg.Vector): Double = {
  0//TODO: implement
  }

  override val uid: String = "dlModel"

  override def copy(extra: ParamMap): H2ODeepLearningModel = defaultCopy(extra)
}

class H2ODeepLearning()(implicit h2oContext: H2OContext)
  extends Estimator[H2ODeepLearningModel] with DeepLearningParams {

  override def fit(dataset: DataFrame): H2ODeepLearningModel = {
    import h2oContext._
    val params = getDeepLearningParams
    params._train = dataset
    val model = new DeepLearning(params).trainModel().get()
    params._train.remove()
    val dlm = new H2ODeepLearningModel(model)
    dlm
  }

  /**
    * Set the param and execute custom piece of code
    */
  private def set[T](param: Param[T], value: T)(f:  => Unit): this.type ={
    f
    set(param,value)
  }

  /** @group setParam */
  def setEpochs(value: Double) = set(epochs, value){getDeepLearningParams._epochs = value}

  /** @group setParam */
  def setL1(value: Double) = set(l1, value){getDeepLearningParams._l1 = value}

  /** @group setParam */
  def setL2(value: Double) = set(l2, value){getDeepLearningParams._l2 = value}

  /** @group setParam */
  def setHidden(value: Array[Int]) = set(hidden, value){getDeepLearningParams._hidden = value}

  /** @group setParam */
  def setResponseColumn(value: String) = set(responseColumn,value){getDeepLearningParams._response_column = value}

  /** @group setParam */
  def setValidKey(value: String) = set(validKey,value){getDeepLearningParams._valid = Key.make[Frame](value)}

  @DeveloperApi
  override def transformSchema(schema: StructType): StructType = ???

  override val uid: String = "dl"

  override def copy(extra: ParamMap): Estimator[H2ODeepLearningModel] = defaultCopy(extra)
}

/**
  * Parameters here can be set as normal and are duplicated to DeepLearningParameters H2O object
  */
trait DeepLearningParams extends Params {

  /**
    * Holder of the parameters, we use it so set default values and then to store the set values by the user
    */
  private val deepLearningParameters = new DeepLearningParameters()
  protected def getDeepLearningParams: DeepLearningParameters = deepLearningParameters

  /**
    * All parameters should be set here along with their documentation and explained default values
    */
  final val epochs = new DoubleParam(this, "epochs", "Explanation")
  final val l1 = new DoubleParam(this, "l1", "Explanation")
  final val l2 = new DoubleParam(this, "l2", "Explanation")
  final val hidden = new Param[Array[Int]](this, "hidden", "Explanation")
  final val responseColumn = new Param[String](this,"responseColumn","Explanation")
  final val validKey = new Param[String](this,"valid","Explanation")

  setDefault(epochs->deepLearningParameters._epochs)
  setDefault(l1->deepLearningParameters._l1)
  setDefault(l2->deepLearningParameters._l2)
  setDefault(hidden->deepLearningParameters._hidden)
  setDefault(responseColumn->deepLearningParameters._response_column)
  setDefault(validKey->deepLearningParameters._valid.toString)

  /** @group getParam */
  def getEpochs: Double = $(epochs)
  /** @group getParam */
  def getL1: Double = $(l1)
  /** @group getParam */
  def getL2: Double = $(l2)
  /** @group getParam */
  def getHidden: Array[Int] = $(hidden)
  /** @group getParam */
  def getResponseColumn: String = $(responseColumn)
  /** @group getParam */
  def getValidKey: String = $(validKey)
}

