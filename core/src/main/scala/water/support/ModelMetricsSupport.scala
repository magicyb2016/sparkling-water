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
package water.support

import hex.{Model, ModelMetrics}
import hex.tree.gbm.GBMModel
import water.fvec.Frame

// FIXME: should be published by h2o-scala interface
trait ModelMetricsSupport {

  def r2(model: GBMModel, fr: Frame) = hex.ModelMetrics.getFromDKV(model, fr)
    .asInstanceOf[hex.ModelMetricsSupervised].r2()

  def modelMetrics[T <: ModelMetrics, M <: Model[M, P, O], P <: hex.Model.Parameters, O <: hex.Model.Output]
  (model: Model[M, P, O], fr: Frame) = ModelMetrics.getFromDKV(model, fr).asInstanceOf[T]

  def binomialMM[M <: Model[M, P, O], P <: hex.Model.Parameters, O <: hex.Model.Output]
  (model: Model[M, P, O], fr: Frame) = modelMetrics[hex.ModelMetricsBinomial, M, P, O](model, fr)

  def multinomialMM[M <: Model[M, P, O], P <: hex.Model.Parameters, O <: hex.Model.Output]
  (model: Model[M, P, O], fr: Frame) = modelMetrics[hex.ModelMetricsMultinomial, M, P, O](model, fr)

  case class R2(name: String, train: Double, test: Double, hold: Double) {
    override def toString: String =
      s"""
         |Results for $name:
         |  - R2 on train = $train
         |  - R2 on test  = $test
         |  - R2 on hold  = $hold
      """.stripMargin
  }
}

// Create companion object
object ModelMetricsSupport extends ModelMetricsSupport

