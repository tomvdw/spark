package com.vistaprint.spark

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.regression.LabeledPoint

object Evaluator {

  def evaluate(testData: RDD[LabeledPoint], model: String, predict: org.apache.spark.mllib.linalg.Vector => Double) {
    val labelsAndPredictions = getLabelsAndPredictions(testData, predict)
    val testErr = labelsAndPredictions.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test correct = " + (1.0 - testErr))
    println("Test incorrect = " + testErr)
    println("Learned classification tree model:\n" + model)
  }

  def evaluateMSE(testData: RDD[LabeledPoint], model: String, predict: org.apache.spark.mllib.linalg.Vector => Double) {
    val labelsAndPredictions = getLabelsAndPredictions(testData, predict)
    val testMSE = labelsAndPredictions.map { case (v, p) => math.pow((v - p), 2) }.mean()
    println("Test correct = " + (1.0 - testMSE))
    println("Test Mean Squared Error = " + testMSE)
    println("Learned regression tree model:\n" + model)
  }
  
  def getLabelsAndPredictions(testData: RDD[LabeledPoint], predict: org.apache.spark.mllib.linalg.Vector => Double) = {
    testData.map { point =>
      val prediction = predict(point.features)
      (point.label, prediction)
    }
  }

}