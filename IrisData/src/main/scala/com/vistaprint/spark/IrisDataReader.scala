package com.vistaprint.spark

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint

object IrisDataReader {
  def getDataSet(sc: SparkContext, path: String) = {
    val irisData = getIrisData(sc, path)
    val distinctClasses = irisData.map(x => x.irisClass).distinct().collect()
    irisData.map(x => toLabeledPoint(x, distinctClasses))
  }

  def getIrisData(sc: SparkContext, path: String) = {
    val rawData = sc.textFile(path)
    rawData.flatMap(line => {
      if (line.isEmpty()) None
      else {
        val p = line.split(",")
        Some(IrisData(p(0).toDouble, p(1).toDouble, p(2).toDouble, p(3).toDouble, p(4)))
      }
    })
  }

  def toLabeledPoint(iris: IrisData, classes: Array[String]): LabeledPoint = {
    val c = classes.indexOf(iris.irisClass)
    new LabeledPoint(c, Vectors.dense(iris.sepalLength, iris.sepalWidth, iris.petalLength, iris.petalWidth))
  }

}