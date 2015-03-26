package com.vistaprint.spark

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.sql.SQLContext
import org.apache.spark.mllib.tree.model.DecisionTreeModel
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.tree.RandomForest

object Tryout extends App {

  run()

  def run() {
    val conf = new SparkConf().setAppName("Decision trees tryout").setMaster("local[4]")
    val sc = new SparkContext(conf)

    val path = "data/iris.data"
    val data = IrisDataReader.getDataSet(sc, path)

    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    val numClasses = 3
    val categoricalFeaturesInfo = Map[Int, Int]()
    val maxDepth = 5
    val maxBins = 32

    val modelGini = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity = "gini", maxDepth, maxBins)
    println("Gini")
    Evaluator.evaluate(testData, modelGini.toDebugString, modelGini.predict)

    println("Entropy")
    val modelEntropy = DecisionTree.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, impurity = "entropy", maxDepth, maxBins)
    Evaluator.evaluate(testData, modelEntropy.toDebugString, modelEntropy.predict)

    println("Ensemble")
    val ensembleModel = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo, numTrees = 4, featureSubsetStrategy = "auto", impurity = "gini", maxDepth, maxBins)
    Evaluator.evaluate(testData, ensembleModel.toDebugString, ensembleModel.predict)

    println("Regression")
    val regressionModel = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo, "variance", maxDepth = 5, maxBins = 32)
    Evaluator.evaluateMSE(testData, regressionModel.toDebugString, regressionModel.predict)
  }

}