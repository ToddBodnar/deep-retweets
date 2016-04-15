//NOT to be used for actual model fitting, but to just get an idea of how vector size effects fitting
//Use word2vec of different vector sizes and a simple linear regression to see if there's much of a performance difference

//Use bag of words like models for prediction
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD}
import org.apache.spark.ml.regression.{DecisionTreeRegressor,DecisionTreeRegressionModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest,GradientBoostedTrees}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel,RandomForestModel,GradientBoostedTreesModel}

import org.apache.spark.h2o._
val h2oContext = H2OContext.getOrCreate(sc)
import h2oContext._
import h2oContext.implicits._


val input = sc.textFile("/Users/toddbodnar/Dropbox/retweets_h7n9.csv")



val parsed = input.map(line => (line.split("\t"))).randomSplit(Array(0.9, 0.1), seed = 42)
val train_set = parsed(0)
val test_set = parsed(1)
//val validate_set = parsed(2)

 val conf = new NeuralNetConfiguration.Builder().momentum(0.9)
.activationFunction(Activations.tanh()).weightInit(WeightInit.VI)
.optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
.iterations(100).visibleUnit(RBM.VisibleUnit.GAUSSIAN)
.hiddenUnit(RBM.HiddenUnit.RECTIFIED).stepFunction(new GradientStepFunction())
.nIn(4).nOut(3).layerFactory(layerFactory)
.list(3).hiddenLayerSizes(3, 2).`override`(classifierOverride )
.build()

val train_set_words = train_set.map(line => line(2).toLowerCase().split(" ").toSeq)

println("Vectors,Train MSE,Train R,Test MSE,Test R,Valid R,Valid MSE")

val numVectors = 100

val word2vec = new Word2Vec()
word2vec.setVectorSize(numVectors)
val model = word2vec.fit(train_set_words)

val train_set_transformed = train_set.map(line => LabeledPoint((Integer.parseInt(line(0))/(1.0+Integer.parseInt(line(1)))>0), Vectors.dense(tweet2vec(model,line(2).toLowerCase().split(" ").toSeq,numVectors).toArray)))
val test_set_transformed = test_set.map(line => LabeledPoint((Integer.parseInt(line(0))/(1.0+Integer.parseInt(line(1)))>0), Vectors.dense(tweet2vec(model,line(2).toLowerCase().split(" ").toSeq,numVectors).toArray)))
//val validate_set_transformed = validate_set.map(line => LabeledPoint((Integer.parseInt(line(0))/(1.0+Integer.parseInt(line(1)))>0), Vectors.dense(tweet2vec(model,line(2).toLowerCase().split(" ").toSeq,numVectors).toArray)))



train_set_transformed.persist()

h2oContext.asH2ODataFrame(train_set_transformed)

val regression = LinearRegressionWithSGD.train(train_set_transformed,100)

val score = {point:LabeledPoint => (point.label, regression.predict(point.features))}
   //val score = {point:LabeledPoint => (point.label, regression.predict(point.features))}

val train_result = train_set_transformed.map(score)
val test_result = test_set_transformed.map(score)
//val validate_result = validate_set_transformed.map(score)

println(numVectors+","+MSE(train_result)+","+coorExtended(train_result)+","+MSE(test_result)+","+coorExtended(test_result));


train_set_transformed.unpersist()

