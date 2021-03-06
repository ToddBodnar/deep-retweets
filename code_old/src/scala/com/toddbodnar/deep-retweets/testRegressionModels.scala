//Try different types of regression models on top of the word2vec model

//Use bag of words like models for prediction
import org.apache.spark.mllib.feature.{Word2Vec, Word2VecModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.regression.{LinearRegressionWithSGD}
import org.apache.spark.ml.regression.{DecisionTreeRegressor,DecisionTreeRegressionModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.BoostingStrategy
import org.apache.spark.mllib.tree.{DecisionTree, RandomForest,GradientBoostedTrees}
import org.apache.spark.mllib.tree.model.{DecisionTreeModel,RandomForestModel,GradientBoostedTreesModel}


val input = sc.textFile("/Users/toddbodnar/Dropbox/retweets_h7n9.csv")



val parsed = input.map(line => (line.split("\t"))).randomSplit(Array(0.8, 0.1, 0.1), seed = 42)
val train_set = parsed(0)
val test_set = parsed(1)
val validate_set = parsed(2)


val train_set_words = train_set.map(line => line(2).toLowerCase().split(" ").toSeq)

println("Model,Vectors,Train MSE,Train R,Test MSE,Test R,Valid R,Valid MSE")

//for (numVectors <- Array(3,10,50,100,300))
for(numVectors <- Array(300))
{

val word2vec = new Word2Vec()
word2vec.setVectorSize(numVectors)
val model = word2vec.fit(train_set_words)

val train_set_transformed = train_set.map(line => LabeledPoint(Integer.parseInt(line(0))/(1.0+Integer.parseInt(line(1))), Vectors.dense(tweet2vec(model,line(2).toLowerCase().split(" ").toSeq,numVectors).toArray)))
val test_set_transformed = test_set.map(line => LabeledPoint(Integer.parseInt(line(0))/(1.0+Integer.parseInt(line(1))), Vectors.dense(tweet2vec(model,line(2).toLowerCase().split(" ").toSeq,numVectors).toArray)))
val validate_set_transformed = validate_set.map(line => LabeledPoint(Integer.parseInt(line(0))/(1.0+Integer.parseInt(line(1))), Vectors.dense(tweet2vec(model,line(2).toLowerCase().split(" ").toSeq,numVectors).toArray)))

train_set_transformed.persist()

val model_names = Array("Linear","Forest_10","Forest_50","Forest_100","Gradient","Decision")
for (model <- model_names)
{
val score = model match{
    case "Linear" =>
        val regression = LinearRegressionWithSGD.train(train_set_transformed,100)

        {point:LabeledPoint => (point.label, regression.predict(point.features))}
    case "Forest_10" => val regression = RandomForest.trainRegressor(train_set_transformed,Map[Int,Int](),10,"auto","variance",4,32)

        {point:LabeledPoint => (point.label, regression.predict(point.features))}

    case "Forest_50" => val regression = RandomForest.trainRegressor(train_set_transformed,Map[Int,Int](),50,"auto","variance",4,32)

        {point:LabeledPoint => (point.label, regression.predict(point.features))}

    case "Forest_100" => val regression = RandomForest.trainRegressor(train_set_transformed,Map[Int,Int](),100,"auto","variance",4,32)

        {point:LabeledPoint => (point.label, regression.predict(point.features))}

    case "Gradient" =>
        val boostingStrategy = BoostingStrategy.defaultParams("Regression")
        boostingStrategy.numIterations = 100
        boostingStrategy.treeStrategy.maxDepth = 5
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        boostingStrategy.treeStrategy.categoricalFeaturesInfo = Map[Int, Int]()
        val regression = GradientBoostedTrees.train(train_set_transformed, boostingStrategy)

        {point:LabeledPoint => (point.label, regression.predict(point.features))}

    case "Decision" => val regression = DecisionTree.trainRegressor(train_set_transformed, Map[Int,Int](),"variance",4,32)

        {point:LabeledPoint => (point.label, regression.predict(point.features))}

}
//val score = {point:LabeledPoint => (point.label, regression.predict(point.features))}

val train_result = train_set_transformed.map(score)
val test_result = test_set_transformed.map(score)
val validate_result = validate_set_transformed.map(score)

println(model+","+numVectors+","+MSE(train_result)+","+coorExtended(train_result)+","+MSE(test_result)+","+coorExtended(test_result)+","+MSE(validate_result)+","+coorExtended(validate_result));

}

train_set_transformed.unpersist()
}
