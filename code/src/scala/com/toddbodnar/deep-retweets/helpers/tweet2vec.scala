/**
 * Converts a tweet's text to a vector array based on the average of the word2vec vectors of each of the words in the tweet
 **/
import scala.util.Try

def tweet2vec( word2vec:org.apache.spark.mllib.feature.Word2VecModel, text:Seq[String]) : List[Float] = {
	//try/catch/filter is to deal with rare words that are not in the model
	val vectors = text.map(word => try{word2vec.getVectors(word)}catch{case e : Exception => None}).filter(_ != None).map(_.asInstanceOf[Array[Float]])
	if (vectors.size == 0)
		return List.fill[Float](100)(0)
	val result = List.tabulate(vectors(0).size)(n => vectors.map(vector => vector(n)).sum / vectors.size)
	return result
}
