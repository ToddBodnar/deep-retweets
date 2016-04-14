//based on ML tutorial
def MSE( data: org.apache.spark.rdd.RDD[(Double, Double)] ): Double = {
  data.map{case(v, p) => math.pow((v - p), 2)}.mean()
}

//coorelation calculation when the dataset is the transpose of what Statistics.coor expects
def coorExtended( data: org.apache.spark.rdd.RDD[(Double, Double)] ): Double = {
  val count = data.map(x => 1.0).reduce(_+_)
  val meanx = data.map(_._1).reduce(_+_) / count
  val meany = data.map(_._2).reduce(_+_) / count
  val dx = data.map(_._1 - meanx)
  val dy = data.map(_._2 - meany)
  val xx = dx.map(x => x*x).reduce(_+_)
  val yy = dy.map(y => y*y).reduce(_+_)
  val xy = data.map(x => (x._1 - meanx)*(x._2 - meany)).reduce(_+_)
  xy / Math.sqrt(xx*yy)
}
