package sentimentanalysis

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.tuning._
import org.apache.spark.sql.functions.{ concat, lit }
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
object SentimentAnalyser {

  def main(args: Array[String]) {
    val time1 = System.nanoTime

    val spark: SparkSession = SparkSession.builder().appName("flightdelay").master("local[*]").getOrCreate()

    var inputfile = "/home/pkanchala1/data/Sports_and_Outdoors_5.json"

   
    var savedmodel: String ="/home/pkanchala1/workspacejava/mapr-sparkml-sentiment-classification-master/sentmodel/"

    if (args.length == 1) {
      inputfile = args(0)

    } else {
      System.out.println("Using file name set in the variable 'inputfile' ")
    }

    import spark.implicits._
   
    // Reading the dataset into data frame
    val dataset = spark.read.format("json").option("inferSchema", "true").load(inputfile)

    // adding review text and summary columns, removing unnecessary columns
    val reviewdata = dataset.withColumn("reviewTS", concat($"summary", lit(" "), $"reviewText")).drop("helpful").drop("reviewerID").drop("reviewerName").drop("reviewTime")

    reviewdata.show(5)

    // Filtering out neutral reviews
    val filteredreviewdata = reviewdata.filter("overall !=3")
    filteredreviewdata.show
    filteredreviewdata.describe("overall").show

    // Bucketizing using Spark Bucketizer
    val bucketizer = new Bucketizer().setInputCol("overall").setOutputCol("label").setSplits(Array(Double.NegativeInfinity, 4.0, Double.PositiveInfinity))

    val transformeddata = bucketizer.transform(filteredreviewdata)
    transformeddata.cache
    transformeddata.groupBy("overall", "label").count.show

    // Performing Stratified Sampling to make sensitive to negative reviews
    val fractions = Map(1.0 -> .1, 0.0 -> 1.0)
    val stratifieddata = transformeddata.stat.sampleBy("label", fractions, 36L)
    stratifieddata.groupBy("label").count.show

    val splitSeed = 5043

    // Splitting dataset, 80% for training and 20% for testing
    val Array(trainingdata, testdata) = stratifieddata.randomSplit(Array(0.8, 0.2), splitSeed)

    trainingdata.cache
    trainingdata.groupBy("label").count.show

    // Tokenizing based on the specified pattern using Regex Tokenizer
    val tokenizer = new RegexTokenizer().setInputCol("reviewTS").setOutputCol("reviewTokensUf").setPattern("\\s+|[,.()\"]")

    // Stop words removal eg 'is','this','that'
    val remover = new StopWordsRemover().setStopWords(StopWordsRemover.loadDefaultStopWords("english")).setInputCol("reviewTokensUf").setOutputCol("reviewTokens")

    // Using count vectorizer for TF-IDF values/ feature extraction

    val vectorizer = new CountVectorizer().setInputCol("reviewTokens").setOutputCol("cv").setVocabSize(200000)

    val idf = new IDF().setInputCol("cv").setOutputCol("features")

    //  To avoid overfitting.
    val lparameter = 0.02
    val aparameter = 0.3

    // Logistic Regresson model
    val classifier = new LogisticRegression().setMaxIter(100).setRegParam(lparameter).setElasticNetParam(aparameter)


    // Placing tokenizer, stop words remover, vectorizer and logistic regressor in a pipeline.
    val steps = Array(tokenizer, remover, vectorizer, idf, classifier)
    val pipeline = new Pipeline().setStages(steps)

    val model = pipeline.fit(trainingdata)

    // to get vocabulary from  Vectorizer
    val vocabulary = model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    
    // to get  logistic regression model 
    val LRModel = model.stages.last.asInstanceOf[LogisticRegressionModel]

    val weights = LRModel.coefficients.toArray

    // To get importance of words
    val wordweight = vocabulary.zip(weights)

    wordweight.sortBy(-_._2).take(5).foreach {
      case (word, weight) =>
        println(s"feature: $word, Importance: $weight")
    }
    wordweight.sortBy(_._2).take(5).foreach {
      case (word, weight) =>
        println(s"feature: $word, Importance: $weight")
    }
    

    //Providing test data to the trained model
    val predictions = model.transform(testdata)

    val evaluator = new BinaryClassificationEvaluator()
    val areaUnderROC = evaluator.evaluate(predictions)

    println("Area under ROC curver " + areaUnderROC)

    val lp = predictions.select("prediction", "label")
    val counttotal = predictions.count().toDouble
    val correct = lp.filter("label == prediction").count().toDouble
    val wrong = lp.filter("label != prediction").count().toDouble
    val ratioWrong = wrong / counttotal
    val ratioCorrect = correct / counttotal
    val truenegative = (lp.filter($"label" === 0.0).filter($"label" === $"prediction").count()) / counttotal
    val truepositive = (lp.filter($"label" === 1.0).filter($"label" === $"prediction").count()) / counttotal
    val falsenegative = (lp.filter($"label" === 0.0).filter(not($"label" === $"prediction")).count()) / counttotal
    val falsepositive = (lp.filter($"label" === 1.0).filter(not($"label" === $"prediction")).count()) / counttotal
    val precision= truepositive / (truepositive + falsepositive)
    val recall= truepositive / (truepositive + falsenegative)
    val fmeasure= 2 *precision *recall / (precision + recall)
    val accuracy=(truepositive + truenegative) / (truepositive + truenegative + falsepositive + falsenegative)

    // Metric

    println("Ratio Correct Value", ratioCorrect)

    println("True Positive Value", truepositive)

    println("False Positive Value", falsepositive)

    println("True Negative Value", truenegative)

    println("False Negative Value", falsenegative)

    println("Precision Value", precision)

    println("Recall Value", recall)

    println("Accuracy Value", accuracy)

    println("Printing Negative Reviews")
    
    predictions.filter($"prediction" === 0.0).select("summary","reviewTokens","overall","prediction").orderBy(desc("rawPrediction")).show(5)

    println("Printing Positive reviews")

    predictions.filter($"prediction" === 1.0).select("summary","reviewTokens","overall","prediction").orderBy("rawPrediction").show(5)
   
    // Saving the model
    model.write.overwrite().save(savedmodel)

    // Calculating the time taken
    val duration = (System.nanoTime - time1) / 1e9d
 
    println("time taken for execution when file is in local file system= "+ duration+ " seconds")
   
  }
}

