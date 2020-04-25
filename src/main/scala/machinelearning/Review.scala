package machinelearning

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql._
import org.apache.spark.ml._
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation._
import org.apache.spark.ml.tuning._
import org.apache.spark.sql.functions.{ concat, lit }

object Review {

  def main(args: Array[String]) {
    val t1 = System.nanoTime

    val spark: SparkSession = SparkSession.builder().appName("SentimentAnalysis").master("local[*]").getOrCreate()

    var inputfile = "/home/hkunta1/workspace-java/project/mapr-sparkml-sentiment-classification/notebooks/Sports_and_Outdoors_5.json"

   // model directory location"
    var modeldirectory: String ="/home/hkunta1/workspace-java/project/mapr-sparkml-sentiment-classification/sentmodel"

    if (args.length == 1) {
      inputfile = args(0)

    } else {
      System.out.println("Using file name set in the variable 'inputfile' ")
    }

    import spark.implicits._
   
    // With the SparkSession read method, read data from a file into a Dataset
    val dataset = spark.read.format("json").option("inferSchema", "true").load(inputfile)

    // add column combining summary and review text, drop some others 
    val reviewdata = dataset.withColumn("reviewTS", concat($"summary", lit(" "), $"reviewText")).drop("helpful").drop("reviewerID").drop("reviewerName").drop("reviewTime")

    reviewdata.show(5)
    val filteredreviewdata = reviewdata.filter("overall !=3")
    filteredreviewdata.show
    filteredreviewdata.describe("overall").show

    val bucketizerdata = new Bucketizer().setInputCol("overall").setOutputCol("label").setSplits(Array(Double.NegativeInfinity, 4.0, Double.PositiveInfinity))


    val transformeddata = bucketizerdata.transform(filteredreviewdata)
    transformeddata.cache
    transformeddata.groupBy("overall", "label").count.show

    val fractionsMap = Map(1.0 -> .1, 0.0 -> 1.0)
    val stratifieddata = transformeddata.stat.sampleBy("label", fractionsMap, 36L)
    stratifieddata.groupBy("label").count.show

    val splitSeed = 5043
    val Array(trainingdata, testdata) = stratifieddata.randomSplit(Array(0.8, 0.2), splitSeed)

    trainingdata.cache
    trainingdata.groupBy("label").count.show
    val tokenizer = new RegexTokenizer().setInputCol("reviewTS").setOutputCol("reviewTokensUf").setPattern("\\s+|[,.()\"]")
    val remover = new StopWordsRemover().setStopWords(StopWordsRemover.loadDefaultStopWords("english")).setInputCol("reviewTokensUf").setOutputCol("reviewTokens")

    val vectorizer = new CountVectorizer().setInputCol("reviewTokens").setOutputCol("cv").setVocabSize(200000) // .setMinDF(4)

    val idf = new IDF().setInputCol("cv").setOutputCol("features")

    //  regularizer parameters to encourage simple models and avoid overfitting.
    val lpar = 0.02
    val apar = 0.3

    // The final element in our ml pipeline is an Logistic Regression estimator  
    val classifier = new LogisticRegression().setMaxIter(100).setRegParam(lpar).setElasticNetParam(apar)

    // Below we chain the stringindexers, vector assembler and logistic regressor in a Pipeline.
    val steps = Array(tokenizer, remover, vectorizer, idf, classifier)
    val pipeline = new Pipeline().setStages(steps)

    val model = pipeline.fit(trainingdata)

    // get vocabulary from the Count Vectorizer
    val vocabulary = model.stages(2).asInstanceOf[CountVectorizerModel].vocabulary
    // get the logistic regression model 
    val classifierModel = model.stages.last.asInstanceOf[LogisticRegressionModel]
    // Get array of coefficients
    val weights = classifierModel.coefficients.toArray
    // get array of word, and corresponding coefficient Array[(String, Double)]
    val word_weight = vocabulary.zip(weights)

    word_weight.sortBy(-_._2).take(5).foreach {
      case (word, weight) =>
        println(s"feature: $word, importance: $weight")
    }
    word_weight.sortBy(_._2).take(5).foreach {
      case (word, weight) =>
        println(s"feature: $word, importance: $weight")
    }
    

    //transform this test set with model pipeline,
    //This will map the features according to the same data
    val predictions = model.transform(testdata)

    val evaluator = new BinaryClassificationEvaluator()
    val areaUnderROC = evaluator.evaluate(predictions)
    println("areaUnderROC " + areaUnderROC)

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

    println("ratio correct", ratioCorrect)

    println("true positive", truepositive)

    println("false positive", falsepositive)

    println("true negative", truenegative)

    println("false negative", falsenegative)

    println("precision", precision)
    println("recall", recall)
    println("accuracy", accuracy)
    println("Negative reviews")
    predictions.filter($"prediction" === 0.0).select("reviewTokens","summary","overall","prediction").orderBy(desc("rawPrediction")).show(5)
    println("Positive reviews")
    predictions.filter($"prediction" === 1.0).select("reviewTokens","summary","overall","prediction").orderBy("rawPrediction").show(5)
   
    model.write.overwrite().save(modeldirectory)
    val duration = (System.nanoTime - t1) / 1e9d
 
    println("time taken for execution when file is in local file system= "+ duration+ " seconds")
    
  }
}
