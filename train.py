from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import when

# Create a Spark session
spark = SparkSession.builder.appName("InsurancePrediction").getOrCreate()

# Load the data into a PySpark DataFrame
df = spark.read.csv('hdfs://localhost:9000/insurance/insurance.csv', header=True, inferSchema=True)

# Encoding categorical columns
df = df.withColumn("sex", when(df["sex"] == "male", 0).otherwise(1)) \
       .withColumn("smoker", when(df["smoker"] == "yes", 0).otherwise(1)) \
       .withColumn("region", when(df["region"] == "southeast", 0)
                               .when(df["region"] == "southwest", 1)
                               .when(df["region"] == "northeast", 2)
                               .otherwise(3))

# Remove outliers using IQR
Q1 = df.approxQuantile("charges", [0.25], 0.0)[0]
Q3 = df.approxQuantile("charges", [0.75], 0.0)[0]

IQR = Q3 - Q1

df = df.filter((df["charges"] >= Q1 - 1.5 * IQR) & (df["charges"] <= Q3 + 1.5 * IQR))


# Create a feature vector
feature_cols = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
df = assembler.transform(df)


# Standardize features
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
scaler_model = scaler.fit(df)
df = scaler_model.transform(df)


# Split the data into training and testing sets
(train_data, test_data) = df.randomSplit([0.8, 0.2], seed=2)


# Build the linear regression model
lr = LinearRegression(labelCol='charges', featuresCol='scaled_features')
pipeline = Pipeline(stages=[lr])

# Train the model
model = pipeline.fit(train_data)

# Make predictions on the test data
predictions = model.transform(test_data)

# Evaluate the model
evaluator = RegressionEvaluator(labelCol='charges', predictionCol='prediction', metricName='r2')
r2 = evaluator.evaluate(predictions)

print('#'*70)

print('''Note: 
-------------------------------------------------------------------
sex -> 0: Male, 1: Female
smoker -> 0: Yes, 1: No
region -> 0: southeast, 1: southwest, 2: northeast, 3: northwest
-------------------------------------------------------------------''')
	 
print("Testing score (R2):", r2)

print('#'*70)

