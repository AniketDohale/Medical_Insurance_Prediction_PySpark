from pyspark.sql import SparkSession
from main import assembler, scaler_model, model
## Make a prediction for a new input

# Create a Spark session
spark = SparkSession.builder.appName("InsurancePrediction").getOrCreate()

new_input = [(31, 1, 25.74, 0, 1, 0)]

new_df = spark.createDataFrame(new_input, ['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

new_df = assembler.transform(new_df)

new_df = scaler_model.transform(new_df)

new_prediction = model.transform(new_df)

predicted_charge = new_prediction.select("prediction").collect()[0][0]

# Save the result to a CSV file
new_prediction.select("prediction").write.save("hdfs://localhost:9000/insurance/Output/result_2", format="csv", header=True)

print("The insurance cost is USD", predicted_charge)
