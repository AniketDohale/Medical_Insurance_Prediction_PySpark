from pyspark.sql import SparkSession
from train import assembler, scaler_model, model
from pyspark.sql.functions import lit
## Make a prediction for a new input

# Create a Spark session
spark = SparkSession.builder.appName("InsurancePrediction").getOrCreate()

new_input = [(18, 0, 27.900, 0, 0, 3)]

new_df = spark.createDataFrame(new_input, ['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

new_df_1 = assembler.transform(new_df)

new_df_1 = scaler_model.transform(new_df_1)

new_prediction = model.transform(new_df_1)

predicted_charge = new_prediction.select("prediction").collect()[0][0]

# Add Predicted Charge to Input Data
result_df = new_df.withColumn("predicted_charge", lit(predicted_charge))


# Save Results to CSV
result_df.select('age', 'sex', 'bmi', 'children', 'smoker', 'region', 'predicted_charge').write.save("hdfs://localhost:9000/insurance/Output/result_1", format="csv", header=True)

print('#'*60)

print("The insurance cost is USD", predicted_charge)

print('#'*60)
