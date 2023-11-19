# Start Hadoop

start-all.sh


# Create HDFS Directory

hdfs dfs -mkdir /insurance
hdfs dfs -mkdir /insurance/Output


# Upload the Dataset into Hadoop

hdfs dfs -copyFromLocal <dataset path> /insurance

example -> hdfs dfs -copyFromLocal /home/ubuntu/Documents/Medical_Insurance_Prediction/Dataset/insurance.csv /insurance/


# Open the "Medical_Insurance_Prediction" directory in terminal
example -> hadoop@Ubuntu:/home/ubuntu/Documents/Medical_Insurance_Prediction$


# Run Python File

spark-submit test_2.py


# Check the Output in Hadoop

hdfs dfs -ls /insurance/Output/
