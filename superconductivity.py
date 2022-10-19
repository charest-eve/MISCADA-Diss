import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext

spark = (SparkSession
    .builder
    .appName('Superconductivity')
    .getOrCreate())

# For user input
# prac_file = sys.argv[1]

num_princ_comp = 10

prac_file = "Volumes/SANDISK/Datasets/superconduct/superconductivity.csv"

sqlContext = SQLContext(spark)
full_df = sqlContext.read.format('csv').options(header=True, inferSchema=True).load(prac_file)

feature_cols = full_df.columns
feature_cols.remove('critical_temp')
feature_cols.remove('material')

vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')

features_df = vector_assembler.transform(full_df).select(['critical_temp', 'features'])

standardizer = StandardScaler(withMean=True, withStd=True, inputCol='features', outputCol='std_features')
standardizer_2 = standardizer.fit(features_df)
std_features_df = standardizer_2.transform(features_df)

pca = PCA(k=num_princ_comp, inputCol='std_features', outputCol='pca_features')
pca_model = pca.fit(std_features_df)
pca_df = pca_model.transform(std_features_df)





