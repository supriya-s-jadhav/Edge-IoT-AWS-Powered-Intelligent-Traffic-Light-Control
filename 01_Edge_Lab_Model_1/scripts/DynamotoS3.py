import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue import DynamicFrame

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
## @type: DataSource
## @args: [database = "dynamodb", table_name = "carstats", transformation_ctx = "datasource0"]
## @return: datasource0
## @inputs: []
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "dynamodb", table_name = "carstats", transformation_ctx = "datasource0")

# Convert to a dataframe and partition based on "partition_col"
partitioned_dataframe = datasource0.toDF().repartition(1)
# Convert back to a DynamicFrame for further processing.
partitioned_dynamicframe = DynamicFrame.fromDF(partitioned_dataframe, glueContext, "partitioned_df")

## @type: ApplyMapping
## @args: [mapping = [("time", "timestamp", "time", "timestamp"), ("mincarspassing", "long", "mincarspassing", "long"), ("maxcarspassing", "long", "maxcarspassing", "long"), ("totalgreenlights", "long", "totalgreenlights", "long"), ("totaltraffic", "long", "totaltraffic", "long")], transformation_ctx = "applymapping1"]
## @return: applymapping1
## @inputs: [frame = datasource0]
applymapping1 = ApplyMapping.apply(frame = partitioned_dynamicframe, mappings = [("time", "timestamp", "time", "timestamp"), ("mincarspassing", "long", "mincarspassing", "long"), ("maxcarspassing", "long", "maxcarspassing", "long"), ("totalgreenlights", "long", "totalgreenlights", "long"), ("totaltraffic", "long", "totaltraffic", "long")], transformation_ctx = "applymapping1")
## @type: DataSink
## @args: [connection_type = "s3", connection_options = {"path": "s3://dynamodb-to-s3-traffic-data/output_glue_2/carStats.csv"}, format = "csv", transformation_ctx = "datasink2"]
## @return: datasink2
## @inputs: [frame = applymapping1]
datasink2 = glueContext.write_dynamic_frame.from_options(frame = applymapping1, connection_type = "s3", connection_options = {"path": "s3://dynamodb-to-s3-traffic-data/output_glue_2/carStats.csv"}, format = "csv", transformation_ctx = "datasink2")
job.commit()