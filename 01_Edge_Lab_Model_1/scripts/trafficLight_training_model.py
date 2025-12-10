#!/usr/bin/env python
# coding: utf-8

# ## Edge demo Model 01 : TrafficLight Simulator Example
# 1. Import necessary libraries
# 2. Create S3 bucket
# 3. Mapping train and test data in S3
# 4. Mapping the path of the models in S3
#

# In[2]:


# Import libraries
import sagemaker
import boto3
from sagemaker.session import s3_input, Session
from sagemaker.amazon.amazon_estimator import get_image_uri
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[3]:


bucket_name = 'edge-demo-sagemaker-model'
my_region = boto3.session.Session().region_name
print(my_region)


# In[4]:


s3 = boto3.resource('s3')
try:
    if my_region == 'us-east-1':
        s3.create_bucket(Bucket=bucket_name)
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error : ', e)


# In[6]:


# set an output path where the trained model will be saved
prefix = 'xgboost-as-a-built-in-algo'
output_path = 's3://{}/{}/output'.format(bucket_name, prefix)
print(output_path)


# ### Read data stored in s3 bucket

# In[16]:


data_key ='carStats.csv'
data_location = 's3://{}/{}'.format(bucket_name,data_key)
traffic_data = pd.read_csv(data_location)


# In[17]:


traffic_data.head(2)


# In[18]:


traffic_data.columns = ['TotalGreenlights (N)', 'Time (S)', 'MinCarsPassing (N)',  'OUTPUT_LABEL','MaxCarsPassing (N)']
traffic_data.head(2)


# In[19]:


traffic_data = traffic_data[['OUTPUT_LABEL','MaxCarsPassing (N)','TotalGreenlights (N)','MinCarsPassing (N)']]
traffic_data.head(2)


# ### Split data into train and test dataset

# In[22]:


train_data, test_data = np.split(traffic_data.sample(frac=1,random_state=1729),[int(0.7*len(traffic_data))])
print(train_data.shape, test_data.shape)


# ### Save train and test data into s3 buckets

# In[31]:


## save train data into s3 bucket
train_data.to_csv('train_data.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train_data.csv')).upload_file('train_data.csv')
s3_input_train = sagemaker.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')


# In[35]:


## save test data into s3 bucket
test_data.to_csv('test_data.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix,'test/test_data.csv')).upload_file('test_data.csv')
s3_input_test = sagemaker.TrainingInput(s3_data='s3://{}/{}/test'.format(bucket_name, prefix), content_type='csv')


# ### Model building : XGBoost Classifier

# In[36]:


# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
# specify the repo_version depending on your preference.

xgboost_container = get_image_uri(boto3.Session().region_name, 'xgboost', repo_version='1.0-1')


# In[37]:


# initialize hyperparameter

hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "objective":"reg:linear",
        "num_round":"50"}


# In[38]:


# construct a SageMaker estimator that calls the xgboost-container
estimator = sagemaker.estimator.Estimator(image_uri=xgboost_container,
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          instance_count=1,
                                          instance_type='ml.m5.2xlarge',
                                          volume_size=5, # 5 GB
                                          output_path=output_path,
                                          train_use_spot_instances=True,
                                          train_max_run=300,
                                          train_max_wait=600)


# In[39]:


estimator.fit({'train': s3_input_train, 'validation': s3_input_test})


# ### Making Inference : Deploy Machine Learning Model As Ednpoints

# In[40]:


# Generating the endpoint for the deployment
xgb_predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')


# In[70]:


cols_input = ['MaxCarsPassing (N)', 'TotalGreenlights (N)', 'MinCarsPassing (N)']
test_data_array = test_data[cols_input]
#t.columns = ['f0', 'f1', 'f2']


# In[56]:


from sagemaker.predictor import csv_serializer
from sagemaker.xgboost.estimator import XGBoost
# Install training algorithm if it is not instalpip install xgboost
get_ipython().system('type python3')
get_ipython().system('/home/ec2-user/anaconda3/envs/python3/bin/python3 -m pip install xgboost==0.90')
import xgboost as xgb


# In[64]:


#xgb_predictor.content_type = 'text/csv' # set the data type for an inference
xgb_predictor.serializer = csv_serializer # set the serializer type


# In[128]:


predictions = xgb_predictor.predict(test_data_array.values).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[:],sep=',')
predictions_array = pd.DataFrame(predictions_array) # and turn the prediction into an array


# In[159]:


t1= test_data[['OUTPUT_LABEL']]
t2= predictions_array
results = pd.concat([t1, t2], axis=1)

#compare_results


# In[118]:

## Model accuracy rate : Confusion Matrix

#cm = pd.crosstab(index=test_data['OUTPUT_LABEL'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
#tn = cm.iloc[0,0]
#fn = cm.iloc[1,0]
#tp = cm.iloc[1,1]
#fp = cm.iloc[0,1]
#p = (tp+tn)/(tp+tn+fp+fn)*100
#print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
#print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
#print("Observed")
#print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
#print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))


# In[ ]:

## Deleting the Endpoint

#sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
#bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
#bucket_to_delete.objects.all().delete()
