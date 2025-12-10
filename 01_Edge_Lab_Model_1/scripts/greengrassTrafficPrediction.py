import sys
import time
import greengrasssdk
import platform
import os
from threading import Timer
import tarfile
import pickle as pkl
import xgboost as xgb
import pandas as pd
import random

client = greengrasssdk.client('iot-data')

model_path = '/greengrass-machine-learning/python/trafficlight/'
global_model = pkl.load(open(model_path + 'xgboost-model', 'rb'))


def greengrass_traffic_prediction_run():
    if global_model is not None:
        try:
            df = pd.DataFrame([{'f0': random.randint(0,20), 'f1': 3, 'f2': random.randint(5,30)}])
            predictions = global_model.predict(xgb.DMatrix(df))
            print("Predicted traffic on the road for next green light signal:", predictions)
            client.publish(topic='traffic/getprediction', payload='Predicted traffic on the road for next green light signal: {}'.format(str(predictions)))
        except:
            e = sys.exc_info()[0]
            print("Exception occured during prediction: %s" % e)

    # Asynchronously schedule this function to be run again in 3 seconds
    Timer(3, greengrass_traffic_prediction_run).start()


# Execute the function above
greengrass_traffic_prediction_run()


# This is a dummy handler and will not be invoked
# Instead the code above will be executed in an infinite loop for our example
def function_handler(event, context):
    return
