# People Detection with Heat Map
This application is based off of the [Store Aisle Monitor Python](https://github.com/intel-iot-devkit/store-aisle-monitor-python). However, this implementation does not require Azure cloud storage. This is so that users who 
don't have an Azure account can easily and quickly run and demonstrate or use this application. 

## Running
Follow all necessary steps for Requirements and Setup in [Store Aisle Monitor Python](https://github.com/intel-iot-devkit/store-aisle-monitor-python) 

Run application using command
    
    python3 main.py -m /opt/intel/openvino/deployment_tools/tools/model_downloader/Retail/object_detection/pedestrian/rmnet_ssd/0013/dldt/person-detection-retail-0013.xml -l /opt/intel/openvino/inference_engine/lib/intel64/libcpu_extension_sse4.so -d CPU -pt 0.7  -i [path to video input]
