# Researching Deployment Methods 

### **Goal:** Run inference with a *large* PyTorch-trained model on a constrained edge device, with acceptable latency, power, and reliability.

### LibTorch

**Libtorch Torch Overview \[[source](https://medium.com/@pouyahallaj/libtorch-the-c-powerhouse-driving-pytorch-ee0d4f7b8743)\]**

- LibTorch is the C++ distribution of PyTorch, designed to bring the same powerful deep learning capabilities of PyTorch’s Python API to the C++ ecosystem  
- Allows PyTorch models to inference in C++ environment (Train models in python \-\> deploy in C++)

**Benefits**

- C++ is known for performance and efficiency and low latency, making it easier for real-time processing   
- Integration is easier as sensor information is collected in C++  
- Works on CPU or CUDA 

**Cons** 

- Recommended to deploy on Jetson Nano ($3000+)  
- C++ Documentation is not ready available   
- Running LibTorch on a Raspberry Pi may be challenging to setup  
  - Precompiled LibTorch is only suitable for x86\_64 machine while the Raspberry Pi running son ARM  
  - Possible solution: [Installing LibTorch on Raspberry Pi 4](https://qengineering.eu/install-pytorch-on-raspberry-pi-4.html) 

### ONNX \[[source](https://onnx.ai/)\]

ONNX Overview

- Open source universal format and ecosystem for machine learning models  
- Allows modeled to be trained in one framework (PyTorch, Tensorflow) and moved to another framework (C++ for example)

**Benefits**

- Makes models easily transferable across different environments  
- Automatically leverages GPU/CPU features  
- Support for many runtimes  
- Fast inference with correct runtime

**Cons:**

- Conversion is not straightforward for more complicated models, depending on architecture you may need to adapt the code

### Running Inference on the cloud 

Overview

- Streaming data to cloud, then sending inference back   
- Possible to be done with AWS Sagemaker  
- Data from device \-\> AWS Sagemaker \-\> Sends back to device \[[source](https://medium.com/data-science/deploy-models-with-aws-sagemaker-endpoints-step-by-step-implementation-1700316afd1d)\]

**Benefits:**

- Able to handle large models   
- Supports real time requests and response streaming

**Cons:**

- Requires connection to internet and latency depends on connectivity  
- Recurring costs  
- Not offline  
- May not be allowed at competition

