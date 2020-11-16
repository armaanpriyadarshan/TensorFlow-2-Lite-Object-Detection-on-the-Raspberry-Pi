# Converting TensorFlow Models to TensorFlow Lite
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
### This Guide Contains Everything you Need to Convert Custom and Pre-trained TensorFlow Models to TensorFlow Lite
I'll be covering two options. Converting a custom model and converting a pre-trained model. If you want to train a custom TensorFlow object detection model, I've made a detailed [GitHub guide](https://github.com/armaanpriyadarshan/Training-a-Custom-TensorFlow-2.X-Object-Detector) and a YouTube video on the topic.

[![Link to my vid](https://github.com/armaanpriyadarshan/Object-Detection-on-Raspberry-Pi/blob/master/doc/Thumbnail2.png)](https://www.youtube.com/watch?v=oqd54apcgGE)

**The following steps for conversion are based off of the directory structure and procedures in this guide. So if you haven't already taken a look at it, I recommend you do so.
To move on, you should have already**
  - **Installed Anaconda**
  - **Setup the Directory Structure**
  - **Compiled Protos and Setup the TensorFlow Object Detection API**
  - **Gathered Training Data**
  - **Trained your Model (without exporting)**
  
## Steps
1. [Creating a New Environment and Installing TensorFlow]()
2. [Exporting the Model]()
3. [Converting the Model to TensorFlow Lite]()
4. [Preparing our Model for Use]()
 

### Creating a New Environment and Installing TensorFlow
To avoid version conflicts, we'll first create a new Anaconda virtual environment to hold all the packages necessary for conversion. To do so, open up a new Anaconda terminal and issue
```
conda create -n tflite pip python=3.7
```

We can now activate our environment with

```
conda activate tflite
```

**Note that whenever you open a new Anaconda Terminal you will not be in the virtual environment. So if you open a new prompt make sure to use the command above to activate the virtual environment**

Now we must install TensorFlow in this virtual environment. However, in this environment we will not just be installing standard TensorFlow. We are going to install tf-nightly. This package is a nightly updated build of TensorFlow. This means it contains the very latest features that TensorFlow has to offer. There is a CPU and GPU version, but if you are only using it conversion I'd stick to the CPU version because it doesn't really matter. We can install it by issuing

```
pip install tf-nightly
```
Now, to test our installation let's use a Python terminal.
```
python
```
Then import the module with
```
Python 3.7.9 (default, Aug 31 2020, 17:10:11) [MSC v.1916 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> import tensorflow as tf
>>> print(tf.__version)
```

**Note: You might get an error with importing the newest version of Numpy. It looks something like this ```RuntimeError: The current Numpy installation ('D:\\Apps\\anaconda3\\envs\\tflite\\lib\\site-packages\\numpy\\__init__.py') fails to pass a sanity check due to a bug in the windows runtime. See this issue for more information: https://tinyurl.com/y3dm3h86```. You can fix this error by installing a previous version of Numpy with ```pip install numpy==1.19.3```.**

If the installation was successful, you should get the version of tf-nightly that you installed. 
```
2.5.0-dev20201111
```
