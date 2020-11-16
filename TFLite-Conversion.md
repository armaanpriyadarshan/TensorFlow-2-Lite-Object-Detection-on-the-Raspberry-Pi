# Converting TensorFlow Models to TensorFlow Lite
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
### This Guide Contains Everything you Need to Convert Custom and Pre-trained TensorFlow Models to TensorFlow Lite
I'll be covering two options. Converting a custom model and converting a pre-trained model. 

**My [YouTube tutorial]() is recommended for this step,  another important step is located [here](https://www.youtube.com/channel/UCT9t2Bug62RDUfSBcPt0Bzg?sub_confirmation=1)!**

## Option 1: Converting Custom Models
If you want to train a custom TensorFlow object detection model, I've made a detailed [GitHub guide](https://github.com/armaanpriyadarshan/Training-a-Custom-TensorFlow-2.X-Object-Detector) and a YouTube video.

[![Link to my vid](https://github.com/armaanpriyadarshan/Object-Detection-on-Raspberry-Pi/blob/master/doc/Thumbnail2.png)](https://www.youtube.com/watch?v=oqd54apcgGE)

The following steps for conversion are based off of the directory structure and procedures in this guide. So if you haven't already taken a look at it, I recommend you do so.
To move on, you should have already
  - Installed Anaconda
  - Setup the Directory Structure
  - Setup the TensorFlow Object Detection API
  - Gathered Training Data
  - Trained your Model (without exporting)

### Preparing our Workspace
To avoid version conflicts, we'll first create a new Anaconda virtual environment to hold all the packages necessary for conversion. To do so, open up a new Anaconda terminal and issue
```
conda create -n tflite
```

Since we have not specified the Python version, Anaconda will use whatever version of Python you have on your machine. I used Python 3.7.8 which was stable for me. If you do get 
errors later on, I'd recommend using this version of Python and it might resolve your issues. We can now activate our environment with

```
conda activate tflite
```

**Note that whenever you open a new Anaconda Terminal you will not be in the virtual environment. So if you open a new prompt make sure to use the command above to activate the virtual environment**

Now we must install TensorFlow in this virtual environment. There is a GPU and CPU version, but since we are just using it for conversion I'd recommend sticking to the CPU version
