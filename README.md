# NN Image Analyzer - Object Detector
![ ](logo.jpg)
<br><br>This is a component of NN Image Analyzer, a distributed system of neural networks developed to be simple and flexible, that can analyze image and detect faces and objects. You can find the other components here:
- [Main Server](https://github.com/CianciarusoCataldo/nn-dispatcher)
- [Face Detector](https://github.com/CianciarusoCataldo/nn-face-detector)

## Summary
- [Introduction](#introduction)
- [Technologies](#technologies)
- [Getting Started](#getting-started)
	- [Prerequisites](#prerequisites)
	- [Install](#installing)
	- [Configure](#configure)
- [Run](#run)
- [The Response Format](#the-response-format)
- [Build your own node](#build-your-own-node)
- [Build your own client](#build-your-own-client)
- [Deploy](#deploy)
- [See in action](#see-in-action)
- [Authors](#authors)
- [License](#license)

## Introduction
NN Image Analyzer is a distributed system, powered by Neural Networks. You can easily analyze images to detect objects and faces. From detected faces, you can also retrieve information about gender, age and emotions. Due to his distributed nature, this projects is divided into multiple components. This is the main component, the **Object detector**. This component handle external requests, receiving image and analyze it to find objects. The user have to simply send a POST request to this webserver, and wait for result. That's it! You can run this application locally, alone or with the other nodes that compose the entire system. To make it modular as most, I designed every node to be independent, so you can easily use them separately in your projects. For every node, there is a dedicated repository:
- [Main Server](https://github.com/CianciarusoCataldo/nn-dispatcher)
- [Object Detector](https://github.com/CianciarusoCataldo/nn-object-detector)
- [Face Detector](https://github.com/CianciarusoCataldo/nn-face-detector)

I decided to not group all nodes under only one repository because every component of this system is independent from each other. This means that you can use the whole system or just one part, it's the same. The idea at the base of this project is to give to other user a tool to build a more complex system, not only to use it.

## Technologies
This project is written with **Python** programming language (**version 3.6.8**). It also use some external dependencies, that are listed into "requirements.txt" file for a fast setup. The Neural Network is based upon [ImageAi](https://github.com/OlafenwaMoses/ImageAI) project, adapted to support multiple request at once, using multithreading provided by Waitress webserver. My personal <a href="https://nn-image-analyzer.herokuapp.com">release</a> contains also a simple website, that I develope using **HTML**, **CSS** and **Javascript**, with also powerful extensions like **Bootstrap** and **Font awesome**, and many others. I decided to not include this website in the public release to let you use your personal website, if you have one.

## Getting Started

You need to clone this repository to your local machine (or just download a compressed zip file from github). You can use **git**:
```
git clone https://www.github.com/CianciarusoCataldo/nn-object-detector
```
Then, move to the new directory:
```
cd nn-object-detector
```
Once you have a local copy, you are ready to start! Follow these steps to be sure everything will work fine at the end. This component of my project is an **Object Detector** node, a webserver that will handle every request incoming and, if there is an image in the request, it will start the detection.  

### Prerequisites

To run this application on your local machine, you need first a Python environment properly configured. So, check your Python installation (eventually, install it from <a href="https://www.python.org/downloads/">official web site</a>). I developed all nodes module under Python 3.6.8, so check if your Python version is equal (or greater) than mine. To check it, just type this in your terminal:
```
python -version
```

### Installing

Because of <a href="https://opencv.org">Open CV</a> module, you must follow some steps, if you are under Linux.

#### Only for Linux users
To work properly in Linux, this application require some external dependencies, directly from Linux repository. I included a file, "Aptfile", which list all these dependencies, for a fast installation. Based on your Linux distribution, you have to use the appropriate package manager. For example, on **Debian** based distribution:
```
apt-get install -y $(cat Aptfile)
```
<br>Now, install all dependencies with pip. Open a terminal and type:

```
pip install -r requirements.txt
```
After that, you are ready to go !


### Configure

Before you start, let's see how this application works. It's designed to be modular and system independent, so it can be easily integrated with every existing system. You can also build your own system, using this module as base. By default, this component use 8082 port, on local, but feel free to change that, you decide. This webserver can show a website to user who reach his url with browser. You can use your own (I provide just a simple index.html, for general purpose), simply put your website files to "static" folder, and be sure to respect the original folder structure. There are many folder, one for every file type:
- Js : put your javascript file here
- Css : put your css file here
- img : put your image files here (depends on browser, be sure that the format is supported)
- Fonts : put your font file here (.woff ecc.)

## Run
Once you have installed all dependencies required, you can start the webserver simply using the startup script provided (start_server.py) that is os-independent. Otherwise, open terminal and type:
```
python start_server.py
```
This will start the Waitress webserver, it will listen to 8082 port by default (you can change this setting in config.ini file). Note that, only at very first run, the program will download the required model for Neural Netowrk from a dedicated cloud storage. Don't worry, it's just a one-shot step, because the model will be permanently stored in "models" directory from here. I prefer to make every repository as small as possible, so the model files that are too big are not included. To To test it, just open your browser and type this in the address bar:
```
http://localhost:8082
```
If everything is working, you will see the index file of the website, into static folder. By now, the webserver is able to handle both POST and GET requests. To start other nodes of the system, just follow their guide, there are only obvious little differences (every node has its own startup script, to make it more immediate):
- [Main Server](https://github.com/CianciarusoCataldo/nn-dispatcher)
- [Face Detector](https://github.com/CianciarusoCataldo/nn-face-detector)

You can easily send a POST request to the server with every language (I also developed a dedicated <a href="">Android Client</a>). In "test" directory you can find a simple python script (send_request.py) which do the job for you.<br>
Just open a terminal, move to "test" directory, and type:
```
python send_request.py <image_path>
```
If you want to send request by your own, just put an image in the HTTP POST request body, associated with an header, "image". The webserver is configured to read the image (in binary form) from the message using this custom header, so make sure to use it, or no request will be processed.

## The response format
The response from webserver follow a predefined pattern, designed to make it readable but also short, to send less data and reduce overhead. The final response, the one that the user will read, has this form:
```
OBJ<Objects list>
```
The < **Objects list** > contains all the objects detected. For every objects, there are two attributes: **name** and **coordinates**. For example, here is a simple objects list:
```
OBJcar-109 1890 560 2300
```
It represents an object "car", which is located at the given coordinates. These coordinates are, in order: 
1. Starting x coordinate
2. Starting y coordinate
3. Final x coordinate
4. Final y coordinate

These coordinates represents the rectangular area where the object is contained, into the original image. Each object is separated from the others by "," delimiter.

## Build your own node
Wanna use your own Neural Network? This module is very fast to adapt to every project. 

- If you want to use the original distributed approach, just start this application on which port you want (check firewall eventually). This nodes will be ready to handle requests as soon as they are dispatched.<br><br>
- If you want to use this webserver "as is", without any external node (centralized approach), you can simply put your code into "post_req" method into "obj_detector.py" module. All nodes use **Flask** micro-framework, so you can easily handle the request as you wish. You can also insert your Neural Network into "detector_server.py" module, changing his "detect" method and, optionally, "get_result" method, which control the result format.


## Build your own client
This application is designed to be flexible and intuitive to use. You can easily build your own client to use this service. Just remember the right [result pattern](#response-format). To give a working example, I developed a dedicated [Android Client](https://www.github.com/CianciarusoCataldo/nn-android-client). Just clone the repository, import to Android studio as project, and build the apk. This client can send requests for only objects, optionally.

## Deploy
Every component of this system is ready to be deployed to some cloud services. In fact, I included a Dockerfile, which automates the build of a Docker image, a self contained virtual machine that run the application, with no need to install anything (you can build by your own using <a href="">Docker</a> and push it to a registry, or you can just simply push this repository to <a href="">Docker Hub</a>, to build in the cloud, and have it stored to the Docker Registry. This image can be, then, hosted to various PaaS service, like <a href="https://azure.microsoft.com">Azure Web App</a> or <a href="https://cloud.google.com/appengine/">Google App Engine</a> (it's also included an app.yaml file, for this purpose). I also included some special files used by <a href="https://www.heroku.com/">Heroku</a> PaaS service (Procfile, for example, contains the correct startup command). If you want to use Heroku, you have to add [Apt Buildpack](https://elements.heroku.com/buildpacks/heroku/heroku-buildpack-apt), to install Linux dependencies in Aptfile before deploying.

## See in action
Wanna see how this system work in real time? I deployed a full working system to a PaaS service, <a href="https://www.heroku.com/">Heroku</a>. You can find it <a href="https://nn-image-analyzer.herokuapp.com">here</a>. Send a request as [described before](#configure), and you'll get the result. Check it out !

Due to Heroku resource management, all the servers will be stopped if they don't receive incoming HTTP traffic within 30 minutes (free plan limit, unfortunately I can't mantain a paid service, right now), so don't worry if the service seems to be offline, just wait 1 minutes (at most) and it will be online again. Until you use it, it will never go offline. Sorry for this drawback.
## Built With

* [Flask](http://flask.pocoo.org) - The web framework used
* [Waitress](https://docs.pylonsproject.org/projects/waitress/en/stable/) - The Web Server used for production (you can use your favourite)
* [ImageAI](https://github.com/OlafenwaMoses/ImageAI), modified to support network requests and consume less resources

## Authors

* **Cataldo Cianciaruso** - *Initial work* - [CianciarusoCataldo](https://github.com/CianciarusoCataldo)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details