##Docker Set up
First of all, install [Docker](https://www.docker.com) on your local computer. The get the 

In the Dockerfile folder run:   
```
docker build -t hanabi .
```  
This will build a Docker image that contains *most* of the dependencies needed and everyone will have the same Ubuntu version. After the image is built, run this to create a Docker container:  
```
docker run -it --name=hanabi --privileged -v <path to your local hanabi directory>:/root/hanabi hanabi /bin/bash
```  
This will create a container called hanabi and will bind your local hanabi directory to it. Make sure there are no cmake files (this is important or you won't be able to run any cmake commands). This way, when you enter your docker container, you can still edit the files locally in whatever editor you pick, and all the changes will show up in the container, where you can run all the programs. 

To exit the container, just type ```exit```and then ```docker stop hanabi``` to stop running the container. Next time when you want to enter the container, you will need to type ```docker start hanabi``` to start the container and then ```docker attach hanabi```.

If you need port forwarding, make sure you have [XQuartz](https://www.xquartz.org) installed. Then do the following:  
```xhost + <your IP>```   
```docker start hanabi```  
```docker exec -it -e DISPLAY=<your IP>:0.0 hanabi /bin/bash```  
This will start the docker container with X11 forwarding.







