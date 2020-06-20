# Fatigue Detection using Deep Learning

This repository contains a project based on the research to detect fatigue levels of a person through a photograph. For this project the main facial cues used to detect fatigue levels are: Undereyes, Eyes, Mouth, Nose and Skin.

# Problem Statement
The complexities of fatigue have drawn much attention from researchers across various disciplines. Short-term fatigue may cause safety issue while driving; thus, dynamic systems were designed to track driver fatigue. Longterm fatigue could lead to chronic syndromes, and eventually affect individuals physical and psychological health. Traditional methodologies of evaluating fatigue not only require sophisticated equipment but also consume enormous time.

# Our Proposal
In this project, we attempt to develop a novel and efﬁcient method to predict individual’s fatigue rate by scrutinising human facial cues. Our goal is to predict fatigue rate based on a single photo. Our work represents a promising way to assess sleep-deprived fatigue, and our project will provide a viable and efﬁcient computational framework for user fatigue modelling in large-scale.

# Architecture
The architecture for this project is shown in the picture below. The image of a face is taken as input and the facial landmarks are detected and cropped out. These cropped out facial landmarks such as eyes, undereyes, nose, mouth along with the entire face image for the skin is fed into individual models trained on these specific features. The individual models return a value which corresponds to the fatigue levels. These values are then taken as a weighted sum (where eyes and undereyes are given more weightage) which is used as the final value to determine the fatigue level of a person.
<p align="center">
<img src="https://user-images.githubusercontent.com/33536225/84591220-466c1700-ae5a-11ea-8c55-776a7c44c459.png" height="400" >
</p>

# Setup Instructions

<li>Clone the entire repository into your local machine.</li>
<li>Download contents of object_detecion folder from zenodo and place it in the folder.</li>
<li>Download models from zenodo and place it in models/image_classification.</li>


<ol>
  <p> Setup a new environment and upgrade pip</p>
   
  ```
   conda create -n tensorflow1 pip python=3.6
   python -m pip install --upgrade pip
  ```

  <p>Install tensorflow-gpu in this environment using:</p>
  
  ```
  pip install --ignore-installed --upgrade tensorflow-gpu
  ```
  <p>All other requirements can be installed using requirements.txt</p>
  
  ```
   pip install -r requirements.txt
  ```

<li> After all the package installation has been done run "app.py":
  
  ```
  python app.py
  ```
<p align="center"> After running the above command you should get a screen that looks like this.
<img src="https://user-images.githubusercontent.com/33536225/84591789-db710f00-ae5e-11ea-99a5-a2460819c80c.png" height="200" >
Copy the url right after Running on and paste it in your browser. 
</p>

<li> After running the python script and copying the link to the browser you should get this screen.</li><br>
<p align="center"> This is the homepage of the project.
<img src="https://user-images.githubusercontent.com/33536225/84591854-4ae6fe80-ae5f-11ea-9d95-6b4dd26b2b1f.png" height="400" >
</p>  
<p> Here you can upload the image using the browse button and selecting the image to check the fatigue levels. Here we have selected an image and are ready to detect the fatigue levels. </p>
<p align="center">
<img src="https://user-images.githubusercontent.com/33536225/84592096-a1a10800-ae60-11ea-98bf-4492f90936bb.png" height="400" >
</p>
<p> After clicking on the predict button this is the result that is found. </p>
<p align="center">Some information based on the results
<img src="https://user-images.githubusercontent.com/33536225/84592141-f9d80a00-ae60-11ea-8ac6-d5b082657820.png" height="400" >
</p>
<br>
<p align="center"> Final score given to the image:
<img src="https://user-images.githubusercontent.com/33536225/84592151-0fe5ca80-ae61-11ea-96bc-600a63c0023a.png" height="400" >
</p>
<br>
<p align="center"> Actual score given to each individual facial landmark
<img src="https://user-images.githubusercontent.com/33536225/84592167-3146b680-ae61-11ea-96fe-1d66b4b57595.png" height="400" >
</p>

### Note : The lower the score, the higher level of fatigue.

Current aggregation used for final score:
  ```
(((sum of left eye and right eye scores) / 2)*0.4) + (((sum of left under-eye and right under-eye scores)/2)*0.55) + (((sum of nose, face and mouth scores)/3)*0.05
  ```
This aggregation has been done through basic intuitive hypothesis. Please feel free to assign weights according your own hypothesis. For eg. - Linear Regression can be used to assign specific weights to each part of the face.
<p> Main contributors of the project:
  <ol>
    <li><a href="https://www.linkedin.com/in/sreyan-ghosh/">Sreyan Ghosh</a></li>
    <li><a href="https://www.linkedin.com/in/sherwinjosephsunny/">Sherwin Joseph<a/></li>
    <li><a href="https://www.linkedin.com/in/rohan-roney-a652a515a/">Rohan Roney</a></li>
    <li><a href="https://www.linkedin.com/in/samden-lepcha/">Samden Lepcha</a></li>
</ol>

# Extras:

### Credits:
<p> You can follow this <a href="https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10" >link</a> for an excellent tutorial to make train your own custom object detection model. Our under-eye object detection as heavily based on this.</p>
<p> You can follow this <a href="https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv-python/" >link</a> from pyimagesearch for facial part extraction using dlib. Our facial part extraction is heavily based on this.</p>


