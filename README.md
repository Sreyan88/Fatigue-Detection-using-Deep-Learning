# Fatigue_Detection
This repository contains a project based on the research to detect fatigue levels of a person through a photograph. For this project the main facial cues used to detect fatigue levels are: Undereyes, Eyes, Mouth, Nose and Skin.
<p align="center">
<img src="https://user-images.githubusercontent.com/33536225/84591184-04db6c00-ae5a-11ea-9005-411dcc9a9d2f.png" height="400" >
</p>

# Problem Statement
The complexities of fatigue have drawn much attention from researchers across various disciplines. Short-term fatigue may cause safety issue while driving; thus, dynamic systems were designed to track driver fatigue. Longterm fatigue could lead to chronic syndromes, and eventually affect individuals physical and psychological health. Traditional methodologies of evaluating fatigue not only require sophisticated equipment but also consume enormous time. 
<p align="center">
<img src="https://user-images.githubusercontent.com/33536225/84591047-fccefc80-ae58-11ea-95cb-1a74fdb9f107.png" height="200" >
</p>

# Our Proposal
In this project, we attempt to develop a novel and efﬁcient method to predict individual’s fatigue rate by scrutinising human facial cues. Our goal is to predict fatigue rate based on a single photo. Our work represents a promising way to assess sleep-deprived fatigue, and our project will provide a viable and efﬁcient computational framework for user fatigue modelling in large-scale.
<p align="center">
<img src="https://user-images.githubusercontent.com/33536225/84591148-c04fd080-ae59-11ea-954e-65f649376e42.png" height="400" >
</p>

# Architecture
The architecture for this project is shown in the picture below. The image of a face is taken as input and the facial landmarks are detected and cropped out. These cropped out facial landmarks such as eyes, undereyes, nose, mouth along with the entire face image for the skin is fed into individual models trained on these specific features. The individual models return a value which corresponds to the fatigue levels. These values are then taken as a weighted sum (where eyes and undereyes are given more weightage) which is used as the final value to determine the fatigue level of a person.
<p align="center">
<img src="https://user-images.githubusercontent.com/33536225/84591220-466c1700-ae5a-11ea-8c55-776a7c44c459.png" height="400" >
</p>
# Setup Instructions

First it is required to clone the entire repository into your local machine. There are two google drive links present inside model folder and the object detection folder. These two links need to be downloaded and placed into these folders respectively.
