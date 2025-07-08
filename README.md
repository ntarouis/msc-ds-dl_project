# msc-ds-dl_project

🚦⭐ This is our project on road object classification. ⭐🚦

## Project Overview

This project focuses on classifying road objects into **three classes**:
- Traffic signs 🛑
- Cars 🚗
- Pedestrians 🚶

To tackle this, we have implemented three different models:
- A simple CNN
- A deeper CNN
- A transfer learning approach using VGG

## Dataset

The dataset we used for training our models is:
*https://www.kaggle.com/datasets/mikoajkoek/traffic-road-object-detection-polish-12k* <br>
Please download the data, create a new folder inside with the name "original" and insert the files there.

## Installation

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

> **Important:**  
> The VGG model checkpoint is too large for the GitHub file size limit, so we've uploaded it to Google Drive.  
> Please use the provided [link](https://drive.google.com/file/d/1U7YZo03henkOrEvazDZ9AlFz3pboSKlR/view?usp=drive_link) to download the `.pth` file and place it inside the `model_checkpoints` folder.
> Thank you :nerd_face:

## Code Structure

- **data/**  
  Where all the raw and processed datasets live. Feed your models from here!

- **model_checkpoints/**
  This is where we’ve saved our trained models `.pth` files—ready to be loaded and tested!

- **src/**  
  Here is all the code we created. Inside you'll find:
  - **models/** – All our neural network architectures and model code.
  - **preprocessing/** – Scripts and utilities for cleaning, cropping, and prepping your data.
  - **training/** – Training loops, utilities, and everything we used to teach your models new tricks.
      <br>*Bonus:* Inside the training folder, we've also included an end-to-end script that takes you from reading the original data,doing all the necessary preprocessing all the way to training the model!
  - **testing/** – Evaluation scripts to see how well our models perform.
      <br>*Bonus:* In the testing folder, you'll find scripts that load the test loader and evaluate all three models in one go.


🎉 **Bonus Fun!**  
    Feeling adventurous? Check out our playground notebook, where you can use our best model and evaluate it on a totally new dataset. 
    `test_on_new/test_on_new_dataset_playground.ipynb`
---
