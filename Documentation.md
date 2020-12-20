# Documentation 
- Requirments 
- Running The Code 
- Code Explanation 
___
##### Python Implementation of [Resutarant Menu Expert](https://web.stanford.edu/class/ee368/Project_Autumn_1516/Reports/Wang_Chen_Lang.pdf)
___
### Requirments
- Python 3
- OpenCV 2.0
- Numpy
- Matplotlib
- Scikit-image 0.17.1
- Pillow 2.2.1
- Pytesseract 0.3.6

Development is done on Linux Environment 
___

### Running The Code

#### Setup
- Clone The Repository
- Install All The Requirments
- Put the dish images inside the **menu_items** folder in **img**
- Install tesseract-ocr
```sh
        $ sudo apt-get install tesseract-ocr
``` 

### Running Script
- Run the script 'main.py' in src with arguments , image path and maximum allowed edit distance  
```sh
        $ python3 main.py <img_path> <max_dist>
```
- Example :
```sh
        $ python3 main.py --img_path '../img/test.jpg' --max_dist 4
```

#### Things to take care of 
- Put all the disk images that you want to visulize in the menu_item folder with .jpeg extension
- Names of dish images must be same as written in the menu
- Don't choose dish images that are very large or small. It may cause streching or compression of the dish image in output  
- Choose clean menu Images that dont have much noise or not blurred 
___

### Code Explanation

##### 1) Resturant Menu Class 
- Class implements the complete pipeline for resturant menu expert 
- Takes two parameters 
    1. img_path = path of image of menu
    2. max_dist = maximum allowed edit distance
- Example:
     ```py
        from resturant_menu import resturant_menu_expert
        resturant_menu_expert('../img/test.jpg' ,4)
    ```

##### 2) get_roi ( img_path )
- Description : Read The Image And provides user interface to get ROI
- Parameters : 
    - img_path :Path Of the Image 
- Output : Return the numpy array of user selected ROI 

##### 2) get_otsu ( img ) 
- Description : Applies OTSU Thresholding on the gray image 
- Parameters : 
    - img : Numpy Array of Image  
- Output : Returns numpy array of Binary Image  

##### 3) invert ( img )
- Description : Applies negative operation on the image
- Parameters:
    - img : Numpy array of Image 
- Output : Returns numpy array of inverted image  

##### 4) disk_dilate( img ,radius):
- Description : Disk Dilation Operation with 
- Parameters :
    - img : Numpy array of Image 
    - radius : Structural element
- Output : Returns numpy array of dilated image  

##### 5) get_bounding_boxes( num_labels , labels_im )
- Description : Generates The Bounding Boxes Around the connected componants ,format ( label ,width ,hight ,x ,y ) 
- Parameters : 
    - num_labels : Connected Componants Labels 
    - labels_im : Labeld image of connected componants
- Output : returns the array of bounding boxes  

##### 6) merge_bounding_boxes( boxes ,dx ,dy)
- Description : Merges bounding boxes that are allmost in one horizontal line 
- Parameters:
    - boxes : Array of bounding boxes 
    - dx : Allowed diffrrance in x direction between two bounding boxes to merge
    - dy : Allowed diffrrance in y direction between two bounding boxes to merge
- Output : returns the array of bounding boxes after merging 

##### 7) rotate_image( mat, angle )
- Description : Rotates the image with given angle without cropping the image 
- Parameters:
    - mat : Numpy array if image 
    - angle : angle to rotate the image 
- Output : Returns numpy array of rotated image 

##### 8) fast_featureless_rotation( img )
- Description : Find the correct angle of rotation that makes image horizontal
- Parameters:
    - img : Numpy array of Image 
- Output : Returns found angle of rotation 

##### 9) line_dilate( img )
- Description : Applies Dilation operation with Rectangular element if (10,2)
- Parameters :
    - img : Numpy array of Image 
- Output : Returns numpy array of dilated image  

##### 10) dish_name_segmentation( dilated_img ,img )
- Description : Merges the bounding boxes and segments the dish names 
- Parameters :
    - dilated_img : Image with line dilation 
    - img : Original Image rotated with angle found in featureless rotation
- Output : Return Segmented images each consist of dish name  

##### 11) ocr( segs )
- Description : Applies Dilation operation with Rectangular element if (10,2)
- Parameters:
    - img : Numpy array of Image 
- Output:
    - Returns numpy array of dilated image 

##### 12) get_database()
- Description : Iterates over dish images folder and generates database 
- Parameters:
- Output : Returns list of dish images names availabel in database  

##### 13) edit_distance( s1 ,s2 ,max_dist)
- Description : Minimum edit distance algoeithm	
- Parameters : 
    - s1 ,s2 : Strings 
    - max_dist : maximum edit distance allowed  
- Output : Returns minimum edit distance to convert s1 to s2  

##### 14) db_lookup(test_str , db ,max_dist)
- Description : Looks up for correct name of dish in the database
- Parameters :
    - test_str : dish name to look up 
    - db : database of the images 
    - max_dist : maximum distance allowed for correction 
- Output:
    - Returns list of dish names with correct text (if found) ,None otherwise 
    
##### 15) OCR_Correction(final_text ,db ,max_dist)
- Description : Runs OCR Correction Algorithm on each detected dish name from OCR
- Parameters : 
    - final_text : List of Dish names Resulted from the OCR  
- Output : Returns List of corrected dish names 

#### 16) get_finla_output( menu , dish_names ,final_angle)
- Description : generates Final Output image with dish images along with dish names in the ROI
- Parameters:
    - final_text : List of Dish names Resulted from the OCR  
- Output : Returns final Output Image 
