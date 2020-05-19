#Title: Full Automous Reconstruction of a Shredded Document Using a CNN
#Author: Jamie Alexander Mallett Skinner
#Date: 21/05/2020
#Inputs Needed: 1) Shreds from a document scanned in, orintated, cropped to smiliar sizes with no back ground boarder on them and placed in same directory as this file.
#               2) A .csv file of the shred IDs


#Libray Import
from keras.models import load_model
import cv2 as cv2
import numpy as np
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import spatial
import csv

#change working directory to correct location
os.chdir('D:\\OneDrive - Loughborough University\\4th year\\FYP\\GitHub folder')

#load in pre-trained model
model = load_model('Xception_CNN.h5')

#merge image function
def merge_images(file1, file2):
    """Merge two images into one, displayed side by side
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = width1 + width2
    result_height = max(height1, height2)

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(width1, 0))

    open_cv_image = np.array(result) 

    return open_cv_image

#load in pre-trained model
model1 = load_model('Page_bulider_CNN.h5')

#merge image function
def merge_images_top_bottom(file1, file2):
    """Crop images to small center line and merge into one image, displayed above and below each other
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = max(width1, width2)
    result_height = 90

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, -(height1-45)))
    result.paste(im=image2, box=(0, 45))

    open_cv_image = np.array(result) 

    return open_cv_image

def merge_images_final(file1, file2):
    """Merge two images into one, displayed above and below each other
    :param file1: path to first image file
    :param file2: path to second image file
    :return: the merged Image object
    """
    image1 = Image.open(file1)
    image2 = Image.open(file2)

    (width1, height1) = image1.size
    (width2, height2) = image2.size

    result_width = max(width1, width2)
    result_height = height1 + height2

    result = Image.new('RGB', (result_width, result_height))
    result.paste(im=image1, box=(0, 0))
    result.paste(im=image2, box=(0, height1))

    open_cv_image = np.array(result) 

    return open_cv_image

#read in image id list from .csv
df = pd.read_csv('Labels_Doc_1.csv')
Row_list = []

for index, rows in df.iterrows(): 
    # Create list for the current row 
    my_list =[rows.Image_ID]
    Row_list.append(my_list)

All_shred_ids = (np.squeeze(Row_list)).tolist()

##################--------- Clustering Function ----------############################
row_img_ids = []
frames = {}
for img_id in All_shred_ids:
    shred_pixel_distribution_df = pd.DataFrame(columns=["Number of Black pixels per row"])
    originalImage = cv2.imread(img_id)
    #convert to black and white
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    (thresh, blackandwhiteImage) = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY)
    #size image shreds
    height, width = blackandwhiteImage.shape
    width = range(0,(width),1)
    height = range(0,(height),1)
    #count number of black pixels in each row
    for column_number in height:
        number_of_black_pixels_in_row = 0
        for row_number in width:
            if blackandwhiteImage.item(column_number,row_number) == 0:
                number_of_black_pixels_in_row = number_of_black_pixels_in_row + 1
            else:
                continue
        #append black pixel number to appropiate row in df
        shred_pixel_distribution_df = shred_pixel_distribution_df.append({"Number of Black pixels per row": number_of_black_pixels_in_row}, ignore_index=True)
    #normalise df numbers using threshold
    shred_pixel_distribution_df[shred_pixel_distribution_df < 3] = 0
    shred_pixel_distribution_df[shred_pixel_distribution_df > 2] = 1
    frames[img_id] = (shred_pixel_distribution_df)
#comparing all shreds against each other
smiliarity_dict = {}
All_ids = All_shred_ids
Row_image_ids_df = pd.DataFrame(columns=["List of Images IDs"])
for img_id in frames.keys():
    pixel_array = frames[img_id].values.tolist()
    score_df = pd.DataFrame(columns=["Cosine Smiliarity Score"])
    Image_ID_df = pd.DataFrame(columns=["Image ID"])
    for img_id2 in All_ids:
        pixel_array2 = frames[img_id2].values.tolist()
        smiliarity_score = 1 - spatial.distance.cosine(pixel_array, pixel_array2)
        score_df = score_df.append({"Cosine Smiliarity Score": smiliarity_score}, ignore_index=True)
        Image_ID_df = Image_ID_df.append({"Image ID": img_id2}, ignore_index=True)
        score_image_id_df = pd.concat([score_df, Image_ID_df], axis=1, sort=False)
        row_images = []
        #sort score_image_id_df into order of highest probability to lowest
        score_image_id_df = score_image_id_df.sort_values(['Cosine Smiliarity Score'], ascending=[False])
        score_image_id_df.dropna(subset=['Cosine Smiliarity Score'], inplace=True)
    for score, img_id in zip(score_image_id_df[["Cosine Smiliarity Score"]].values, score_image_id_df[["Image ID"]].values):
        if score > 0.75:
            #append row to new dataframe if smiliarity above threshold
            row_images.extend(img_id.tolist())
            if len(row_images) > (36):
                break
        else:
            continue
    Row_image_ids_df = Row_image_ids_df.append({"List of Images IDs": row_images}, ignore_index=True)


Row_image_ids_df2 = Row_image_ids_df
a = []
index1 = 0
row_lists_df = pd.DataFrame(columns=["List of Rows"])
#take list of all best matching shreds and merge lists with matching shred IDs in them
while index1 < len(Row_image_ids_df2):
    index2 = 0
    c = []
    rows = Row_image_ids_df2.iloc[index1]
    rows = (np.squeeze(rows))
    if (type(rows)!=list):
        Row_image_ids_df2 = Row_image_ids_df2.drop([Row_image_ids_df2.index[0]])
        continue
    Row_image_ids_df3 = Row_image_ids_df2
    #compare row to all over rows
    while index2 < len(Row_image_ids_df3):
        rows2 = Row_image_ids_df3.iloc[index2]
        rows2 = (np.squeeze(rows2))
        if (type(rows2)!=list):
            index2 += 1
            continue
        #compare c to current row
        b = set(rows) & set(rows2)
        #if there are any matching elements in rows append together
        if any(b) == True:
            Row_image_ids_df2[index2:(index2+1)] = pd.DataFrame(columns=["List of Rows"])
            c = rows
            c.extend(rows2)
            c = list(set(c))
            c = sorted(c, key=str.lower)
        index2 += 1
    row_lists_df = row_lists_df.append({"List of Rows": c}, ignore_index=True)
    Row_image_ids_df2 = Row_image_ids_df2.drop([Row_image_ids_df2.index[0]])
row_lists_df = row_lists_df[row_lists_df['List of Rows'].map(lambda d: len(d)) > 0]


#comparing shreds in lists below 4 in lenght against all other lists with a lower theshold smiliarity. This helps sort edge pieces of rows into correct lists
row_lists_final_df = pd.DataFrame(columns=["List of Rows"])
for list_r in row_lists_df.values:
    if len((np.squeeze(list_r)).tolist()) < 4:
        for img_id_edge in np.squeeze(list_r).tolist():
            #compare to other lists
            s = ""
            pixel_array_edge = frames[s.join(img_id_edge)].values.tolist()
            for list_r2 in row_lists_df.values:
                if len((np.squeeze(list_r2)).tolist()) > 2:
                    for img_id in ((np.squeeze(list_r2)).tolist()):
                        pixel_array = frames[img_id].values.tolist()
                        smiliarity_score = 1 - spatial.distance.cosine(pixel_array, pixel_array_edge)
                        if smiliarity_score > 0.58:
                            row_img_ids = ((np.squeeze(list_r2)).tolist()) + img_id_edge.split('  ')
                            row_lists_final_df = row_lists_final_df.append({"List of Rows": row_img_ids}, ignore_index=True)
                            break
    else:
        row_img_ids = (np.squeeze(list_r)).tolist()
        row_lists_final_df = row_lists_final_df.append({"List of Rows": row_img_ids}, ignore_index=True)


#another loop to merge rows with matching shreds ids in them
Row_image_ids_df2 = row_lists_final_df
a = []
index1 = 0
row_lists_df = pd.DataFrame(columns=["List of Rows"])
while index1 < len(Row_image_ids_df2):
    index2 = 0
    c = []
    rows = Row_image_ids_df2.iloc[index1]
    rows = (np.squeeze(rows))
    Row_image_ids_df3 = Row_image_ids_df2
    list_index = []
    #compare row to all over rows
    for rows2 in Row_image_ids_df3.values:
        rows2 = (np.squeeze(rows2).tolist())
        #compare c to current row
        b = set(rows) & set(rows2)
        #if there are any matching elements in rows append together
        if any(b) == True:
            list_index.append(index2)
        index2 += 1
    for index3 in list_index:
        c.extend(np.squeeze(Row_image_ids_df3.iloc[index3]))
    c = list(set(c))
    c = sorted(c, key=str.lower)
    row_lists_df = row_lists_df.append({"List of Rows": c}, ignore_index=True)
    Row_image_ids_df2 = Row_image_ids_df2.drop(Row_image_ids_df2.index[list_index])
row_lists_final_df = row_lists_df[row_lists_df['List of Rows'].map(lambda d: len(d)) > 0]


######################--------- Edge Piece Detection Function ------------###################
number = 0
row_list = []
for row_of_images in row_lists_final_df.values:
    row_of_images = ((np.squeeze(row_of_images)).tolist())
    Left_Hand_Piece = False
    Right_Hand_Piece = False
    #identify left hand piece
    for img_id in row_of_images:
        number_of_white_pixels = 0
        originalImage = cv2.imread(img_id)
        #convert to black and white
        grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
        (thresh, blackandwhiteImage) = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY)
        #size image shreds
        height, width = blackandwhiteImage.shape
        #number of pixels to loop through for a column 
        loop_list = range(0,(height),1)
        #loop to total number of pixels on left hand side
        for pixel_number in loop_list:
            if blackandwhiteImage.item(pixel_number,1) == 0:
                break
            else:
                number_of_white_pixels = number_of_white_pixels + 1
        if number_of_white_pixels == height:
            file1 = img_id
            Left_Hand_Piece = True
            break
    #if left hand piece can't be identified - identify right hand piece
    if Left_Hand_Piece == False:
        for img_id in row_of_images:
            number_of_white_pixels = 0
            originalImage = cv2.imread(img_id)
            grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
            (thresh, blackandwhiteImage) = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY)
            #size image shreds
            height, width = blackandwhiteImage.shape
            #number of pixels to loop through for a column 
            loop_list = range(0,(height),1)
            #loop to total number of pixels on right hand side
            for pixel_number in loop_list:
                if blackandwhiteImage.item(pixel_number,(width-1)) == 0:
                    break
                else:
                    number_of_white_pixels = number_of_white_pixels + 1
            if number_of_white_pixels == height:
                file1 = img_id
                Right_Hand_Piece = True
                #Left_Hand_Piece = False
                break
    #if not edge piece detected use first shred in list and bulid of this
    #if Right_Hand_Piece == False & Left_Hand_Piece == False:
        #file1 = row_of_images[0]
        #Left_Hand_Piece = True
        #Right_Hand_Piece = False

    ######################--------- Row Construction function ---------######################

    #total number of shreds in row minus the starting lefthand piece
    row_of_images.remove(file1)
    #create dataframe to save final shred order to
    Final_df_of_matches = pd.DataFrame(columns=["File1","File2","Prob"])
    #loop which changes file 1 in each loop for the shred to right hand side of inital chosen shred
    if Left_Hand_Piece == True:
        while len(row_of_images) > 0:
            #make inital empty list
            answer = []
            #take one shred and compare to all others
            for file2 in row_of_images: #all other images in file
                #read in merged image and shape as needed for CNN
                img = merge_images(file1, file2)
                img = cv2.resize(img, (90, 335))
                img = np.reshape(img,[1,335,90,3])
                #predict what class the merged image is in
                prob = model.predict(img)
                prob = prob[:,1]
                prob = np.squeeze(prob)
                classes = prob.argmax(axis=-1)
                #save file1 in column 1 and file2 in column 2 and predicted probability of match in column 3
                answer.append([file1,file2,(prob)])
        
            #convert to dataframe
            df = pd.DataFrame(answer, columns=["File1","File2","Prob"])
            #find row with best match
            best_matching_shreds = df[df.Prob == df.Prob.max()]
            #Append that match to new dataframe
            Final_df_of_matches = Final_df_of_matches.append(best_matching_shreds)
            file1 = best_matching_shreds.iloc[0,1]
            row_of_images.remove(best_matching_shreds.iloc[0,1])

        #Merge Shreds in correct order and save
        number+=1
        row_name = 'Row_' + str(number) + '.jpg'
        row_list.append(row_name)
        list_1 = Final_df_of_matches['File2'].to_list()
        list_1.pop(0)
        img = merge_images((Final_df_of_matches.iloc[0,0]),(Final_df_of_matches.iloc[0,1]))
        plt.imsave(row_name, img, cmap='Greys')
        for file2 in list_1:
            img = merge_images(row_name,file2)
            plt.imsave(row_name, img, cmap='Greys')
    #If right hand piece was identified not left
    if Right_Hand_Piece == True:
        while len(row_of_images) > 0:
            #make inital empty list
            answer = []
            #take one shred and compare to all others
            for file2 in row_of_images: #all other images in file
                #read in merged image and shape as needed for CNN
                img = merge_images(file2,file1)
                img = cv2.resize(img, (90, 335))
                img = np.reshape(img,[1,335,90,3])
                #predict what class the merged image is in
                prob = model.predict(img)
                prob = prob[:,1]
                prob = np.squeeze(prob)
                classes = prob.argmax(axis=-1)
                #save file1 in column 1 and file2 in column 2 and predicted probability of match in column 3
                answer.append([file1,file2,(prob)])
        
            #convert to dataframe
            df = pd.DataFrame(answer, columns=["File1","File2","Prob"])
            #find row with best match
            best_matching_shreds = df[df.Prob == df.Prob.max()]
            #Append that match to new dataframe
            Final_df_of_matches = Final_df_of_matches.append(best_matching_shreds)
            file1 = best_matching_shreds.iloc[0,1]
            row_of_images.remove(best_matching_shreds.iloc[0,1])

        #Merge Shreds in correct order and save
        number+=1
        row_name = 'Row_' + str(number) + '.jpg'
        row_list.append(row_name)
        list_1 = Final_df_of_matches['File2'].to_list()
        list_1.pop(0)
        img = merge_images((Final_df_of_matches.iloc[0,1]),(Final_df_of_matches.iloc[0,0]))
        plt.imsave(row_name, img, cmap='Greys') #will need to actively change FinalRow name for each row or after bulid resave under another name.
        for file2 in list_1:
            img = merge_images(file2,row_name)
            plt.imsave(row_name, img, cmap='Greys')


######################--------- Page Construction function ---------######################

#way of identifying top row or bottom row -- user input method -- run code to this point and view rows and enter id
#file1_pg = 'Row_1.jpg'
#Top_Piece = True
#Bottom_Piece = False

#automatic top row finder
for row_id in row_list:
    black_pixel_no = 0
    originalImage = cv2.imread(row_id)
    #convert to black and white
    grayImage = cv2.cvtColor(originalImage, cv2.COLOR_BGR2GRAY)
    (thresh, blackandwhiteImage) = cv2.threshold(grayImage, 200, 255, cv2.THRESH_BINARY)
    #size image shreds
    height, width = blackandwhiteImage.shape
    width = range(0,(width),1)
    height = range(0,5,1)
    #count number of black pixels in each row
    for row_number in height:
        for column_number in width:
            if blackandwhiteImage.item(row_number,column_number) == 0:
                black_pixel_no+=1
            else:
                continue
    if black_pixel_no < 2:
        file1_pg = row_id
        Top_Piece = True
        Bottom_Piece = False
        break
    else:
        continue


number = 0
row_list.remove(file1_pg)
#create dataframe to save final shred order to
Final_df_of_matches = pd.DataFrame(columns=["file1_pg","file2_pg","Prob"])
#loop which changes file 1 in each loop for the shred below the matched shred
if Top_Piece == True:
    while len(row_list) > 0:
        #make inital empty list
        answer = []
        #take one shred and compare to all others
        for file2_pg in row_list:
            #read in merged image and shape as needed for CNN
            img = merge_images_top_bottom(file1_pg, file2_pg)
            img = cv2.resize(img, (1100, 90))
            img = np.reshape(img,[1,90,1100,3])
            #predict what class the merged image is in
            prob = model1.predict(img)
            prob = prob[:,1]
            prob = np.squeeze(prob)
            classes = prob.argmax(axis=-1)
            #save file1_pg in column 1 and file2_pg in column 2 and predicted probability of match in column 3
            answer.append([file1_pg,file2_pg,(prob)])
        
        #convert to dataframe
        df = pd.DataFrame(answer, columns=["file1_pg","file2_pg","Prob"])
        #find row with best match
        best_matching_shreds = df[df.Prob == df.Prob.max()]
        #Append that match to new dataframe
        Final_df_of_matches = Final_df_of_matches.append(best_matching_shreds)
        file1_pg = best_matching_shreds.iloc[0,1]
        row_list.remove(best_matching_shreds.iloc[0,1])

    #Merge Shreds in correct order and save
    number+=1
    page_name = 'Page_' + str(number) + '.jpg'
    list_1 = Final_df_of_matches['file2_pg'].to_list()
    list_1.pop(0)
    img = merge_images_final((Final_df_of_matches.iloc[0,0]),(Final_df_of_matches.iloc[0,1]))
    plt.imsave(page_name, img, cmap='Greys')
    for file2_pg in list_1:
        img = merge_images_final(page_name,file2_pg)
        plt.imsave(page_name, img, cmap='Greys')

if Bottom_Piece == True:
    while len(row_list) > 0:
        #make inital empty list
        answer = []
        #take one shred and compare to all others
        for file2_pg in row_list:
            #read in merged image and shape as needed for CNN
            img = merge_images_top_bottom(file2_pg,file1_pg)
            img = cv2.resize(img, (1100, 90))
            img = np.reshape(img,[1,1100,90,3])
            #predict what class the merged image is in
            prob = model1.predict(img)
            prob = prob[:,1]
            prob = np.squeeze(prob)
            classes = prob.argmax(axis=-1)
            #save file1_pg in column 1 and file2_pg in column 2 and predicted probability of match in column 3
            answer.append([file1_pg,file2_pg,(prob)])
        
        #convert to dataframe
        df = pd.DataFrame(answer, columns=["file1_pg","file2_pg","Prob"])
        #find row with best match
        best_matching_shreds = df[df.Prob == df.Prob.max()]
        #Append that match to new dataframe
        Final_df_of_matches = Final_df_of_matches.append(best_matching_shreds)
        file1_pg = best_matching_shreds.iloc[0,1]
        row_list.remove(best_matching_shreds.iloc[0,1])

    #Merge Shreds in correct order and save
    number+=1
    page_name = 'Page_' + str(number) + '.jpg'
    list_1 = Final_df_of_matches['file2_pg'].to_list()
    list_1.pop(0)
    img = merge_images_final((Final_df_of_matches.iloc[0,1]),(Final_df_of_matches.iloc[0,0]))
    plt.imsave(page_name, img, cmap='Greys')
    for file2_pg in list_1:
        img = merge_images_final(file2_pg,page_name)
        plt.imsave(page_name, img, cmap='Greys')
