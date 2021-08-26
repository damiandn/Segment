
#things to automate
#automatically remove seeds for small areas (> 500)
#autoamtically remove seeds that are more than 200 pixels from the nearest seed


from aicsimageio import AICSImage
from aicsimageio.writers import ome_tiff_writer
import cv2
import h5py
from IPython.display import clear_output, display
from magicgui import magicgui
import math
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import napari
import numpy as np
import operator
import os
import pandas as pd
from pathlib import Path
import pickle
from qtpy.QtWidgets import QSlider
from qtpy.QtCore import Qt
import random
from scipy import ndimage as ndi
from scipy.ndimage.morphology import distance_transform_edt
import scipy.misc
from scipy.interpolate import interpn
from scipy.ndimage import zoom
from skimage import io
from skimage.filters import threshold_otsu
from skimage.filters import threshold_mean
import skimage.filters
from skimage.measure import label, regionprops, marching_cubes_lewiner
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import tifffile
import time
import shutil

plt.rcParams['figure.figsize'] = (16.0, 12.0)

surf_vol = 0.325 * 0.325

#make a random color map
import vispy
import vispy.color


colList = np.load(os.path.join('..', 'ColorDict.npy'))
with open (os.path.join('..', 'CellColorDict'), 'rb') as fp:
                cellColorDict = pickle.load(fp)


random_cmap = vispy.color.colormap.Colormap(colList, controls=None, interpolation='zero')

WT_01_05_19_sheath_cells = [3, 5, 6, 8, 10, 14, 15, 16, 18, 28, 29, 33, 34, 36, 38, 42, 50, 51, 56, 58, 76, 81, 83, 87, 95, 96, 97, 99, 106, 122, 127, 130, 133, 135, 136, 152, 159]
WT_01_05_19_leading_cells = [63, 64, 73, 74, 101, 104, 111, 115, 119, 120, 121, 134, 137, 138, 139, 140, 141, 142, 143, 144, 145]
WT_01_05_19_trailing_cells = [7, 19, 20, 23, 26, 27, 30, 32, 45, 46, 52, 61, 62, 67, 68, 71, 80 ,90 ,93, 161, 150, 152]
WT_01_05_19_neuromast_cells = [4, 9, 11, 12, 13, 17, 21, 22, 24, 25, 31, 35, 37, 39, 40, 41, 43, 44, 49, 53, 54, 55, 57, 59, 65, 66, 70, 72, 75, 77, 78, 79, 82, 84, 85, 86, 88, 89, 91, 92, 94, 98, 100, 102, 103, 105, 107, 108, 109, 110, 112, 113, 114, 116, 117, 118, 123, 124, 125, 126, 128, 129, 131, 132, 146, 147, 148, 149, 153, 154, 155, 156, 157, 158, 160]


WT_12_04_20_4hr_sheath_cells = [9, 12, 22, 40, 46, 48, 52, 58, 75, 94, 96, 98, 102, 105, 109, 111, 112, 113, 118, 120, 123, 124, 126, 127, 128, 129, 149]
WT_12_04_20_4hr_leading_cells = [59, 60, 61, 62, 72, 73, 74, 77, 78, 79, 81, 91, 95, 99, 100, 103, 110, 117, 119, 130, 132, 133, 134, 139, 140]
WT_12_04_20_4hr_trailing_cells = [4, 28, 67]
WT_12_04_20_4hr_neuromast_cells = [5, 6, 7, 10, 11, 13, 15, 16, 17, 18, 19, 20, 23, 24, 26, 27, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 47, 49, 50, 51, 53, 54, 55, 56, 57, 63, 64, 65, 66, 68, 69, 70, 71, 76, 80, 82, 83, 84, 85, 86, 87, 88, 89, 90, 92, 93, 97, 101, 104, 106, 107, 108, 114, 115, 116, 121, 122, 125, 131, 135, 136, 137, 138, 141, 142, 143, 144, 145, 146, 147, 148, 150, 151] 

WT_08_21_20_sheath_cells = [3, 23, 29, 35, 38, 44, 52, 54, 55, 60, 74, 83, 85, 91, 92, 95, 96, 100, 101, 102, 103, 109, 114]
WT_08_21_20_leading_cells = [4, 5, 7, 8, 11, 13, 34, 108, 113]
WT_08_21_20_trailing_cells = [87, 88]
WT_08_21_20_neuromast_cells = [6, 9, 10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 36, 37, 39, 40, 41, 42, 43, 45, 46, 47, 48, 49, 51, 53, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82, 84, 86, 89, 90, 93, 94, 97, 98, 99, 115, 116, 118]

print('Functions Running')
        

class DDN:

    class PreProcess:
    
        def RotateImages(path, total_file_nums):
            im = np.load(os.path.join(path, 'processed_images', 'mem', 'temp', 'mem_0_temp.npz'))        
            im = im['arr_0']
            viewer = napari.Viewer()
            viewer.add_image(im)

            @magicgui(call_button='Reset rotation')
            def Reset_rotation():
                viewer.layers.pop(0)
                im = np.load(os.path.join(path, 'processed_images', 'mem',  'temp', 'mem_0_temp.npz'))        
                im = im['arr_0']
                viewer.add_image(im)
            viewer.window.add_dock_widget(Reset_rotation, area='right')

            @magicgui(call_button='Rotate around long axis of PLLp')
            def Rotate_long(rotation:str):
                rotation = int(rotation)
                im = viewer.layers['im'].data
                viewer.layers.pop(0)
                im = scipy.ndimage.rotate(im, rotation, axes=(0, 2))
                viewer.add_image(im)
                #change this to save to save folder
            viewer.window.add_dock_widget(Rotate_long, area='right')

            @magicgui(call_button='Rotate around short axis of PLLp')
            def Rotate_short(rotation:str):
                rotation = int(rotation)
                im = viewer.layers['im'].data
                viewer.layers.pop(0)
                im = scipy.ndimage.rotate(im, rotation, axes=(0, 1))
                viewer.add_image(im)
                #change this to save to save folder
            viewer.window.add_dock_widget(Rotate_short, area='right')

            @magicgui(call_button='Rotate in plane')
            def Rotate_plane(rotation:str):
                rotation = int(rotation)    
                im = viewer.layers['im'].data
                viewer.layers.pop(0)
                im = scipy.ndimage.rotate(im, rotation, axes=(1, 2))
                viewer.add_image(im)
                #change this to save to save folder
            viewer.window.add_dock_widget(Rotate_plane, area='right')

            @magicgui(call_button='Rotate All')
            def Rotate_all(long_axis:str, short_axis:str, plane_axis:str):

                for i in range(0, total_file_nums-1):
                    clear_output(wait=True)
                    print('running timepoint ' + str(i))
                    im = np.load(os.path.join(path, 'processed_images', 'mem', 'temp', 'mem_' + str(i) + '_temp.npz'))        
                    im = im['arr_0']
                    im_nuc = np.load(os.path.join(path, 'processed_images', 'nuc', 'temp', 'nuc_' + str(i) + '_temp.npz'))        
                    im_nuc = im_nuc['arr_0']
                    if int(long_axis) != 0:
                        im = scipy.ndimage.rotate(im, int(long_axis), axes=(0, 2))
                        im_nuc = scipy.ndimage.rotate(im_nuc, int(long_axis), axes=(0, 2))
                    if int(short_axis) != 0:
                        im = scipy.ndimage.rotate(im, int(short_axis), axes=(0, 1))
                        im_nuc = scipy.ndimage.rotate(im_nuc, int(short_axis), axes=(0, 1))

                    if int(plane_axis) != 0:                        
                        im = scipy.ndimage.rotate(im, int(plane_axis), axes=(1, 2))
                        im_nuc = scipy.ndimage.rotate(im_nuc, int(plane_axis), axes=(1, 2))

                    np.savez_compressed(os.path.join(path, 'processed_images', 'mem',  'temp', 'mem_' + str(i) + '_temp.npz'), im)        
                    np.savez_compressed(os.path.join(path, 'processed_images', 'nuc',  'temp', 'nuc_' + str(i) + '_temp.npz'), im_nuc)        

            viewer.window.add_dock_widget(Rotate_all, area='right')



    
    
        def AlignRG(path, total_file_nums):
            im_mem = np.load(os.path.join(os.path.join(path, 'processed_images', 'mem', 'temp', 'mem_' + str(0) + '_temp.npz')))
            im_mem = im_mem['arr_0']
            im_nuc = np.load(os.path.join(os.path.join(path, 'processed_images', 'nuc',  'temp','nuc_' + str(0) + '_temp.npz')))
            im_nuc = im_nuc['arr_0']

            viewer = napari.Viewer()
            viewer.add_image(im_mem, name = 'mem', blending='additive', colormap='green')
            viewer.add_image(im_nuc, name = 'nuc', blending='additive', colormap='red')

            @magicgui(call_button='TranslateAllImages')

            def TranslateAllImages(pixelsUp:int, pixelsDown:int, pixelsLeft:int, pixelsRight:int):
                for i in range(0, total_file_nums):
                
                    im_nuc = np.load(os.path.join(path, 'processed_images', 'nuc', 'temp', 'nuc_' + str(i) + '_temp.npz'))
                    im_nuc = im_nuc['arr_0']
                    
                    
                    if pixelsUp != 0:                    
                        translation_matrix = np.float32([ [1,0,0-pixelsUp], [0,1,0] ])
                        im_nuc = cv2.warpAffine(im_nuc, translation_matrix, (0,0))

                    if pixelsDown != 0:                    
                        translation_matrix = np.float32([ [1,0,pixelsDown], [0,1,0] ])
                        im_nuc = cv2.warpAffine(im_nuc, translation_matrix, (0,0))
                        print('translated down')
                    
                    if pixelsLeft != 0:                    
                        translation_matrix = np.float32([ [1,0,0-pixelsLeft], [0,1,0] ])
                        for i in range(0, im_nuc.shape[0]):
                            temp_subset = im_nuc[i,:,:]
                            translated_subset = cv2.warpAffine(temp_subset, translation_matrix, (0,0))
                            im_nuc[i,:,:] = translated_subset
                            
                    if pixelsRight != 0:                    
                        translation_matrix = np.float32([ [1,0,pixelsRight], [0,1,0] ])
                        for i in range(0, im_nuc.shape[0]):
                            temp_subset = im_nuc[i,:,:]
                            translated_subset = cv2.warpAffine(temp_subset, translation_matrix, (0,0))
                            im_nuc[i,:,:] = translated_subset
                            print('translated right')
                    
                    np.savez_compressed(os.path.join(path, 'processed_images', 'nuc', 'temp', 'nuc_' + str(i) + '_temp.npz'), im_nuc)
                    


            viewer.window.add_dock_widget(TranslateAllImages, area='right')


            @magicgui(call_button='TranslateNucleiUp')
            def TranslateNucleiUp():
                
                im_nuc = viewer.layers['nuc'].data

                translation_matrix = np.float32([ [1,0,-1], [0,1,0] ])
                
                im_nuc = cv2.warpAffine(im_nuc, translation_matrix, (0,0))
                viewer.layers.pop(1)
                viewer.add_image(im_nuc, name = 'nuc', blending='additive', colormap='red')
                
            viewer.window.add_dock_widget(TranslateNucleiUp, area='right')

            @magicgui(call_button='TranslateNucleiDown')

            def TranslateNucleiDown():

                im_nuc = viewer.layers['nuc'].data

                translation_matrix = np.float32([ [1,0,1], [0,1,0] ])

                
                im_nuc = cv2.warpAffine(im_nuc, translation_matrix, (0,0))
                viewer.layers.pop(1)
                viewer.add_image(im_nuc, name = 'nuc', blending='additive', colormap='red')

                
            viewer.window.add_dock_widget(TranslateNucleiDown, area='right')


            @magicgui(call_button='TranslateNucleiLeft')
            def TranslateNucleiLeft():

                im_nuc = viewer.layers['nuc'].data

                translation_matrix = np.float32([ [1,0,-1], [0,1,0] ])

                for i in range(0, im_nuc.shape[0]):
                    temp_subset = im_nuc[i,:,:]
                    translated_subset = cv2.warpAffine(temp_subset, translation_matrix, (0,0))
                    im_nuc[i,:,:] = translated_subset
                

                #im_nuc = cv2.warpAffine(im_nuc, translation_matrix, (0,0))
                viewer.layers.pop(1)
                viewer.add_image(im_nuc, name = 'nuc', blending='additive', colormap='red')
              
                
            viewer.window.add_dock_widget(TranslateNucleiLeft, area='right')


            @magicgui(call_button='TranslateNucleiRight')

            def TranslateNucleiRight():

                im_nuc = viewer.layers['nuc'].data

                translation_matrix = np.float32([ [1,0,1], [0,1,0] ])

                for i in range(0, im_nuc.shape[0]):
                    temp_subset = im_nuc[i,:,:]
                    translated_subset = cv2.warpAffine(temp_subset, translation_matrix, (0,0))
                    im_nuc[i,:,:] = translated_subset
                
                viewer.layers.pop(1)
                viewer.add_image(im_nuc, name = 'nuc', blending='additive', colormap='red')

                    
            viewer.window.add_dock_widget(TranslateNucleiRight, area='right')

        
        def OpenAlignmentViewer(im, path, chunk_nums):
            viewer = napari.Viewer()
            viewer.add_image(im, name = 'im', blending='additive', colormap='green')
            points_layer = viewer.add_points(name='align_2', ndim =4)
            points_layer = viewer.add_points(name='align_1', ndim =4)
            
            @magicgui(call_button='Align All Images')
            def getPoints():
                align_point_1 = viewer.layers['align_1'].data.astype(int)
                align_point_2 = viewer.layers['align_2'].data.astype(int)
                print('point 1 = ' + str(align_point_1))
                print('point 2 = ' + str(align_point_2))
                delta = align_point_2 - align_point_1 
                delta=delta[0]
                translation_matrix = np.float32([ [1,0,-delta[3]], [0,1,-delta[2]]])
                DDN.PreProcess.translate_all_images(translation_matrix, delta[1], delta[2], delta[3], path, chunk_nums)
            
            @magicgui(call_button='Check Alignment')
            def checkAlignment():
                align_point_1 = viewer.layers['align_1'].data.astype(int)
                align_point_2 = viewer.layers['align_2'].data.astype(int)
                print('point 1 = ' + str(align_point_1))
                print('point 2 = ' + str(align_point_2))
                delta = align_point_2 - align_point_1 
                delta=delta[0]
                data = viewer.layers['im'].data
                data_part_2 = data[1,:,:,:]
                temp_im = np.copy(data_part_2)
                translation_matrix = np.float32([ [1,0,-delta[3]], [0,1,-delta[2]]])
                
                print(translation_matrix)
                for i in range(0, temp_im.shape[0]):
                            temp_subset = temp_im[i,:,:]
                            translated_subset = cv2.warpAffine(temp_subset, translation_matrix, (0,0))
                            temp_im[i,:,:] = translated_subset
                blank_im = np.zeros(data_part_2.shape)
                for i in range(0, data_part_2.shape[0]):
                    try:
                        blank_im[i,:,:] = temp_im[i + delta[1],:,:]
                    except:
                        print("")
                part_1 = data[0,:,:,:]
                viewer.layers.remove('im')
                viewer.add_image(data[0,:,:,:], name = 'part_1', blending='additive', colormap='green');
                viewer.add_image(blank_im, name = 'part_2', blending='additive', colormap='red');
            
            viewer.window.add_dock_widget(checkAlignment)    
            viewer.window.add_dock_widget(getPoints)    
            return viewer

        def OpenCropViewer(path, total_file_nums):
    
            im1 = np.load(os.path.join(path, 'processed_images', 'mem', 'temp', 'mem_0.npz'))
            im1 = im1['arr_0']
            im2 = np.load(os.path.join(path, 'processed_images', 'mem', 'temp', 'mem_' + str(total_file_nums) + '.npz'))
            im2 = im2['arr_0']
    
              
            viewer = napari.Viewer()
            viewer.add_image(im1, name = 'first_image', blending='additive', colormap='gray')
            viewer.add_image(im2, name = 'last_image', blending='additive', colormap='gray')

            
            points_layer = viewer.add_points(name='x_min', ndim =4)
            points_layer = viewer.add_points(name='x_max', ndim =4)
            points_layer = viewer.add_points(name='y_min', ndim =4)
            points_layer = viewer.add_points(name='y_max', ndim =4)
            points_layer = viewer.add_points(name='z_min', ndim =4)
            points_layer = viewer.add_points(name='z_max', ndim =4)

            @magicgui(call_button='Preview Crop')
            def PreviewCrop():
                x_min = viewer.layers['x_min'].data.astype(int)[0][3]
                x_max = viewer.layers['x_max'].data.astype(int)[0][3]
                y_min = viewer.layers['y_min'].data.astype(int)[0][2]
                y_max = viewer.layers['y_max'].data.astype(int)[0][2]
                z_min = viewer.layers['z_min'].data.astype(int)[0][1]
                z_max = viewer.layers['z_max'].data.astype(int)[0][1]

                data_1 = viewer.layers['first_image'].data
                data_2 = viewer.layers['last_image'].data

                data_1 = data_1[z_min:z_max, y_min:y_max, x_min:x_max]
                data_2 = data_2[z_min:z_max, y_min:y_max, x_min:x_max]

                viewer.layers.remove('first_image')
                viewer.layers.remove('last_image')

                viewer.add_image(data_1, name = 'first_image_crop', blending='additive', colormap='gray')
                viewer.add_image(data_2, name = 'last_image_crop', blending='additive', colormap='gray')
            
            viewer.window.add_dock_widget(PreviewCrop, area='right')
            
            
            @magicgui(call_button='Crop All Frames')
            def CropAll():
                x_min = viewer.layers['x_min'].data.astype(int)[0][3]
                x_max = viewer.layers['x_max'].data.astype(int)[0][3]
                y_min = viewer.layers['y_min'].data.astype(int)[0][2]
                y_max = viewer.layers['y_max'].data.astype(int)[0][2]
                z_min = viewer.layers['z_min'].data.astype(int)[0][1]
                z_max = viewer.layers['z_max'].data.astype(int)[0][1]

                for i in range(0, total_file_nums):
                        clear_output(wait=True)
                        print('running image ' + str(i))
                        im = np.load(os.path.join(path, 'processed_images', 'mem', 'temp', 'mem_' + str(i) + '.npz'))
                        im = im['arr_0']
                        im = im[z_min:z_max, y_min:y_max, x_min:x_max]
                        np.savez_compressed(os.path.join(path, 'processed_images', 'mem',  'temp', 'mem_' + str(i) + '_temp'), im)
                        
                        im = np.load(os.path.join(path, 'processed_images', 'nuc',  'temp', 'nuc_' + str(i) + '.npz'))
                        im = im['arr_0']
                        im = im[z_min:z_max, y_min:y_max, x_min:x_max]
                        np.savez_compressed(os.path.join(path, 'processed_images', 'nuc',  'temp', 'nuc_' + str(i) + '_temp'), im)
                
                
                        #change this to save to save folder
            viewer.window.add_dock_widget(CropAll, area='right')
           
            
            return viewer

        def CheckAllFiles(path, start = 0, end = 119):
            #check the sequence
            im_mem = np.load(os.path.join(path, 'processed_images', 'mem', 'temp', 'mem_0_temp.npz'))
            im_mem = im_mem['arr_0']
            #im_nuc = np.load(os.path.join(path, 'processed_images', 'nuc', 'temp', 'nuc_0_temp.npz'))
            #im_nuc = im_nuc['arr_0']

            shape = im_mem.shape

            image_mem = np.zeros((end, shape[0], shape[1], shape[2]), dtype=np.uint16)
            image_nuc = np.zeros((end, shape[0], shape[1], shape[2]), dtype=np.uint16)
            

            for i in range(start, end):
                clear_output(wait=True)
                print('running image ' + str(i))
                im_mem = np.load(os.path.join(path, 'processed_images', 'mem', 'temp', 'mem_' + str(i) + '_temp.npz'))
                im_mem = im_mem['arr_0']
                im_nuc = np.load(os.path.join(path, 'processed_images', 'nuc', 'temp', 'nuc_' + str(i) + '_temp.npz'))
                im_nuc = im_nuc['arr_0']
                image_mem[i,:,:,:] = im_mem
                image_nuc[i,:,:,:] = im_nuc

            viewer = napari.Viewer()
            viewer.add_image(image_mem, name = 'im_mem', blending='additive', colormap='green')
            viewer.add_image(image_nuc, name = 'im_nuc', blending='additive', colormap='red')

            @magicgui(call_button='Finalize Images')
            def Finalize():
                for i in range(start, end): 
                    src = os.path.join(path, 'processed_images', 'mem', 'temp', 'mem_' + str(i) + '_temp.npz')
                    dst = os.path.join(path, 'processed_images', 'mem', 'mem_' + str(i) + '.npz')
                    shutil.copy(src, dst)
                    src = os.path.join(path, 'processed_images', 'nuc', 'temp', 'nuc_' + str(i) + '_temp.npz')
                    dst = os.path.join(path, 'processed_images', 'nuc', 'nuc_' + str(i) + '.npz')
                    shutil.copy(src, dst)
        

            viewer.window.add_dock_widget(Finalize, area='right')
           

            return viewer
                
        def OpenErosionTest(path, tp):
            
            im = DDN.Segmentation.SimpleSegmentNuclei(path, tp)
            mem_im = np.load(os.path.join(path, 'processed_images', 'mem', 'mem_' + str(tp) + '.npz'))
            mem_im = mem_im['arr_0']
            
            viewer = napari.Viewer()
            labels, _ = ndi.label(im)
            viewer.add_image(mem_im, name = 'mem', blending='additive', colormap='gray')   
            layer = viewer.add_image(labels.astype(np.uint8), name = 'seeds', blending='additive', colormap=random_cmap)
            
            @magicgui(call_button='Erode Seeds')
            def Erode():
                data = viewer.layers['seeds'].data
                data = skimage.morphology.binary_erosion(data)
                data = skimage.morphology.remove_small_objects(data, min_size=10)
                data = skimage.morphology.binary_dilation(data)
                data = skimage.morphology.binary_erosion(data)
                labels, _ = ndi.label(data)
                layer.data = labels
                        
            viewer.window.add_dock_widget(Erode, area='right')

         
            
            return viewer

    class Segmentation:
        
        def CleanUpSinglePointSegmentation(seg):
            print('removing small objects inside other non-background objects')
            
           
        
        
        def get_3dseed_from_all_frames(bw, stack_shape, hole_min=0, bg_seed = True):
            from skimage.morphology import remove_small_objects
            out = remove_small_objects(bw>0, hole_min)
            out1 = label(out)
            stat = regionprops(out1)
            seed = np.zeros(stack_shape)
            seed_count=0
            if bg_seed:
                seed[0,:,:] = 1
                seed_count += 1
            for idx in range(len(stat)):
                pz, py, px = np.round(stat[idx].centroid)
                seed_count+=1
                seed[int(pz),int(py),int(px)]=seed_count
            return seed, seed_count
        
        def SimpleSegmentNuclei(path, timepoint, save=False):
            mem_im = np.load(os.path.join(path, 'processed_images', 'mem', 'mem_' + str(timepoint) + '.npz'))
            mem_im = mem_im['arr_0']          
            nuc_im = np.load(os.path.join(path, 'processed_images', 'nuc', 'nuc_' + str(timepoint) + '.npz'))
            nuc_im = nuc_im['arr_0']           
            sub = cv2.subtract(nuc_im,mem_im)           
            gauss = skimage.filters.gaussian(sub, sigma=1)
            thresh = skimage.filters.threshold_otsu(gauss)
            binary = gauss <= thresh
            binary = np.invert(binary)

            if save:
                np.savez_compressed(os.path.join(path, 'seg', 'nuc', 'nuc_' + str(timepoint) + '.npz'), binary)

            return binary

    
        def ErodeNucleiForSeeds(path, data, num_rounds, timepoint, save=False):
            
            for i in range(0, num_rounds):
                data = skimage.morphology.binary_erosion(data)
                data = skimage.morphology.remove_small_objects(data, min_size=10)
                data = skimage.morphology.binary_dilation(data)
                data = skimage.morphology.binary_erosion(data)
            if save:
                np.savez_compressed(os.path.join(path, 'seg', 'seeds', 'seed_' + str(timepoint) + '.npz'), data)
               
            return data
       
        def SegmentImage(path, timepoint, gauss_sigma=1, save=True):
            clear_output(wait=True)
            print('running timepoint ' + str(timepoint))
            mem = np.load(os.path.join(path, 'processed_images', 'mem', 'mem_' + str(timepoint) + '.npz'))
            mem = mem['arr_0']
            seed_image = np.load(os.path.join(path, 'seg', 'seeds', 'seed_' + str(timepoint) + '.npz'))
            seed_image = seed_image['arr_0']
            mem_img = skimage.filters.gaussian(mem, sigma=gauss_sigma) 
            seed, seed_count = DDN.Segmentation.get_3dseed_from_all_frames(seed_image, seed_image.shape)
            segmented = watershed(mem_img, seed.astype(int), watershed_line=True)
            lines = segmented>0
            lines = skimage.util.invert(lines)
            if save:
                np.savez_compressed(os.path.join(path, 'seg', 'mem', 'seg_' + str(timepoint) + '.npz'), lines)
            return lines

        def SegmentFromSeedArray(path, shape, timepoint, seed_array, gauss_sigma = 1, save=False):

            clear_output(wait=True)
            print('running timepoint ' + str(timepoint))    
            newSeed = np.zeros(shape)
            newSeed[0,:,:] = 1
            seed_int = 2
            for seed in seed_array:
                if seed[0] == timepoint:
                    try:
                        newSeed[int(seed[1]), int(seed[2]), int(seed[3])] = seed_int
                        seed_int = seed_int + 1
                    except IndexError as e:
                        print('skipping seed')            
            mem = np.load(os.path.join(path, 'processed_images', 'mem', 'mem_' + str(timepoint) + '.npz'))
            mem = mem['arr_0']
            mem_img = skimage.filters.gaussian(mem, sigma=gauss_sigma) 
            segmented = watershed(mem_img, newSeed.astype(int), watershed_line=True)
            lines = segmented>0
            lines = skimage.util.invert(lines)
            if save:
                np.savez_compressed(os.path.join(path, 'seg', 'mem', 'seg_' + str(timepoint) + '.npz'), lines)
            return lines

        def GetSeedArray(path, min_tp, max_tp):

            all_centroids = []
            Path(os.path.join(path, 'seg', 'seeds', 'array')).mkdir(parents=True, exist_ok=True)
            for i in range(min_tp, max_tp):
                clear_output(wait=True)
                print('running timepoint ' + str(i))
                labs = np.load(os.path.join(path, 'seg', 'seeds', 'seed_' + str(i) + '.npz'))
                labs = labs['arr_0']
                seed, seed_count = DDN.Segmentation.get_3dseed_from_all_frames(labs, labs.shape)
                currentProps = regionprops(seed.astype(np.uint8))
                for prop in currentProps:
                    centroid = prop['centroid']
                    new_centroid = [i, centroid[0], centroid[1], centroid[2]]
                    all_centroids.append(new_centroid)
            
            np.save(os.path.join(path, 'seg', 'seeds', 'array', 'seeds_array.npy'), all_centroids)
            return all_centroids

    class Mesh:
    
        def GetMesh(myDict, track, timepoint, linewidth = 0.2, alpha = 0.3, edgecolor = 'k', facecolor = 'w'):
            verts = myDict[timepoint][track]['verts']
            faces = myDict[timepoint][track]['faces']
            mesh = Poly3DCollection(verts[faces], linewidths=linewidth, alpha=alpha)
            mesh.set_edgecolor(edgecolor)
            mesh.set_facecolor(facecolor)
            return mesh


        def ViewSingleMesh(mesh_dict, cell_dict, cell, tp):
            myMesh = DDN.Mesh.GetMesh(mesh_dict, cell, tp)

            centroid = cell_dict[tp][cell]['Centroid']
            max_range = 35
            
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('fjdksl')
            ax.add_collection3d(myMesh)
 
            #myMesh = DDN.Mesh.GetMesh(mesh_dict, cell, tp+10)

            #centroid = cell_dict[tp][cell]['Centroid']
            #max_range = 35
    
            #ax.add_collection3d(myMesh)

            ax.set_xlim(centroid[0] - max_range, centroid[0] + max_range)
            ax.set_ylim(centroid[1] - max_range, centroid[1] + max_range)
            ax.set_zlim(centroid[2] - max_range, centroid[2] + max_range)
    
    
    
        def MakeWholeTimepointImage(path, myDict, meshDict, tp, centroid = -1, saveFig = False, showFig = True, max_range = 200, folder = os.path.join('F:', 'segmentation', 'analysis', 'surface'), colorDict = colList):
    
            myList = meshDict[tp].keys()
            
            myCentroidList = myDict[tp].keys()
            #myList = [4,5,6]
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            if centroid == -1:
                centroids = []
                for cell in myCentroidList:
                    centroids.append(myDict[tp][cell]['Centroid'])
              
                centroids = np.asarray(centroids)
                            
                whole_centroid_z = np.average(centroids[:,0])
                whole_centroid_y = np.average(centroids[:,1])
                whole_centroid_x = np.average(centroids[:,2])
                ax.set_xlim(whole_centroid_x - max_range, whole_centroid_x + max_range)
                ax.set_ylim(whole_centroid_y - max_range, whole_centroid_y + max_range)
                ax.set_zlim(whole_centroid_z - max_range, whole_centroid_z + max_range)
                    
            else:
                ax.set_xlim(centroid[2] - max_range, centroid[2] + max_range)
                ax.set_ylim(centroid[1] - max_range, centroid[1] + max_range)
                ax.set_zlim(centroid[0] - max_range, centroid[0] + max_range)
            
            
            #ax.set_axis_off()
            
            for cell in myList:
                mesh = DDN.Mesh.GetMesh(meshDict, cell, tp, linewidth=0.1, facecolor = colorDict[cell])
                ax.add_collection3d(mesh)

            #plt.show(fig) #260 works ok
            ax.azim = 235
            if saveFig == True:
                savefile = os.path.join(path, 'analysis', 'whole_nuc', 'whole_nuc_im' + str(tp))
                plt.savefig(savefile,bbox_inches='tight', dpi=800)
            
            if showFig == False:
                plt.close(fig)
            


    class Utils:
    
        def checkDifferenceBetweenSeedArrays(tp, path):
            seed_new_temp = np.load(os.path.join(path, 'seg', 'seeds', 'array', 'seeds_array.npy'))
            seed_old_temp = np.load(os.path.join(path, 'seg', 'seeds', 'array', 'old_seeds.npy'))
            new_list = []
            old_list = []
            for coord in seed_new_temp:
                if coord[0] == tp:
                    new_list.append(coord.tolist())
            for coord in seed_old_temp:
                if coord[0] == tp:
                    old_list.append(coord.tolist())
            for coord in new_list:
                if coord not in old_list:
                    return True
            for coord in old_list:
                if coord not in new_list:
                    return True
            return False

    
        def GetShape(path):
            im = np.load(os.path.join(path, 'processed_images', 'mem', 'mem_0.npz'))
            im = im['arr_0']
            shape = im.shape
            return shape
    
    
        def AreThereNaN(colorMap):
            for element in colorMap:
                if np.isnan(element):
                    return True
            return False
    
        def SaveDict(filename, dict):
            """SaveDict(filename, dict). 
            saves a dictionary to a filename. 
            Params:
            filename: path and filename to save to
            dict: the dict to save
            """
            with open (filename, 'wb') as fp:
                pickle.dump(dict, fp)

        def LoadDict(path, filename):
            """Loads a specific Dictionary by filename"""
            with open (os.path.join(path, 'dicts', filename), 'rb') as fp:
                Dict = pickle.load(fp)
            return Dict

        def LoadLabs(path, filename):
            labs = np.load(os.path.join(path, 'labs', '' + filename + '.npz'))
            labs = labs['arr_0']
            return labs
        
        def calculateDistance(p1, p2):
            """return the distance between two np array centroids, p1 and p2
            """
            p1 = np.asarray(p1)
            p2 = np.asarray(p2)
            squared_dist = np.sum((p1-p2)**2, axis=0)
            dist = np.sqrt(squared_dist)
            return dist

        def makeImageFromDict(dict, cells, timepoint, shape):
            """ Takes input cell array and timepoint, returns npArray image of cells in cell array and timepoint.
                coordinates: A list of Z,Y,Z coordinates
                shape: a tuple of the Z,Y,Z dimensions of the image'''
            """
            img = np.ma.array(np.zeros(shape))
                
            if isinstance(cells, int):
                if cells in dict[timepoint].keys():
                    if 'Coordinates' in dict[timepoint][cells].keys():
                        coord = dict[timepoint][cells]['Coordinates'].astype(int)
                        for inp in coord:
                            img[inp[0],inp[1],inp[2]]=int(cells)
                return img
    
            for cell in cells:
                if cell in dict[timepoint].keys():
                    if 'Coordinates' in dict[timepoint][cell].keys():
                        coord = dict[timepoint][cell]['Coordinates']
                        for inp in coord:
                            img[inp[0],inp[1],inp[2]]=int(cell)
            return img


        

        def makeImageFromCoordinates(coordinates, shape, value = 1):
            """ Takes input coordinates from a dict or prop, and an image shape, and returns a numpy array containing the image
                coordinates: A list of Z,Y,Z coordinates
                shape: a tuple of the Z,Y,Z dimensions of the image'''
            """
            img = np.ma.array(np.zeros(shape))
            for inp in coordinates:#.astype(int):
                img[inp[0],inp[1],inp[2]]=value
            return img

        def isPixelEdge(labs, z, y, x):
            label = labs[z,y,x] 
            top=labs[z-1, y, x]
            bottom=labs[z+1, y, x]
            up=labs[z, y-1, x]
            down=labs[z, y+1,x]
            left=labs[z,y,x-1]
            right=labs[z,y,x+1]  
            if top==label:
                if bottom==label:
                    if up==label:
                        if down==label:
                            if left==label:
                                if right==label:
                                    return False
            return True

        def getBoundingBox(labs, coords, dist):
            """Returns a dist X dist matrix of a the labelled image labs around the pixel coords
            input: 
                labs: a labelled image
                coords: a tuple of Z, Y, Z coordinates
                dist: the edge size of the bounding box
            returns: 
                N: a dist X dist matrix, where (2 X dist + 1) is the edge length
                lab: the label at position defined by coords.
            """    
            N = labs[coords[0]-dist:coords[0]+(dist+1), coords[1]-dist:coords[1]+(dist+1), coords[2]-dist:coords[2]+(dist+1)]
            lab = int(labs[coords[0], coords[1], coords[2]])
            return N, lab


        def GetSmallestBoxWithBoundary(labs, coords):
            """Takes an input coordinate and returns an NXN matrix of the smallest bounding box around that pixel which contans at least one other non-zero label
               input: A labelled image, and a tuple containing Z,Y,X coordinates
               output: an NXN matrix of the smallest bounding box with at least one other non-zero label and the label of the pixel at coords
               returns -1 if no bounding box can be found with size < 4.
            """
            lab = labs[coords[0], coords[1], coords[2]]
            N, lab = DDN.Utils.getBoundingBox(labs, coords, 1)
            if DDN.Neighbors.ContainsOtherNonZeroLabel(N, lab):
                return N, lab
            
            N2, lab = DDN.Utils.getBoundingBox(labs, coords, 2)
            if DDN.Neighbors.ContainsOtherNonZeroLabel(N2, lab):
                return N2, lab
            N3, lab = DDN.Utils.getBoundingBox(labs, coords, 3)
            if DDN.Neighbors.ContainsOtherNonZeroLabel(N3, lab):
                return N3, lab
           
            return -1

    class Neighbors:
        def ContainsOtherNonZeroLabel(array, lab):
            """takes an array and returns True if it contains at least one other non-zero label other than the one specified by lab
            input:
                array: an array of ints
                lab: a label to check against, int
            returns:
                True if the array contains at least one other non-zero label
                False if the array does not contain at least one other non-zero label
            """
            unique = np.unique(array, return_index=False, return_inverse=False, return_counts=True, axis=None)
            for i in unique[0]:
                i = int(i)
                if i != 0 and i != lab:
                    return True
            return False

        def getMaxNonLabelBounary(array, label):
            maxCount = 0;
            maxVal = 0;
            unique, counts = np.unique(array[0], return_index=False, return_inverse=False, return_counts=True, axis=None)
            for i in range(0, len(unique)):
                if int(unique[i]) != label and int(unique[i]) != 0:
                    if counts[i] > maxVal:
                        maxVal = unique[i]
                        maxCount = counts[i]
            return maxVal              

    class Images:
        def makeBinaryNonBackground(data):
            data = data >1
            out=data.astype(np.uint8)
            out[out>0]=255
            out=data.astype(np.uint8)
            return out
        
        def MakeSingleImageFromDict(tp, cells, shape, labs):
            im = np.zeros(shape)
            if type(cells) == int:
                im = np.where(labs[tp,:,:,:] == cells, labs[tp,:,:,:], im)
                return im
            
            if type(cells) == list:
                for cell in cells:
                    im = np.where(labs[tp,:,:,:] == cell, labs[tp,:,:,:], im)
                return im
            
        def MakeImageFromCoordnates(coords, shape, label=255):
            im = np.zeros(shape)
            for coord in coords:
                im[coord[0], coord[1], coord[2]] = label
            return im
        
        def AddImageFromCoordnates(im, coords, shape, label=255):
            for coord in coords:
                im[coord[0], coord[1], coord[2]] = label
            return im
        
        
        def MakeMovieFromCells(cells, shape, total_frames, labs):
  
            new_shape = (total_frames, shape[0], shape[1], shape[2])
            im = np.zeros(new_shape)
            if type(cells) == int:
                    im = np.where(labs == cells, labs, im)
                    return im
                
            if type(cells) == list:
                for cell in cells:
                    im = np.where(labs == cell, labs, im)
                return im
        
        def MakeMovieFromCellsDict(cells, shape, total_frames, myDict):
  
            new_shape = (total_frames, shape[0], shape[1], shape[2])
            im = np.zeros(new_shape)
            if type(cells) == int:
                for tp in range(0, total_frames):
                    if cells in myDict[tp].keys():
                        coords = myDict[tp][cells]['Coordinates']
                        image = im[tp,:,:,:]
                        for coord in coords:
                            image[coord[0], coord[1], coord[2]] = cells
           
            if type(cells) == list:
                for cell in cells:
                    clear_output(wait=True)
                    print('running cell ' + str(cell))
                    for tp in range(0, total_frames):
                        if cell in myDict[tp].keys():
                            coords = myDict[tp][cell]['Coordinates']
                            image = im[tp,:,:,:]
                            for coord in coords:
                                image[coord[0], coord[1], coord[2]] = cell
                            
                    #im = np.where(labs == cells, labs, im)
            return im
                
            if type(cells) == list:
                for cell in cells:
                    im = np.where(labs == cell, labs, im)
                return im
        
        
        def ViewStackedImages(path, min_tp, max_tp, membranes=True, small_areas=False, segmentation=True, confidence=False, changes=False, average=False, nuc=True, nuc_intersect=False, labels=False):
            
            mem, seg, mask, out, avg_img, small_areas, nuc, nuc_mask, labs, labs_bin = DDN.Images.MakeStackedImages(path, min_tp, max_tp, membranes, small_areas, segmentation, confidence, changes, average, nuc, nuc_intersect, labels)
            
            viewer = napari.Viewer()
            
            if type(mem) == np.ndarray:
                viewer.add_image(mem, name = 'membranes', blending='additive', colormap='green')
            if type(seg) == np.ndarray:
                viewer.add_image(seg, name = 'segmentation', blending='additive', colormap='red')
            if type(mask) == np.ndarray:
                viewer.add_image(mask, name = 'segmentation confidence', blending='additive', colormap='blue')
            if type(out) == np.ndarray:
                viewer.add_image(out, name = 'changes', blending='additive', colormap='blue')
            if type(avg_img) == np.ndarray:
                viewer.add_image(avg_img, name = 'average', blending='additive', colormap='gray')
            if type(small_areas) == np.ndarray:
                viewer.add_image(small_areas, name = 'small areas', blending='additive', colormap='blue')
            if type(nuc) == np.ndarray:
                viewer.add_image(nuc, name = 'nuclei', blending='additive', colormap='gray')
            if type(nuc_mask) == np.ndarray:
                viewer.add_image(nuc_mask, name = 'nuclear confidence', blending='additive', colormap='gray')
            if type(labs) == np.ndarray:
                viewer.add_image(labs, name = 'labels', blending='additive', colormap=random_cmap)
               
            return viewer
            
           
        def GetMembraneImage(path, shape, min_tp=0, max_tp=120):

            mem = np.zeros((max_tp, shape[0],shape[1],shape[2]), dtype=np.int16)
            for i in range(min_tp, max_tp):
                clear_output(wait=True)
                print('loading membrane image part {}'.format(i))
                mem_temp = np.load(os.path.join(path, 'processed_images', 'mem', 'mem_' + str(i) + '.npy'))
                mem[i, :,:,:] = mem_temp
            return mem
            
        def GetNuclearImage(path, shape, min_tp=0, max_tp=120):
 
            nuc = np.zeros((max_tp, shape[0],shape[1],shape[2]), dtype=np.int16)
            for i in range(min_tp, max_tp):
                clear_output(wait=True)
                print('loading membrane image part {}'.format(i))
                nuc_temp = np.load(os.path.join(path, 'processed_images', 'nuc', 'nuc_' + str(i) + '.npy'))
                nuc[i, :,:,:] = nuc_temp
            return nuc
           
        def MakeStackedImages(path, min_tp = 0, max_tp = 120, membranes_bool=True, small_areas_bool=False, segmentation_bool=True, confidence_bool=False, changes_bool=False, average_bool=False, nuc_bool=False, nuc_intersect_bool=False, labels_bool=False, labs_bin_bool = False):

            mem_temp = np.load(os.path.join(path, 'processed_images', 'mem', 'mem_0.npz'))
            mem_temp = mem_temp['arr_0']
            shape = mem_temp.shape
            mem = -1 
            seg = -1 
            mask = -1 
            nuc_mask = -1
            out = -1 
            avg_img = -1 
            small_area_image = -1
            seg_filled = -1
            labs = -1
            nuc = -1
            labs_bin = -1
            assert min_tp >=0
           
            if nuc_bool:
                nuc = np.zeros((max_tp, shape[0],shape[1],shape[2]), dtype=np.int16)
                for i in range(min_tp, max_tp):
                    clear_output(wait=True)
                    print('loading nuclear image part {}'.format(i))
                    nuc_temp = np.load(os.path.join(path, 'processed_images', 'nuc', 'nuc_' + str(i) + '.npz'))
                    nuc_temp = nuc_temp['arr_0']
                    nuc[i, :,:,:] = nuc_temp
            
            if segmentation_bool:
                seg = np.zeros((max_tp, shape[0],shape[1],shape[2]), dtype=np.bool)
                if labels_bool:
                    labs = np.zeros((max_tp, shape[0],shape[1],shape[2]))
                for i in range(min_tp, max_tp):
                    clear_output(wait=True)
                    print('loading segmentation image part {}'.format(i))
                    im_array = np.load(os.path.join(path, 'seg', 'mem', 'seg_' + str(i) + '.npz'))
                    im_temp = im_array['arr_0']
                    seg[i,:,:,:] = im_temp    
                    if labels_bool:
                        im_inv = skimage.util.invert(im_temp)
                        im, count = ndi.label(im_inv)
                        labs[i, :,:,:] = im
            
            if labs_bin_bool:
                labs_bin = np.zeros((max_tp, shape[0],shape[1],shape[2]), dtype = np.uint8)
                for i in range(min_tp, max_tp):
                    clear_output(wait=True)
                    print('making binary image part {}'.format(i))
                    im_array = np.load(os.path.join(path, 'seg', 'mem', 'seg_' + str(i) + '.npz'))
                    im_temp = im_array['arr_0']
                    im_inv = skimage.util.invert(im_temp)
                    im, count = ndi.label(im_inv)
                    im[im > 1] = 255
                    labs_bin[i,:,:,:] = im
      
                    
            if membranes_bool:
                mem = np.zeros((max_tp, shape[0],shape[1],shape[2]), dtype=np.int16)
                for i in range(min_tp, max_tp):
                    clear_output(wait=True)
                    print('loading membrane image part {}'.format(i))
                    mem_temp = np.load(os.path.join(path, 'processed_images', 'mem', 'mem_' + str(i) + '.npz'))
                    mem_temp = mem_temp['arr_0']
                    mem[i, :,:,:] = mem_temp

            if confidence_bool:
                if type(seg) != int and type(mem) != int:
                    clear_output(wait=True)
                    print('making membrane confidence image')
                    img_screened = np.where(seg, mem, 0)
                    mask = (img_screened < 200) & (img_screened > 1)
            if nuc_intersect_bool:
                if type(seg) != int and type(nuc) != int:
                    clear_output(wait=True)
                    print('making nuclear confidence image')
                    img_screened = np.where(seg, nuc, 0)
                    nuc_mask = (img_screened > 100) & (img_screened > 1)
                                
            if small_areas_bool:
                small_area_image = np.zeros((max_tp, shape[0],shape[1],shape[2]), dtype=np.uint8)
                if type(seg) == np.ndarray:
                   
                    for i in range(min_tp, max_tp):
                        flag = False
                        clear_output(wait=True)
                        print('loading small areas image part {}'.format(i))
                        seg_temp_raw = seg[i, :, :, :]
                        seg_temp = skimage.util.invert(seg_temp_raw)
                        labels, count = ndi.label(seg_temp)
                        currentProps = regionprops(labels)
                        for prop in currentProps:
                            if prop['Area'] < 1000:
                                flag = True
                                centroid = prop['centroid']
                                for coord in prop['coords']:
                                    small_area_image[i, coord[0], coord[1], coord[2]] = 255
                                try:
                                    small_area_image[i, int(centroid[0]), 0:int(shape[1]), int(centroid[2])] = 255
                                    small_area_image[i, int(centroid[0]), int(centroid[1]), 0:int(shape[2])] = 255
                                    small_area_image[i, int(centroid[0])+1, 0:int(shape[1]), int(centroid[2])] = 255
                                    small_area_image[i, int(centroid[0])+1, int(centroid[1]), 0:int(shape[2])] = 255
                                    small_area_image[i, int(centroid[0])-1, 0:int(shape[1]), int(centroid[2])] = 255
                                    small_area_image[i, int(centroid[0])-1, int(centroid[1]), 0:int(shape[2])] = 255
                                    small_area_image[i, :, 0:20, :] = 255
                                except IndexError:
                                    print('index error')
         
            if changes_bool:         
                lines1_temp = np.load(os.path.join(path, 'seg', 'seg_' + str(min_tp) + '.npz'))
                lines1 = lines1_temp['arr_0']
                lines2_temp = np.load(os.path.join(path, 'seg', 'seg_' + str(min_tp + 1) + '.npz'))
                lines2 = lines2_temp['arr_0']
                im, count = ndi.label(lines1)
                im2, count, = ndi.label(lines2)                     
                binary = DDN.Images.makeBinaryNonBackground(im)
                binary1 = lines1 + binary
                binary = DDN.Images.makeBinaryNonBackground(im2)
                binary2 = lines2 + binary
                subtracted = binary1 - binary2
                subtracted2 = binary2 - binary1
                subtracted_final = subtracted & subtracted2
                subtracted_final  = subtracted_final > 0
                out=subtracted_final.astype(np.uint8)
                out[out>0]=255

                out = np.reshape(out, (out.shape[0],out.shape[1],out.shape[2],1))
                for i in range(min_tp+1, max_tp):
                    clear_output(wait=True)
                    print('loading segmentation errors part {}'.format(i))
                    lines1_temp = np.load(os.path.join(path, 'seg', 'seg_' + str(i) + '.npz'))
                    lines1 = lines1_temp['arr_0']
                    lines2_temp = np.load(os.path.join(path, 'seg', 'seg_' + str(i-1) + '.npz'))
                    lines2 = lines2_temp['arr_0']                  
                    lines1 = skimage.util.invert(lines1)
                    lines2 = skimage.util.invert(lines2)
                    im, count = ndi.label(lines1)
                    im2, count = ndi.label(lines2)                 
                    binary = DDN.Images.makeBinaryNonBackground(im)
                    binary1 = lines1 + binary
                    binary = DDN.Images.makeBinaryNonBackground(im2)
                    binary2 = lines2 + binary
                    subtracted = binary1 - binary2
                    subtracted2 = binary2 - binary1
                    subtracted_final = subtracted & subtracted2
                    subtracted_final  = subtracted_final > 0
                    img=subtracted_final.astype(np.uint8)
                    img[img>0]=255
                    img = np.reshape(img, (img.shape[0],img.shape[1],img.shape[2],1))
                    out = np.append(out , img , axis = 3)
                out = np.rollaxis(out, 3)
         
            if average_bool:           
                if type(seg) != int:               
                    print('running max projection')
                    IM_AVG= np.average(seg, axis=1)
                    avg_img = np.zeros(seg.shape)    
                    for i in range(min_tp,max_tp):
                        clear_output(wait=True)
                        print('building max projection of timepoint {}'.format(i))
                        for j in range(0, 161):
                            avg_img[i,j,:,:] = IM_AVG[i, :, :]
        
            print('Finished!')
            
            return mem, seg, mask, out, avg_img, small_area_image, nuc, nuc_mask, labs, labs_bin

  
        def makeContactImageWholeTimepoint(shape, surface_contacts_dict, timepoint):
            im = np.zeros(shape)
            for track in surface_contacts_dict[0].keys():
                for contact in surface_contacts_dict[0][track].keys():
                    coords = surface_contacts_dict[0][track][contact]
                    for coor in coords:
                        im[coor[0], coor[1], coor[2]] = contact
            return im

     
    class Tracking:
        
        def FindClosestCentroidInFrame(centroid, centroid_list):
            """ given a centroid and a list of all other centroids, finds the closest centroid in the current frame to the passed centroid"""
            tp = centroid[0]
            centroid1_zyx = [centroid[1], centroid[2], centroid[3]]
            min_dist = 9999
            min_label = -1
            min_centroid = -1
            index = -1
            for i, next_tp_centroid in enumerate(centroid_list):
                if next_tp_centroid[0] == tp: 
                    centroid2_zyx = [next_tp_centroid[1], next_tp_centroid[2], next_tp_centroid[3]]
                    if centroid2_zyx != centroid1_zyx:
                        distance = DDN.Utils.calculateDistance(np.array(centroid1_zyx), np.array(centroid2_zyx))
                        if distance < min_dist and next_tp_centroid[5] == -1:
                            min_dist = distance
                            min_centroid = next_tp_centroid
                            min_label = next_tp_centroid[4]
                            index = i
            return min_dist, min_label, min_centroid, index  
    
        def FindAverageDistanceOfClosestCentroid(path, tp, centroid_positions):
            """given a timepoint and a list of centroid positions, finds the average smallest distance between all centroids in that timepoint"""
            min_dist_sum = 0
            num = 0
            for centroid in centroid_positions:
                if centroid[0] == tp:
                    min_dist, min_label, min_centroid, index  = DDN.Tracking.FindClosestCentroidInFrame(centroid, centroid_positions)
                    min_dist_sum = min_dist_sum + min_dist
                    num = num + 1
            print('minimum average inter-centroid position = ' + str(min_dist_sum / num))
            return (min_dist_sum / num)
         
        def makeLabsImage(path, timepoints, shape, save=False):
            labs_whole = np.zeros((timepoints, shape[0],shape[1],shape[2]), dtype=np.uint8)
            centroid_positions = []
            for timepoint in range(0, timepoints):
                clear_output(wait=True)
                print('Running timepoint ' + str(timepoint))
                temp = np.load(os.path.join(path, 'seg', 'mem', 'seg_' + str(timepoint) + '.npz'))
                seg = temp['arr_0']
                seg = skimage.util.invert(seg)
                labs, _ = ndi.label(seg)
                props = regionprops(labs)
                labs_whole[timepoint,:,:,:] = labs.astype(np.uint8)
                for prop in props:
                    if prop['Area'] < 10000000:
                        centroid = prop['centroid']
                        label = prop['label']
                        temp_centroid = [timepoint, int(centroid[0]),int(centroid[1]),int(centroid[2]), prop['label']]
                        centroid_positions.append(temp_centroid)
            temp = np.full(len(centroid_positions), -1)
            centroid_positions = np.c_[centroid_positions,temp]
            if save:
                np.savez_compressed(os.path.join(path, 'tracking', 'centroid_positions.npz', centroid_positions))
                np.savez_compressed(os.path.join(path, 'tracking', 'labs_whole.npz', labs_whole))
            return labs_whole, centroid_positions
    
        def FindClosestCentroidWithDisplacement(centroid, timepoint1, timepoint2, centroid_list, disp = (0, -3, 0), untracked_only = False):

            centroid1_zyx = [centroid[1], centroid[2], centroid[3]]
            centroid1_zyx = tuple(map(operator.add, centroid1_zyx, disp))   
            min_dist = 9999
            min_label = -1
            min_centroid = -1
            index = -1
            min_disp = (3,3,3)

            for i, next_tp_centroid in enumerate(centroid_list):
                if next_tp_centroid[0] == timepoint2:
                    centroid2_zyx = [next_tp_centroid[1], next_tp_centroid[2], next_tp_centroid[3]]
                    distance = DDN.Utils.calculateDistance(np.array(centroid1_zyx), np.array(centroid2_zyx))
                    if distance < min_dist:
                        if untracked_only == True:
                            if next_tp_centroid[5] == -1:
                                min_dist = distance
                                min_centroid = next_tp_centroid
                                min_label = next_tp_centroid[4]
                                index = i
                                min_disp = np.asarray(centroid2_zyx) - np.asarray(centroid1_zyx)
                        else:
                            min_dist = distance
                            min_centroid = next_tp_centroid
                            min_label = next_tp_centroid[4]
                            index = i
                            min_disp = np.asarray(centroid2_zyx) - np.asarray(centroid1_zyx)

            return min_dist, min_label, min_centroid, index, min_disp 
    
        def GetDisplacementMatrix(path, tp, centroid_list):
            min_disp = [0,0,0]
            n = 0
            for centroid in centroid_list:
                if centroid[0] == tp:
                    min_dist, min_label, min_centroid, index, disp = DDN.Tracking.FindClosestCentroidWithDisplacement(centroid, tp, tp+1, centroid_list, disp=(0,0,0), untracked_only = False)
                    min_disp = min_disp + disp
                    n = n + 1
            return min_disp / n
        
        def TrackCells(centroid_positions, displacement, start_tp = 0, end_tp = 120, dist_cutoff = 10):

            print('running TrackCells')
            for i, centroid in enumerate(centroid_positions):
                if centroid[0] == 0 and centroid[4] > 1:
                    centroid_positions[i, 5] = centroid_positions[i,4]
                    centroid_positions = DDN.Tracking.TrackSingleCell(centroid, centroid_positions, displacement = displacement, start_timepoint = start_tp, end_timepoint = end_tp, dist_cutoff = dist_cutoff)
            return centroid_positions.astype(int)
                
        
        def TrackSingleCell(initial_centroid, centroid_positions, displacement=(0,-3,0), start_timepoint = 0, trackNum = 999, end_timepoint = 120, dist_cutoff=10, untracked_only = False):

            track_length = (end_timepoint - start_timepoint) + 1
            if start_timepoint == 0:
                trackLabel = initial_centroid[4]
            else:
                trackLabel = trackNum
                
            for i, track in enumerate(centroid_positions):
                if track[0] == initial_centroid[0] and track[4] == initial_centroid[4]:
                    centroid_positions[i, 5] = trackLabel

            current_centroid = initial_centroid
            clear_output(wait=True)
            print('tracking cell number ' + str(initial_centroid[4]) + ' as track ' + str(trackLabel) + ' starting at position ' + str(start_timepoint))
            
            for i in range(start_timepoint, end_timepoint):          
                    min_dist, label, centroid, track_num, disp_temp = DDN.Tracking.FindClosestCentroidWithDisplacement(current_centroid, i, i+1, centroid_positions, disp = displacement, untracked_only = untracked_only)        
                    if min_dist <= dist_cutoff:
                        centroid_positions[track_num, 5] = trackLabel
                        current_centroid = centroid_positions[track_num]
                    elif min_dist > dist_cutoff:
                        if i < end_timepoint - 1:
                            min_dist, label, centroid, track_num, disp_temp = DDN.Tracking.FindClosestCentroidWithDisplacement(current_centroid, i, i+2, centroid_positions, disp = displacement, untracked_only = untracked_only)
                            if min_dist < dist_cutoff:
                                centroid_positions[track_num, 5] = trackLabel
                                current_centroid = centroid_positions[track_num]
            print('finished track')
            return centroid_positions.astype(int)

        def editTracks(path, cell_tracks, labs_whole, displacement, dist_cutoff): 

            lineage_to_highlight = 0
            lineage_to_track = 0

            #make a subset of the zyc coordinates
            cell_track_list_zyx = cell_tracks[:,0:4]
            cell_track_list_zyx = cell_tracks[:,0:4]

            #make a list of only the cell tracks
            track_list = cell_tracks[:,5]
            
            viewer = napari.Viewer()

            viewer.add_image(labs_whole, name = 'labs_whole', blending='additive', colormap='gray');

            #add lineage markers
            linmark_x = np.arange(0, 160, 10)
            linmark_x = linmark_x * 10
            linmark_x = linmark_x - 1500
            linmark_y = np.full(len(linmark_x), -5)
            lineage_marks = np.column_stack((linmark_y,linmark_x))
            linmark_50_100 = [[-15, -1000], [-15, -500]]
            lineage_marks = np.row_stack((lineage_marks, linmark_50_100))    
                        
            points_layer = viewer.add_points(
                lineage_marks,
                face_color='red',
                size=8,
                name = 'lineage marks'
            )
               
            empty_tracks = []

            for track in cell_tracks:
                if track[5] == -1:
                    empty_tracks.append(track)
            empty_tracks = np.asarray(empty_tracks)
            
            empty_tracks_TZYX = empty_tracks[:,0:4]

            points_layer = viewer.add_points(
                empty_tracks_TZYX,
                face_color='red',             
                size=20,
                name = 'empty tracks'
            )
            
            lineages_tracked = cell_tracks[cell_tracks[:,5] != -1]
            lineage_track_list = lineages_tracked[:,5]
                 
            tp = lineages_tracked[:,0]
            lineage = np.vstack((tp, lineage_track_list)).T
            lineage[:,1] = lineage[:,1] * 10
            lineage[:,1] = lineage[:,1] - 1500            
            lineage[:,0] = lineage[:,0] * 5
                       
            lineage_point_properties = {
                'deets':lineages_tracked,
                'track': lineage_track_list,
            }

            points_layer = viewer.add_points(
                lineage,    
                size=3,
                name='lineage',
                face_color='track',
                face_colormap = random_cmap,
                properties=lineage_point_properties,
            )
            
            point_properties2 = {
                'track': track_list,
                'deets':cell_tracks,
            }

            points_layer = viewer.add_points(
                cell_track_list_zyx,
                properties=point_properties2,
                face_color='track',
                face_colormap=random_cmap,
                size=5,
                name = 'all_centroids'
            )
            
            
            @magicgui(call_button='TrackForward')
            def TrackForward(lineage_to_track):
                cell_to_track = int(lineage_to_track)
                current_tp = viewer.dims.current_step[0]
                current_z = viewer.dims.current_step[1]
                #find the centroid at the next timepoint
                
                for track in cell_tracks:
                    
                    if track[0] == current_tp+1:
                        if track[5] == cell_to_track:
                            current_z = track[1]
                            viewer.dims.set_point(0, current_tp + 1)
                            viewer.dims.set_point(1, current_z)
            viewer.window.add_dock_widget(TrackForward.Gui())           
            
            @magicgui(call_button='TrackToLastPoint')
            def TrackToLastPoint(lineage_to_track):
                cell_to_track = int(lineage_to_track)
                current_tp = viewer.dims.current_step[0]
                for i in range(0, 240):
                    current_tp = viewer.dims.current_step[0]
                    current_z = viewer.dims.current_step[1]
                    for track in cell_tracks:

                        if track[0] == current_tp+1:
                            if track[5] == cell_to_track:
                                current_z = track[1]
                                viewer.dims.set_point(0, current_tp + 1)
                                viewer.dims.set_point(1, current_z)                                 
            viewer.window.add_dock_widget(TrackToLastPoint.Gui())
  
            @magicgui(call_button='GoToPoint')
            def GoToPoint():
                current_properties = viewer.layers['lineage'].current_properties
                centroid = current_properties['deets'][0]
                print(centroid)
                viewer.dims.set_point(1, centroid[1])
                viewer.dims.set_point(0, centroid[0])
                
                highlight_track_number = centroid[5]
                
                highlight_track = []

                for track in cell_tracks:
                    if track[5] == highlight_track_number:
                        highlight_track.append(track)
                highlight_track = np.asarray(highlight_track)
                highlight_track_TZYX = highlight_track[:,0:4]
                
                points_layer = viewer.add_points(
                    highlight_track_TZYX,
                    face_color='green',
                    size=15,
                    name = 'highlighted track'
                    )                
            viewer.window.add_dock_widget(GoToPoint.Gui())
           
            @magicgui(call_button='PrintDetails')
            def PrintDetails():
                current_properties = viewer.layers['all_centroids'].current_properties
                centroid = current_properties['deets']
                print(centroid)              
                #TestUpdate(cell_tracks)             
            viewer.window.add_dock_widget(PrintDetails.Gui())

            @magicgui(call_button='PrintLineageDetails')
            def PrintLineageDetails():
                clear_output(wait=True)
                current_properties = viewer.layers['lineage'].current_properties
                centroid = current_properties['deets'][0]
                print('timepoint: ' +str(centroid[0]))
                print('centroid: z:' +str(centroid[1]) + ' y: ' + str(centroid[2]) + ' x: ' + str(centroid[3]))
                print('track number: ' +str(centroid[5]))
            viewer.window.add_dock_widget(PrintLineageDetails.Gui())
                       
            @magicgui(call_button='AddPointToTrack')
            def AddPointToTrack(track_list):
                current_properties = viewer.layers['all_centroids'].current_properties
                centroid = current_properties['deets']
                new_centroid = centroid[0]
                trackToAddTo = track_list              
                cell_tracks[:,5][(cell_tracks[:,5] == new_centroid[5]) & (cell_tracks[:,0] == new_centroid[0]) & (cell_tracks[:,1] == new_centroid[1]) & (cell_tracks[:,2] == new_centroid[2]) & (cell_tracks[:,3] == new_centroid[3])] = trackToAddTo
                SaveSeeds()                
                TestUpdate(cell_tracks)               
            viewer.window.add_dock_widget(AddPointToTrack.Gui())
            
            @magicgui(call_button='RemoveSinglePointFromTrack')
            def RemoveSinglePointFromTrack():
                current_properties = viewer.layers['all_centroids'].current_properties
                centroid = current_properties['deets']
                untrack_track_number = centroid[0][5]
                untrack_timepoint = centroid[0][0]
                
                
                cell_tracks[:,5][(cell_tracks[:,5] == untrack_track_number) & (cell_tracks[:,0] == untrack_timepoint)] = -1
                
                TestUpdate(cell_tracks)
                SaveSeeds()
            viewer.window.add_dock_widget(RemoveSinglePointFromTrack.Gui())
      
            @magicgui(call_button='UnTrackLineage')
            def UnTrackLineage():
                current_properties = viewer.layers['all_centroids'].current_properties
                centroid = current_properties['deets']
                untrack_track_number = centroid[0][5]
                untrack_timepoint = centroid[0][0]
                
                
                cell_tracks[:,5][(cell_tracks[:,5] == untrack_track_number) & (cell_tracks[:,0] >= untrack_timepoint)] = -1
                
                TestUpdate(cell_tracks)
                SaveSeeds()              
            viewer.window.add_dock_widget(UnTrackLineage.Gui())
     
            @magicgui(call_button='HighlightLineageFromID')
            def HighlightLineageFromID(lineage_to_highlight):
                        
                highlight_track = []
                lineage_to_highlight = int(lineage_to_highlight)
                
                for track in cell_tracks:
                    if track[5] == lineage_to_highlight:
                        highlight_track.append(track)
                highlight_track = np.asarray(highlight_track)
                highlight_track_TZYX = highlight_track[:,0:4]
                
                points_layer = viewer.add_points(
                    highlight_track_TZYX,
                    face_color='green',
                    size=15,
                    name = 'highlighted track'
                    )
            viewer.window.add_dock_widget(HighlightLineageFromID.Gui())
          
            @magicgui(call_button='HighlightLineage')
            def HighlightLineage():
                current_properties = viewer.layers['lineage'].current_properties
                centroid = current_properties['deets']
                print(centroid)
                highlight_track_number = centroid[0][5]
                
                highlight_track = []

                for track in cell_tracks:
                    if track[5] == highlight_track_number:
                        highlight_track.append(track)
                highlight_track = np.asarray(highlight_track)
                highlight_track_TZYX = highlight_track[:,0:4]
                
                points_layer = viewer.add_points(
                    highlight_track_TZYX,
                    face_color='green',
                    size=15,
                    name = 'highlighted track'
                    )  
            viewer.window.add_dock_widget(HighlightLineage.Gui())

            @magicgui(call_button='SaveSeeds')
            def SaveSeeds():
                properties = viewer.layers['all_centroids'].properties
                deets = properties['deets']
                np.savez_compressed(os.path.join(path, 'tracking', 'new_tracks.npz'), deets)
            viewer.window.add_dock_widget(SaveSeeds.Gui())

            @magicgui(call_button='TrackAndAdd')
            def TrackAndAdd(track_list):
                centroid_positions_data = viewer.layers['all_centroids'].data  
                properties = viewer.layers['all_centroids'].properties
                current_properties = viewer.layers['all_centroids'].current_properties
                centroid = current_properties['deets']
                maxTrack = np.amax(properties['deets'][:,5])
                startTime = centroid[0][0]
                temp = DDN.Tracking.TrackSingleCell(centroid[0], properties['deets'], start_timepoint = startTime,  trackNum = track_list, untracked_only = True)
                SaveSeeds()
                TestUpdate(temp)
            viewer.window.add_dock_widget(TrackAndAdd.Gui())

            @magicgui(call_button='TrackNewTrack')
            def TrackNewTrack():

                centroid_positions_data = viewer.layers['all_centroids'].data  
                properties = viewer.layers['all_centroids'].properties
                current_properties = viewer.layers['all_centroids'].current_properties
                centroid = current_properties['deets']
                maxTrack = np.amax(properties['deets'][:,5])
                startTime = centroid[0][0]
                temp = DDN.Tracking.TrackSingleCell(centroid[0], properties['deets'], start_timepoint = startTime,  trackNum = maxTrack+1,  untracked_only = True)
                SaveSeeds()
                TestUpdate(temp)
            viewer.window.add_dock_widget(TrackNewTrack.Gui())

           # @magicgui(call_button='TestUpdate')
            def TestUpdate(cell_tracks):
                cell_track_list_temp = cell_tracks[:,0:4]
                track_list = cell_tracks[:,5]
                viewer.layers.remove('empty tracks')
                viewer.layers.remove('all_centroids')
                viewer.layers.remove('lineage')
                viewer.layers.remove('lineage marks')
                points_layer = viewer.add_points(
                    lineage_marks,
                    face_color='red',
                    size=8,
                    name = 'lineage marks'
                )        
                lineages_tracked = cell_tracks[cell_tracks[:,5] != -1]
                lineage_track_list = lineages_tracked[:,5]
                tp = lineages_tracked[:,0]
                lineage = np.vstack((tp, lineage_track_list)).T
                lineage[:,1] = lineage[:,1] * 10
                lineage[:,1] = lineage[:,1] - 1500
                lineage[:,0] = lineage[:,0] * 5
                lineage_point_properties = {
                    'deets':lineages_tracked,
                    'track': lineage_track_list,
                }
                
                points_layer = viewer.add_points(
                    lineage,    
                    size=3,
                    name='lineage',
                    face_color='track',
                    face_colormap = random_cmap,
                    properties=lineage_point_properties,
                )
                  
                point_properties3 = {
                'track': track_list,
                'deets':cell_tracks,
                }
                
                empty_tracks = []
                for track in cell_tracks:
                    if track[5] == -1:
                        empty_tracks.append(track)
                empty_tracks = np.asarray(empty_tracks)
                empty_tracks_TZYX = empty_tracks[:,0:4]

                points_layer = viewer.add_points(
                    empty_tracks_TZYX,
                    face_color='red',
                    face_colormap=random_cmap,
                    size=20,
                    name = 'empty tracks'
                )
                points_layer = viewer.add_points(
                    cell_track_list_temp,
                    properties=point_properties3,
                    face_color='track',
                    face_colormap=random_cmap,
                    size=5,
                    name = 'all_centroids'
                )

            return viewer


        def MakeTrackedImage(labs_tracked, tracked_cells, labs_whole):
            for centroid in tracked_cells:                
                        if centroid[5] != -1:
                            clear_output(wait=True)
                            print('running timepoint ' + str(centroid[0]) + ' cell ' + str(centroid[5]) )
                            timepoint = int(centroid[0])
                            labs = labs_whole[timepoint,:,:,:]
                            tracked = labs_tracked[timepoint,:,:,:]
                            tracked = np.where(labs==int(centroid[4]), centroid[5], tracked)
                            labs_tracked[timepoint,:,:,:] = tracked
                      
            return labs_tracked

    class Interpolation:
        
        def interpolate_single_frame(frame1, frame2):
            mid_frame = cv2.addWeighted(frame1,0.5,frame2,1-0.5,0)   
            core = mid_frame == 255
            frame1_erode = skimage.morphology.binary_erosion(frame1)
            frame2_erode = skimage.morphology.binary_erosion(frame2)
            core = core + frame1_erode + frame2_erode
            core[core>0] = 255
            return core
    
        def interpolate(track, missing_frame, tracked_labs):

            print('starting interpolation')
            single_frame_before = np.copy(tracked_labs[missing_frame-1,:,:,:])
            single_frame_after = np.copy(tracked_labs[missing_frame+1,:,:,:])
            single_frame_middle = np.copy(tracked_labs[missing_frame,:,:,:])
            single_frame_actual = np.copy(tracked_labs[missing_frame,:,:,:])
            single_frame_before[single_frame_before != track] = 0
            single_frame_before[single_frame_before == track] = 255
            single_frame_after[single_frame_after !=  track] = 0
            single_frame_after[single_frame_after ==  track] = 255
            single_frame_interpolated = np.zeros(single_frame_before.shape)              
            for i in range (0, single_frame_before.shape[0]):
                if np.amax(single_frame_before[i,:,:]) > 0 and np.amax(single_frame_after[i,:,:]) > 0:
                    single_frame_interpolated[i,:,:] = DDN.Interpolation.interpolate_single_frame(single_frame_before[i],single_frame_after[i])  
            single_frame_interpolated_new = np.where(single_frame_actual == 0, single_frame_interpolated, 0)
            single_frame_interpolated_new[single_frame_interpolated_new > 0] = track            
            props = regionprops(single_frame_interpolated_new.astype(int))
            for prop in props:
                if int(prop['label']) == int(track):
                    interpolated_centroid = prop['centroid']         
            whole_interpolated_frame = np.where(single_frame_interpolated_new > 0, single_frame_interpolated_new, single_frame_middle)          
            for prop in props:
                if int(prop['label']) == int(track):
                    for i in prop['coords']:
                        if DDN.Interpolation.isPixelEdgeWithOtherLab(whole_interpolated_frame, i[0], i[1], i[2]):
                            whole_interpolated_frame[i[0], i[1], i[2]] = 0                  
            return whole_interpolated_frame, interpolated_centroid

        def isPixelEdgeWithOtherLab(labs, z, y, x):
                    label = labs[z,y,x] 
                    top=labs[z-1, y, x]
                    bottom=labs[z+1, y, x]
                    up=labs[z, y-1, x]
                    down=labs[z, y+1,x]
                    left=labs[z,y,x-1]
                    right=labs[z,y,x+1]  
                    if top==label or top == 0:
                        if bottom==label or bottom == 0:
                            if up==label  or up == 0:
                                if down==label or down == 0:
                                    if left==label or left == 0:
                                        if right==label or right == 0:
                                            return False
                    return True
           
        def Interpolater(path, tracked_labs, membranes, track_list, additional_lab = -1):

                tp = '0'
                mySlice = '0'
                viewer = napari.Viewer()
                viewer.add_image(membranes, name = 'membranes', blending='additive');
                viewer.add_labels(tracked_labs, name = 'tracked_labs', blending='additive');
                if type(additional_lab) == np.ndarray:
                    viewer.add_labels(additional_lab, name = 'subset_labs', blending='additive');


                def update_display():
           
                    current_z = viewer.dims.current_step[1]
                    viewer.dims.set_point(1, current_z+1)
                    viewer.dims.set_point(1, current_z)

                #viewer.dims.set_point(1, current_z)


                @viewer.bind_key('.')
                def z_plus(viewer):
                    current_z = viewer.dims.current_step[1]
                    viewer.dims.set_point(1, current_z + 1)
                    
                @viewer.bind_key(',')
                def z_minus(viewer):
                    current_z = viewer.dims.current_step[1]
                    viewer.dims.set_point(1, current_z - 1)

                @magicgui(call_button='Save')
                def Save(savename = "tracked_labs"):
                    new_labs = viewer.layers['tracked_labs'].data
                    np.savez_compressed(os.path.join(path, 'tracking', savename + '.npz'), new_labs)
                viewer.window.add_dock_widget(Save, area='right')

                @magicgui(call_button='Fix All Boundaries')
                def refine_boundaries(tp:int):   
                    tp = int(tp)
                    print('running timepoint ' + str(tp))
                    new_labs = viewer.layers['tracked_labs'].data
                    imtemp = new_labs[tp,:,:,:].astype(int)
                    props=regionprops(imtemp)
                    for prop in props:
                        for i in prop['coords']:
                            if DDN.Interpolation.isPixelEdgeWithOtherLab(imtemp, i[0], i[1], i[2]):
                                imtemp[i[0], i[1], i[2]] = 0
                    new_labs[tp,:,:,:] = imtemp
                viewer.window.add_dock_widget(refine_boundaries, area='right')

                @magicgui(call_button='Fix Boundaries')
                def refine_boundaries_single(tp:int, track:int):   
                    tp = int(tp)
                    track = int(track)
                    print('running timepoint ' + str(tp) + ' track ' + str(track))
                    new_labs = viewer.layers['tracked_labs'].data
                    imtemp = new_labs[tp,:,:,:].astype(int)
                    props=regionprops(imtemp)
                    for prop in props:
                        if prop['label'] == track:
                            for i in prop['coords']:
                                if DDN.Interpolation.isPixelEdgeWithOtherLab(imtemp, i[0], i[1], i[2]):
                                    imtemp[i[0], i[1], i[2]] = 0
                    new_labs[tp,:,:,:] = imtemp
                viewer.window.add_dock_widget(refine_boundaries_single, area='right')

                @magicgui(call_button='InterpolateFrame')
                def InterpolateFrame(label:int, timepoint:int):
                    print('running interpolate frame from button call')
                    track_list = int(label)
                    timepoint_list = int(timepoint)
                    all_labs = viewer.layers['tracked_labs'].data
                    new_frame, interpolated_centroid = DDN.Interpolation.interpolate(track_list, timepoint_list, all_labs)
                    interpolated_centroid_new = [int(timepoint_list), interpolated_centroid[0], interpolated_centroid[1], interpolated_centroid[2], int(track_list), int(track_list)]
                    all_labs[timepoint_list,:,:,:] = new_frame        
                    update_display()
                viewer.window.add_dock_widget(InterpolateFrame, area='right')

                @magicgui(call_button='Copy Frame')
                def CopyFrame(label = "0", orig_tp = "0", new_tp = "0"):
                    label = int(label)
                    orig_tp = int(orig_tp)
                    new_tp = int(new_tp)
                    

                    all_labs = viewer.layers['tracked_labs'].data   
                    
                    orig_frame = all_labs[orig_tp,:,:,:]
                    new_frame = np.copy(all_labs[new_tp,:,:,:])
                    
                    new_im = np.zeros(orig_frame.shape)
                    print(new_im.shape)
                    new_im[orig_frame == label] = label
                    #all_labs[new_tp,:,:,:] = new_im                      

                    new_frame = np.where(new_frame == 0, new_im, new_frame)
                    
                    props = regionprops(new_frame.astype(int))
                    for prop in props:
                       if int(prop['label']) == label:
                            for i in prop['coords']:
                                if DDN.Interpolation.isPixelEdgeWithOtherLab(new_frame, i[0], i[1], i[2]):
                                    new_frame[i[0], i[1], i[2]] = 0 
                                    
                    all_labs[new_tp,:,:,:] = new_frame     
                    update_display()                    
                                    
                viewer.window.add_dock_widget(CopyFrame, area='right')
                
                 


          #      single_frame_interpolated_new = np.where(single_frame_actual == 0, single_frame_interpolated, 0)
          #  single_frame_interpolated_new[single_frame_interpolated_new > 0] = track            
          #  props = regionprops(single_frame_interpolated_new.astype(int))
          #  for prop in props:
          #      if int(prop['label']) == int(track):
          #          interpolated_centroid = prop['centroid']         
           # whole_interpolated_frame = np.where(single_frame_interpolated_new > 0, single_frame_interpolated_new, single_frame_middle)          
           # for prop in props:
           #     if int(prop['label']) == int(track):
           #         for i in prop['coords']:
           #             if DDN.Interpolation.isPixelEdgeWithOtherLab(whole_interpolated_frame, i[0], i[1], i[2]):
           #                 whole_interpolated_frame[i[0], i[1], i[2]] = 0                  

    
                
                #@magicgui(call_button='TrackForward')
                #def TrackForward(lineage_to_track):
                #    cell_to_track = int(lineage_to_track)
                #    current_tp = viewer.dims.current_step[0]
                #    current_z = viewer.dims.current_step[1]
                    #find the centroid at the next timepoint
                
                
                #@magicgui(call_button='Generate Tracks')
                #def InterpolateFrame(trackstart:'0', trackend='999'):
                #    trackstart = int(trackstart)
                #    trackend = int(trackend)    
                #    labs_to_check = list(track_list[trackstart:trackend])
                #    labs_subset = np.zeros(tracked_labs.shape, dtype=np.uint8)
                #    for i in labs_to_check:
                #        print('running ' + str(i))
                #        labs_subset[tracked_labs==i] = i
                #    viewer.add_labels(labs_subset, name='labs_subset', blending='additive')

                 #   IM_AVG= np.average(labs_subset, axis=1)
                   
                    
                 #   average_image = np.zeros(tracked_labs.shape)
                    
                 #   for i in range(0,tracked_labs.shape[0]):
                 #           average_image[i,:,:,:] = IM_AVG[i, :, :]

                 #   viewer.add_image(average_image, name='avg', blending='additive', colormap='gray')

    
               # viewer.window.add_dock_widget(InterpolateFrame.Gui())

                
                @magicgui(call_button='Change Label All')
                def InterpolateFrames(old_label:int, new_label:int, timepoint_start:int, timepoint_end:int):
                    old_track = int(old_label)
                    new_track = int(new_label)
                    timepoint_start = int(timepoint_start)
                    timepoint_end = int(timepoint_end)
                    all_labs = viewer.layers['tracked_labs'].data
                    for i in range(timepoint_start, timepoint_end+1):
                        all_labs[i,:,:,:][all_labs[i,:,:,:] == old_track] = new_track    
                    update_display()
                viewer.window.add_dock_widget(InterpolateFrames, area='right')

                @magicgui(call_button='Change Label')
                def ChangeLabel(old_label:int, new_label:int, timepoint_list:int):
                    old_track = int(old_label)
                    new_track = int(new_label)
                    timepoint_list = int(timepoint_list)
                    all_labs = viewer.layers['tracked_labs'].data
                    all_labs[timepoint_list,:,:,:][all_labs[timepoint_list,:,:,:] == old_track] = new_track     
                    update_display()
                viewer.window.add_dock_widget(ChangeLabel, area='right')
                

                @magicgui(call_button='Delete Label')
                def DeleteLabel(label:int, timepoint:int):
                    track_list = int(label)
                    timepoint_list = int(timepoint)
                    all_labs = viewer.layers['tracked_labs'].data
                    all_labs[timepoint_list,:,:,:][all_labs[timepoint_list,:,:,:] == track_list] = 0     
                    update_display()
                viewer.window.add_dock_widget(DeleteLabel, area='right')

                return viewer
                
    class OBJ:
        
         def makeMultipleOBJ(meshDict, cells, timepoint, path, colorMap = cellColorDict, meshlab_path = os.path.join('D:', 'Segment', 'meshlab')):
            """
            makes a combined OBJ file of the passed cell array, in the 'path' folder. Each cell is colored according to the colorMap (an n-array of RGB tuples)
            meshDict: A dictionary containing the cell meshes
            cells: a list of cells to include in the OBJ. If an empty list is passed ([]), all cells are included
            timepoint: timepoint to plot
            path: Path to store the data. Relative to the current working directory
            colorMap: an array of RGB tuples to color each cell as
            
            """
            
            
            #path = 'output\\OBJ\\' + str(path)
            filenames = []
            filenames_pc = []
            objnames = []
            print(path)
            try:
                os.mkdir(path)
            except OSError:
                print ("Creation of the directory %s failed" % path)
            else:
                print ("Successfully created the directory %s " % path)
            
            
            if len(cells) == 0:
                for cell in meshDict[timepoint].keys():
                    cells.append(cell)
            
            if len(cells) != 0:
                for cell in cells:
                    if cell in meshDict[timepoint].keys():
                        if len(meshDict[timepoint][cell]['verts']) > 100:
                            clear_output(wait=True)
                            print('making pointcloud of  ' + str(cell) + ' timepoint ' + str(timepoint))
                            DDN.OBJ.MakePointcloud(meshDict, cell, timepoint, path)
                            
                            print('making mesh of  ' + str(cell) + ' timepoint ' + str(timepoint))
                            
                            filename_pc = 'cell_' + str(cell) + '_tp_' + str(timepoint) + '.obj'
                            filenames_pc.append(filename_pc)
                            filename = 'cell_' + str(cell) + '_tp_' + str(timepoint) + '_mesh.obj'
                            filenames.append(filename)
                            
                            objnames.append(cell)
                            command = meshlab_path + '\\meshlabserver.exe -i ' + path + '\\cell_' + str(cell) + '_tp_' + str(timepoint) + '.obj -o ' + path + '\\' + filename + ' -s ' + meshlab_path + '\\makesurface.mlx'
                            os.system(command)
                print(filenames)
                print(objnames)
                 
                DDN.OBJ.combine_cells(filenames, objnames, path, path, colorMap)
                
                #for filename in filenames:
                #    file = path + '\\' + filename
                #    os.remove(file)
                for filename_pc in filenames_pc:
                    print(filename_pc)
                    file_pc = path + '\\' + filename_pc
                    os.remove(file_pc)
                    
 
        

         def MakePointcloud(mesh_dict, cell, timepoint, path):
                    #path = 'output\\pointclouds\\cell' + str(cell)
                if not os.path.isdir(path):
                    os.makedirs('output\\pointclouds\\cell' + str(cell))
                      
                verts = mesh_dict[timepoint][cell]['verts']
                    
                thefile = open(path + '\\cell_' + str(cell) + '_tp_' + str(timepoint) + '.obj', 'w')
                    
                for item in verts:
                    thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))
                    
                thefile.close()
  
         def combine_cells(filenames, objnames, Inpath, outPath, colorMap, filename = None):
            
            try:
                os.mkdir(outPath + "\\whole")
            except OSError:
                print ("Creation of the directory %s failed" % outPath)
                return
            else:
                print ("Successfully created the directory %s " % outPath)
            
            
            if type(filename) != str:
                f_new = open(outPath + "\\whole\\combined_obj.obj","w")
                f_mtl = open(outPath + "\\whole\\combined_mtl.mtl","w")
                f_new.write('mtllib combined_mtl.mtl')
                
            if type(filename) == str:
                f_new = open(outPath + "\\whole\\" + filename + ".obj","w")
                f_mtl = open(outPath + "\\whole\\" + filename + ".mtl","w")
                f_new.write("mtllib " + filename + ".mtl")
            
            offset = 0
            total_offset = 0
            for filename in zip(filenames, objnames):
                mtlName = 'cell_' +str(filename[1])
                file = open(Inpath + '\\' + filename[0],"r")
                total_offset = offset
                file.seek(0)
                lines = file.readlines()

                f_new.write('o ' + str(filename[1]))
                f_new.write("\n")         
                for line in lines:
                    if line[0] == 'v':                          
                        f_new.write(line)                      
                        offset = offset + 1
                f_new.write('usemtl ' + mtlName)
                f_new.write("\n") 
                for line in lines:
                    if line[0] == 'f':
                        line_subset = line[2:]
                        line_split = [int(s) for s in line_subset.split(' ')]
                        newString = 'f ' + str(line_split[0] + total_offset) + ' ' + str(line_split[1] + total_offset) + ' ' + str(line_split[2] + total_offset)
                        f_new.write(newString)
                        f_new.write("\n")   
                        
                #temp_sum = np.sum(colorMap[int(filename[1])])        
                
                if DDN.Utils.AreThereNaN(colorMap[int(filename[1])]):
                    colorVals = 'Kd 0 0 0'
                else:
                    colorVals = 'Kd ' + str(colorMap[int(filename[1])][0]) + ' ' + str(colorMap[int(filename[1])][1]) + ' ' + str(colorMap[int(filename[1])][2])
                    
                f_mtl.write('newmtl ' + mtlName)
                f_mtl.write("\n")
                f_mtl.write(colorVals)
                f_mtl.write("\n")
            f_new.close()
            f_mtl.close() 
            
            
    class Tracking_old:
    
        def DisplaceCentroidAndCoords(myDict, track, timepoint, disp = (0, 0, 0)):
    
            if timepoint in myDict[track].keys():
                if 'Centroid' in myDict[track][timepoint].keys():
                    centroid = myDict[track][timepoint]['Centroid']
                    coords = myDict[track][timepoint]['Coordinates']

                    new_centroid = (centroid[0] + disp[0], centroid[1] + disp[1], centroid[2] + disp[2])
                    myDict[track][timepoint]['Centroid'] = new_centroid

                    newCoords = np.zeros(myDict[track][timepoint]['Coordinates'].shape)
                    for i in range(0, len(myDict[track][timepoint]['Coordinates'])):
                            newCoords[i] = myDict[track][timepoint]['Coordinates'][i] + disp
                    myDict[track][timepoint]['Coordinates'] = newCoords.astype(int)
    
            return myDict
    
        def TrackAndAddNewTrack(myDict, track, start_frame, end_frame, tracked_dict, displacement = False, disp = (0,0,0)):
            
            max_track = max(tracked_dict.keys())
            newtrack = max_track+1
            print('new track number is ' + str(newtrack) + ', adding old track ' + str(track) + ' at timepoint ' + str(start_frame))
            cell_track = DDN.Tracking.TrackSingleCell(myDict, track, start_frame, end_frame)
            currentTrack = cell_track[start_frame]
            tracked_dict[newtrack] = {}
            for timepoint in myDict.keys():
                if cell_track[timepoint] != -1 and cell_track[timepoint] != 0:
                    tracked_dict[newtrack][timepoint] = {}
                    tracked_dict[newtrack][timepoint]['Centroid'] = myDict[timepoint][cell_track[timepoint]]['Centroid']
                    tracked_dict[newtrack][timepoint]['Area'] = myDict[timepoint][cell_track[timepoint]]['Area']
                    tracked_dict[newtrack][timepoint]['Coordinates'] = myDict[timepoint][cell_track[timepoint]]['Coordinates']#
                if cell_track[timepoint] == -1 or cell_track[timepoint] == 0:
                    tracked_dict[newtrack][timepoint] = {}
                    
            if displacement == True:
                for i in range(start_frame, end_frame+1):
                    tracked_dict = DDN.Tracking.DisplaceCentroidAndCoords(tracked_dict, track, i, disp)
                
                
            return tracked_dict
    
        def TrackAndFuseTracks(myDict, trackedTrack, unTrackedTrack, start_frame, end_frame, tracked_dict, displacement = False, disp = (0,0,0)):
            cell_track = DDN.Tracking.TrackSingleCell(myDict, unTrackedTrack, start_frame, end_frame)
            print('adding ' + str(unTrackedTrack) + ' to track ' + str(trackedTrack))
            for timepoint in myDict.keys():
                if cell_track[timepoint] != -1 and cell_track[timepoint] != 0:
                    tracked_dict[trackedTrack][timepoint] = {}
                    tracked_dict[trackedTrack][timepoint]['Centroid'] = myDict[timepoint][cell_track[timepoint]]['Centroid']
                    tracked_dict[trackedTrack][timepoint]['Area'] = myDict[timepoint][cell_track[timepoint]]['Area']
                    tracked_dict[trackedTrack][timepoint]['Coordinates'] = myDict[timepoint][cell_track[timepoint]]['Coordinates']
                    
            if displacement == True:
                for i in range(start_frame, end_frame+1):
                    tracked_dict = DDN.Tracking.DisplaceCentroidAndCoords(tracked_dict, trackedTrack, i, disp)
                       
             
            return tracked_dict

        def AddSingleTimepointToTrack(myDict, tracked_cell_dict, trackedTrack, unTrackedTrack, timepoint):
            coords = myDict[timepoint][unTrackedTrack]['Coordinates']
            centroid = myDict[timepoint][unTrackedTrack]['Centroid']
            area = myDict[timepoint][unTrackedTrack]['Area']
            
            tracked_cell_dict[trackedTrack][timepoint]['Coordinates'] = coords
            tracked_cell_dict[trackedTrack][timepoint]['Centroid'] = centroid
            tracked_cell_dict[trackedTrack][timepoint]['Area'] = area
           
            
            return tracked_cell_dict
        
        def UpdateMeshDicts(tracked_cell_dict, meshDict, nucDict, track):
            print('updating cell mesh')
            meshDict[track] = {}
            for tp in tracked_cell_dict[track].keys():
                if 'Area' in tracked_cell_dict[track][tp]:
                    if tracked_cell_dict[track][tp]['Area'] > 10:
                        clear_output(wait=True)
                        print('running track ' + str(track) + ' timepoint ' + str(tp))
                        meshDict[track][tp] = {}
                        data = DDN.Utils.makeImageFromCoordinates(tracked_cell_dict[track][tp]['Coordinates'])
                        verts, faces, normals, values = measure.marching_cubes_lewiner(data, 0)
                        meshDict[track][tp]['verts'] = verts
                        meshDict[track][tp]['faces'] = faces
                        meshDict[track][tp]['normals'] = normals

            print('updating nuc mesh')
            nucDict[track] = {}
            for tp in tracked_cell_dict[track].keys():
                if 'Nucleus' in tracked_cell_dict[track][tp]:
                    if 'Area' in tracked_cell_dict[track][tp]['Nucleus']:
                        if tracked_cell_dict[track][tp]['Nucleus']['Area'] > 10:
                            clear_output(wait=True)
                            print('running track ' + str(track) + ' timepoint ' + str(tp))
                            nucDict[track][tp] = {}
                            data = DDN.Utils.makeImageFromCoordinates(tracked_cell_dict[track][tp]['Nucleus']['Coordinates'])
                            verts, faces, normals, values = measure.marching_cubes_lewiner(data, 0)
                            nucDict[track][tp]['verts'] = verts
                            nucDict[track][tp]['faces'] = faces
                            nucDict[track][tp]['normals'] = normals

            return tracked_cell_dict, meshDict, nucDict


        def FindAverageAndMinimumDistancesWithinFrame(myDict, frame):
            """
            returns the average and minimum distance between all object centroids in a frame.
            """
            
            
            min_dist_array = []
            min_dist_global= 99999

            for cell1 in myDict[frame].keys():
                min_dist, label, centroid = DDN.Tracking.findClosestCentroid(myDict, cell1, frame, frame)
                min_dist_array.append(min_dist)

            print('average minuimum distance = ' + str(np.average(min_dist_array)))
            print('absolute minuimum distance = ' + str(np.amin(min_dist_array)))

            return np.average(min_dist_array), np.amin(min_dist_array)

        def CalculateAverageDisplacementBetweenFrames(myDict, frame1, frame2):
            """
            returns the average displacement of all centroids between two frames (assuming objects are tracked in a dict). Used to refine tracking procedure to get the displacement. In this function, the dict is [timepoint][cell]
            """
            displacement_array_z = []
            displacement_array_y = []
            displacement_array_x = []
            for cell1 in myDict[frame1].keys():
                centroid_frame1 = myDict[frame1][cell1]['Centroid']
                min_dist, label, centroid_frame2 = DDN.Tracking.findClosestCentroid(myDict, cell1, frame1, frame2)
                displacement_array_z.append(centroid_frame1[0] - centroid_frame2[0])
                displacement_array_y.append(centroid_frame1[1] - centroid_frame2[1])
                displacement_array_x.append(centroid_frame1[2] - centroid_frame2[2])

            print('average z displacement = ' + str(np.average(displacement_array_z)))
            print('average y displacement = ' + str(np.average(displacement_array_y)))
            print('average x displacement = ' + str(np.average(displacement_array_x)))
            return np.average(displacement_array_z),np.average(displacement_array_y), np.average(displacement_array_x)

        def findClosestCentroid(myDict, cell, timepoint1, timepoint2):
            centroid1 = myDict[timepoint1][cell]['Centroid']

            min_dist = 9999
            label = -1
            centroid = -1
            for cell2 in myDict[timepoint2].keys():
                centroid2 = myDict[timepoint2][cell2]['Centroid']
                distance = DDN.Utils.calculateDistance(np.array(centroid1), np.array(centroid2))
                if distance > 0:
                    if distance < min_dist:
                        min_dist = distance
                        label = cell2
                        centroid = myDict[timepoint2][cell2]['Centroid']
            return min_dist, label, centroid

        def findClosestCentroidWithDisplacement(myDict, cell, timepoint1, timepoint2, disp = (0, 0, 0)):


            centroid1 = myDict[timepoint1][cell]['Centroid']

            centroid1 = tuple(map(operator.add, centroid1, disp))

            min_dist = 9999
            label = -1
            centroid = -1
            for cell2 in myDict[timepoint2].keys():
                if 'Centroid' in myDict[timepoint2][cell2]:
                        centroid2 = myDict[timepoint2][cell2]['Centroid']
                        distance = DDN.Utils.calculateDistance(np.array(centroid1), np.array(centroid2))
                        if distance > 0:
                            if distance < min_dist:
                                min_dist = distance
                                label = cell2
                                centroid = myDict[timepoint2][cell2]['Centroid']
            return min_dist, label, centroid

        def TrackSingleCell(myDict, cell, start_timepoint, end_timepoint, displacement = (0, -3, 0), dist_cutoff = 10):

            track_length = end_timepoint - start_timepoint + 1

            cell_track = np.zeros(300)

            #cell_track[0] = cell
            cell_track[start_timepoint] = cell
            cell_label = cell
            #cell_label = myDict[0][28]['Old_Label']

            for i in range(start_timepoint, end_timepoint):
                #print('running cell ' + str(cell) + ' timepoint ' + str(i) + ' finding ' + str(cell_track[i]))
                if cell_track[i] > 0:
                    min_dist, label, centroid = DDN.Tracking.findClosestCentroidWithDisplacement(myDict, cell_track[i], i, i+1, disp = displacement)

                    if min_dist <= dist_cutoff:
                        cell_track[i+1] = label

                    elif min_dist > dist_cutoff:  #if there isn't a cell within 10 units, check the next timepoint
                        cell_track[i+1] = -1
                        if i < end_timepoint - 1:
                            min_dist, label, centroid = DDN.Tracking.findClosestCentroidWithDisplacement(myDict, cell_track[i], i, i+2, disp = (displacement[0] + displacement[0], displacement[1] + displacement[1],displacement[2] + displacement[2]))
                            if min_dist < dist_cutoff:
                                cell_track[i+2] = label


            return cell_track

        def MatchNuclei(nuc_prop, labs):
            """ Take a labelled nuclear prop, and match the coordinates to a labelled cell image
                nuc_prop: a properties object  generated by regionprops of a nucleus
                labs: a labelled cell image to match the nucleus to
                returns: a cell label which the nucleus is matched to, or -1 if not label is found
            """
            matched_label = -1
            if nuc_prop['Area'] > 10:
                empty_list = []
                for coordinate in nuc_prop['Coordinates']:
                    empty_list.append(labs[coordinate[0], coordinate[1], coordinate[2]])
                matched_label = mode(empty_list)
            return matched_label

        def FindErrorTracksFromDict(myDict):
            smallestTimepoint = 999
            largestTimepoint = 0
            for track in myDict.keys():
                for timepoint in myDict[track].keys():
                    if timepoint < smallestTimepoint:
                        smallestTimepoint = timepoint
                    if timepoint > largestTimepoint:
                        largestTimepoint = timepoint
            largestTimepoint = largestTimepoint -1
            error_track = []
            error_timepoint = []
            incomplete_tracks = []
            
            #get the incomplete tracks
            for track in myDict.keys():
                if largestTimepoint in myDict[track].keys():
                    if 'Centroid' not in myDict[track][largestTimepoint].keys():    
                        if 'Centroid' not in myDict[track][largestTimepoint-1].keys():    
                            if 'Centroid' not in myDict[track][largestTimepoint-2].keys():    
                                if 'Centroid' not in myDict[track][largestTimepoint-3].keys():    
                                    if 'Centroid' not in myDict[track][largestTimepoint-4].keys():   
                                        if 'Centroid' not in myDict[track][largestTimepoint-5].keys():
                                            if track not in incomplete_tracks:
                                                incomplete_tracks.append(track)                  

            if largestTimepoint not in myDict[track].keys():
                if largestTimepoint-1 not in myDict[track].keys():
                        if largestTimepoint-2 not in myDict[track].keys():
                            if track not in incomplete_tracks:
                                incomplete_tracks.append(track)     
                     
                                             
            for track in myDict.keys():
                if track not in incomplete_tracks:
                    for timepoint in myDict[track].keys():
                            if 'Centroid' not in myDict[track][timepoint].keys():
                                error_track.append(track)
                                error_timepoint.append(timepoint)
                            if 'Area' in myDict[track][timepoint].keys():
                                if myDict[track][timepoint]['Area'] < 20:
                                    error_track.append(track)
                                    error_timepoint.append(timepoint)
            return error_track, error_timepoint, incomplete_tracks

            
    class Interpolation_old:

        """Functions for interpolating missing data
        """

        def InterpolateSingleFrameNucleusFromDict(track, missingframe, dict):

            coords_before = dict[track][missingframe-1]['Nucleus']['Coordinates']
            coords_after = dict[track][missingframe+1]['Nucleus']['Coordinates']


            img_before = DDN.Utils.makeImageFromCoordinates(coords_before)
            img_after = DDN.Utils.makeImageFromCoordinates(coords_after)
            interpolated = np.zeros(shape)

            for i in range (0, shape[0]):
                interpolated[i] = DDN.Interpolation.interp_shape(img_before[i],img_after[i], 0.5)

            data = interpolated >0
            out=data.astype(np.uint8)
            out[out>0]=255
            props = regionprops(out)

            #from the props we need to return Area, Centroid and Coordinates
            Area = props[0]['Area']
            Centroid = props[0]['Centroid']
            Coordinates = props[0]['Coordinates']

            dict[track][missingframe]['Nucleus'] = {}
            dict[track][missingframe]['Nucleus']['Area'] = Area
            dict[track][missingframe]['Nucleus']['Centroid'] = Centroid
            dict[track][missingframe]['Nucleus']['Coordinates'] = Coordinates
        
            print('Fixed  track ' + str(track) + ' at timepoint ' + str(missingframe))

            return dict
        
        def InterpolateSingleFrameCellFromDict(track, missingframe, dict, shape):

            coords_before = dict[track][missingframe-1]['Coordinates']
            coords_after = dict[track][missingframe+1]['Coordinates']


            img_before = DDN.Utils.makeImageFromCoordinates(coords_before, shape)
            img_after = DDN.Utils.makeImageFromCoordinates(coords_after, shape)
            interpolated = np.zeros(shape)

            for i in range (0, shape[0]):
                interpolated[i] = DDN.Interpolation.interp_shape(img_before[i],img_after[i], 0.5)

            data = interpolated >0
            out=data.astype(np.uint8)
            out[out>0]=255
            props = regionprops(out)

            #from the props we need to return Area, Centroid and Coordinates
            Area = props[0]['Area']
            Centroid = props[0]['Centroid']
            Coordinates = props[0]['Coordinates']

            dict[track][missingframe] = {}
            dict[track][missingframe]['Area'] = Area
            dict[track][missingframe]['Centroid'] = Centroid
            dict[track][missingframe]['Coordinates'] = Coordinates
        
            print('Fixed  track ' + str(track) + ' at timepoint ' + str(missingframe))

            return dict


        def InterpolateTwoFramesFromDict(track, missingframe1, missingframe2, dict, shape):

            coords_before = dict[track][missingframe1-1]['Coordinates']
            coords_after = dict[track][missingframe2+1]['Coordinates']


            img_before = DDN.Utils.makeImageFromCoordinates(coords_before, shape)
            img_after = DDN.Utils.makeImageFromCoordinates(coords_after, shape)
            interpolated_middle = np.zeros(shape)


            for i in range (0, shape[0]):
                interpolated_middle[i] = DDN.Interpolation.interp_shape(img_before[i],img_after[i], 0.5)


            data = interpolated_middle >0
            out=data.astype(np.uint8)
            out[out>0]=255
            props = regionprops(out)

            interpolated_middle_img = DDN.Utils.makeImageFromCoordinates(props[0]['Coordinates'], shape)

            interpolated_1 = np.zeros(shape)
            interpolated_2 = np.zeros(shape)

            for i in range (0, 154):
                interpolated_1[i] = DDN.Interpolation.interp_shape(img_before[i],interpolated_middle_img[i], 0.5)

            for i in range (0, 154):
                interpolated_2[i] = DDN.Interpolation.interp_shape(interpolated_middle_img[i],img_after[i], 0.5)


            data = interpolated_1 >0
            out=data.astype(np.uint8)
            out[out>0]=255
            props1 = regionprops(out)


            data = interpolated_2 >0
            out=data.astype(np.uint8)
            out[out>0]=255
            props2 = regionprops(out)




            #from the props we need to return Area, Centroid and Coordinates
            Area = props1[0]['Area']
            Centroid = props1[0]['Centroid']
            Coordinates = props1[0]['Coordinates']

            dict[track][missingframe1] = {}
            dict[track][missingframe1]['Area'] = Area
            dict[track][missingframe1]['Centroid'] = Centroid
            dict[track][missingframe1]['Coordinates'] = Coordinates
            
          


            Area = props2[0]['Area']
            Centroid = props2[0]['Centroid']
            Coordinates = props2[0]['Coordinates']

            dict[track][missingframe2] = {}
            dict[track][missingframe2]['Area'] = Area
            dict[track][missingframe2]['Centroid'] = Centroid
            dict[track][missingframe2]['Coordinates'] = Coordinates
            

            print('Fixed  track ' + str(track) + ' at timepoint ' + str(missingframe1) + ' '+ str(missingframe2))
            return dict
    
    
        __all__ = ['bwperim']

        def bwperim(bw, n=4):
            """
            perim = bwperim(bw, n=4)
            Find the perimeter of objects in binary images.
            A pixel is part of an object perimeter if its value is one and there
            is at least one zero-valued pixel in its neighborhood.
            By default the neighborhood of a pixel is 4 nearest pixels, but
            if `n` is set to 8 the 8 nearest pixels will be considered.
            Parameters
            ----------
              bw : A black-and-white image
              n : Connectivity. Must be 4 or 8 (default: 8)
            Returns
            -------
              perim : A boolean image
            """

            if n not in (4,8):
                raise ValueError('mahotas.bwperim: n must be 4 or 8')
            rows,cols = bw.shape

            # Translate image by one pixel in all directions
            north = np.zeros((rows,cols))
            south = np.zeros((rows,cols))
            west = np.zeros((rows,cols))
            east = np.zeros((rows,cols))

            north[:-1,:] = bw[1:,:]
            south[1:,:]  = bw[:-1,:]
            west[:,:-1]  = bw[:,1:]
            east[:,1:]   = bw[:,:-1]
            idx = (north == bw) & \
                  (south == bw) & \
                  (west  == bw) & \
                  (east  == bw)
            if n == 8:
                north_east = np.zeros((rows, cols))
                north_west = np.zeros((rows, cols))
                south_east = np.zeros((rows, cols))
                south_west = np.zeros((rows, cols))
                north_east[:-1, 1:]   = bw[1:, :-1]
                north_west[:-1, :-1] = bw[1:, 1:]
                south_east[1:, 1:]     = bw[:-1, :-1]
                south_west[1:, :-1]   = bw[:-1, 1:]
                idx &= (north_east == bw) & \
                       (south_east == bw) & \
                       (south_west == bw) & \
                       (north_west == bw)
            return ~idx * bw



        def signed_bwdist(im):
            '''
            Find perim and return masked image (signed/reversed)
            '''    
            im = -DDN.Interpolation.bwdist(DDN.Interpolation.bwperim(im))*np.logical_not(im) + DDN.Interpolation.bwdist(DDN.Interpolation.bwperim(im))*im
            return im

        def bwdist(im):
            '''
            Find distance map of image
            '''
            dist_im = distance_transform_edt(1-im)
            return dist_im

        def interp_shape(top, bottom, precision):
            '''
            Interpolate between two contours

            Input: top 
                    [X,Y] - Image of top contour (mask)
                   bottom
                    [X,Y] - Image of bottom contour (mask)
                   precision
                     float  - % between the images to interpolate 
                        Ex: num=0.5 - Interpolate the middle image between top and bottom image
            Output: out
                    [X,Y] - Interpolated image at num (%) between top and bottom

            '''
            if precision>2:
                print("Error: Precision must be between 0 and 1 (float)")

            top = DDN.Interpolation.signed_bwdist(top)
            bottom = DDN.Interpolation.signed_bwdist(bottom)

            # row,cols definition
            r, c = top.shape

            # Reverse % indexing
            precision = 1+precision

            # rejoin top, bottom into a single array of shape (2, r, c)
            top_and_bottom = np.stack((top, bottom))

            # create ndgrids 
            points = (np.r_[0, 2], np.arange(r), np.arange(c))
            xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r*c, 2))
            xi = np.c_[np.full((r*c),precision), xi]

            # Interpolate for new plane
            out = interpn(points, top_and_bottom, xi)
            out = out.reshape((r, c))

            # Threshold distmap to values above 0
            out = out > 0

            return out, top, bottom
     
        



def saveImageFile(data, savefile):

        out=data.astype(np.uint8)
        
        writer = ome_tiff_writer.OmeTiffWriter(savefile, overwrite_file=True)
        writer.save(out)
        
        
def makeImageFromCoordinates(coordinates, shape, value = 255):
    """ Takes input coordinates from a dict or prop, and an image shape, and returns a numpy array containing the image
        coordinates: A list of Z,Y,Z coordinates
        shape: a tuple of the Z,Y,Z dimensions of the image'''
    """
    img = np.ma.array(np.zeros(shape))
    for inp in coordinates:#.astype(int):
        img[inp[0],inp[1],inp[2]]=value
    return img
    
    



def testFunctions():
    print('functions loaded')




print('Functions Loaded')


    

def get_3dseed_from_all_frames(bw, stack_shape, hole_min, bg_seed = True, min_area = 15):
    from skimage.morphology import remove_small_objects
    #out = remove_small_objects(bw>0, hole_min)
    out1,_ = ndi.label(bw)
    stat = regionprops(out1)
    seed = np.zeros(stack_shape)
    seed_count=0
    if bg_seed:
        seed[0,:,:] = 1
        seed_count += 1
    for idx in range(len(stat)):
        if stat[idx].area > min_area:
            pz, py, px = np.round(stat[idx].centroid)
            seed_count+=1
            seed[int(pz),int(py),int(px)]=seed_count
    return seed, seed_count


def TrackCells(centroid_positions, start_timepoint = 0, end_timepoint = 120):
    
    #cell_track_list = []
    
    
    
    print('running TrackCells')
    for i, centroid in enumerate(centroid_positions):
        #print('running cell track ' + str(centroid[4]))
        if centroid[0] == 0 and centroid[4] > 1:
            centroid_positions[i, 5] = centroid_positions[i,4]
            centroid_positions = TrackSingleCell(centroid, centroid_positions, start_timepoint=start_timepoint, end_timepoint=end_timepoint)
            #cell_track_list.append(cell_track)
    #cell_track_list_np = np.asarray(cell_track_list)
    
    return centroid_positions.astype(int)

def TrackSingleCell(initial_centroid, centroid_positions, displacement=(0,-3,0), start_timepoint = 0, trackNum = 999, end_timepoint = 240, dist_cutoff=10):
    
    
    #hack to set the first position the same
    clear_output(wait=True)
    #print('initial centroid passed to track single cell = ' + str(initial_centroid))
    
    
    track_length = (end_timepoint - start_timepoint) + 1


    if start_timepoint == 0:
        trackLabel = initial_centroid[4]
        #initial_centroid = np.append(initial_centroid, trackLabel)
    else:
        trackLabel = trackNum
        #initial_centroid = np.append(initial_centroid, trackLabel)

        
    for i, track in enumerate(centroid_positions):
        
        if track[0] == initial_centroid[0] and track[4] == initial_centroid[4]:
            centroid_positions[i, 5] = trackLabel
        
   
    current_centroid = initial_centroid
    
    print('tracking cell number ' + str(initial_centroid[4]) + ' as track ' + str(trackLabel) + ' starting at position ' + str(start_timepoint))
    
    for i in range(start_timepoint, end_timepoint):
            
            clear_output(wait=True)
            print('tracking cell ' + str(trackLabel)  + ' at timepoint ' + str(i))
            min_dist, label, centroid, track_num = FindClosestCentroidWithDisplacement(current_centroid, i, i+1, centroid_positions, disp = displacement)
            if min_dist <= dist_cutoff:
                #centroid = np.append(centroid, trackLabel)
                #cell_track[i+1] = centroid
                centroid_positions[track_num, 5] = trackLabel
                current_centroid = centroid_positions[track_num]
                
                #print('matched centroid at ' + str(i+1) + ' is ' + str(centroid) + ' at position ' + str(track_num))
                #print(centroid_positions[track_num])

                #print(cell_track[i+1])
            elif min_dist > dist_cutoff:  #if there isn't a cell within 10 units, check the next timepoint
                #cell_track[i+1] = -1
                #entroid_positions[track_num, 5] = trackLabel
                if i < end_timepoint - 1:
                    min_dist, label, centroid, track_num = FindClosestCentroidWithDisplacement(current_centroid, i, i+2, centroid_positions, disp = displacement)
                    if min_dist < dist_cutoff:
                        centroid_positions[track_num, 5] = trackLabel
                        current_centroid = centroid_positions[track_num]
                        #cell_track[i+2] = centroid
                    
    #for centroid in cell_track:
    #    centroid[5] = trackLabel
                        
    #return cell_track.astype(int), centroid_positions
    print('finished track')
    return centroid_positions
    
def FindClosestCentroidWithDisplacement(centroid, timepoint1, timepoint2, centroid_list, disp = (0, -3, 0)):

            
            #print(centroid)
            centroid1_zyx = [centroid[1], centroid[2], centroid[3]]

            centroid1_zyx = tuple(map(operator.add, centroid1_zyx, disp))
            
            
            min_dist = 9999
            min_label = -1
            min_centroid = -1
            index = -1
            
           #temp = (centroid_list[:,0] == timepoint2)
           #subset = centroid_list[temp]
            
            for i, next_tp_centroid in enumerate(centroid_list):
                if next_tp_centroid[0] == timepoint2:
                    centroid2_zyx = [next_tp_centroid[1], next_tp_centroid[2], next_tp_centroid[3]]

                    #print('centroid 1 = ' + str(centroid1_zyx))
                    #print('centroid 2 = ' + str(centroid2_zyx))
                    distance = DDN.Utils.calculateDistance(np.array(centroid1_zyx), np.array(centroid2_zyx))
                    if distance < min_dist and next_tp_centroid[5] == -1:
                        min_dist = distance
                        min_centroid = next_tp_centroid
                        min_label = next_tp_centroid[4]
                        index = i
            #print(min_dist)
            #print(min_centroid)
            #print(min_label)
            
            return min_dist, min_label, min_centroid, index  

#min_dist, min_label, min_centroid = findClosestCentroidWithDisplacement([0, 33, 1029, 83, 12], 0, 1, centroid_position)    

    
    
    
def makeLabsImage(timepoints, shape, path):
    labs_whole = np.zeros((timepoints, shape[0],shape[1],shape[2]), dtype=np.uint8)
    centroid_positions = []

    for timepoint in range(0, timepoints):
        
        clear_output(wait=True)
        print('Running timepoint ' + str(timepoint))
        temp = np.load(os.path.join(path, 'seg', 'mem', 'seg_' + str(timepoint) + '.npz'))
        seg = temp['arr_0']
        seg = skimage.util.invert(seg)
        labs, _ = ndi.label(seg)
        props = regionprops(labs)
        #np.savez_compressed(path + '\\labs\\labs_' + str(timepoint) + '.npz', labs)
        labs_whole[timepoint,:,:,:] = labs
        for prop in props:
            if prop['Area'] < 10000000:
                centroid = prop['centroid']
                label = prop['label']
                temp_centroid = [timepoint, int(centroid[0]),int(centroid[1]),int(centroid[2]), prop['label']]
                centroid_positions.append(temp_centroid)

    #add a track list of -1 to centroid_positions
    temp = np.full(len(centroid_positions), -1)
    centroid_positions = np.c_[centroid_positions,temp]
    return labs_whole, centroid_positions

        

#edit the tracks



lineage_to_highlight = 0


def editTracks(cell_tracks, labs, path): 

    
    #cell_tracks = np.load(FileLocation + 'tracking\\new_tracks.npz')
    #cell_tracks = cell_tracks['arr_0']

    #make a subset of the zyc coordinates
    cell_track_list_zyx = cell_tracks[:,0:4]

    #make a list of only the cell tracks
    track_list = cell_tracks[:,5]
    
    viewer = napari.Viewer()

    viewer.add_image(labs, name = 'labs', blending='additive', colormap='gray');
    #viewer.add_image(labs_tracked, name = 'tracked', blending='additive', colormap='gray');



    #add lineage markers
    linmark_x = np.arange(0, 160, 10)
    linmark_x = linmark_x * 10
    linmark_x = linmark_x - 1500
    linmark_y = np.full(len(linmark_x), -5)
    lineage_marks = np.column_stack((linmark_y,linmark_x))
    linmark_50_100 = [[-15, -1000], [-15, -500]]
    lineage_marks = np.row_stack((lineage_marks, linmark_50_100))    
    
    #lineage_marks = np.vstack((linmark_x, linmark_y)).T
    
    points_layer = viewer.add_points(
        lineage_marks,
        face_color='red',
        size=8,
        name = 'lineage marks'
    )
       
    empty_tracks = []

    for track in cell_tracks:
        if track[5] == -1:
            empty_tracks.append(track)
    empty_tracks = np.asarray(empty_tracks)
    empty_tracks_TZYX = empty_tracks[:,0:4]

    points_layer = viewer.add_points(
        empty_tracks_TZYX,
        face_color='red',
        face_colormap=random_cmap,
        size=20,
        name = 'empty tracks'
    )


    #add the lineage points
    
    lineages_tracked = cell_tracks[cell_tracks[:,5] != -1]
    lineage_track_list = lineages_tracked[:,5]
    
    
    tp = lineages_tracked[:,0]
    lineage = np.vstack((tp, lineage_track_list)).T
    lineage[:,1] = lineage[:,1] * 10
    lineage[:,1] = lineage[:,1] - 1500
    
    lineage[:,0] = lineage[:,0] * 5
    
    lineage_name = []
    for i in range(0, len(lineage_track_list)):
        lineage_name.append(0)
    lineage_point_properties = {
        'type': lineage_name,
        'track': lineage_track_list,
        'tp': lineages_tracked[:,0],
        'z': lineages_tracked[:,1],
        'y': lineages_tracked[:,2],
        'x': lineages_tracked[:,3],
        'old_track': lineages_tracked[:,4]     
    }
    
    
    points_layer = viewer.add_points(
        lineage,    
        size=3,
        name='lineage',
        face_color='track',
        face_colormap = random_cmap,
        properties=lineage_point_properties,
    )
    
    centroid_name = []
    for i in range(0, len(cell_tracks[:,5])):
        centroid_name.append(1)
    #add tracked cells
    point_properties2 = {
        'type': centroid_name,
        'track': cell_tracks[:,5],
        'z': cell_tracks[:,1],
        'y': cell_tracks[:,2],
        'x': cell_tracks[:,3],
        'tp': cell_tracks[:,0],
        'old_track': cell_tracks[:,4]     
}

    points_layer = viewer.add_points(
        cell_track_list_zyx,
        properties=point_properties2,
        face_color='track',
        face_colormap=random_cmap,
        size=5,
        name = 'all_centroids'
    )



    @magicgui(call_button='LoadTracksFromSavedAndUpdate')
    def LoadTracksFromSavedAndUpdate():
        cell_tracks = np.load(os.path.join(path, 'tracking', 'new_tracks.npz'))
        cell_tracks = cell_tracks['arr_0']
        TestUpdate(cell_tracks)
    #viewer.window.add_dock_widget(LoadTracksFromSavedAndUpdate.Gui())
    #LoadTracksFromSavedAndUpdate.show(run=True)
    viewer.window.add_dock_widget(LoadTracksFromSavedAndUpdate, area='right')   

    @magicgui(call_button='GoToPoint')
    def GoToPoint():
        current_properties = viewer.layers['lineage'].current_properties
        #centroid = current_properties['deets'][0]
        #print(centroid)
        
        print('track number ' + str(current_properties['track'][0]) + ' a timepoint ' + str(current_properties['tp'][0]))
        
        viewer.dims.set_point(1, current_properties['z'][0])
        viewer.dims.set_point(0, current_properties['tp'][0])
        
        highlight_track_number = current_properties['track']
        
        highlight_track = []

        for track in cell_tracks:
            if track[5] == highlight_track_number:
                highlight_track.append(track)
        highlight_track = np.asarray(highlight_track)
        highlight_track_TZYX = highlight_track[:,0:4]
        
        points_layer = viewer.add_points(
            highlight_track_TZYX,
            face_color='transparent',
            edge_width=12,
            edge_color='green',
            size=15,
            name = 'highlighted track'
    )
    viewer.window.add_dock_widget(GoToPoint, area='right')

    
    
    @magicgui(call_button='PrintDetails')
    def PrintDetails():
        
        
        current_properties = viewer.layers['all_centroids'].current_properties
        centroid = (current_properties['z'][0], current_properties['y'][0], current_properties['x'][0])
        print('track ' + str(current_properties['track'][0]) + ' with centroid at ' + str(centroid) + ', selected timepoint ' + str(current_properties['tp'][0]))
    viewer.window.add_dock_widget(PrintDetails, area='right')

    @magicgui(call_button='PrintLineageDetails')
    def PrintLineageDetails():    
        current_properties = viewer.layers['lineage'].current_properties
        centroid = (current_properties['z'][0], current_properties['y'][0], current_properties['x'][0])
        print('track ' + str(current_properties['track'][0]) + ' with centroid at ' + str(centroid) + ', selected timepoint ' + str(current_properties['tp'][0]))
    viewer.window.add_dock_widget(PrintLineageDetails, area='right')

    @magicgui(call_button='AddPointToTrack')

    def AddPointToTrack(trackToAdd:int):
        current_properties = viewer.layers['all_centroids'].current_properties
        x = current_properties['x'][0]
        y = current_properties['y'][0]
        z = current_properties['z'][0]
        tp = current_properties['tp'][0]
        track = current_properties['track'][0]  
    
        cell_tracks_del = np.delete(cell_tracks, 4, 1)
        #cell_tracks_del

        temp = np.where((cell_tracks_del == (int(tp),int(z),int(y),int(x),int(track))).all(axis=1))
        #cell_tracks[temp]

        cell_tracks[temp,5] = trackToAdd

        SaveSeeds()
        TestUpdate(cell_tracks)
    viewer.window.add_dock_widget(AddPointToTrack, area='right')
    
    @magicgui(call_button='RemoveSinglePointFromTrack')
    def RemoveSinglePointFromTrack():
        current_properties = viewer.layers['all_centroids'].current_properties
        x = current_properties['x'][0]
        y = current_properties['y'][0]
        z = current_properties['z'][0]
        tp = current_properties['tp'][0]
        track = current_properties['track'][0]         
        
        cell_tracks[:,5][(cell_tracks[:,5] == track) & (cell_tracks[:,0] == tp)] = -1
        
        TestUpdate(cell_tracks)
        SaveSeeds()
    viewer.window.add_dock_widget(RemoveSinglePointFromTrack, area='right')

    
    
    @magicgui(call_button='UnTrackLineage')
    def UnTrackLineage():
        
        current_properties = viewer.layers['all_centroids'].current_properties
        x = current_properties['x'][0]
        y = current_properties['y'][0]
        z = current_properties['z'][0]
        tp = current_properties['tp'][0]
        track = current_properties['track'][0]        
     
        
        cell_tracks[:,5][(cell_tracks[:,5] == track) & (cell_tracks[:,0] >= tp)] = -1
        
        TestUpdate(cell_tracks)
        SaveSeeds()
    viewer.window.add_dock_widget(UnTrackLineage, area='right')

    
    @magicgui(call_button='TrackForward')
    def TrackForward(lineage_to_track:int):
        cell_to_track = int(lineage_to_track)
        current_tp = viewer.dims.current_step[0]
        current_z = viewer.dims.current_step[1]
        #find the centroid at the next timepoint

        for track in cell_tracks:

            if track[0] == current_tp+1:
                if track[5] == cell_to_track:
                    current_z = track[1]
                    viewer.dims.set_point(0, current_tp + 1)
                    viewer.dims.set_point(1, current_z)
    viewer.window.add_dock_widget(TrackForward, area='right')           




    @magicgui(call_button='SaveSeeds')
    def SaveSeeds():


        properties = viewer.layers['all_centroids'].properties
        tp = properties['tp']
        z = properties['z']
        y = properties['y']
        x = properties['x']
        old_track = properties['old_track']
        track = properties['track']
        
  
        deets=np.zeros((len(z), 6))

        deets[:,0] = tp
        deets[:,1] = z
        deets[:,2] = y
        deets[:,3] = x
        deets[:,4] = old_track    
        deets[:,5] = track
      
        
        np.savez_compressed(os.path.join(path, 'tracking', 'new_tracks.npz'), deets.astype(int))
        print('seeds saved')
    viewer.window.add_dock_widget(SaveSeeds, area='right')




    @magicgui(call_button='TrackAndAdd')
    def TrackAndAdd(track_list:int):
        track_list = int(track_list)
        centroid_positions_data = viewer.layers['all_centroids'].data  
        properties = viewer.layers['all_centroids'].properties

        current_properties = viewer.layers['all_centroids'].current_properties
        
        x = current_properties['x'][0]
        y = current_properties['y'][0]
        z = current_properties['z'][0]
        tp = int(current_properties['tp'][0])
        track = current_properties['track'][0]        
        
        all_tracks = properties['track']
        
        
        last_tp = int(np.amax(properties['tp']))
        
        maxTrack = np.amax(all_tracks)

        temp = TrackSingleCell((int(tp), int(z), int(y), int(x), int(track_list), int(track_list)), cell_tracks, start_timepoint = tp,  end_timepoint = last_tp, trackNum = int(track_list))
        
        
        SaveSeeds()
        TestUpdate(temp)


    #TrackAndAdd.show(run=True)
    viewer.window.add_dock_widget(TrackAndAdd, area='right')



    @magicgui(call_button='TrackNewTrack')
    def TrackNewTrack():

        centroid_positions_data = viewer.layers['all_centroids'].data  
        properties = viewer.layers['all_centroids'].properties
        
        current_properties = viewer.layers['all_centroids'].current_properties
        x = current_properties['x'][0]
        y = current_properties['y'][0]
        z = current_properties['z'][0]
        tp = int(current_properties['tp'][0])
        
        alltracks = properties['track']
        print(alltracks)
        maxTrack = np.amax(alltracks)
        print('--------')
        print(maxTrack)
        current_properties['track'] = maxTrack        

        temp = TrackSingleCell((int(tp), int(z), int(y), int(x), maxTrack, maxTrack), cell_tracks, start_timepoint = tp,  trackNum = maxTrack+1)
        SaveSeeds()
        TestUpdate(temp)
        #print(cell_track_list)
    #TrackNewTrack.show(run=True)
    viewer.window.add_dock_widget(TrackNewTrack, area='right')


   # @magicgui(call_button='TestUpdate')
    def TestUpdate(cell_tracks):
        cell_track_list_temp = cell_tracks[:,0:4]
        track_list = cell_tracks[:,5]
        
        
        
        viewer.layers.remove('empty tracks')
        viewer.layers.remove('all_centroids')
        viewer.layers.remove('lineage')
        viewer.layers.remove('lineage marks')

        
        

        points_layer = viewer.add_points(
            lineage_marks,
            face_color='red',
            size=8,
            name = 'lineage marks'
    )
        
        lineages_tracked = cell_tracks[cell_tracks[:,5] != -1]
        lineage_track_list = lineages_tracked[:,5]


        tp = lineages_tracked[:,0]
        lineage = np.vstack((tp, lineage_track_list)).T
        lineage[:,1] = lineage[:,1] * 10
        lineage[:,1] = lineage[:,1] - 1500

        lineage[:,0] = lineage[:,0] * 5
    
        
        lineage_name = []
        for i in range(0, len(lineage_track_list)):
            lineage_name.append(0)
        lineage_point_properties = {
            'type': lineage_name,
            'track': lineage_track_list,
            'tp': lineages_tracked[:,0],
            'z': lineages_tracked[:,1],
            'y': lineages_tracked[:,2],
            'x': lineages_tracked[:,3],
            'old_track': lineages_tracked[:,4]     
        }

    
        points_layer = viewer.add_points(
            lineage,    
            size=3,
            name='lineage',
            face_color='track',
            face_colormap = random_cmap,
            properties=lineage_point_properties,
        )




        point_properties3 = {
            'type': centroid_name,
            'track': cell_tracks[:,5],
            'z': cell_tracks[:,1],
            'y': cell_tracks[:,2],
            'x': cell_tracks[:,3],
            'tp': cell_tracks[:,0],
            'old_track': cell_tracks[:,4]     
        }

        
        empty_tracks = []
        for track in cell_tracks:
            if track[5] == -1:
                empty_tracks.append(track)
        empty_tracks = np.asarray(empty_tracks)
        empty_tracks_TZYX = empty_tracks[:,0:4]

        points_layer = viewer.add_points(
            empty_tracks_TZYX,
            face_color='red',
            face_colormap=random_cmap,
            size=20,
            name = 'empty tracks'
        )


        points_layer = viewer.add_points(
            cell_track_list_zyx,
            properties=point_properties3,
            face_color='track',
            face_colormap=random_cmap,
            size=5,
            name = 'all_centroids'
        )
        
  #  viewer.window.add_dock_widget(TestUpdate.Gui())

   