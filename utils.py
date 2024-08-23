import numpy as np
import random
import glob
from PIL import Image
import matplotlib.pyplot as plt 

# to initialize centroids
def init_centroids(num_clusters, image):
    pixels,color_sys = image.shape
    centroids_init = np.empty([num_clusters, color_sys])
    for i in range(num_clusters):
        rand_pixel = random.randint(0,pixels)
        centroids_init[i] = image[rand_pixel, :]
    return centroids_init


# Assign closest centroid to pixel 
def find_closest_centroids(image,centroids):
    num_pixels, color_sys = image.shape
    #index of cluster centroids 
    idx= np.zeros(image.shape[0], dtype = int)
    #find closest centroids index
    for i in range(num_pixels):
        distance = []
        for j in range(centroids.shape[0]):
            distance_cal = np.linalg.norm(image[i] - centroids[j])
            distance.append(distance_cal) 
        idx[i] = np.argmin(distance)
    return idx


# move cluster centroids by finding the mean of all assigned pixels 
def compute_centroid(idx, image, k):
    num_pixels, color_sys = image.shape
    updated_centroids = np.zeros((k,color_sys))
    for i in range(k):
        points = image[idx == i]
        updated_centroids[i] =np.mean(points, axis = 0)
    return updated_centroids


# assigns points and moves cluster centroids in a loop 
def run_kmeans(image,init_centroids, max_iters = 20):
    num_pixels, color_sys = image.shape
    idx = np.zeros(num_pixels)
    k = init_centroids.shape[0]
    centroids = init_centroids
    # collecting idx and centroids for each iteration 
    # we will be needed this ploting the compression process
    all_idx = np.zeros((num_pixels,max_iters))
    all_centroids = np.zeros((k * max_iters,color_sys))
    
    for i in range(max_iters):
        print('K-means iteration %d/%d' %(i, max_iters - 1))
        # assigns pixels to cluster centroids
        idx = find_closest_centroids(image,centroids)
        # update cluster centroids 
        centroids = compute_centroid(idx,image,k)
        # for ploting later 
        all_idx[:,i] = idx
        all_centroids = np.append(all_centroids,centroids, axis = 0)
    return idx, centroids, all_idx, all_centroids


#This function will generarte an image for each iteration of K-means
def image_generator(all_centroids_p,all_idx,max_iters,original_image,input_image,folder_name):
    # lets first save the original image 
    # This will be the first frame of our gif
    original_img =Image.open(input_image)
    original_img.save("iterations/%s/image_before_compression.png"%folder_name)
    print("original image saved")
    #get the centroids for each iteration of K-means
    centroids_split = np.array_split(all_centroids_p, max_iters, axis = 0)
   #generate the image in a loop
    for i  in range(max_iters):
        print('iteration %d/%d' %(i, max_iters - 1))
        #get the associated idx vector 
        idx = all_idx[:,i].astype(int)
        centroids = centroids_split[i]
        #create image
        image= centroids[idx,:]
        # reshape image and save 
        image= np.reshape(image,original_image.shape)
        save_file_name_as = "iterations/%s/image_iteration_%d.png"%(folder_name,i)
        img = Image.fromarray((image * 225).astype(np.uint8))
        img.save(save_file_name_as)
        print("image %d generated successfully" %i)


# a function to display cluster centroids 
def display_cluster_centroids(centroids,title):
    palette = np.expand_dims(centroids, axis = 0)
    #code to display it in a square grid
    #palette = palette.reshape((4,-1,3))
    num = np.arange(0,len(centroids))
    plt.figure(figsize = (16,5))
    plt.xticks(num)
    plt.yticks([])
    plt.imshow(palette)
    plt.title(title)


# display all cluster centroids
def display_all_centroids(initial_centroids,all_centroids_p,max_iters):
    centroids_split = np.array_split(all_centroids_p, max_iters, axis = 0)
    display_cluster_centroids(initial_centroids, 'Initial Cluster Centroids before training ')
    for i in range(max_iters):
        centroids = centroids_split[i]
        display_cluster_centroids(centroids, 'Cluster Centroid at iteration %d'%i)
    

# creating an animated gif that will illustrate the compression process                   
# Creates a GIF of all the images in a folder
def make_gif(frame_folder,save_in):
    frames = []
    # sort by name so proper order is preserved 
    imgs = sorted(glob.glob("{}/*.png".format(frame_folder)))
    print(imgs)
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)
    # Save into a GIF file that loops forever
    frames[0].save(save_in, format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=500, loop=0)

        
        
   