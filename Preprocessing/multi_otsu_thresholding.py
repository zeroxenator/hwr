def compute_class_stats(histogram, lower_limit, upper_limit):
    class_occurance = 0
    class_mean = 0
    #compute class occurance
    for i in range(lower_limit, upper_limit):
        if(i +1 in histogram):
            class_occurance += histogram[i+1]
    #compute class mean
    for i in range(lower_limit, upper_limit):
        if(i +1 in histogram):
            class_mean += (i * histogram[i+1]) / class_occurance
        
    return class_occurance, class_mean
        
    
    

def multi_otsu_thresholding(image):
    #count all the different greylevels in the image
    histogram = {}
    height, width = image.shape
    N = height * width
    for i in range(height):
        for j in range(width):
            if(image[i,j] not in histogram):
                histogram[image[i,j]] = image[i,j]
            else:
                 histogram[image[i,j]] += 1
    #calculate p_i: divide each count by the total number of pixels
    for key, value in histogram.items():
        histogram[key] = histogram[key] / N
    #print(histogram)
    k1 = 2
    k2 = 3
    
    total_mean = 0
    total_variance = 0
    #compute the total mean
    for i in range(255):
        if(i + 1 in histogram):
            total_mean += i * histogram[i + 1]
    
    #compute the total variance
    for i in range(255):
        if(i +1 in histogram):
            total_variance += (i+1 - total_mean) * (i+1 - total_mean) * histogram[i + 1]
    
    maximum_variance = 0
    thresholds = [1, 2]
    #for every possible thresholds k1 and k2:
    for k1 in range(1,255):
        for k2 in range(1,255):    
            #1. compute w0, w1 and w2: class occurence of all three classes
            #2. compute u1, u2 and u3: class means of all three classes
            w0, u0 = compute_class_stats(histogram, 1, k1)
            w1, u1 = compute_class_stats(histogram, k1 + 1, k2)
            w2, u2 = compute_class_stats(histogram, k2 + 1, 255)        
            #3. compute and store the between classes variance.
            #variance = w0 * w1 * w2 * (u2 - u1 - u0) * (u2 - u1 - u0)
            variance01 = w0 * w1 * (u1 - u0) * (u1 - u0)
            variance02 = w0 * w2 * (u2 - u0) * (u2 - u0)
            variance12 = w1 * w2 * (u2 - u1) * (u2 - u1)
            variance = variance01 + variance02 + variance12
            #print(variance)
            #4. overwrite current optimal thresholds if variance is higher
            #than current maximum
            if(variance > maximum_variance):
                maximum_variance = variance
                thresholds[0] = k1
                thresholds[1] = k2
        if(round(k1/255 * 100, 1) % 10 == 0):        
        	print("Progress:", round(k1/255 * 100, 1), "%")
    print(thresholds[0],thresholds[1], maximum_variance) 
    #threshold the image using the optimal k1 and k2
    for i in range(height):
        for j in range(width):
            if(image[i,j] < thresholds[0]):
                image[i,j] = 0
            elif(image[i,j] > thresholds[1]):
                image[i,j] = 255
            else:
                image[i,j] = 125         
    #return the image
    return image
