import python.darknet
import os
from PIL import Image
from numpy import linalg as LA
import locale
import cv2
import matplotlib.pyplot as plt
import numpy as np
net = python.darknet.load_net(b"/home/dell/Downloads/darknet-master/cfg/yolov3.cfg",b"/home/dell/Downloads/darknet-master/yolov3.weights",0)
meta = python.darknet.load_meta(b"/home/dell/Downloads/darknet-master/cfg/coco.data")
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
list_dir = ['/home/dell/data/raman',
        '/home/dell/data/sadguru',
            '/home/dell/data/sandeep',
            '/home/dell/data/saurabh',
            '/home/dell/data/shailendra'
            ,'/home/dell/data/atul']
for rootdir in list_dir:
    folder = (os.listdir(rootdir))
    folder.sort(key = int)
    for dirs in folder:
        path1 = os.path.join(rootdir,dirs,'data/')
        subfolder = os.listdir(path1)
        subfolder.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for image_path in subfolder:            
            #print(path1+image)
            #print(image)
            flag = 2
            path = bytes(os.path.join(path1, image_path), encoding="utf-8")
            print(path)
            r = python.darknet.detect(net, meta, path) 
            k =0 
            for i in r:
                k+=1
                if i[0] ==b'person' and i[1]>.98:
                    path_2 = os.path.join(rootdir,dirs,'person/')
                    saving_path = path_2+image_path
                    save_file = open(saving_path, 'w')
                    image = Image.open(path)
                    print('person')
                    image.save(saving_path)
                    save_file.close()
                    flag = 0
                    if(flag==0):
                        x = i[2][0] 
                        y = i[2][1]
                        w = i[2][2]
                        z = i[2][3]
                        x_max = (2*x+w)/2
                        x_min = (2*x-w)/2
                        y_min = (2*y-z)/2
                        y_max = (2*y+z)/2
                        image = Image.open(path)
                        cropped = image.crop((x_min, y_min, x_max, y_max))
                        #cropped = image.crop((x, y, w, z))
                        cropped = cropped.resize((300,300), Image.ANTIALIAS)
                        path2 = os.path.join(rootdir,dirs,'one_person/')
                        saving_path = path2+image_path
                        save_file = open(saving_path, 'w')
                        print('crop image')
                        plt.imshow(cropped)
                        cropped.save(saving_path)
                        save_file.close()
                        break   
            if flag ==2: 
                path2 = os.path.join(rootdir,dirs,'no_person/')
                saving_path = path2+image_path
                save_file = open(saving_path, 'w')
                image = Image.open(path)
                print('no person')
                image.save(saving_path)
                save_file.close()
        ##for frame wise pixel comparision
        path3 = os.path.join(rootdir,dirs,'one_person/')
        files = os.listdir(path3)
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        norm_list = []
        images = []
        differ = []
        def convertToRGB(img): 
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        for f in files:
            path4 = os.path.join(path3,f)
            test1 = cv2.imread(path4)
            gray_img = cv2.cvtColor(test1,cv2.COLOR_BGR2GRAY)
            images.append(gray_img)
            norm2 = LA.norm(gray_img,1)
            norm_list.append(norm2)
            for i in range(len(norm_list)):
                differ.append(norm_list[i]-(np.mean(norm_list)))
        i = 0
        j = 0
        path5 = os.path.join(rootdir,dirs,'one_person_/')
        for f in files:
            path4 = os.path.join(path3,f)
            image = Image.open(path4)
            if abs(differ[i])<10000:
                saving_path_6 = path5+f
                image.save(saving_path_6)
                j +=1
            i+=1
            print(j) 
        
            