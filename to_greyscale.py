from PIL import Image
import  os


dataset_train_path = './Heiro_val_grey/'
folders = [folder for folder in os.listdir(dataset_train_path) if folder!= ".DS_Store"]

for folder in folders:
    label_dir = os.path.join(dataset_train_path, folder)
    os.chdir(label_dir)
    os.system("for old in *.png; do mv $old `basename $old .png`.jpg; done")
    os.chdir("../..")
    
    
##    images = [image for image in os.listdir(label_dir) if image!= ".DS_Store"]
##
##    for img in images:
##        img_path = os.path.join(label_dir, img)
##        if img_path[-3:] == 'jpg':
##            os.remove(img_path)
##            
