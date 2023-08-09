import os
import glob
import random
import numpy as np
import darknet
import tensorflow as tf
from PIL import Image, ImageFilter

class Result:
    def __init__(self, image, detect_label, detections, filename):
        self.image = image
        self.detect_label = detect_label
        self.detections = detections
        self.filename = filename




def show_console_result(result_vec, obj_names, frame_id=-1):
    if frame_id >= 0:
        print("Frame:", frame_id)
    
    for i in result_vec:
        if len(obj_names) > i.obj_id:
            print(obj_names[i.obj_id], "-", end=" ")
        
        print("obj_id =", i.obj_id, ", x =", i.x, ", y =", i.y, ", w =", i.w, ", h =", i.h, end=" ")
        print(f", prob = {i.prob:.3f}")
        


def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    a = len(images_path)
    if len(images_path) > 1:
        return images_path
    
    if len(images_path) == 1:
        images_path = images_path[0]
    
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))



def image_detection(image_or_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    if isinstance(image_or_path, str):
        #image = cv2.imread(image_or_path)
        image = Image.open(image_or_path)
    else:
        image = image_or_path
    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image_rgb = image.convert("RGB")
    #image_shape = image_rgb.size
    h_, w_ = image_rgb.size
    image_shape = [w_, h_]

    #image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    image_resized = image_rgb.resize((width, height), resample=Image.BILINEAR)
    image_resized = np.array(image_resized)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes()) 
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    #return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections, image_shape, image_rgb
    return image, detections, image_shape, image_rgb, image_or_path



def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    image = np.array(image)
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def load_network_at_begining():
    data_file = "obj.data"
    weights = "yolov3-custom_best.weights"
    config_file = "yolov3-custom.cfg"
  
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        config_file,
        data_file,
        weights,
    )

    return network, class_names, class_colors
    


def main_detect(input, network, class_names, class_colors):
    res = []

    thresh = 0.25
    data_file = "obj.data"
    weights = "yolov3-custom_best.weights"
    config_file = "yolov3-custom.cfg"
  

    random.seed(3)  # deterministic bbox colors



    images = load_images(input)

    index = 0
    help_ = 0
    while True:
        # loop asking for new image paths if no list is given
        if input:
            if index >= len(images) or help_ == 1:
                break
    
            if isinstance(images, str):
                image_name = images
                help_ = 1
            else:
                image_name = images[index]


        image, detections, image_shape, image_rgb, path = image_detection(
            image_name, network, class_names, class_colors, thresh
            )

        index += 1

        # poslat na zpracování do další CNN pro rozpoznávání číslic
        def key_func(item):
            return item[2][0]

        l = []
        sorted_detections = sorted(detections, key=key_func)

        # extrakce jednotlivých číslic z detekce číslic
        for label, confidence, bbox in sorted_detections:
            if label == '1':
                xx, yy, ww, hh = convert2relative(image, bbox)
                xx = xx * image_shape[1] # 1488
                yy = yy * image_shape[0] # 1261
                ww = ww * image_shape[1] # 85
                hh = hh * image_shape[0] # 141
                if not isinstance(image_rgb, Image.Image):
                    image_rgb = Image.fromarray(image_rgb)

                roi = image_rgb.crop((int(xx-ww/2),int(yy-hh/2), int(xx+ww/2),int(yy+hh/2)))


                # Preprocess the ROI
                gray_image = roi.convert('L')
                gray_image = gray_image.filter(ImageFilter.GaussianBlur(radius=3))
                from skimage.filters import threshold_otsu
                gray_array = np.array(gray_image)
                threshold_value = threshold_otsu(gray_array)
                binary_array = (gray_array > threshold_value).astype(np.uint8) * 255
                gray_image = Image.fromarray(binary_array)


                # teď má výřez tvar obdélníku. Když ho konvertuju na 28x28, tak se smrští a čísla nejdou rozpoznat
                # proto konvertuju na čtverec a čísla potom mají stejný poměr i když udělám resize
                # Define the desired final shape
                # Get the dimensions of the original image
                width, height = gray_image.size # 86, 141
                # Determine the longer side
                longer_side = max(height, width) # 141
                # Compute the margin sizes
                margin_height = (longer_side - height) // 2
                margin_width = (longer_side - width) // 2
                # Create a new image with the desired shape and fill it with black
                new_image = np.zeros((longer_side, longer_side), dtype=np.uint8)
                # Place the original image in the center of the new image
                new_image[margin_height:margin_height + height, margin_width:margin_width + width] = gray_image
                
                # konvertuju na velikost 28x28, která je nutná pro rozpoznávání mnist čísel
                new_image = Image.fromarray(new_image)
                resized_image = new_image.resize((28, 28))
                resized_image = np.array(resized_image)
                test_images = resized_image / 255.0
                test_images = np.reshape(test_images, (1, 28, 28, 1))
                model = tf.keras.models.load_model("MNIST_keras_CNN.h5")

                # predikuj extrahované číslo
                predictions = model.predict(test_images)
                predicted_labels = [tf.argmax(pred).numpy() for pred in predictions]
                
                d = str(predicted_labels)[1:-1]
                l.append(d)



        res.append(Result(image, l, sorted_detections, path))

    return res

