import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

class ARTModel:
    def __init__(self):
        self.weights = []
        self.model = []
        self.model_number = 0
        self.traning_mode = 0
        self.final_output = None  

    def art_f1(self, input_matrix):
        if self.traning_mode != 0 and self.model_number != 0:
            sums = np.array([np.sum(self.model[x] * self.weights[x] * input_matrix) for x in range(self.model_number)])
            max_value = np.argmax(sums)
            self.art_f2(input_matrix, max_value)
        else:
            self.add_model(input_matrix)

    def art_f2(self, input_matrix, max_value):
        xi = self.model[max_value] * input_matrix
        sum_xi = np.sum(xi)
        total_sum = np.sum(input_matrix)
        if sum_xi / total_sum <= 0.75:
            self.add_model(input_matrix)
        else:
            self.output_model(max_value)

    def add_model(self, input_matrix):
        self.model.append(input_matrix)
        xi = input_matrix * input_matrix
        sum_xi = np.sum(xi)
        new_b = (2 * xi) / (sum_xi + 1)
        self.weights.append(new_b)
        self.model_number += 1
        self.output_model(self.model_number-1)

    def output_model(self, max_value):
       
        if self.final_output is None:
            self.final_output = self.model[max_value]
        else:
            self.final_output = np.concatenate((self.final_output, self.model[max_value]), axis=1)

    def show_final_output(self):
    
        if self.final_output is not None:
            plt.imshow(self.final_output, cmap='gray')
            plt.colorbar()
            plt.show()

    def input(self, input_group, tm):
        self.traning_mode = tm
        self.final_output = None 
        for input_matrix in input_group:
            self.art_f1(input_matrix)


def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError("The image path is incorrect or the file does not exist.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

def detect_text(binary):
    reader = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(binary, paragraph=False)
    return results

def process_text_blocks(results, binary):
    input = []
    for (bbox, text, prob) in results:
        if prob > 0.1:
            x, y, w, h = cv2.boundingRect(np.array([bbox]).astype(int))
            if w > 0 and h > 0: 
                roi = binary[y:y+h, x:x+w]
                if roi.size == 0:
                    print("Empty ROI detected")
                    continue 
                num_labels, labels_im = cv2.connectedComponents(roi)
                for label in range(1, num_labels): 
                    letter_mask = (labels_im == label).astype(np.uint8) * 255
                    x, y, w, h = cv2.boundingRect(letter_mask)
                    if w > 0 and h > 0: 
                        letter_img = roi[y:y+h, x:x+w]
                        resized_img = cv2.resize(letter_img, (300, 300), interpolation=cv2.INTER_AREA)
                        input.append(resized_img / 255.0)
            else:
                print("Invalid bounding box dimensions detected")
    return input



image_path = 'harf10.jpg'
binary = preprocess_image(image_path)
results = detect_text(binary)
input_data = process_text_blocks(results, binary)

art = ARTModel()
art.input(input_data, 0)



def display_all_models(art_model):
    print("\n")
    print("Total models stored:", art_model.model_number)

display_all_models(art)
art.show_final_output()
image_path = 's2.jpg'
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError("The image path is incorrect or the file does not exist.")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)


reader = easyocr.Reader(['en'], gpu=False)


results = reader.readtext(binary, paragraph=False)


results_sorted = sorted(results, key=lambda item: cv2.boundingRect(np.array([item[0]]).astype(int))[0])

def resize_matrix(matrix, size=(300, 300)):
    """Resize the letter image to a specific size."""
    return cv2.resize(matrix, size, interpolation=cv2.INTER_AREA)


input = []


for (bbox, text, prob) in results_sorted:
    if prob > 0.1:  
        x, y, w, h = cv2.boundingRect(np.array([bbox]).astype(int))
        roi = binary[y:y+h, x:x+w]
        num_labels, labels_im = cv2.connectedComponents(roi)

       
        letter_bboxes = []
        for label in range(1, num_labels): 
            letter_mask = (labels_im == label).astype(np.uint8) * 255
            x, y, w, h = cv2.boundingRect(letter_mask)
            letter_bboxes.append((x, letter_mask, y, w, h))

       
        letter_bboxes_sorted = sorted(letter_bboxes, key=lambda b: b[0])

       
        for x, letter_mask, y, w, h in letter_bboxes_sorted:
            letter_img = roi[y:y+h, x:x+w]
            resized_img = resize_matrix(letter_img) 
            input.append(resized_img / 255.0)

art.input(input,1)  
art.show_final_output()