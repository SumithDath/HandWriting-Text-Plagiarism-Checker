import cv2
import os
import io
from numpy import vectorize 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
num = int(input("Enter Number of Students: "))
for i in range(num):

     roll = input("Enter roll number: ")
     files = [coc for coc in os.listdir() if coc.startswith(roll) if coc.endswith('.png')]
     print(files)


     for file in files:

         img = cv2.imread(file)
         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         adaptive_threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 91, 11)
         text = pytesseract.image_to_string(gray)

         roll_txt = (str(roll) + '.txt') 
         f = open(roll_txt, 'a')
         f.write(text)
         f.close()

sample_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
sample_contents = [open(File).read() for File in sample_files]

vectorize = lambda Text: TfidfVectorizer().fit_transform(Text).toarray()
similarity = lambda doc1, doc2: cosine_similarity([doc1, doc2])

vectors = vectorize(sample_contents)
s_vectors = list(zip(sample_files, vectors))

def check_plagiarism():
    results = set()
    global s_vectors
    for sample_a, text_vector_a in s_vectors:
        new_vectors = s_vectors.copy()
        current_index = new_vectors.index((sample_a, text_vector_a))
        del new_vectors[current_index]
        for sample_b, text_vector_b in new_vectors:
            sim_score = similarity(text_vector_a, text_vector_b)[0][1]
            sample_pair = sorted((sample_a, sample_b))
            score = sample_pair[0], sample_pair[1], sim_score*100
            results.add(score)
    return results

for data in check_plagiarism():
    print(data)    

