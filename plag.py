import os
from numpy import vectorize 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import pytesseract
import io

def process_image(image_name, lang_code):
    pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    return pytesseract.image_to_string(Image.open(image_name), lang=lang_code)

def print_data(data):
    print(data)

def output_file(filename, data):
    file = io.open(filename, "w+", encoding="utf-8")
    file.write(data)
    file.close()

def main():
    num = int(input("Enter Number of Images: "))
    for i in range(num):
     img = input("Enter image name: ")
     data_eng = process_image(img, "eng")
    
     name = input("Enter file name: ")
     print(data_eng)
     output_file(name, data_eng)

if __name__ == "__main__":
    main()

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

