# **HandWriting-Text-Plagiarism-Checker**

## Installation
• Clone Repository

``` 
   git clone https://github.com/SumithDath/HandWriting-Text-Plagiarism-Checker/
```




• Install Python modules

``` pip install opencv-python 
    pip install numpy
    pip install pytessract
    pip install scikit-learn
    pip install pillow 
```

• Download Tesseract-OCR

https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.1.20220118.exe

## Usage

Run Python script

```
    python script.py
```

• Input Values

``` Enter Number of Students:                      // integer ```

``` Enter Roll Number:                             // char or int ```

## Description

Images(.png) of a students should end with integer.
```

    EX: Student1-1.png
        Student1-2.png
        Student1-3.png
        .
        .
        Student2-1.png
        Student2-2.png
        Student2-3.png
        .
        .
```

Each image is scanned and converted into text.

Content of ```Student1``` are stored in ```Student1.txt```

From list of ```Outputs```, each pair of ``.txt`` files are compared to check for similarities and plagiarism percentage is shown.
