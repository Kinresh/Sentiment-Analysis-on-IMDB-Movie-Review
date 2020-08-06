
# -*- coding: utf-8 -*-
"""

@author: Krunal Katrodiya
"""
import tarfile
import glob

my_tar = tarfile.open('aclImdb_v1.tar.gz')
print ("Extracting All Files...")
my_tar.extractall('./RAW_Data') # specify which folder to extract to
print ("Extraction Completed...")
my_tar.close()


print ("Merging all positive and negative reviews for training in full_train.txt...")

#read positive reviews for training and store in full_train.txt. 
read_files = glob.glob("RAW_Data/aclImdb/train/pos/*.txt")
with open("RAW_Data/full_train.txt", 'w',encoding="utf-8") as mergedFile:
    for f in read_files:
        with open(f, "r",encoding="utf-8") as infile:
            mergedFile.write(infile.read())
        mergedFile.write("\n")
    
    #read negative reviews for training and store it in full_train.txt. 
    read_files = glob.glob("RAW_Data/aclImdb/train/neg/*.txt")
    for f in read_files:
        with open(f, "r",encoding="utf-8") as infile:
            mergedFile.write(infile.read())
        mergedFile.write("\n")

print ("Merged...")



print ("Merging all positive and negative reviews for testing in full_test.txt...")

#read positive reviews for training and store it in full_test.txt. 
read_files = glob.glob("RAW_Data/aclImdb/test/pos/*.txt")

with open("RAW_Data/full_test.txt", "w",encoding="utf-8") as mergedFile:
    for f in read_files:
        with open(f, "r",encoding="utf-8") as infile:
            mergedFile.write(infile.read())
        mergedFile.write("\n")
    
    #read negative reviews for training and store it in full_test.txt. 
    read_files = glob.glob("RAW_Data/aclImdb/test/neg/*.txt")
    for f in read_files:
        with open(f, "r",encoding="utf-8") as infile:
            mergedFile.write(infile.read())
        mergedFile.write("\n")
print ("Merged...")

