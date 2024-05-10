import json
import os

print(os.getcwd())

data_dir = '/Users/tanguybosser/Documents/PhD/phd management/Code/neuralTPPs/data'
file_dir = data_dir + '/baseline/mimic2/split_1/train.json'


f = open(file_dir)
data = json.load(f)
print(data)