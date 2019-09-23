import os
import pandas as pd

from Util import Util
from Functions import Functions


data_path = './data/known_peoples'
output_path = './data/known_peoples.csv'

print('Pre-processing of images in', data_path)

names = []
descriptors = []

for file in os.listdir(data_path):

    print('Processing', file)
    image = Util.load_image(os.path.join(data_path, file))

    name = file[:-4]
    encodings = Functions.extract_descriptors(image)

    if len(encodings) == 1:

        print(file, 'has been successfully processed.')
        names.append(name)
        descriptors.append(encodings[0])
        print(encodings[0], type(encodings[0]))

    else:

        print('ERROR : The image', file, 'contains 0 or more than 1 face.')


print('Creating dataset...')
dataset = pd.DataFrame({'name':names, 'descriptor':descriptors})
dataset.to_pickle(output_path)
print("Dataset saved at", output_path)
print("Done.")
