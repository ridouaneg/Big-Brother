# Imports
import os
import numpy as np
import h5py
import utils
import functions


def descriptors_from_database():
    # Generate descriptors from images in the known_peoples folder
    # Input : nothing
    # Output : two numpy arrays
    #          descriptors : each line correspond to the descriptor of an image in the database
    #          names : names associated to the descriptors

    descriptors = []
    names = []

    # For each image in ./ressources/known_peoples/
    for file in os.listdir("./ressources/known_peoples"):

        print("Processing", file)

        # Generate the descriptor of the (only) face
        image = utils.load_image("./ressources/known_peoples/"+file)
        faces = functions.face_locations(image)
        encodings = functions.face_descriptors(image, faces)

        # If there's one and only one face on the image
        if len(encodings) == 1:
            # Add the descriptor and the name associated to lists
            descriptors.append(encodings[0])
            names.append(file)
            print(file, "has been successfully added.\n")

        # Else, raise error
        else:
            print("ERROR : The image ", file, " contains 0 or more than 1 face.\n")

    # Convert lists to numpy arrays and reshape
    descriptors = np.array(descriptors) # (number of images, 128)
    names = np.array(names)
    names = np.reshape(names, (names.shape[0], 1)) # (number of images, 1)

    return descriptors, names


# Create dataset (h5py file)
print("Creating dataset...\n")
dataset_path = './ressources/datasets/known_peoples.hdf5'
known_peoples = h5py.File(dataset_path, 'w')

# Generate data from the previous function
known_peoples_descriptors, known_peoples_labels = descriptors_from_database()

# Add data to the dataset
n1, p1 = np.shape(known_peoples_descriptors)
known_peoples.create_dataset('known_peoples_descriptors', (n1, p1), dtype=float, data=known_peoples_descriptors)
n2, p2 = np.shape(known_peoples_labels)
#known_peoples.create_dataset('known_peoples_labels', (n2, p2), dtype=int, data=known_peoples_labels)

# Save and close the file
known_peoples.flush()
known_peoples.close()

# Alternative : we can save names in a text file
names_file = open('./ressources/datasets/known_peoples_names.txt', 'w')
for i in range(n2):
    names_file.write(known_peoples_labels[i][0][:-4]+'\n')
names_file.close()

print("Dataset saved at", dataset_path)
print("Done.")
