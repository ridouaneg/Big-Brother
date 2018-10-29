# Imports
import os
import h5py
import numpy as np


def descriptors_from_database():
    # Generate descriptors from images in the known_peoples folder
    # Input : nothing
    # Output : two numpy arrays
    #          descriptors : each line correspond to the descriptor of an image in the database
    #          names : names associated to the descriptors

    descriptors = []
    names = []

    for folder_name in os.listdir("./ressources/known_peoples"):

        name = folder_name
        for file in os.listdir("./ressources/known_peoples/" + folder_name):

            descriptor = [0] * 128

            descriptors.append(descriptor)
            names.append(name)

    descriptors = np.array(descriptors)
    names = np.array(names)
    names = np.reshape(names, (names.shape[0], 1))

    return descriptors, names


known_peoples_descriptors, known_peoples_labels = descriptors_from_database()

n1, p1 = np.shape(known_peoples_descriptors)
n2, p2 = np.shape(known_peoples_labels)

known_peoples = h5py.File('./ressources/known_peoples/known_peoples.hdf5', 'w')

known_peoples.create_dataset('known_peoples_descriptors', (n1, p1), dtype='<U9', data=known_peoples_descriptors)
known_peoples.create_dataset('known_peoples_labels', (n2, p2), dtype=int, data=known_peoples_labels)

known_peoples.flush()
known_peoples.close()
