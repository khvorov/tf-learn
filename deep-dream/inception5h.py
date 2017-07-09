import numpy as np
import tensorflow as tf
import download
import os

########################################################################
# Various directories and file-names.

# The pre-trained Inception model is taken from this tutorial:
# https://github.com/pkmital/CADL/blob/master/session-4/libs/vgg16.py

# The class-names are available in the following URL:
# https://s3.amazonaws.com/cadl/models/synset.txt

# Internet URL for the file with the Inception model.
# Note that this might change in the future and will need to be updated.
data_url = "http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"

# Directory to store the downloaded data.
data_dir = "inception5h/"

# File containing the TensorFlow graph definition. (Downloaded)
path_graph_def = "tensorflow_inception_graph.pb"

########################################################################


def maybe_download():
    print("Downloading Inception 5h Model ...")

    # The file on the internet is not stored in a compressed format.
    # This function should not extract the file when it does not have
    # a relevant filename-extensions such as .zip or .tar.gz
    download.maybe_download_and_extract(url=data_url, download_dir=data_dir)


########################################################################


class Inception5h:
    """
    The Inception model is a Deep Neural Network which has already been
    trained for classifying images into 1000 different categories.

    When you create a new instance of this class, the Inception model
    will be loaded and can be used immediately without training.
    """

    # Name of the tensor for feeding the input image.
    tensor_name_input_image = "input:0"

    # Names for the convolutional layers in the model for use in Style Transfer.
    layer_names = ['conv2d0', 'conv2d1', 'conv2d2',
                   'mixed3a', 'mixed3b',
                   'mixed4a', 'mixed4b', 'mixed4c', 'mixed4d', 'mixed4e',
                   'mixed5a', 'mixed5b']

    def __init__(self):
        # Now load the model from file. The way TensorFlow
        # does this is confusing and requires several steps.

        # Create a new TensorFlow computational graph.
        self.graph = tf.Graph()

        # Set the new graph as the default.
        with self.graph.as_default():

            # TensorFlow graphs are saved to disk as so-called Protocol Buffers
            # aka. proto-bufs which is a file-format that works on multiple
            # platforms. In this case it is saved as a binary file.

            # Open the graph-def file for binary reading.
            path = os.path.join(data_dir, path_graph_def)
            with tf.gfile.FastGFile(path, 'rb') as file:
                # The graph-def is a saved copy of a TensorFlow graph.
                # First we need to create an empty graph-def.
                graph_def = tf.GraphDef()

                # Then we load the proto-buf file into the graph-def.
                graph_def.ParseFromString(file.read())

                # Finally we import the graph-def to the default TensorFlow graph.
                tf.import_graph_def(graph_def, name='')

                # Now self.graph holds the Inception model from the proto-buf file.

            # Get a reference to the tensor for inputting images to the graph.
            self.input = self.graph.get_tensor_by_name(self.tensor_name_input_image)

            # Get references to the tensors for the commonly used layers.
            self.layer_tensors = [self.graph.get_tensor_by_name(name + ":0") for name in self.layer_names]

    def create_feed_dict(self, image):
        """
        Create and return a feed-dict with an image.

        :param image:
            The input image is a 3-dim array which is already decoded.
            The pixels MUST be values between 0 and 255 (float or int).

        :return:
            Dict for feeding to the graph in TensorFlow.
        """

        # Expand 3-dim array to 4-dim by prepending an 'empty' dimension.
        # This is because we are only feeding a single image, but the
        # Inception model was built to take multiple images as input.
        image = np.expand_dims(image, axis=0)

        # Create feed-dict for inputting data to TensorFlow.
        return {self.tensor_name_input_image: image}

    def get_gradient(self, tensor):
        """
        Get gradient for the given tensor
        """

        with self.graph.as_default():
            # TODO: as per Hvass it's worth trying to remove the square operation
            tensor = tf.square(tensor)
            tensor_mean = tf.reduce_mean(tensor)
            gradient = tf.gradients(tensor_mean, self.input)[0]

        return gradient

########################################################################
