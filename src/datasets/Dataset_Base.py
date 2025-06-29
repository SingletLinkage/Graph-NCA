from torch.utils.data import Dataset
from src.datasets.Data_Instance import Data_Container
import torch
class Dataset_Base(Dataset):
    r"""Base class for any dataset within this project
        .. WARNING:: Not to be used directly!
    """
    def __init__(self, resize=True): 
        self.resize = resize
        self.count = 42
        self.data = Data_Container()

    def set_size(self, size):
        r"""Set size of images
            #Args
                size (int, int): Size of images
        """
        self.size = tuple(size)

    def set_experiment(self, experiment):
        r"""Set experiment
            #Args
                experiment: The experiment class
        """
        self.exp = experiment

    def setPaths(self, images_path, images_list, labels_path, labels_list):
        r"""Set the important image paths
            #Args
                images_path (String): The path to the images
                images_list ([String]): A list of the names of all images
                labels_path (String): The path to the labels
                labels_list ([String]): A list of the names of all labels
            .. TODO:: Refactor
        """
        self.images_path = images_path
        self.images_list = images_list
        self.labels_path = labels_path
        self.labels_list = labels_list
        self.length = len(self.images_list)

    def getImagePaths(self):
        r"""Get a list of all images in dataset
            #Returns:
                list ([String]): List of images
        """
        return self.images_list

    def __len__(self):
        r"""Get number of items in dataset"""
        return self.length

    def getItemByName(self, name):
        r"""Get item by its name
            #Args
                name (String)
            #Returns:
                item (tensor): The image tensor
        """
        idx = self.images_list.index(name)
        return self.__getitem__(idx)

    def getFilesInPath(self, path):
        raise NotImplementedError("Subclasses should implement this!")

    # def __getitem__(self, idx):
    #     raise NotImplementedError("Subclasses should implement this!")
    def __getitem__(self, idx):
        """Get an image and its corresponding label"""
        # Get the image path and label path
        img_id = self.images_list[idx]  # Changed from self.images to self.images_list
        
        # Load and process the image
        img = self.load_image(img_id)
        
        # Load and process the label
        label = self.load_label(img_id)
        
        # Apply any transformations
        if hasattr(self, 'transform') and self.transform:
            img = self.transform(img)
        
        return img_id, img, label
    # def __getitem__(self, idx):
    #     """Get an image and its corresponding label"""
    #     # Get the image path and label path
    #     img_id = self.images_list[idx]  # Changed from self.images to self.images_list
        
    #     # Load and process the image
    #     img = self.load_image(img_id)
        
    #     # Load and process the label
    #     label = self.load_label(img_id)
        
    #     # Apply any transformations only if not already a tensor
    #     if hasattr(self, 'transform') and self.transform and not isinstance(img, torch.Tensor):
    #         img = self.transform(img)
        
    #     return img_id, img, label

    def setState(self, state):
        self.state = state