from PIL import Image
import torch
from torch.utils.data import Dataset

# Used for Slide.inferClassifier()
class WholeSlideImageDataset(Dataset):
    """WholeSlideImage dataset."""

    def __init__(self, slideClass, tissueLevelThreshold=False, foregroundLevelThreshold=False, otsuLevelThreshold=False, triangleLevelThreshold=False, maskLevelThreshold=False, transform=None, segmenting=False):
        self.slideClass = slideClass
        self.tileSize = self.slideClass.tileSize
        self.tissueLevelThreshold = tissueLevelThreshold
        self.foregroundLevelThreshold = foregroundLevelThreshold
        self.otsuLevelThreshold = otsuLevelThreshold
        self.triangleLevelThreshold=triangleLevelThreshold
        self.maskLevelThreshold = maskLevelThreshold
        self.transform = transform
        self.segmenting = segmenting
        self.suitableTileAddresses = self.slideClass.suitableTileAddresses(tissueLevelThreshold=self.tissueLevelThreshold,
                                                                            foregroundLevelThreshold=self.foregroundLevelThreshold,
                                                                            otsuLevelThreshold=self.otsuLevelThreshold,
                                                                            triangleLevelThreshold=self.triangleLevelThreshold,
                                                                             maskLevelThreshold=self.maskLevelThreshold)


    def __len__(self):
        #if self.tissueLevelThreshold or self.foregroundLevelThreshold:
        return len(self.suitableTileAddresses)
        #else:
        #    return self.slideClass.getTileCount(foregroundOnly=self.foregroundOnly)

    # returns HWC 0-255 numpy array of tile if segmenting is False,
    # otherwise returns CHW 0-1 torch tensor. No transforms recommended for segmenting case
    def __getitem__(self, idx):

        tileAddress = self.suitableTileAddresses[idx]

        if self.segmenting:
            img = self.slideClass.getTile(tileAddress, writeToNumpy=True, useFetch=True)
            img = img[...,:3] # clip off the transparency channel
        else:
            img = Image.fromarray(self.slideClass.getTile(
                tileAddress, writeToNumpy=True, useFetch=True)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # process numpy image for input into segmentation architecture
        if self.segmenting:
            img = img.transpose((2, 0, 1))
            if img.max() > 1:
                img = img / 255
            img = torch.from_numpy(img).type(torch.FloatTensor)

        sample = {'image': img, 'tileAddress': tileAddress}
        return sample

