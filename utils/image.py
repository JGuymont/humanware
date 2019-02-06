from torchvision import transforms
from PIL import Image


class SVHNImage:
    """
    A utility classd for an image from the SVHN dataset
    """
    def __init__(self, metadata, image_path, crop_percent=None, transform=None):
        self._crop_percent = crop_percent
        self._transform = transform
        self._image = Image.open(image_path)
        self._min_left = min(metadata['left'])
        self._max_left = max(metadata['left']) + max(metadata['width'])
        self._min_top = min(metadata['top'])
        self._max_top = max(metadata['top']) + max(metadata['height'])
    
    def _crop(self, image):
        """
        Crop an Pil image around the bounding box that contains all
        the digits

        :param image: the image to crop
        :return: the image cropped
        """
        image = image.crop((self._min_left, self._min_top, self._max_left,
                            self._max_top))
        return image

    def _crop_expand(self, image):
        """
        Crop an Pil image arround the bounding box that contains all
        the digits and expand the box by 30%

        :param image: the image to crop
        """
        image = image.crop((
            (1 - self._crop_percent) * self._min_left, 
            (1 - self._crop_percent) * self._min_top,
            (1 + self._crop_percent) * self._max_left, 
            (1 + self._crop_percent) * self._max_top))
        return image

    def image(self):
        transform = transforms.Compose([transforms.ToTensor()])
        return transform(self._image)
    
    def bounded_image(self):
        transform = transforms.Compose([transforms.ToTensor()])
        image = self._crop(self._image)
        return transform(image)
    
    def cropped_image(self):
        transform = transforms.Compose([transforms.Resize((64, 64)),
                                        transforms.ToTensor()])
        image = self._crop_expand(self._image)
        return transform(image)
    
    def transformed_image(self):
        return self._transform(self._crop_expand(self._image))

