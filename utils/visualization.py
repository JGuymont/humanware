from torchvision import transforms

def tensor_to_image(tensor):
    img = transforms.ToPILImage(mode='RGB')(tensor)
    img.show()