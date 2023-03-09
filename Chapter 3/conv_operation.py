import torch
from PIL import Image
import torchvision.transforms as T

# Read input image
img = Image.open('lion_10006.jpg')

# convert the input image to torch tensor
img = T.ToTensor()(img)
print("Input image size:", img.size()) # size = [3, 640, 640]

# unsqueeze the image to make it 4D tensor
img = img.unsqueeze(0) # image size = [1, 3, 640, 640]
# define convolution layer
# conv = nn.Conv2d(in_channels, out_channels, kernel_size)
conv = torch.nn.Conv2d(3, 3, 3)

# apply convolution operation on image
img = conv(img)

# squeeze image to make it 3D
img = img.squeeze(0) #now size is again [3, 640, 640]

# convert image to PIL image
img = T.ToPILImage()(img[0,:,:]) # this is channel 0, by changing the index, I can visualize othr outputs

# display the image after convolution
img.show()