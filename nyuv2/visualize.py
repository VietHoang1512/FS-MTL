import random
import torch
from collections import namedtuple
from PIL import Image
import matplotlib.pyplot as plt

CityscapesClass = namedtuple('CityscapesClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                 'has_instances', 'ignore_in_eval', 'color'])
classes = [
        CityscapesClass('unlabeled', 0, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('ego vehicle', 1, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('rectification border', 2, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('out of roi', 3, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('static', 4, 255, 'void', 0, False, True, (0, 0, 0)),
        CityscapesClass('dynamic', 5, 255, 'void', 0, False, True, (111, 74, 0)),
        CityscapesClass('ground', 6, 255, 'void', 0, False, True, (81, 0, 81)),
        CityscapesClass('road', 7, 0, 'flat', 1, False, False, (128, 64, 128)),
        CityscapesClass('sidewalk', 8, 1, 'flat', 1, False, False, (244, 35, 232)),
        CityscapesClass('parking', 9, 255, 'flat', 1, False, True, (250, 170, 160)),
        CityscapesClass('rail track', 10, 255, 'flat', 1, False, True, (230, 150, 140)),
        CityscapesClass('building', 11, 2, 'construction', 2, False, False, (70, 70, 70)),
        CityscapesClass('wall', 12, 3, 'construction', 2, False, False, (102, 102, 156)),
        CityscapesClass('fence', 13, 4, 'construction', 2, False, False, (190, 153, 153)),
        CityscapesClass('guard rail', 14, 255, 'construction', 2, False, True, (180, 165, 180)),
        CityscapesClass('bridge', 15, 255, 'construction', 2, False, True, (150, 100, 100)),
        CityscapesClass('tunnel', 16, 255, 'construction', 2, False, True, (150, 120, 90)),
        CityscapesClass('pole', 17, 5, 'object', 3, False, False, (153, 153, 153)),
        CityscapesClass('polegroup', 18, 255, 'object', 3, False, True, (153, 153, 153)),
        CityscapesClass('traffic light', 19, 6, 'object', 3, False, False, (250, 170, 30)),
        CityscapesClass('traffic sign', 20, 7, 'object', 3, False, False, (220, 220, 0)),
        CityscapesClass('vegetation', 21, 8, 'nature', 4, False, False, (107, 142, 35)),
        CityscapesClass('terrain', 22, 9, 'nature', 4, False, False, (152, 251, 152)),
        CityscapesClass('sky', 23, 10, 'sky', 5, False, False, (70, 130, 180)),
        CityscapesClass('person', 24, 11, 'human', 6, True, False, (220, 20, 60)),
        CityscapesClass('rider', 25, 12, 'human', 6, True, False, (255, 0, 0)),
        CityscapesClass('car', 26, 13, 'vehicle', 7, True, False, (0, 0, 142)),
        CityscapesClass('truck', 27, 14, 'vehicle', 7, True, False, (0, 0, 70)),
        CityscapesClass('bus', 28, 15, 'vehicle', 7, True, False, (0, 60, 100)),
        CityscapesClass('caravan', 29, 255, 'vehicle', 7, True, True, (0, 0, 90)),
        CityscapesClass('trailer', 30, 255, 'vehicle', 7, True, True, (0, 0, 110)),
        CityscapesClass('train', 31, 16, 'vehicle', 7, True, False, (0, 80, 100)),
        CityscapesClass('motorcycle', 32, 17, 'vehicle', 7, True, False, (0, 0, 230)),
        CityscapesClass('bicycle', 33, 18, 'vehicle', 7, True, False, (119, 11, 32)),
        CityscapesClass('license plate', -1, -1, 'vehicle', 7, False, True, (0, 0, 142)),
    ]

map_id_to_category_id = [x.category_id for x in classes]
map_id_to_category_id = torch.tensor(map_id_to_category_id)


COLORS = [x.color for x in classes]
COLORS = torch.tensor(COLORS)

coarse_COLORS = torch.tensor(
    [(0, 0, 0),
     (128, 64, 128),
     (70, 70, 70),
     (153, 153, 153),
     (107, 142, 35),
     (70, 130, 180),
     (220, 20, 60),
     (0, 0, 142)
     ]
)

def onehot_segmentation_to_img(onehot, colors):
    indices = torch.argmax(onehot, dim=1)
    return indices_segmentation_to_img(indices, colors)


def indices_segmentation_to_img(indices, colors):
    indices = indices.long()
    if indices.size(1) == 1:
        # Remove single channel axis.
        indices = indices[:, 0]
    rgbs = colors[indices]
    return rgbs

def random_crop(img, label, depth, h=184, w=256):
    height, width = img.shape[-2:]
    i = random.randint(0, height - h)
    j = random.randint(0, width - w)
    img = img[:, i : i + h, j : j + w]
    label = label[ i : i + h, j : j + w]
    depth = depth[ :, i : i + h, j : j + w]
    return img, label, depth

if __name__=="__main__":
    from data import NYUv2
    
    dataset = NYUv2(root="data", train=False, augmentation=False)
    
    for idx in range(10):
        image, semantic, depth, _ = dataset[idx]
        
        image, semantic, depth = random_crop(image, semantic, depth)
        

        rgb_semantic = indices_segmentation_to_img(semantic, COLORS)
        rgb_semantic = rgb_semantic.numpy().astype('uint8')
        rgb_semantic = Image.fromarray(rgb_semantic)
        rgb_semantic.save(f'plot/semantic_{idx}.png')
        rgb_image = (255*image).permute(1,2,0).numpy().astype('uint8')        
        rgb_image = Image.fromarray(rgb_image)
        rgb_image.save(f'plot/image_{idx}.png')

        cmap = plt.cm.jet
        cmap.set_bad(color="black")
        depth0 = depth[0]
        depth0 = depth0/10
        plt.imshow(depth0)
        plt.axis('off')
        plt.savefig(f'plot/depth_{idx}.png')