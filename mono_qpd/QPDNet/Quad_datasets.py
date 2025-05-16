import numpy as np
import torch
import torch.utils.data as data
import torch.nn.functional as F
import logging
import os
import re
import copy
import math
import random
from pathlib import Path
from glob import glob
import os.path as osp

from mono_qpd.QPDNet.utils import frame_utils
from mono_qpd.QPDNet.utils.augmentor import QuadAugmentor, SparseQuadAugmentor
from mono_qpd.QPDNet.utils.transforms import RandomBrightness


class QuadDataset(data.Dataset):
    def __init__(self, datatype, gt_types, aug_params=None, sparse=False, reader=None, lrtb='', is_test=False, image_set = 'train', preprocess_params=None):
        self.augmentor = None
        self.sparse = sparse
        
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                raise NotImplementedError('Not implemented yet')
            else:
                self.augmentor = QuadAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        if datatype == 'dual':
            self.lrtb_image_num = 2
        elif datatype == 'quad':
            self.lrtb_image_num = 4

        self.preprocess_params = preprocess_params
        self.gt_types = gt_types
        self.init_seed = False
        self.disparity_list = []
        self.image_list = []
        self.lrtb = lrtb
        self.image_set = image_set
    
    def _crop(self, img: torch.Tensor, crop_h: int, crop_w: int) -> torch.Tensor:
        h, w = img.shape[-2], img.shape[-1]

        h_start = h // 2 - crop_h // 2
        h_end = h_start + crop_h
        w_start = w // 2 - crop_w // 2
        w_end = w_start + crop_w

        return img[..., h_start:h_end, w_start:w_end]

    def _resize(self, img: torch.Tensor, resize_h: int, resize_w: int) -> torch.Tensor:
        if img.dim() == 3:
            return F.interpolate(img.unsqueeze(0), size=(resize_h, resize_w), mode='bilinear', align_corners=False).squeeze(0)
        elif img.dim() == 4:
            return F.interpolate(img, size=(resize_h, resize_w), mode='bilinear', align_corners=False)
        else:
            raise ValueError("Invalid dimension of input image img.dim() :", img.dim())

    def preprocess(self, items, crop_h, crop_w, resize_h, resize_w):
        """
        Preprocess the items.
        First crop and resize cropped part into desginated size.
        """
        image_list = []
        
        # unpack items
        image_list.append(items['center']) # 3 x h x w
        image_list.append(items['lrtb_list']) # image_num x 3 x h x w
        if 'disp' in items:
            image_list.append(items['disp']) # 1 x h x w
        if 'inv_depth' in items:
            image_list.append(items['inv_depth'])  # 1 x h x w
        if 'AiF' in items:
            image_list.append(items['AiF']) # 3 x h x w

        # crop and resize
        image_list = list(map(lambda x: self._crop(x, crop_h, crop_w), image_list))
        image_list = list(map(lambda x: self._resize(x, resize_h, resize_w), image_list))

        # pack items
        items['center'] = image_list.pop(0)
        items['lrtb_list'] = image_list.pop(0)
        if 'disp' in items:
            items['disp'] = image_list.pop(0)
        if 'inv_depth' in items:
            items['inv_depth'] = image_list.pop(0)
        if 'AiF' in items:
            items['AiF'] = image_list.pop(0)

        return items

    def __getitem__(self, index):
        """
        Retrieve an item from the dataset at the specified index.
        Parameters:
        index (int): The index of the item to retrieve.
        Returns:
        dict: A dictionary containing the following keys:
            - 'image_list': The list of image paths corresponding to the index.
            - 'center': The center image as a torch tensor of shape (3, H, W).
            - 'lrtb_list': A list of left, right, top, and bottom images as a torch tensor of shape (image_num, 3, H, W).
            - 'disp' (optional): The disparity map as a torch tensor of shape (1, H, W), if 'disp' is in self.gt_types.
            - 'disp_valid' (optional): A validity mask for the disparity map as a torch tensor of shape (1, H, W), if 'disp' is in self.gt_types.
            - 'inv_depth' (optional): The inverse depth map as a torch tensor of shape (1, H, W), if 'inv_depth' is in self.gt_types.
            - 'inv_depth_valid' (optional): A validity mask for the inverse depth map as a torch tensor of shape (1, H, W), if 'inv_depth' is in self.gt_types.
            - 'AiF' (optional): The all-in-focus image as a torch tensor of shape (3, H, W), if 'AiF' is in self.gt_types.
        
        """
        
        # Set different seed for each worker
        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        # index can be larger than image_list as these dataset can be mutiplied for certain purpose. This is normalization for such case.
        index = index % len(self.image_list)

        center = frame_utils.read_gen(self.image_list[index][0])
        center = np.array(center).astype(np.uint8)[..., :3]
        
        lrtb_list = []
        for i in range(1,self.lrtb_image_num + 1):            
            img = frame_utils.read_gen(self.image_list[index][i])
            img = np.array(img).astype(np.uint8)[..., :3]
            lrtb_list.append(img)

        # Load the resources
        if 'disp' in self.gt_types:
            disp = self.disparity_reader(self.disparity_list[index]) # h x w
            disp = np.expand_dims(- disp, axis=-1) # h x w x 1
            # flow = np.stack([-disp, np.zeros_like(disp)], axis=-1)
            disp_valid = (np.abs(disp) < 512) # h x w

        if 'inv_depth' in self.gt_types:
            inv_depth = self.disparity_reader(self.inv_depth_list[index]) # h x w x 1
            inv_depth = np.expand_dims(inv_depth, axis=-1) # h x w x 1
            inv_depth_valid = (inv_depth > 0) # h x w x 1 
     
        if 'AiF' in self.gt_types:
            AiF = frame_utils.read_gen(self.aif_list[index])
            AiF = np.array(AiF).astype(np.uint8)[..., :3]
        
        # Pack the items into dictionary
        items = {}
        items['image_list'] = self.image_list[index]
        items['center'] = center
        items['lrtb_list'] = lrtb_list
        if 'disp' in self.gt_types:
            items['disp'] = disp

        if 'inv_depth' in self.gt_types:
            items['inv_depth'] = inv_depth

        if 'AiF' in self.gt_types:
            items['AiF'] = AiF
                    
        # Augmentation for training
        if self.augmentor is not None:
            if self.sparse:
                raise NotImplementedError('Not implemented yet')
                items = self.augmentor(items)
            else:
                items = self.augmentor(items)

        # Convert numpy to torch tensor
        items['lrtb_list'] = np.stack(items['lrtb_list'][0:self.lrtb_image_num], axis=-1)
        items['center'] = torch.from_numpy(items['center']).permute(2, 0, 1).float() # 3 x h x w
        items['lrtb_list'] = torch.from_numpy(items['lrtb_list']).permute(3, 2, 0, 1).float() # image_num x 3 x h x w
        if 'disp' in self.gt_types:
            items['disp' ]= torch.from_numpy(items['disp']).permute(2, 0, 1).float() # 2 x h x w
        if 'inv_depth' in self.gt_types:
            items['inv_depth'] = torch.from_numpy(items['inv_depth']).permute(2, 0, 1).float()
        if 'AiF' in self.gt_types:
            items['AiF'] = torch.from_numpy(items['AiF']).permute(2, 0, 1).float()
        
        # Preprocess
        if self.preprocess_params is not None:
            items = self.preprocess(items, **self.preprocess_params)

        # Add validity mask
        if 'disp' in self.gt_types:
            items['disp_valid'] = torch.ones(1, items['disp'].shape[-2], items['disp'].shape[-1]).float()
        if 'inv_depth' in self.gt_types:
            items['inv_depth_valid'] = torch.ones(1, items['inv_depth'].shape[-2], items['inv_depth'].shape[-1]).float()

        return items

    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)

class QPD(QuadDataset):

    def __init__(self, datatype='dual', gt_types=['disp'], aug_params=None, root='', image_set='train', preprocess_params=None):
        super(QPD, self).__init__(aug_params=aug_params, datatype='dual', gt_types=gt_types, sparse=False, lrtb='', image_set = image_set, preprocess_params=preprocess_params)
        assert os.path.exists(root)
        self.aif_list = []

        imagel_list = sorted(glob(os.path.join(root, image_set+'_l','source', 'seq_*/*.png')))
        imager_list = sorted(glob(os.path.join(root, image_set+'_r','source', 'seq_*/*.png')))

        
        imaget_list = sorted(glob(os.path.join(root, image_set+'_t','source', 'seq_*/*.png')))
        imageb_list = sorted(glob(os.path.join(root, image_set+'_b','source', 'seq_*/*.png')))
        imagec_list = sorted(glob(os.path.join(root, image_set+'_c','source', 'seq_*/*.png')))
        disp_list = sorted(glob(os.path.join(root, image_set+'_c','target_disp', 'seq_*/*.npy')))
        aif_list = sorted(glob(os.path.join(root, image_set+'_c','target', 'seq_*/*.png')))

        for idx, (imgc, imgl, imgr, imgt, imgb, disp, aif) in enumerate(zip(imagec_list, imagel_list, imager_list, imaget_list, imageb_list, disp_list, aif_list)):
            if datatype == 'dual':
                self.image_list += [ [imgc, imgl, imgr] ]
            elif datatype == 'quad':
                self.image_list += [ [imgc, imgl, imgr, imgt, imgb] ]
            self.disparity_list += [ disp ]
            self.aif_list += [ aif ]

class Real_QPD(QuadDataset):
    def __init__(self, datatype='dual', aug_params=None, root='', image_set='train', preprocess_params=None):
        super(Real_QPD, self).__init__(datatype=datatype, gt_types=[], aug_params=aug_params, sparse=False, lrtb='', image_set = image_set, is_test=True, preprocess_params=preprocess_params)
        assert os.path.exists(root)
        imagel_list = sorted(glob(os.path.join(root, '**', 'scale3', image_set+'_l','source', 'seq_*/*.png')))
        imager_list = sorted(glob(os.path.join(root, '**', 'scale3', image_set+'_r','source', 'seq_*/*.png')))
        
        imaget_list = sorted(glob(os.path.join(root, '**', 'scale3', image_set+'_t','source', 'seq_*/*.png')))
        imageb_list = sorted(glob(os.path.join(root, '**', 'scale3', image_set+'_b','source', 'seq_*/*.png')))

        imagec_list = sorted(glob(os.path.join(root, '**', 'scale3', image_set+'_c','source', 'seq_*/*.png')))
        
        for idx, (imgc, imgl, imgr, imgt, imgb) in enumerate(zip(imagec_list, imagel_list, imager_list, imaget_list, imageb_list)):
            if datatype == 'dual':
                self.image_list += [ [imgc, imgl, imgr] ]
            elif datatype == 'quad':
                self.image_list += [ [imgc, imgl, imgr, imgt, imgb] ]

class DPD_Blur(QuadDataset):
    def __init__(self, datatype='dual', gt_types=['AiF'], aug_params=None, root='', image_set='train', resize_ratio = None, preprocess_params=None):
        super(DPD_Blur, self).__init__(aug_params=aug_params, datatype='dual', gt_types=gt_types, sparse=False, lrtb='', image_set = image_set, preprocess_params=preprocess_params)
        
        self.resize_ratio = resize_ratio

        assert os.path.exists(root)
        imagel_list = sorted(glob(os.path.join(root, image_set+'_l','source', '*.png')))
        imager_list = sorted(glob(os.path.join(root, image_set+'_r','source', '*.png')))
        imagec_list = sorted(glob(os.path.join(root, image_set+'_c','source', '*.png')))
        
        for idx, (imgc, imgl, imgr) in enumerate(zip(imagec_list, imagel_list, imager_list)):
            if datatype == 'dual':
                self.image_list += [ [imgc, imgl, imgr] ]
            elif datatype == 'quad':
                raise NotImplementedError

class DPD_Disp(QuadDataset):
    def __init__(self, datatype='dual', gt_types=['inv_depth'], aug_params=None, root='', image_set='train', resize_ratio = None, preprocess_params=None):
        super(DPD_Disp, self).__init__(aug_params=aug_params, datatype='dual', gt_types=gt_types, sparse=False, lrtb='', image_set = image_set, preprocess_params=preprocess_params)
        self.inv_depth_list = []

        self.resize_ratio = resize_ratio

        assert os.path.exists(root)
        imagel_list = sorted(glob(os.path.join(root, image_set+'_l','source', 'seq_*/*.jpg')))
        imager_list = sorted(glob(os.path.join(root, image_set+'_r','source', 'seq_*/*.jpg')))
        imagec_list = sorted(glob(os.path.join(root, image_set+'_c','source', 'seq_*/*.jpg')))
        depth_list = sorted(glob(os.path.join(root, image_set+'_c','target_depth', 'seq_*/*.TIF')))

        for idx, (imgc, imgl, imgr, depth) in enumerate(zip(imagec_list, imagel_list, imager_list, depth_list)):
            self.image_list += [ [imgc, imgl, imgr] ]
            self.inv_depth_list += [ depth ] # depth

  
def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    
    for dataset_name in args.train_datasets:
        if dataset_name.startswith("QPD"):
            new_dataset = QPD(datatype=args.datatype, gt_types=args.qpd_gt_types, aug_params=aug_params, root=args.datasets_path)
        train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=True, shuffle=True, num_workers=int(os.environ.get('SLURM_CPUS_PER_TASK', 6))-2, drop_last=True)
    
    # : Below line is only for debugging
    # train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
    #     pin_memory=True, shuffle=True, num_workers=0, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader
