import torch
import torch.nn as nn
import torch.nn.functional as F
from mono_qpd.FMDP.update import BasicMultiUpdateBlock
from mono_qpd.FMDP.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from mono_qpd.FMDP.corr import CorrBlock1D
from mono_qpd.FMDP.utils.utils import coords_grid, upflow8
from mono_qpd.FMDP.FFA import Block, Group, default_conv
from mono_qpd.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from mono_qpd.feature_converter import InterpConverter
import matplotlib.pyplot as plt

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


# class MonoQPD(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         # else_args = args['else']
#         # da_v2_args = args['da_v2']

#         self.da_v2_output_condition = 'enc_features'        
#         self.feature_converter = InterpConverter()
#         self.da_v2 = DepthAnythingV2(args.encoder, output_condition=self.da_v2_output_condition)
#         self.qpdnet = QPDNet(args)

#     def resize_to_14_multiples(self, image):
#         h, w = image.shape[2], image.shape[3]
#         new_h = (h // 14) * 14
#         new_w = (w // 14) * 14

#         resized_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
#         return resized_image
    
#     def normalize_image(self, image):
#         # Normalization
#         mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
#         std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
#         image = image / image.max()
#         image = (image - mean) / std
#         return image
        
#     def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
#         h, w = image1.shape[2], image1.shape[3]
#         assert h % 112 == 0 and w % 112 == 0, "Image dimensions must be multiples of 224"

#         image1_normalized = self.normalize_image(image1)
#         # enc_features, depth = self.da_v2(image1_normalized) # Original
#         if self.da_v2_output_condition == 'enc_features':
#             ret_features = self.da_v2(image1_normalized)
#             ret_features = ret_features[1:]
            
#         elif self.da_v2_output_condition == 'dec_features':
#             ret_features = self.da_v2(image1_normalized)
#             ret_features = ret_features[1:]
        
#         ret_features = self.feature_converter(ret_features)
#         ret_features = ret_features[::-1] # Reverse the order of the features

#         if test_mode:
#             disp, deblur = self.qpdnet(ret_features, image1, image2, iters=iters, test_mode=test_mode, flow_init=None) # [[original_disp, upsampled_disp], [original_deblur, upsampled_deblur]]
#             return disp, deblur
#         else:
#             predictions = self.qpdnet(ret_features, image1, image2, iters=iters, test_mode=test_mode, flow_init=None)
#             return predictions # [disp_predictions, deblur_predictions]


class FMDP(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.feature_converter = InterpConverter()
        self.da_v2 = DepthAnythingV2(args.encoder, output_condition='enc_features')
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        
        ######CAPA initial
        if self.args.CAPA:
            if self.args.input_image_num==4:
                self.FFAGroup = Group(conv=default_conv, dim=36*4, kernel_size=3, blocks=3).cuda()
            else:
                self.FFAGroup = Group(conv=default_conv, dim=36*2, kernel_size=3, blocks=3).cuda()

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
            if self.args.input_image_num==4:
                self.fnet2 = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)

    def resize_to_14_multiples(self, image):
        h, w = image.shape[2], image.shape[3]
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14

        resized_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return resized_image

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, H, W = img.shape

        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, D, H, W = flow.shape
        factor = 2 ** self.args.n_downsample
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, D, 9, 1, 1, H, W)
 
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, D, factor*H, factor*W)

    def imagenet_normalize(self, image):
        # Normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        image = image / image.max()
        image = (image - mean) / std
        return image

    def dav2_forward_align(self, image):
        image = self.imagenet_normalize(image)
        inter_features = self.da_v2(image)
        inter_features = inter_features[1:]                    
        aligned_features = self.feature_converter(inter_features) # Reverse the order of the features

        return aligned_features[::-1]

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        # image1_numpy = image1.cpu().squeeze().permute(1, 2, 0).numpy()
        # left_numpy = image2[0].cpu().permute(1, 2, 0).numpy()
        # plt.imsave('image1.png', image1_numpy.astype('uint8'))
        # plt.imsave('image2.png', left_numpy.astype('uint8'))

        h, w = image1.shape[2], image1.shape[3]
        assert h % 112 == 0 and w % 112 == 0, "Image dimensions must be multiples of 224"

        image1_min_max_normalized = (image1 / 255.0).contiguous()
        image2_min_max_normalized = (image2 / 255.0).contiguous()

        h, w = image1.shape[-2:]
        deblur = F.interpolate(image1_min_max_normalized, size=(h//4, w//4), mode='bilinear', align_corners=False)
        # deblur = image1_min_max_normalized

        image1 = (2 * image1_min_max_normalized - 1.0).contiguous()
        image2 = (2 * image2_min_max_normalized - 1.0).contiguous()
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            if self.args.shared_backbone:
                *cnet_list, x = self.cnet(torch.cat((image1, image2), dim=0), dual_inp=True, num_layers=self.args.n_gru_layers)
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0]//2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.args.n_gru_layers)
                if self.args.input_image_num==4:
                    image_num = image2.shape[0]//self.args.input_image_num
                    flr = self.fnet([image1, image2[:2*image_num]])
                    ftb = self.fnet2([image1, image2[2*image_num:]])
                    fmap1 = torch.stack([flr[0], ftb[0]],dim=1)
                    flr = torch.stack(flr[1:],dim=1)
                    ftb = torch.stack(ftb[1:],dim=1)
                    fmap2 = torch.cat([flr, ftb], dim=1)
                else:
                    fmap = self.fnet([image1, image2])
                    fmap1 = fmap[0]
                    fmap2 = torch.stack(fmap[1:],dim=1)

            # ori_inp_list = [torch.relu(x[1]) for x in cnet_list] # Original

            aligned_features = self.dav2_forward_align(image1_min_max_normalized)       
            net_list = [x for x in aligned_features[::-1]]
            # net_list = [torch.tanh(x[0]) for x in cnet_list] # Oriiginal net_list(inital hidden states)
            inp_list = [x for x in aligned_features[::-1]]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        if self.args.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        else:
            quit()
        # elif self.args.corr_implementation == "alt": # More memory efficient than reg
        #     corr_block = PytorchAlternateCorrBlock1D
        #     fmap1, fmap2 = fmap1.float(), fmap2.float()
        # elif self.args.corr_implementation == "reg_cuda": # Faster version of reg
        #     corr_block = CorrBlockFast1D
        # elif self.args.corr_implementation == "alt_cuda": # Faster version of alt
        #     corr_block = AlternateCorrBlock
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels, input_image_num=self.args.input_image_num)

        coords0, coords1 = self.initialize_flow(net_list[0])

        
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        deblur_predictions = []
        
        for itr in range(iters):
            # print(f"Iteration {itr+1}/{iters}")

            # Depth Anything V2 feature for next itertion
            if itr > 0:
                aligned_features = self.dav2_forward_align(deblur_up)
                inp_list = [x for x in aligned_features[::-1]]
                inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

            coords1 = coords1.detach()
            corr = corr_fn(coords1, coords0) # index correlation volume
            if self.args.CAPA:
                corr = self.FFAGroup(corr)
            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, masks, deltas = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)
            
            up_mask = masks[0]
            deblur_mask = masks[1]

            delta_flow = deltas[0]
            delta_deblur = deltas[1]

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
            deblur = deblur + delta_deblur

            # upsample deblurred image using the same convex upsample algorithm
            deblur_up = self.upsample_flow(deblur, deblur_mask)

            # But we do not need to upsample or output intermediate flows in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]



            flow_predictions.append(flow_up)
            deblur_predictions.append(deblur_up)



        if test_mode:
            return [[coords1 - coords0, flow_up], [deblur, deblur_up]]

        return [flow_predictions, deblur_predictions]
