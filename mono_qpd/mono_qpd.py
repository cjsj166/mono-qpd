import torch
import torch.nn as nn
import torch.nn.functional as F
from mono_qpd.QPDNet.qpd_net import QPDNet
from mono_qpd.Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
from mono_qpd.feature_converter import PixelShuffleConverter, ConvConverter, DecConverter, FixedConvConverter, InterpConverter, SkipConvConverter


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

class MonoQPD(nn.Module):
    def __init__(self, args):
        super().__init__()
        # else_args = args['else']
        # da_v2_args = args['da_v2']

        self.da_v2_output_condition = 'enc_features'
        if args.feature_converter == 'pixelshuffle':
            self.feature_converter = PixelShuffleConverter()
            self.da_v2_output_condition = 'enc_features'

        elif args.feature_converter == 'conv':
            self.feature_converter = ConvConverter()
            self.da_v2_output_condition = 'enc_features'

        elif args.feature_converter == 'fixed-conv':
            self.feature_converter = FixedConvConverter()
            self.da_v2_output_condition = 'enc_features'
        
        elif args.feature_converter == 'interp':
            self.feature_converter = InterpConverter()
            self.da_v2_output_condition = 'enc_features'
        
        elif args.feature_converter == 'skipconv-interp':
            self.feature_converter = SkipConvConverter()
            self.da_v2_output_condition = 'enc_features'

        elif args.feature_converter == 'decoder_features':
            self.feature_converter = DecConverter()
            self.da_v2_output_condition = 'dec_features'

        self.da_v2 = DepthAnythingV2(args.encoder, output_condition=self.da_v2_output_condition)

        self.qpdnet = QPDNet(args)
    def resize_to_14_multiples(self, image):
        h, w = image.shape[2], image.shape[3]
        new_h = (h // 14) * 14
        new_w = (w // 14) * 14

        resized_image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
        return resized_image
    
    def normalize_image(self, image):
        # Normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
        image = image / image.max()
        image = (image - mean) / std
        return image
        
    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        h, w = image1.shape[2], image1.shape[3]
        assert h % 224 == 0 and w % 224 == 0, "Image dimensions must be multiples of 224"
        # image1_resized = self.resize_to_14_multiples(image1)

        # center_normalized = self.normalize_image(image1)
        left_normalized = self.normalize_image(image2[0:1])
        right_normalized = self.normalize_image(image2[1:])
        # enc_features, depth = self.da_v2(image1_normalized) # Original
        if self.da_v2_output_condition == 'enc_features':
            # c_features = self.da_v2(center_normalized)
            # c_features = c_features[1:]
            l_features = self.da_v2(left_normalized)
            l_features = l_features[1:]
            r_features = self.da_v2(right_normalized)
            r_features = r_features[1:]
        
        lr_features = [torch.cat([l, r], dim=1) for l, r in zip(l_features, r_features)]
        lr_features = self.feature_converter(lr_features)
        # for f in ret_features:
        #     print(f.shape)
        lr_features = lr_features[::-1] # Reverse the order of the features

        if test_mode:
            original_disp, upsampled = self.qpdnet(lr_features, image1, image2, iters=iters, test_mode=test_mode, flow_init=None)
            return original_disp, upsampled
        else:
            disp_predictions = self.qpdnet(lr_features, image1, image2, iters=iters, test_mode=test_mode, flow_init=None)
            return disp_predictions

