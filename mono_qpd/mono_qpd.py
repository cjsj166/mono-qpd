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

        self.feature_converter = InterpConverter(args.extra_channel_conv)
        self.da_v2 = DepthAnythingV2(args.encoder, output_condition='enc_features')
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
        center, deblur = image1
        
        h, w = center.shape[2], center.shape[3]
        assert h % 112 == 0 and w % 112 == 0, "Image dimensions must be multiples of 224"

        center_normalized = self.normalize_image(center)
        center_ret_features = self.da_v2(center_normalized)
        center_ret_features = center_ret_features[1:]

        deblur_normalized = self.normalize_image(deblur)
        deblur_ret_features = self.da_v2(deblur_normalized)
        deblur_ret_features = deblur_ret_features[1:]

        ret_features = [torch.concat((c_f, d_f), dim=1) for c_f, d_f in zip(center_ret_features, deblur_ret_features)]
        ret_features = self.feature_converter(ret_features)
        # for f in ret_features:
        #     print(f.shape)
        ret_features = ret_features[::-1] # Reverse the order of the features

        if test_mode:
            original_disp, upsampled = self.qpdnet(ret_features, center, image2, iters=iters, test_mode=test_mode, flow_init=None)
            return original_disp, upsampled
        else:
            disp_predictions = self.qpdnet(ret_features, center, image2, iters=iters, test_mode=test_mode, flow_init=None)
            return disp_predictions

