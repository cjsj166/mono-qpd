import torch
import torch.nn as nn
import torch.nn.functional as F
from mono_qpd.QPDNet.update import BasicMultiUpdateBlock
from mono_qpd.QPDNet.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from mono_qpd.QPDNet.corr import CorrBlock1D
from mono_qpd.QPDNet.utils.utils import coords_grid, upflow8
from mono_qpd.QPDNet.FFA import Block, Group, default_conv
from torch.profiler import profile, record_function, ProfilerActivity



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

class QPDNet(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        context_dims = args.hidden_dims

        self.cnet = MultiBasicEncoder(output_dim=[args.hidden_dims, context_dims], norm_fn=args.context_norm, downsample=args.n_downsample)
        self.update_block = BasicMultiUpdateBlock(self.args, hidden_dims=args.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, 3, padding=3//2) for i in range(self.args.n_gru_layers)])
        
        # fmap2 lookup
        self.fmap2_reduce_dim = nn.Sequential(
            nn.Conv2d(256, 16, 1, padding=0),
            nn.ReLU(inplace=False),
        )


        ######CAPA initial
        if self.args.CAPA:
            if self.args.input_image_num==4:
                self.FFAGroup = Group(conv=default_conv, dim=36*4, kernel_size=3, blocks=3).cuda()
            else:
                # self.FFAGroup = Group(conv=default_conv, dim=36*2, kernel_size=3, blocks=3).cuda()
                self.FFAGroup = Group(conv=default_conv, dim=36*5, kernel_size=3, blocks=3).cuda()

        if args.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1))
        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
            if self.args.input_image_num==4:
                self.fnet2 = BasicEncoder(output_dim=256, norm_fn='instance', downsample=args.n_downsample)
        

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

    def bilinear_sampler(self, img, coords, mode='bilinear', mask=False):
        H, W = img.shape[-2:]
        xgrid, ygrid = coords.split([1,1], dim=-1)
        xgrid = 2*xgrid/(W-1) - 1
        if H > 1:
            ygrid = 2*ygrid/(H-1) - 1

        grid = torch.cat([xgrid, ygrid], dim=-1)
        img = F.grid_sample(img, grid, align_corners=True)

        if mask:
            mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
            return img, mask.float()

        return img

    def fmaps_lookup(self, coords1, coords0, fmaps_list):
        batch, _, h1, w1 = coords1.shape # batch, xy, h1, w1
        _, _, c, _, _ = fmaps_list[0].shape

        coords0 = coords0[:, :1].reshape(batch, h1, w1, 1)
        coords1 = coords1[:, :1].reshape(batch, h1, w1, 1)

        disp = coords1 - coords0
        r = self.args.corr_radius
        
        dx = torch.linspace(-r, r, 2*r+1)
        dx = dx.view(2*r+1, 1).to(coords1.device)

        # creating grid
        r_x = (coords0+disp).reshape(batch*h1*w1, 1, 1, 1)
        l_x = (coords0-disp).reshape(batch*h1*w1, 1, 1, 1)
        c_x = coords0.reshape(batch*h1*w1, 1, 1, 1)

        feats = []
        for i, fmap in enumerate(fmaps_list): # fmap: [b, t, c, h, w]
            b, t, c, h, w = fmap.shape

            r_x0 = dx + r_x / 2**i
            l_x0 = dx + l_x / 2**i
            c_x0 = dx + c_x / 2**i
            r_x0 = r_x0.reshape(batch * h1, 1, w1*(2*r+1), 1) # b*h, 1, w1*(2*r+1), 1
            l_x0 = l_x0.reshape(batch * h1, 1, w1*(2*r+1), 1)
            c_x0 = c_x0.reshape(batch * h1, 1, w1*(2*r+1), 1)
            y0 = torch.zeros_like(r_x0)
            
            # # concat along batch size and grid sample only once
            # c_coords_lvl = torch.cat([c_x0, y0], dim=-1)
            # r_coords_lvl = torch.cat([r_x0,y0], dim=-1)
            # l_coords_lvl = torch.cat([l_x0,y0], dim=-1)

            # all_coords_lvl = torch.cat([c_coords_lvl, r_coords_lvl, l_coords_lvl], dim=0) # b*h*3, 2, w1*(2*r+1), 2
            # all_fmap = fmap.permute(0, 3, 1, 2, 4).reshape(b*h*t, c, 1, w) # b
            # all_feat = self.bilinear_sampler(all_fmap, all_coords_lvl) # b*h*3, c, 1, w1*(2*r+1)
            # all_feat = all_feat.reshape(b*h, 3, c, w1, 2*r+1) # b*h, 3, c, w1, 2*r+1
            # c_feat = all_feat[:, 0, :, :, :] # b*h, c, w1, 2*r+1
            # c_feat_flip = c_feat.flip(dims=[3]) # b*h, c, w1, 2*r+1
            # r_feat = all_feat[:, 1, :, :, :]
            # l_feat = all_feat[:, 2, :, :, :]
            # l_feat_flip = l_feat.flip(dims=[3]) # b*h, c, w1, 2*r+1

            # feat1 = torch.cat([c_feat, r_feat, l_feat_flip], dim=1) # b*h, 3*c, w1, 2*r+1
            # feat2 = torch.cat([r_feat, l_feat_flip, c_feat_flip], dim=1) # b*h, 3*c, w1, 2*r+1
            # dot = feat1 * feat2 # b*h, 3*c, w1, 2*r+1
            # dot = dot.reshape(b, h1, w1, 3, c, 2*r+1).contiguous() # b, h1, w1, 3, c, 2*r+1
            # dot = torch.sum(dot, dim=4, keepdim=False) / c # b, h1, w1, 3, 1, 2*r+1
            # dot = dot.reshape(b, h1, w1, 3 * (2*r+1))
            # feats.append(dot)

            # aligning fmap shape
            fmap = fmap.permute(0, 3, 2, 1, 4).unsqueeze(2)  # [b, t, c, h, w] -> [b, h, 1, c, t, w]
            fmap = fmap.reshape(b*h, c, 1, t, w)

            # bilinear sampling with grid_sample function
            center_fmap = fmap[:, :, 0, 0:1, :].contiguous() # b*h, c, 1, w
            coords_lvl = torch.cat([c_x0, y0], dim=-1)
            c_feat = self.bilinear_sampler(center_fmap, coords_lvl) # b*h, c, 1, w1*(2*r+1)
            c_feat = c_feat.reshape(b*h, c, w1, 2*r+1) # b*h, c, w1, 2*r+1
            c_feat_flip = c_feat.flip(dims=[3]) # b*h, c, w1, 2*r+1

            right_fmap = fmap[:, :, 0, 1:2, :].contiguous() # b*h, c, 1, w
            coords_lvl = torch.cat([r_x0,y0], dim=-1)
            r_feat = self.bilinear_sampler(right_fmap, coords_lvl) # b*h, c, 1, w1*(2*r+1)
            r_feat = r_feat.reshape(b*h, c, w1, 2*r+1)

            left_fmap = fmap[:, :, 0, 2:3, :].contiguous()
            coords_lvl = torch.cat([l_x0,y0], dim=-1)
            l_feat = self.bilinear_sampler(left_fmap, coords_lvl)
            l_feat = l_feat.reshape(b*h, c, w1, 2*r+1)
            l_feat_flip = l_feat.flip(dims=[3])

            dot = torch.sum(c_feat * r_feat, dim=1, keepdim=True) / c # b*h, 1, w1, 2*r+1
            dot = dot.reshape(b, h1, w1, 2*r+1).contiguous()
            feats.append(dot)

            dot = torch.sum(c_feat_flip * l_feat_flip, dim=1, keepdim=True) / c # b*h, 1, w1, 2*r+1
            dot = dot.reshape(b, h1, w1, 2*r+1).contiguous()
            feats.append(dot)

            dot = torch.sum(l_feat_flip * r_feat, dim=1, keepdim=True) / c # b*h, 1, w1, 2*r+1
            dot = dot.reshape(b, h, w1, 2*r+1).contiguous() # b, h, w1, 2*r+1
            feats.append(dot)

        out = torch.cat(feats, dim=3)
        out = out.permute(0, 3, 1, 2) # b, c*(2*r+1), h, w1
        return out.contiguous().float()


            

    def forward(self, int_features, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
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

            net_list = [torch.tanh(x[0]) for x in cnet_list]
            ori_inp_list = [torch.relu(x[1]) for x in cnet_list] # Original
            inp_list = [x for x in int_features[::-1]]

            # Rather than running the GRU's conv layers on the context features multiple times, we do it once at the beginning
            inp_list = [list(conv(i).split(split_size=conv.out_channels//3, dim=1)) for i,conv in zip(inp_list, self.context_zqr_convs)]

        if self.args.corr_implementation == "reg": # Default
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        else:
            quit()

        fmaps = torch.cat([fmap1.unsqueeze(1), fmap2], dim=1) # [b, t, c, h, w] t=3

        b, t, c, h, w = fmaps.shape
        reduce_fmaps = fmaps.reshape(b*t, c, h, w).contiguous() # [b*t, c, h, w] # .permute(0, 2, 3, 1)
        reduce_fmaps_2 = F.interpolate(reduce_fmaps, size=(h, w//2), mode='bilinear', align_corners=False)
        reduce_fmaps_4 = F.interpolate(reduce_fmaps, size=(h, w//4), mode='bilinear', align_corners=False)
        reduce_fmaps_8 = F.interpolate(reduce_fmaps, size=(h, w//8), mode='bilinear', align_corners=False)

        _, new_c, _, _ = reduce_fmaps.shape
        reduce_fmaps = reduce_fmaps.reshape(b, t, new_c, h, w)
        reduce_fmaps_2 = reduce_fmaps_2.reshape(b, t, new_c, h, w//2)
        reduce_fmaps_4 = reduce_fmaps_4.reshape(b, t, new_c, h, w//4)
        reduce_fmaps_8 = reduce_fmaps_8.reshape(b, t, new_c, h, w//8)

        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels, input_image_num=self.args.input_image_num)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        for itr in range(iters):
            coords1 = coords1.detach()

            # torch.cuda.synchronize()
            
            # with record_function("volume_lookup"):
            volume_corr = corr_fn(coords1, coords0) # index correlation volume
            #     # torch.cuda.synchronize()
            # with record_function("fmaps_lookup"):
            feature_corr = self.fmaps_lookup(coords1, coords0, [reduce_fmaps, reduce_fmaps_2, reduce_fmaps_4, reduce_fmaps_8])
                # torch.cuda.synchronize()

            corr = torch.cat([volume_corr, feature_corr], dim=1) # [b, c*(2*r+1), h, w] # 2*r+1 = 9
            # corr = lrcorr
            if self.args.CAPA:
                corr = self.FFAGroup(corr)

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                if self.args.n_gru_layers == 3 and self.args.slow_fast_gru: # Update low-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=True, iter16=False, iter08=False, update=False)
                if self.args.n_gru_layers >= 2 and self.args.slow_fast_gru:# Update low-res GRU and mid-res GRU
                    net_list = self.update_block(net_list, inp_list, iter32=self.args.n_gru_layers==3, iter16=True, iter08=False, update=False)
                net_list, up_mask, delta_flow = self.update_block(net_list, inp_list, corr, flow, iter32=self.args.n_gru_layers==3, iter16=self.args.n_gru_layers>=2)

            # in stereo mode, project flow onto epipolar
            delta_flow[:,1] = 0.0

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # We do not need to upsample or output intermediate results in test_mode
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_up = flow_up[:,:1]
            flow_predictions.append(flow_up)

        # key_avgs = prof.key_averages()
        # selected = [e for e in key_avgs if e.key in ("fmaps_lookup", "volume_lookup")]
        # for e in key_avgs:
        #     if e.key not in ("fmaps_lookup", "volume_lookup"):
        #         continue
        #     print(f"{e.key}: {e.cpu_time_total:.2f} ms")
        
        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
