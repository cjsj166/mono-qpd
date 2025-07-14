import torch
import torch.nn as nn
import torch.nn.functional as F
from mono_qpd.QPDNet.update import BasicMultiUpdateBlock
from mono_qpd.QPDNet.extractor import BasicEncoder, MultiBasicEncoder, ResidualBlock
from mono_qpd.QPDNet.corr import CorrBlock1D
from mono_qpd.QPDNet.utils.utils import coords_grid, upflow8
from mono_qpd.QPDNet.FFA import Block, Group, default_conv



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
                self.FFAGroup = Group(conv=default_conv, dim=36*3, kernel_size=3, blocks=3).cuda()

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

    def fmap2_lookup(self, coords1, coords0, fmap2_list):
        batch, _, h1, w1 = coords1.shape
        _, _, c, _, _ = fmap2_list[0].shape

        coords0 = coords0[:, :1].permute(0, 2, 3, 1)
        coords1 = coords1[:, :1].permute(0, 2, 3, 1)

        disp = coords1 - coords0
        r = self.args.corr_radius
        
        dx = torch.linspace(-r, r, 2*r+1)
        dx = dx.view(2*r+1, 1).to(coords1.device)

        # creating grid

        feats = []
        for i, fmap in enumerate(fmap2_list):
            b, t, c, h, w = fmap.shape
            level_base = coords0
            # shift = (2**i - 1) / 2.0
            # level_base = coords0 - shift

            cr_x = (level_base+disp).reshape(batch*h1*w1, 1, 1, 1)
            cl_x = (level_base-disp).reshape(batch*h1*w1, 1, 1, 1)

            cr_x0 = dx + cr_x / 2**i
            cl_x0 = dx + cl_x / 2**i
            cr_x0 = cr_x0.reshape(batch * h1, 1, w1*(2*r+1), 1) # b*h, 1, w1*(2*r+1), 1
            cl_x0 = cl_x0.reshape(batch * h1, 1, w1*(2*r+1), 1)
            y0 = torch.zeros_like(cr_x0)

            # aligning fmap shape
            fmap = fmap.permute(0, 3, 2, 1, 4).unsqueeze(2)  # [b, t, c, h, w] -> [b, h, 1, c, t, w]
            fmap = fmap.reshape(b*h, c, 1, t, w)

            # bilinear sampling with grid_sample function
            right_fmap = fmap[:, :, 0, 1:, :].contiguous() # b*h, c, 1, w
            coords_lvl = torch.cat([cr_x0,y0], dim=-1)
            cr_feat = self.bilinear_sampler(right_fmap, coords_lvl) # b*h, c, 1, w1*(2*r+1)
            cr_feat = cr_feat.reshape(b*h, c, w1, 2*r+1)

            left_fmap = fmap[:, :, 0, :1, :].contiguous()
            coords_lvl = torch.cat([cl_x0,y0], dim=-1)
            cl_feat = self.bilinear_sampler(left_fmap, coords_lvl)
            cl_feat = cl_feat.reshape(b*h, c, w1, 2*r+1)
            cl_feat = cl_feat.flip(dims=[3])

            dot = torch.sum(cl_feat * cr_feat, dim=1, keepdim=True) # b*h, 1, w1, 2*r+1
            dot = dot.reshape(b, h, w1, 2*r+1) # b, h, w1, 2*r+1
            # dot = dot / torch.sqrt(torch.tensor(c, dtype=torch.float32, device=dot.device)) # Normalize by channel size

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

        # with torch.no_grad():
        #     B, T, C, H, W = fmap2.shape
            
        #     x_idx = torch.arange(W, device=fmap2.device, dtype=fmap2.dtype)     # (W,)
        #     x_idx = x_idx.view(1, 1, 1, 1, W)                                   # (1,1,1,1,W)
        #     x_idx = x_idx.expand(B, T, C, H, W)                                 # (B,T,C,H,W)
        #     fmap2 = x_idx.clone()                                             # 원하는 값 저장
        #     fmap2 = fmap2.detach()

        b, t, c, h, w = fmap2.shape
        reduce_fmap2 = fmap2.reshape(b*t, c, h, w).contiguous() # [b*t, c, h, w] # .permute(0, 2, 3, 1)
        reduce_fmap2_2 = F.avg_pool2d(reduce_fmap2, kernel_size=(1, 2), stride=(1, 2)) # reduce_fmap2_2[0, :10, 56, 28]
        reduce_fmap2_4 = F.avg_pool2d(reduce_fmap2_2, kernel_size=(1, 2), stride=(1, 2))
        reduce_fmap2_8 = F.avg_pool2d(reduce_fmap2_4, kernel_size=(1, 2), stride=(1, 2))

        reduce_fmap2 = reduce_fmap2.reshape(b, t, c, h, w)
        reduce_fmap2_2 = reduce_fmap2_2.reshape(b, t, c, h, w//2)
        reduce_fmap2_4 = reduce_fmap2_4.reshape(b, t, c, h, w//4)
        reduce_fmap2_8 = reduce_fmap2_8.reshape(b, t, c, h, w//8)
        corr_fn = corr_block(fmap1, fmap2, radius=self.args.corr_radius, num_levels=self.args.corr_levels, input_image_num=self.args.input_image_num)

        coords0, coords1 = self.initialize_flow(net_list[0])

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            # coords1 = coords0 + 0.01
            # coords0 = coords0 + 0.1
            # coords1 = coords1 / 4.0
            # coords0 = coords0 / 4.0
            coords1 = coords1.detach()
            corr = corr_fn(coords1, coords0) # index correlation volume
            # volume_lrcorr = corr[:, -36:]
            # lrcorr = self.fmap2_lookup(coords1, coords0, [reduce_fmap2, reduce_fmap2_2, reduce_fmap2_4, reduce_fmap2_8])
            # corr[:, -36:] = lrcorr

            # def debug_print(name, tensor_list):
            #     print(name)
            #     for num in tensor_list:
            #         print(f"{num:.8f}", end=' ')
            #     print('')

            # for i in [0, 1, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]:
            #     debug_print(f"lrcorr[{i}]", lrcorr[0, :, 56, i].tolist())
            #     debug_print(f"volume_lrcorr[{i}]", volume_lrcorr[0, :, 56, i].tolist())
            #     debug_print(f"diff[{i}]", (lrcorr[0, :, 56, i] - volume_lrcorr[0, :, 56, i]).tolist())
            #     print("")

            # for i in [0, 1, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]:
            #     debug_print(f"lrcorr[{i}]", lrcorr[0, :, 0, i].tolist())
            #     debug_print(f"volume_lrcorr[{i}]", volume_lrcorr[0, :, 0, i].tolist())
            #     debug_print(f"diff[{i}]", (lrcorr[0, :, 0, i] - volume_lrcorr[0, :, 0, i]).tolist())
            #     print("")
            
            # ref   = lrcorr        # feature‑sampling 버전
            # test  = volume_lrcorr # 보간 volume 버전

            # diff_abs  = (ref - test).abs()
            # peak      = ref.abs().max()

            # rtol = 1e-4           # 값이 ~6e4 까지 가는 경우
            # atol = 1e-6

            # same = (diff_abs <= atol + rtol * peak).all()
            # print('동일?', same)
            # print('max diff', diff_abs.max().item(),
                # 'mean rel', (diff_abs / (ref.abs()+1e-12)).mean().item())

                        
            # with torch.no_grad():
            #     diff = (lrcorr - volume_lrcorr).abs()
            #     flat   = diff.view(-1)                       # (total,)
            #     k      = int(flat.numel() * 0.10 + 0.5)      # 반올림해 개수 결정
            #     k      = max(k, 1)                           # 최소 1개 보장

            #     # ② top-k 추출
            #     topv, topi = torch.topk(flat, k, largest=True, sorted=False)

            #     # ③ 다차원 인덱스로 변환
            #     multi_idx  = torch.unravel_index(topi, diff.shape)
            #     # multi_idx 는 (dim, ) 튜플 ─ 예: (b_idx, c_idx, y_idx, x_idx)

            #     # ④ 필요하면 스택해 (k, ndim) 형태로 보기 좋게 정리
            #     coords = torch.stack(multi_idx, dim=1)       # 각 행: [b, c, y, x]
            #     # print('상위 10 % 개수:', k)
            #     # print('샘플 인덱스 5개:\n', coords[:5])
            #     # print('샘플 값:\n', topv[:5])

            #     import matplotlib.pyplot as plt
            #     # diff_level1 = diff[:, :9, :, :]
            #     # diff_offset = torch.sum(diff_level1, dim=(2, 3))
            #     # plt.imsave(f'result/diff_offset_iter{itr}.png', diff_offset.cpu().numpy(), cmap='gray')
            #     # for i in range(9):
            #     #     plt.imsave(f'result/diff_level1_iter{itr}_ch{i}.png', diff_level1[0, i, :, :].cpu().numpy(), cmap='gray')

            #     diff_level1 = torch.sum(diff[:, :9, :, :], dim=(0, 1))
            #     diff_level2 = torch.sum(diff[:, 9:18, :, :], dim=(0, 1))
            #     diff_level3 = torch.sum(diff[:, 18:27, :, :], dim=(0, 1))
            #     diff_level4 = torch.sum(diff[:, 27:36, :, :], dim=(0, 1))

            #     plt.imsave(f'result/diff_level1_iter{itr}.png', diff_level1.cpu().numpy(), cmap='gray')
            #     plt.imsave(f'result/diff_level2_iter{itr}.png', diff_level2.cpu().numpy(), cmap='gray')
            #     plt.imsave(f'result/diff_level3_iter{itr}.png', diff_level3.cpu().numpy(), cmap='gray')
            #     plt.imsave(f'result/diff_level4_iter{itr}.png', diff_level4.cpu().numpy(), cmap='gray')

            #     lrcorr_level1 = torch.sum(lrcorr[:, :9, :, :], dim=(0, 1))
            #     lrcorr_level2 = torch.sum(lrcorr[:, 9:18, :, :], dim=(0, 1))
            #     lrcorr_level3 = torch.sum(lrcorr[:, 18:27, :, :], dim=(0, 1))
            #     lrcorr_level4 = torch.sum(lrcorr[:, 27:36, :, :], dim=(0, 1))
            #     volume_lrcorr_level1 = torch.sum(volume_lrcorr[:, :9, :, :], dim=(0, 1))
            #     volume_lrcorr_level2 = torch.sum(volume_lrcorr[:, 9:18, :, :], dim=(0, 1))
            #     volume_lrcorr_level3 = torch.sum(volume_lrcorr[:, 18:27, :, :], dim=(0, 1))
            #     volume_lrcorr_level4 = torch.sum(volume_lrcorr[:, 27:36, :, :], dim=(0, 1))

            #     plt.imsave(f'result/lrcorr_level1_iter{itr}.png', lrcorr_level1.cpu().numpy(), cmap='gray')
            #     plt.imsave(f'result/lrcorr_level2_iter{itr}.png', lrcorr_level2.cpu().numpy(), cmap='gray')
            #     plt.imsave(f'result/lrcorr_level3_iter{itr}.png', lrcorr_level3.cpu().numpy(), cmap='gray')
            #     plt.imsave(f'result/lrcorr_level4_iter{itr}.png', lrcorr_level4.cpu().numpy(), cmap='gray')
            #     plt.imsave(f'result/volume_lrcorr_level1_iter{itr}.png', volume_lrcorr_level1.cpu().numpy(), cmap='gray')
            #     plt.imsave(f'result/volume_lrcorr_level2_iter{itr}.png', volume_lrcorr_level2.cpu().numpy(), cmap='gray')
            #     plt.imsave(f'result/volume_lrcorr_level3_iter{itr}.png', volume_lrcorr_level3.cpu().numpy(), cmap='gray')
            #     plt.imsave(f'result/volume_lrcorr_level4_iter{itr}.png', volume_lrcorr_level4.cpu().numpy(), cmap='gray')
                
            # print(lrcorr[0, :19, 56, 56] - volume_lrcorr[0, :19, 56, 56])
            # diff_rel = (lrcorr - volume_lrcorr) / lrcorr
            # print("diff mean:", diff_rel.mean(),"diff std:", diff_rel.std())
            # corr = torch.cat([corr, lrcorr], dim=1)
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

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
