import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

class MaskComponents(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, x, masks):
        device = x.device
        x = x.detach().cpu().numpy()
        bs = x.shape[0]

        batch_idx = []
        new_mask = []

        for b in range(bs):
            mask = masks[b]
            num_valid = torch.tensor((mask == 1).sum().item())
            if torch.sum(mask) != 0 and num_valid != 0:
                batch_idx.append(b)
                new_mask.append(masks[b])


        batch_idx = torch.tensor(batch_idx).to(device)
        new_mask = torch.stack(new_mask).to(device)

        return batch_idx, new_mask



class AuxLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.mc = MaskComponents()

    def forward(self, x, masks, is_front=True):
        with torch.cuda.amp.autocast(enabled=False):

            x = x.squeeze(dim=-1)
            x = x.mean(dim=-1)

            bs = x.shape[0]
            l = x.shape[1]
            h = w = int(math.sqrt(l))
            masks = F.interpolate(masks, size=(h, w), mode='bicubic')
            masks = (masks != 0).long()
            masks = masks.squeeze(1)
            # Detect blobs
            m = torch.mean(x, dim=-1, keepdim=True)
            masks = masks.view(bs, -1)
            # Compute connected components
            with torch.no_grad():
                B_mask = (x >= m).long()
                b_id, masks = self.mc(x, masks)
            if not is_front:
                masks = (masks==0).long()
            inner = torch.logical_and(B_mask[b_id], masks).long()
            outer = torch.logical_or(B_mask[b_id], masks).long()
            p_1 = torch.sum(x[b_id] * inner, dim=-1)
            B_1 = torch.sum(x[b_id] * outer, dim=-1)
            p1 = p_1/B_1
            h = -1 * torch.log(p1 + 1e-4)
            loss = torch.sum(h) / masks.shape[0]

            if not math.isfinite(loss):
                return None
            else:
                return loss

