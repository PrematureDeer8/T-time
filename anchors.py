from torch import nn
import torch
import numpy as np

class Anchors(nn.Module):
        def __init__(self, pyramid_levels=[2,3,4,5,6],strides=None, sizes=None, ratios=np.array([0.5,1,2]), \
                     scales=np.array([1, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])):
            super(Anchors,self).__init__();
            self.pyramid_levels = pyramid_levels;
            self.ratios = ratios;
            self.scales = scales;

            if(strides is None):
                self.strides = [2 ** i for i in self.pyramid_levels];
            if(sizes is None):
                 self.sizes = [2 ** (i + 2) for i in self.pyramid_levels];

        def forward(self, x):
            image_shape = np.array(x.shape[2:]);
            image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels];

            candidate_anchors = np.zeros((0, 4)).astype(np.float32);
            for i, p in enumerate(self.pyramid_levels):
                # create our anchors
                anchors = np.zeros((len(self.ratios) * len(self.scales), 4));
                anchors[..., 2:] = self.sizes[i] * np.transpose(np.tile(self.scales, (2, len(self.ratios))));
                areas = anchors[..., 2] * anchors[..., 3];
                anchors[:, 2] = np.sqrt(areas / np.repeat(self.ratios, len(self.scales)));
                anchors[:, 3] = anchors[:, 2] * np.repeat(self.ratios, len(self.scales));
                anchors[:, 0::2] -= np.transpose(np.tile(anchors[:, 2] * 0.5, (2, 1)));
                anchors[:, 1::2] -= np.transpose(np.tile(anchors[:, 3] * 0.5, (2, 1)));

                # shift anchors so that they don't have negative values
                shift_x = (np.arange(0, image_shapes[i][1]) + 0.5) * self.strides[i];
                shift_y = (np.arange(0, image_shapes[i][0]) + 0.5) * self.strides[i];
                shift_x, shift_y = np.meshgrid(shift_x, shift_y);
                shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose();
                A = anchors.shape[0];
                K = shifts.shape[0];
                shifted_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))).reshape((K*A, 4));
                
                candidate_anchors = np.append(candidate_anchors, shifted_anchors, axis=0);

            candidate_anchors = np.expand_dims(candidate_anchors, axis=0);
            return torch.from_numpy(candidate_anchors.astype(np.float32));

