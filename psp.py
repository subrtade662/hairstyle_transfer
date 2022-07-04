import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
from Generator import Generator
import psp_encoders as psp_enc

# modified from https://github.com/eladrich/pixel2style2pixel/blob/master/models/psp.py

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]:v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class MypSp(nn.Module):

    def __init__(self, opts):
        super(MypSp, self).__init__()
        self.set_opts(opts)
        self.encoder = psp_enc.MyHairEncoderGradualStyleWPlus(50, 'ir_se', self.opts)
        self.decoder = Generator(1024, 512, 8).to(self.opts.device)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.load_weights()

    def load_weights(self):
        print('Loading pSp from checkpoint: {}'.format(self.opts.checkpoint_path))
        ckpt = torch.load((self.opts.checkpoint_path), map_location='cpu')
        enc = get_keys(ckpt, 'encoder')
        self.encoder.load_state_dict(enc, strict=True)
        self.decoder.load_state_dict((get_keys(ckpt, 'decoder')), strict=True)

    def forward(self, face, hair, resize=True, randomize_noise=True, inject_latent=None, return_latents=False):
        codes = self.encoder(face, hair)
        input_is_latent = True
        images, result_latent = self.decoder([codes], input_is_latent=input_is_latent,
          randomize_noise=randomize_noise,
          return_latents=return_latents)
        if resize:
            images = self.face_pool(images)
        if return_latents:
            return (images, result_latent)
        else:
            return images

    def set_opts(self, opts):
        self.opts = opts

    def forw_from_codes(self, codes, resize=True, randomize_noise=True, inject_latent=None, return_latents=False):
        input_is_latent = True
        images, result_latent = self.decoder([codes], input_is_latent=input_is_latent,
          randomize_noise=randomize_noise,
          return_latents=return_latents)
        if resize:
            images = self.face_pool(images)
        if return_latents:
            return (images, result_latent)
        else:
            return images

    def forward_v_i(self, face_vec, hair_im, resize=False, randomize_noise=True, inject_latent=None, return_latents=False):
        codes = self.encoder.forward_vec_im(face_vec, hair_im)
        return self.forw_from_codes(codes, resize, randomize_noise, inject_latent, return_latents)

    def forward_i_v(self, face_im, hair_vec, resize=False, randomize_noise=True, inject_latent=None, return_latents=False):
        codes = self.encoder.forward_im_vec(face_im, hair_vec)
        return self.forw_from_codes(codes, resize, randomize_noise, inject_latent, return_latents)

    def forward_v_v(self, face_vec, hair_vec, resize=False, randomize_noise=True, inject_latent=None, return_latents=False):
        codes = self.encoder.forward_vec_vec(face_vec, hair_vec)
        return self.forw_from_codes(codes, resize, randomize_noise, inject_latent, return_latents)

    def get_embedding(self, input_im, is_face):
        return self.encoder.get_embedding(input_im, is_face)
