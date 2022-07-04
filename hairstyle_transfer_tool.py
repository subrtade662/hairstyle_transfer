from fnmatch import filter
from psp import MypSp
from argparse import Namespace
import utils, os, torch, imageio, numpy as np, cv2, string, pickle as pkl



class Tool:

    def __init__(self, opts=None, result_path='./data/results/', checkpoint_path='./best_model.pt'):
        if opts is None:
            self.opts = Namespace()
            self.opts.input_nc = 3
            self.opts.device = 'cuda:0'
            self.opts.test_batch_size = 2
            self.opts.checkpoint_path = checkpoint_path
            self.opts.result_path = result_path
        else:
            self.opts = opts
            self.opts.result_path = result_path
            os.makedirs(result_path, exist_ok=True)
        self.net = self.load_network()
        self.uploaded_images_path = './data/uploaded_images/'
        self.predefined_images_path = './data/'
        os.makedirs((self.uploaded_images_path), exist_ok=True)
        self.chars = np.array((list(string.ascii_letters + string.digits + '_')), dtype=str)
        self.char_inds = np.arange(len(self.chars))
        self.alpha_blend_sigma = 5.0

    def get_predefined_images(self):
        image_files_ = os.listdir(self.predefined_images_path)
        image_files = []
        for suff in ('jpg', 'jpeg', 'png'):
            image_files += filter(image_files_, f"*.{suff}")

        return image_files

    def get_previously_uploaded_images(self):
        image_files_ = os.listdir(self.uploaded_images_path)
        image_files = []
        for suff in ('jpg', 'jpeg', 'png'):
            image_files += filter(image_files_, f"*.{suff}")

        return image_files

    def get_new_filename(self):
        fname_len = 20

        def get_rand_fname():
            ids = np.random.choice((self.char_inds), size=fname_len, replace=True)
            return ''.join(self.chars[ids])

        fname_ = get_rand_fname()
        print(f"generated name: {fname_}")
        while os.path.exists(os.path.join(self.uploaded_images_path, f"u{fname_}.png")):
            fname_ = get_rand_fname()

        return f"u{fname_}.png"

    def get_all_images(self):
        predefined_images = self.get_predefined_images()
        uploaded = self.get_previously_uploaded_images()
        return predefined_images + uploaded

    def load_network(self):
        net = MypSp(self.opts)
        return net

    def get_absolute_path(self, fname):
        fname = fname.split('/')[(-1)]
        if fname.startswith('u'):
            return os.path.join(self.uploaded_images_path, fname)
        else:
            return os.path.join(self.predefined_images_path, fname)

    def __prepare_filename(self, nm1, nm2, blend, kwargs={'phase': 'ht'}):
        fname1_ws = nm1.split('/')[(-1)]
        fname1_wos = fname1_ws.split('.')[0]
        if nm2 is not None:
            fname2_ws = nm2.split('/')[(-1)]
            fname2_wos = fname2_ws.split('.')[0]
        if 'ht' in kwargs['phase']:
            new_filename = os.path.join(self.opts.result_path, f"{fname1_wos}_{fname2_wos}_blend_{blend}.jpg")
        else:
            if 'interp' in kwargs['phase']:
                os.makedirs((os.path.join(self.opts.result_path, 'interp')), exist_ok=True)
                new_filename = os.path.join(self.opts.result_path, 'interp', f"{fname1_wos}_{fname2_wos}_{kwargs['curr_step']}_{kwargs['max_step']}_blend_{blend}_hair_{kwargs['hair']}.jpg")
            else:
                if 'manip' in kwargs['phase']:
                    os.makedirs((os.path.join(self.opts.result_path, 'manip')), exist_ok=True)
                    new_filename = os.path.join(self.opts.result_path, 'manip', f"{fname1_wos}_{kwargs['dir']}_{kwargs['strength']}_blend_{blend}.jpg")
        return new_filename

    def hairstyle_transfer(self, im1_path, im2_path, alpha_blend=True):
        nonblended_name = self.__prepare_filename(im1_path, im2_path, False)
        blended_name = self.__prepare_filename(im1_path, im2_path, True)
        thefilename = blended_name if alpha_blend else nonblended_name
        if os.path.exists(thefilename):
            return thefilename
        else:
            if self.net is None:
                print('Model was not loaded. Loading default model')
                self.load_network()
            else:
                self.net.to(self.opts.device)
                if alpha_blend:
                    inner_face1, hair1, mask1 = utils.load_and_split_image(im1_path, return_mask=True)
                    inner_face2, hair2, mask2 = utils.load_and_split_image(im2_path, return_mask=True)
                else:
                    inner_face1, hair1 = utils.load_and_split_image(im1_path)
                inner_face2, hair2 = utils.load_and_split_image(im2_path)
            # print(f"Range of inner face: [{inner_face2.min()}, {inner_face2.max()}]")
            im2_orig = inner_face2 + hair2
            input_face = utils.numpy_uint8_to_torch(inner_face2.astype(np.uint8))
            input_hair = utils.numpy_uint8_to_torch(hair1.astype(np.uint8))
            if alpha_blend:
                masks = [
                 mask2]
                im_orig = [im2_orig]
            output = self.net.forward(input_face, input_hair, resize=False)
            output_images = output.cpu()
            for ii in range(output_images.shape[0]):
                img_generated = utils.torch_to_numpy_uint8(output_images[ii:ii + 1])
                if alpha_blend:
                    img_original = im_orig[ii]
                    im_mask = utils.gaussian_filter((masks[ii]), sigma=(self.alpha_blend_sigma))
                    # print(f"Range of image mask: [{im_mask.min()}, {im_mask.max()}]")
                    if img_original.shape[0] < 1024:
                        img_original = cv2.resize(img_original, (1024, 1024))
                    if im_mask.shape[0] < 1024:
                        im_mask = cv2.resize(im_mask, (1024, 1024))
                    masked_output = utils.alpha_blend_images(img_generated, img_original, im_mask)
                    # print(f"Range of masked output: [{masked_output.min()}, {masked_output.max()}]")
                    imageio.imwrite(blended_name, masked_output)
                imageio.imwrite(nonblended_name, img_generated)

            return thefilename

    def __interpolate_from_inputs(self, net, in1, in2, are_faces=False, n_interp_steps=5):
        with torch.no_grad():
            e1 = net.get_embedding(in1, are_faces).cpu()
            e2 = net.get_embedding(in2, are_faces).cpu()
            linspace = np.linspace(0, 1, num=n_interp_steps)
            out = list(map(lambda x: (1 - x) * e1 + x * e2, linspace))
            return torch.cat(tuple(out), 0)

    def interpolation_single_pair(self, im1_path, im2_path, n_steps=10, interpolate_hair=True, alpha_blend=True):
        if os.path.exists(self.__prepare_filename(im1_path, im2_path, alpha_blend, kwargs={'phase':'interp',  'hair':interpolate_hair,  'curr_step':0,  'max_step':n_steps})):
            filenames = list(map(lambda x: self.__prepare_filename(im1_path, im2_path, alpha_blend, kwargs={'phase':'interp',  'hair':interpolate_hair,  'curr_step':x,  'max_step':n_steps}), np.arange(n_steps)))
            return filenames
        else:
            if self.net is None:
                print('Model was not loaded. Loading default model')
                self.load_network()
            else:
                self.net.to(self.opts.device)
                if alpha_blend:
                    inner_face1, hair1, mask1 = utils.load_and_split_image(im1_path, return_mask=True)
                    inner_face2, hair2, mask2 = utils.load_and_split_image(im2_path, return_mask=True)
                else:
                    inner_face1, hair1 = utils.load_and_split_image(im1_path)
                    inner_face2, hair2 = utils.load_and_split_image(im2_path)
                im1_orig = inner_face1 + hair1
                im2_orig = inner_face2 + hair2
                inner_face1_t = utils.numpy_uint8_to_torch(inner_face1.astype(np.uint8))
                inner_face2_t = utils.numpy_uint8_to_torch(inner_face2.astype(np.uint8))
                hair1_t = utils.numpy_uint8_to_torch(hair1.astype(np.uint8))
                hair2_t = utils.numpy_uint8_to_torch(hair2.astype(np.uint8))
                if interpolate_hair:
                    input_hair_embeddings = self.__interpolate_from_inputs((self.net),
                      hair1_t, hair2_t, n_interp_steps=n_steps)
                else:
                    input_face_embeddings = self.__interpolate_from_inputs((self.net),
                      inner_face1_t, inner_face2_t, are_faces=True, n_interp_steps=n_steps)
            paths_blended = []
            paths_nonblended = []
            for n in range(n_steps):
                with torch.no_grad():
                    if interpolate_hair:
                        start_result_batch = self.net.forward_i_v(inner_face1_t,
                          (input_hair_embeddings[n].unsqueeze(0).cuda()), resize=True)
                        end_result_batch = self.net.forward_i_v(inner_face2_t,
                          (input_hair_embeddings[n].unsqueeze(0).cuda()), resize=True)
                    else:
                        start_result_batch = self.net.forward_v_i((input_face_embeddings[n].unsqueeze(0).cuda()),
                          hair1_t, resize=True)
                        end_result_batch = self.net.forward_v_i((input_face_embeddings[n].unsqueeze(0).cuda()),
                          hair2_t, resize=True)
                    start_output_im = utils.torch_to_numpy_uint8(start_result_batch)
                    end_output_im = utils.torch_to_numpy_uint8(end_result_batch)
                    nonblended_name_A = self.__prepare_filename(im1_path, im2_path, False, kwargs={'phase':'interp',  'hair':interpolate_hair,  'curr_step':n,  'max_step':n_steps})
                    nonblended_name_B = self.__prepare_filename(im2_path, im1_path, False, kwargs={'phase':'interp',  'hair':interpolate_hair,  'curr_step':n,  'max_step':n_steps})
                    imageio.imwrite(nonblended_name_A, start_output_im)
                    imageio.imwrite(nonblended_name_B, end_output_im)
                    paths_nonblended.append(nonblended_name_A)
                    if alpha_blend:
                        img_original_A = im1_orig
                        img_original_B = im2_orig
                        im_mask1 = utils.gaussian_filter(mask1, sigma=(self.alpha_blend_sigma))
                        im_mask2 = utils.gaussian_filter(mask2, sigma=(self.alpha_blend_sigma))
                        masked_output_A = utils.alpha_blend_images(start_output_im, img_original_A, im_mask1)
                        masked_output_B = utils.alpha_blend_images(end_output_im, img_original_B, im_mask2)
                        blended_name_A = self.__prepare_filename(im1_path, im2_path, True, kwargs={'phase':'interp',  'hair':interpolate_hair,  'curr_step':n,  'max_step':n_steps})
                        blended_name_B = self.__prepare_filename(im2_path, im1_path, True, kwargs={'phase':'interp',  'hair':interpolate_hair,  'curr_step':n,  'max_step':n_steps})
                        imageio.imwrite(blended_name_A, masked_output_A)
                        imageio.imwrite(blended_name_B, masked_output_B)
                        paths_blended.append(blended_name_A)

            if alpha_blend:
                paths_blended
            return paths_nonblended

    def __manipulate_inputs(self, net, in1, latent_dir, are_faces=False, coeffs=[10]):
        with torch.no_grad():
            e1 = net.get_embedding(in1, are_faces).cpu()
        embeddings = []
        for c in coeffs:
            e = e1 + c * latent_dir
            embeddings.append(e.unsqueeze(0))

        return torch.cat(tuple(embeddings), 0)

    def hair_manipulation_single(self, im1_path, direction_name, coeffs=np.arange(-25, 25.5, 0.5), alpha_blend=True):
        assert direction_name in ('color', 'structure'), 'Direction not recognized'
        if os.path.exists(self.__prepare_filename(im1_path, None, alpha_blend, kwargs={'phase':'manip',  'dir':direction_name,  'strength':coeffs[0]})):
            filenames = list(map(lambda x: self.__prepare_filename(im1_path, None, alpha_blend, kwargs={'phase':'manip',  'dir':direction_name,  'strength':x}), coeffs))
            return filenames
        else:
            if self.net is None:
                print('Model was not loaded. Loading default model')
                self.load_network()
            else:
                self.net.to(self.opts.device)
                if alpha_blend:
                    inner_face1, hair1, mask1 = utils.load_and_split_image(im1_path, return_mask=True)
                else:
                    inner_face1, hair1 = utils.load_and_split_image(im1_path)
            im1_orig = inner_face1 + hair1
            with open('./latent_directions.pkl', 'rb') as (f):
                directions = pkl.load(f)
            latent_dir = directions[direction_name]
            inner_face1_t = utils.numpy_uint8_to_torch(inner_face1.astype(np.uint8))
            hair1_t = utils.numpy_uint8_to_torch(hair1.astype(np.uint8))
            input_hair_embeddings = self.__manipulate_inputs((self.net),
              hair1_t, latent_dir, coeffs=coeffs)
            imagepaths = []
            for ii in range(len(coeffs)):
                with torch.no_grad():
                    manipulated_img = self.net.forward_i_v(inner_face1_t,
                      (input_hair_embeddings[ii].unsqueeze(0).cuda().float()), resize=True)
                    manipulated_img = utils.torch_to_numpy_uint8(manipulated_img)
                    nonblended_name = self.__prepare_filename(im1_path, None, False, kwargs={'phase':'manip',  'dir':direction_name,  'strength':coeffs[ii]})
                    imageio.imwrite(nonblended_name, np.uint8(manipulated_img))
                    if alpha_blend:
                        im_mask = utils.gaussian_filter(mask1, sigma=(self.alpha_blend_sigma))
                        masked_output = utils.alpha_blend_images(manipulated_img, im1_orig, im_mask)
                        blended_name = self.__prepare_filename(im1_path, None, True, kwargs={'phase':'manip',  'dir':direction_name,  'strength':coeffs[ii]})
                        imageio.imwrite(blended_name, np.uint8(masked_output))
                        imagepaths.append(blended_name)
                    else:
                        imagepaths.append(nonblended_name)

            return imagepaths
# okay decompiling __pycache__/hairstyle_transfer_tool.cpython-36.pyc