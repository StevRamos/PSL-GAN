import os
import math
import sys

import io
import cv2
import torch.nn.init as init
import torch
from pytorch_fid.inception import InceptionV3
from torch.autograd import Variable
from torch.nn.functional import adaptive_avg_pool2d
import wandb
import numpy as np
from matplotlib import pyplot as plt
import glob
from scipy import linalg
from PIL import Image

LCOLOR = "#A7ABB0"
RCOLOR = "#2E477D"

# start points
I_dict  = {
    "24": np.array([0,0,0,0 ,1,2,3,4,5,6,9 ,9 ,10,11,11,12,12,13,14,15,15,15,16,16,16,17,18]), 
    "27": np.array([0,0,0,0,3,4,5,6,7,7 ,7 ,7 ,7 ,8 ,8 ,8 ,8 ,8 ,11,13,15,17,19,21,23,25])
}

# end points
J_dict  = {
    "24": np.array([1,4,9,10,2,3,7,5,6,8,10,23,23,13,23,14,23,15,16,17,19,21,18,20,22,19,20]),
    "27": np.array([1,2,3,4,5,6,7,8,9,11,13,15,17,10,19,21,23,25,12,14,16,18,20,22,24,26]) 
}

#left right
LR_dict = {
    "24": np.array([0,1,0,1 ,0,0,0,1,1,1,1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1], dtype=bool),
    "27": np.array([0,1,0,1,0,1,0,1,0,0 ,0 ,0 ,0 ,1 ,1 ,1 ,1 ,1 ,0 ,0 ,0 ,0 ,1 ,1 ,1 , 1], dtype=bool)
}




def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def init_optimizer(generator, discriminator, lr, b1=0.5, b2=0.999):
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

    return opt_g, opt_d


def save_weights(generator, discriminator, path, use_wandb):
    torch.save(generator.state_dict(), os.path.join(path, 'generator.pth'))
    torch.save(discriminator.state_dict(), os.path.join(path, 'discriminator.pth'))

    if use_wandb:
        wandb.save(os.path.join(path, '*.pth'),
                    base_path='/'.join(path.split('/')[:-2]))   



def trunc(latent, mean_size, truncation):  # Truncation trick on Z
    FloatTensor     = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    t = Variable(FloatTensor(np.random.normal(0, 1, (mean_size, *latent.shape[1:]))))
    m = t.mean(0, keepdim=True)

    for i,_ in enumerate(latent):
        latent[i] = m + truncation*(latent[i] - m)

    return latent


def make_gif(outpath, duration, input_frames, format_frames="jpg"):
    if isinstance(input_frames, (np.ndarray, list, tuple)):
        images = (Image.fromarray(np.transpose(i, (1,2,0))) for i in input_frames)         
    elif isinstance(input_frames, (str)):
        input_path = os.path.join(input_frames, "*." + format_frames)
        images = (Image.open(f) for f in sorted(glob.glob(input_path), key=os.path.getmtime))
    else:
        images = None
        print("This type file is no supported")
        sys.exit(0)

    img = next(images)  # extract first image from iterator
    img.save(fp=outpath, format='GIF', append_images=images,
                save_all=True, duration=duration, loop=0)

    print("Gif has been created ...")
    return outpath


def plot_action(name_labels, path_saved_weights, data_numpy, n_samples, n_samples_plot, version_lm=24, duration=1000):
    I = I_dict[str(version_lm)]
    J = J_dict[str(version_lm)]
    LR = LR_dict[str(version_lm)]

    array_videos = {}

    #fig, ax = plt.subplots()
    
    for i, label in enumerate(name_labels):

        array_video_label = []

        print(f"Sign {label}")
        path_signs_label = os.path.join(path_saved_weights, f"sign_{label}")
        try:
            os.mkdir(path_signs_label)
        except OSError:
            pass

        lim_inf = i * n_samples
        lim_sup = lim_inf + n_samples
        #print(f"{lim_inf=}")
        #print(f"{lim_sup=}")
        fake_samples = data_numpy[lim_inf:lim_sup]
        indexes = np.random.choice(fake_samples.shape[0],
                                    n_samples_plot, 
                                    replace=False)
        fake_samples = fake_samples[indexes]
        #print(fake_samples.shape) 

        left_samples = int((n_samples_plot-1)/2)
        right_samples = (n_samples_plot - 1) - left_samples        
            
        #print(f"{left_samples=}")
        for pos in range(1, 1 + left_samples):
            #print(f"y axis of {pos} sample minus {pos}")
            fake_samples[pos,:,:,0] = fake_samples[pos,:,:,0] - pos*2 

        #print(f"{right_samples=}")
        for pos in range(1 + left_samples, 1 + left_samples + right_samples):
            #print(f"y axis of {pos} sample plus {pos-left_samples}")
            fake_samples[pos,:,:,0] = fake_samples[pos,:,:,0] + (pos - left_samples) * 2 

        
        

        for frame_idx in range(fake_samples.shape[1]):
        
            #plt.cla()
            fig, ax = plt.subplots()
            #plt.axis("off")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_title(f"{n_samples_plot} samples - Sign {label} - Frame: {frame_idx}")

            ax.set_xlim([-1*(2*left_samples + 1), 1*(2*right_samples + 1)])
            ax.set_ylim([-1, 1])

            for data in fake_samples:
                x = data[frame_idx, :, 0]
                y = data[frame_idx, :, 1]

                for index in range( len(I) ):
                    x_plot = [x[I[index]], x[J[index]]]
                    y_plot = [y[I[index]], y[J[index]]]
                    ax.plot(x_plot, y_plot, lw=1, c=LCOLOR if LR[index] else RCOLOR)

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.invert_yaxis()
            #plt.show()
            plt.savefig(os.path.join(path_signs_label,"frame_"+str(frame_idx)+".jpg"))
            
            #plt.close(fig)
            print("The {} frame 2d skeleton......".format(frame_idx))


            frame_fig = get_img_from_fig(fig)
            frame_fig = np.transpose(frame_fig, (2,0,1))
            array_video_label.append(frame_fig)
            plt.cla()
            plt.close(fig)

        array_video_label = np.array(array_video_label)
        
        #make gif
        outpath_gif = os.path.join(path_signs_label, f"sign_{label}.gif")
        make_gif(outpath_gif, duration, array_video_label)
        
        #array_videos[f"samples_{label}"] = path_signs_label 
        array_videos[f"samples_{label}"] = array_video_label

        #save 

    #plt.close(fig)

    return array_videos


def make_video_action(array_videos):

    for label in array_videos.keys():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        name_video = f"{label}.mp4"
        path_video = os.path.join(array_videos[label], name_video)
        
        img_array = []
        frames = os.path.join(array_videos[label], "*.jpg")
        for filename in sorted(glob.glob(frames), key = os.path.getmtime):
            print(filename)
            img = cv2.imread(filename)
            img_array.append(img)

        h, w, _ = img.shape
        print(img.shape)
        out = cv2.VideoWriter(path_video, fourcc, 1, (w, h))

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        array_videos[label] = path_video
        print(path_video)

    return array_videos


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def get_activations(samples, model, device):
    """Calculates the activations of the pool_3 layer for all images.
    Params:
    -- model       : Instance of inception model
    -- device      : Device to run calculations
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()


    #GETTING INCEPTION MAP
    # addinng dimension 3 to get fid
    with torch.no_grad():
        if samples.shape[1]!=3:
            zero_z_coord = torch.zeros(samples.shape[0], 1,
                                        samples.shape[2],
                                        samples.shape[3]).to(device)
            samples_zero_coord = torch.cat((samples,
                                            zero_z_coord),
                                            1)                                        
            pred = model(samples_zero_coord)[0]
        else:
            pred = model(samples)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        #END INCEPTION MAP

    return pred





def compute_statistics_of_samples(act):
    """Calculation of the statistics used by the FID.
    Params:
    -- act       : List of activations got by inception model
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma



def calculate_fid_given_arrays(fake, real):

    m1, s1 = compute_statistics_of_samples(fake)
    m2, s2 = compute_statistics_of_samples(real)
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value



def sample_action(n_samples, n_samples_plot, latent_dim, name_labels, 
                label_encoder, generator, device, mean_size, 
                time, joints, dataset_real, load_weights=False,
                truncation=None, path_saved_weights=None, batch_size=32,
                dims_fid=2048):

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims_fid]
    inceptionv3 = InceptionV3([block_idx]).to(device)
    inceptionv3.eval()

    generator.eval()
    if load_weights:
        generator.load_state_dict(torch.load(os.path.join(path_saved_weights, 'generator.pth'), map_location=torch.device(device)))

    Tensor     = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    name_labels = np.unique(name_labels)
    labels_to_sample = label_encoder.transform(name_labels)

    classes = np.array([i for i in labels_to_sample for _ in range(n_samples)])
    #print(classes)

    n_batches = math.ceil(len(classes)/batch_size)
    
    labels_batch = []
    z = []
    gen_imgs = []

    pred_arr = np.empty((len(classes), dims_fid))
    start_idx = 0

    #Fake samples
    with torch.no_grad():
        for n in range(n_batches):
            lim_inf = n * batch_size
            lim_sup = lim_inf + batch_size
            classes_batch = classes[lim_inf: lim_sup]
            z_batch = Variable(Tensor(np.random.normal(0, 1, (len(classes_batch), latent_dim)))) 
            z_batch = trunc(z_batch, mean_size, truncation) if truncation is not None else z_batch  
            
            labels_batch_b = Variable(LongTensor(classes_batch))

            gen_imgs_batch  = generator(z_batch, labels_batch_b, truncation)
            
            #START INCEPTION MAP
            pred = get_activations(gen_imgs_batch, inceptionv3, device)
            pred_arr[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]
            #END INCEPTION MAP

            gen_imgs_batch   = gen_imgs_batch.data.cpu()
            labels_batch_b = labels_batch_b.data.cpu()
            z_batch = z_batch.cpu()

            z.append(z_batch)
            labels_batch.append(labels_batch_b)
            gen_imgs.append(gen_imgs_batch)

    z = np.concatenate(z)
    labels_batch = np.concatenate(labels_batch)
    gen_imgs = np.concatenate(gen_imgs)


    #Real samples
    params = {
        "batch_size": batch_size,
        "num_workers": 4
    }
    train_dataloader = torch.utils.data.DataLoader(dataset_real, 
                                                    **params, 
                                                    shuffle=False)

    pred_arr_real = np.empty((len(dataset_real), dims_fid))
    start_idx = 0
    for i, (imgs, _, _) in enumerate(train_dataloader):
        real_imgs = Variable(imgs.type(Tensor))
        real_imgs = real_imgs.to(device)
        #START INCEPTION MAP
        pred = get_activations(real_imgs, inceptionv3, device)
        pred_arr_real[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]
        #END INCEPTION MAP
    # end real samples

    data_numpy = np.transpose(gen_imgs[:,:,:time,:joints], (0, 2, 3, 1))
    print(data_numpy.shape)

    # Hip to 0 like other methods
    tmp = []
    for d in data_numpy:
        tmp = d[:,0,:]
        z = torch.zeros((tmp.shape[0], tmp.shape[1] ))
        d[:,0,:] = z

    print(data_numpy.max())
    print(data_numpy.min())

    array_videos = plot_action(name_labels, path_saved_weights, data_numpy, n_samples, n_samples_plot)
    #array_videos = make_video_action(array_videos)
    
    fid = calculate_fid_given_arrays(pred_arr, pred_arr_real)
    
    print(data_numpy.max())
    print(data_numpy.min())

    return data_numpy, array_videos, fid
    

