import os

import io
import cv2
import torch.nn.init as init
import torch
from torch.autograd import Variable
import wandb
import numpy as np
from matplotlib import pyplot as plt
import glob

LCOLOR = "#A7ABB0"
RCOLOR = "#2E477D"

I  = np.array([0,0,0,0 ,1,2,3,4,5,6,9 ,9 ,10,11,11,12,12,13,14,15,15,15,16,16,16,17,18]) # start points
J  = np.array([1,4,9,10,2,3,7,5,6,8,10,23,23,13,23,14,23,15,16,17,19,21,18,20,22,19,20]) # end points
LR = np.array([0,1,0,1 ,0,0,0,1,1,1,1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1], dtype=bool)




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



def plot_action(name_labels, path_saved_weights, data_numpy, n_samples):
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
        #print(fake_samples.shape) 

        left_samples = int((n_samples-1)/2)
        right_samples = (n_samples - 1) - left_samples        
            
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
            ax.set_title(f"Sign {label} - Frame: {frame_idx}")

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

        #array_videos[f"samples_{label}"] = path_signs_label 
        array_videos[f"samples_{label}"] = array_video_label

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

def sample_action(n_samples, latent_dim, name_labels, 
                label_encoder, generator, device, mean_size, 
                time, joints, load_weights=False,
                truncation=None, path_saved_weights=None):

    generator.eval()
    if load_weights:
        generator.load_state_dict(torch.load(os.path.join(path_saved_weights, 'generator.pth'), map_location=torch.device(device)))

    Tensor     = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

    name_labels = np.unique(name_labels)
    labels_to_sample = label_encoder.transform(name_labels)

    classes = np.array([i for i in labels_to_sample for _ in range(n_samples)])
    #print(classes)
    z         = Variable(Tensor(np.random.normal(0, 1, (len(classes), latent_dim)))) 
    z = trunc(z, mean_size, truncation) if truncation is not None else z

    labels_batch = Variable(LongTensor(classes))

    gen_imgs  = generator(z, labels_batch, truncation)
    gen_imgs   = gen_imgs.data.cpu()
    z_s = z.cpu()

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
    #fig, ax = plt.subplots()

    array_videos = plot_action(name_labels, path_saved_weights, data_numpy, n_samples)
    #array_videos = make_video_action(array_videos)

    print(data_numpy.max())
    print(data_numpy.min())

    return data_numpy, array_videos



