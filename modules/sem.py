import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
# os.chdir(os.path.abspath(".."))
from modules.module import Module_Class
import config
try:
    import torch
    import torchvision.models.segmentation
    import torchvision.transforms as tf
    print("Module exists.")
except ImportError:
    print("Module does not exist.")

class SEM(Module_Class):
    def __init__(self,module=config.MODULES):
        super().__init__(module)
        self.device = None
        self.model = None
        self.IMGSIZE = 800
        self.transformImg = tf.Compose([tf.ToPILImage(), 
                               tf.Resize((self.IMGSIZE, self.IMGSIZE)), 
                               tf.ToTensor(),
                               tf.Normalize((0.485, 0.456, 0.406),
                                            (0.229, 0.224, 0.225))])
        self.label_color = {"void":[0,0,0],
                             "cross":[255,255,255],
                             "tsv":[255, 255,0],
                             "cross_staircase":[255, 0, 255]
                             }
        self.load_model()

    def load_model(self):

        if not os.path.exists(self.paths["model_path"]):
            raise FileNotFoundError(f"""Cannot find the file {self.paths["model_path"]}""")
        
        modelPath = self.paths["model_path"]
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
        self.model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)  
        self.model.classifier[4] = torch.nn.Conv2d(256, 4, kernel_size=(1, 1), stride=(1, 1)) 
        self.model = self.model.to(self.device)  # Set net to GPU or CPU
        self.model.load_state_dict(torch.load(modelPath,map_location=torch.device('cpu'))) # Load trained model
        self.model.eval() # Set to evaluation mode
        self.LOGGER.info(f"model is loaded.")

    def prediction(self,imagePath):
        
        img = cv2.imread(imagePath) # load test image
        height_orgin , widh_orgin ,d = img.shape # Get image original size 
        Img = self.transformImg(img)  # Transform to pytorch
        Img = torch.autograd.Variable(Img, requires_grad=False).to(self.device).unsqueeze(0)
        with torch.no_grad():
            Prd = self.model(Img)['out']  # Run net
        # resize to orginal size
        Prd = tf.Resize((height_orgin,widh_orgin))(Prd[0])
        #Convert probability to class map
        seg = torch.argmax(Prd, 0).cpu().detach().numpy()
        
        
        class_name = list(self.label_color.keys())[np.max(seg)]
        
        self.LOGGER.info(f"The pattern is: {class_name}")
        
        seg = SEM.post_process_mask(seg)
    
        fig, patterns = SEM.draw_contour(img,seg)

        self.LOGGER.info(f"The CDs are: {str(patterns)}")
        
        return fig, patterns,class_name

    @staticmethod
    def post_process_mask(seg, min_size = 150, median_thr = 0.8):
        seg = seg.astype('uint8') 
        nb_blobs, im_with_separated_blobs, stats, _ = cv2.connectedComponentsWithStats(seg)
        # stats (and the silenced output centroids) gives some information about the blobs. See the docs for more information. 
        # here, we're interested only in the size of the blobs, contained in the last column of stats.
        sizes = stats[:, -1]
        # the following lines result in taking out the background which is also considered a component, which I find for most applications to not be the expected output.
        # you may also keep the results as they are by commenting out the following lines. You'll have to update the ranges in the for loop below. 
        sizes = sizes[1:]
        nb_blobs -= 1
        # minimum size of particles we want to keep (number of pixels).
        # here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever.
        # output image with only the kept components
        im_result = np.zeros_like(im_with_separated_blobs)
        # for every component in the image, keep it only if it's above min_size
        all_size,filtered_blobs = [],[]
        for blob in range(nb_blobs):
            if sizes[blob] >= min_size:
                all_size.append(sizes[blob])
                filtered_blobs.append(blob)
                # see description of im_with_separated_blobs above
        size_median = np.median(all_size)
        for size,blob in zip(all_size,filtered_blobs):
            if size > median_thr*size_median:
                im_result[im_with_separated_blobs == blob + 1] = 255
        return im_result
    
    @staticmethod
    def draw_contour(img,mask,getcds=True):
        mask = cv2.convertScaleAbs(mask)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # for c in cnts:
        #     cv2.drawContours(img, [c], -1, (0, 0, 12), thickness=1)
        fig, ax = plt.subplots() 
        ax.imshow(img)
        ax.axis('off')
        patterns = []
        if not getcds:
            return fig,patterns

        for i,cnt in enumerate(cnts):
            ax.plot(cnt[:,:,0],cnt[:,:,1],'-',color='orange')
            M = cv2.moments(cnt)
            centroid = [int(M["m10"] / M["m00"]) , int(M["m01"] / M["m00"])]
            edges_dic= SEM.calculate_edges(cnt,centroid)
            cds = []
            for key, edges in edges_dic.items():
            
                X =np.linspace(edges[0],edges[1]).astype(int) 
                Y = np.ones_like(X)*edges[2]
                ax.plot(X,Y,color='red')
                cds.append(abs(edges[0]-edges[1]))
            patterns.append(cds)
        patterns = np.array(patterns)

        return fig,patterns

   
    @staticmethod
    def calculate_edges(cnt,centroid,thrs = 0.3):
        edges = {}
        centroid_x = centroid[0]
        centroid_y = centroid[1]
        left_bool = cnt[:,:,0] < centroid_x
        left_side = cnt[:,:,0][left_bool]
        right_bool = cnt[:,:,0] > centroid_x
        right_side = cnt[:,:,0][right_bool]
        
        y_left = cnt[:,:,1][left_bool]
        y_right = cnt[:,:,1][right_bool]
        
        threshold_top_left = np.min(y_left)+abs(np.min(y_left)-centroid_y)*thrs
        index_top_left = np.argmin(abs(y_left-threshold_top_left))

        threshold_top_right = np.min(y_right)+abs(np.min(y_right)-centroid_y)*thrs
        index_top_right = np.argmin(abs(y_right-threshold_top_right))
        
        
        threshold_bottom_left = np.max(y_left)-abs(np.max(y_left)-centroid_y)*thrs
        index_bottom_left = np.argmin(abs(y_left-threshold_bottom_left))

        threshold_bottom_right = np.max(y_right)-abs(np.max(y_right)-centroid_y)*thrs
        index_bottom_right = np.argmin(abs(y_right-threshold_bottom_right))

        
        edge_mid_left = left_side[np.argmin(abs(y_left-centroid_y))]
        edge_top_left = left_side[index_top_left]
        edge_bott_left = left_side[index_bottom_left]
        
        edge_mid_right = right_side[np.argmin(abs(y_right-centroid_y))]
        edge_top_right = right_side[index_top_right]
        edge_bott_right = right_side[index_bottom_right]
        
        edges["bottom"] = [edge_bott_left,edge_bott_right,y_left[index_bottom_left]]
        edges["mid"] = [edge_mid_left,edge_mid_right,centroid_y]
        edges["top"] = [edge_top_left,edge_top_right,y_left[index_top_left]]
        
        
        return edges



if __name__ == "__main__":
    sem = SEM()
    data_images =  [x for x in os.listdir(sem.paths['image_dir']) if x.endswith('.png') 
              or x.endswith('.jpg') ]
    idx=np.random.randint(0,len(data_images)) 
    im_path = os.path.join(sem.paths['image_dir'],data_images[idx])
    fig, patterns = sem.prediction(im_path)
