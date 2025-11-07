# Abstrct

Graph Convolutional Network (GCN) has achieved remarkable result in skeleton-based action recognition. In GCNs, multi-order information has shown notable improvement for recognition and the graph topology, which is the key to fusing and extracting representative features. 

However, the GCN-based methods still face the following problems: 
-  Nodes will have over-smooth problems in deep and complex networks. 
- Lack of efficient methods to fuse data streams of different modalities. 

In our work, we proposed a novel data-fusing method, Feedback Directed Graph Convolution (FD-GC), to dynamically construct diverse correlation matrices and effectively aggregate both joint and bone features in different hierarchical update state and utilize them as feedback loops to participate in aggregation respectively for both streams. Our methods significantly reduce the difficulty of modeling multi-streams features at a small parameter cost. Furthermore, the experimental results indicate FD-GC alleviates the over-smooth effect via the feedback mechanism, constructing stronger representation capabilities of fine-grained actions, and performs as well as most skeletal motion recognition algorithms on two large public datasets NTU RGB+D 60, NTU RGB+D 120 and Northwestern-UCLA.

Highlights:
- We propose a feedback directed graph convolution method FD
GC, which can effectively construct the dynamic correlation be
tween multi-stream data through feedback information, to effec
tively aggregate and update features, and verified the effective
ness of contribution of high-level feedback feature information.
-  We construct a novel feedback directed graph convolutional net
work FD-GCN to verify and visualize the effective representation 
of motion information by directed topology, and reduce the influ
ence of over-smooth on feature aggregation in deep networks at 
a small parameter cost.
- The extensive experimental result highlight the benefits of FD
GC, which performs as well as most skeletal motion recognition 
algorithms on two large public datasets NTU RGB+D 60, NTU 
RGB+D 120 and Northwestern-UCLA.

Access to our work: [FD-GCN: Feedback Directed Graph Convolutional Network for skeleton-based action recognition](https://www.sciencedirect.com/science/article/pii/S1524070325000530)

Theoretical verification in Supplemental Material: ['Oversmooth' effect in Graph Convolutinal Networks](https://ars.els-cdn.com/content/image/1-s2.0-S1524070325000530-mmc1.pdf)

## Details
Overall:
![Overall](\resources\Overall.jpg)

Visaualize:
![Visualize](resources\highlights.jpg)

Results:
![Results](resources\results.jpg)

# Note

~~PyTorch version should be 0.3! For PyTorch0.4 or higher, the codes need to be modified.~~ \
Now we have updated the code to >=Pytorch0.4. \

# Data Preparation

- Download the raw data from [NTU-RGB+D](https://github.com/shahroudy/NTURGB-D) and [Skeleton-Kinetics](https://github.com/yysijie/st-gcn). Then put them under the data directory:

       -data\
         -kinetics_raw\
           -kinetics_train\
             ...
           -kinetics_val\
             ...
           -kinetics_train_label.json
           -keintics_val_label.json
         -nturgbd_raw\
           -nturgb+d_skeletons\
             ...
           -samples_with_missing_skeletons.txt

[https://github.com/shahroudy/NTURGB-D]: NTU-RGB+D
[https://github.com/yysijie/st-gcn]: Skeleton-Kinetics

- Preprocess the data with

  `python data_gen/ntu_gendata.py`

  `python data_gen/kinetics-gendata.py.`

- Generate the bone data with:
  `python data_gen/gen_bone_data.py`

# Training & Testing

Change the config file depending on what you want.

    `python main.py --config ./config/nturgbd-cross-view/train_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-view/train_bone.yaml`

To ensemble the results of joints and bones, run test firstly to generate the scores of the softmax layer.

    `python main.py --config ./config/nturgbd-cross-view/test_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-view/test_bone.yaml`



# Citation

Please cite the following paper if you use this repository in your reseach.

    @article{RAN2025101306,
        title = {FD-GCN: Feedback Directed Graph Convolutional Network for skeleton-based action recognition},
        journal = {Graphical Models},
        volume = {142},
        pages = {101306},
        year = {2025},
        issn = {1524-0703},
        doi = {https://doi.org/10.1016/j.gmod.2025.101306},
        url = {https://www.sciencedirect.com/science/article/pii/S1524070325000530},
        author = {Ruixi Ran and Wenlu Yang},
        keywords = {Skeleton-based action recognition, Graph neural network, Pattern recognition, Deep learning}
        }
    

# Contact

For any questions, feel free to contact: `944295628@qq.com`
