# Attention based 3D Point Semantic Segmentation using Graph Neural Networks

## Dataset
- The code is trained on S3DIS dataset for 3D Semantic Segmentation/
- To download the data, run `download_data.sh` and save dataset in `./indoor3d_sem_seg_hdf5_data/`

## Running the code
- For the ease of running the code, we also have a jupyter notebook which can be run easily by uploading it on google colab and performing all computations on the GPU.
    - ```674_project.ipynb```

## Training
- Run ```python train_semseg.py```

## Testing 
- Run ```python test_semseg.py```
- For testing, consider changing the checkpoint value from the checkpoints folder.
    - For example : ```model.load_state_dict(torch.load('./checkpoints/GACNet_000_0.3636.pth', map_location=torch.device('cpu')))```

### Contributions
- Aman Bansal : Contributed to Voxel Sampling (model.py, line 101-122), created test_semseg.py for performing testing (test_semseg.py, line 1-66)
- Ankita Sahoo : Contributed to the encoding of point clouds vertices in model.py using NewPointNetFeaturePropagation (model.py, line 350-366), modification of the GACNet model (model.py, line 382-398, line 400-423)
- Vishnu Sabbavarapu : Contributed to Dilute Furthest Point Sampling (model.py, line 124-158) and changes associated with it in GACNet model (model.py, line 425-429, line 372-374), sample_and_group function (model.py, line 279), GraphAttentionConvLayer (model.py, line 176)
