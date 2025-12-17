<h1 align="center">FAKE DETECTOR</h1>

<h3 align="left">Tasks</h3>
<p align="left">

The developed library fulfills 3 main detection tasks:
- **Generating**: detection of generated images containing a person's face.
- **Modifying**: detection of modified images containing a person's face.
- **Swapping**: detection of images containing a person's face with a faceswap.
</p>

<h3 align="left">Documentation</h3>
<p align="left">

Online documentation is available at Read the Docs: https://fake-detector.readthedocs.io.
</p>

<h3 align="left">Installation</h3>
<p align="left">

Our library was tested with Python 3.11.

```pip install -i https://test.pypi.org/simple/ detect-ai```

```pip install -r .\requirements.txt```

The correct versions of the libraries are presented in the ```requirements.txt``` 
file.

To install the library, firstly, download the repository.

Then, create the virtual environment.

After that, use pip to install the libraries from the ```requirements.txt``` file.

The final structure of the project must be as follows:
```
│   .gitignore
│   LICENSE
│   main.py
│   README.md
│   requirements.txt
├───datasets
├───detect_ai
├───interface
├───models
├───tests
│   │   generating.py
│   │   modifying.py
│   │   swapping.py
│   │   swapping_performance.py
│   ├───generating_data
│   ├───swapping_data
└───tutorials
```

Folders ```models``` and ```datasets``` are required to be created manually.

Run the project in the Python IDE like PyCharm and check the ```tutorials``` to start the work.
</p>

<h3 align="left">Models</h3>
<p align="left">

Pre-trained models are available here: https://disk.yandex.ru/d/QVQL-HBabVkUoA.

After their download and unzipping, you should have the following structure of the ```models``` folder: 
```
models
├───generating
│       convnext_model.pth
│       eva_model.pth
│       final_decisiontree.pkl
│
├───modifying
│   │   face_landmarker_v2_with_blendshapes.task
│   │
│   ├───binary
│   │   ├───bald_gan
│   │   │   └───eff_net_b3
│   │   │           class.json
│   │   │           eff_net_b3.keras
│   │   │
│   │   ├───beauty_gan
│   │   │   ├───eff_net_b3
│   │   │   │       class.json
│   │   │   │       eff_net_b3.keras
│   │   │   │
│   │   │   └───inception_v3
│   │   │           class.json
│   │   │           inception_v3.keras
│   │   │
│   │   ├───b_lfw
│   │   │   └───effnetv2s
│   │   │           class.json
│   │   │           effnetv2s.keras
│   │   │
│   │   ├───makeup_wild
│   │   │   └───eff_net_b3
│   │   │           class.json
│   │   │           eff_net_b3.keras
│   │   │
│   │   ├───pilgram
│   │   │   └───eff_net_b3
│   │   │           class.json
│   │   │           eff_net_b3.keras
│   │   │
│   │   └───qwen
│   │       └───eff_net_b3
│   │               class.json
│   │               eff_net_b3.keras
│   │
│   └───multiclass
│       ├───pilgram
│       │   └───eff_net_b3
│       │           class.json
│       │           eff_net_b3.keras
│       │
│       └───tool
│           └───eff_net_b3
│                   class.json
│                   eff_net_b3.keras
│
└───swapping
    │   face_detection_yunet_2023mar.onnx
    │   shape_predictor_68_face_landmarks.dat
    │
    ├───efficientnet
    │       effnet_github.pth
    │       effnet_rgb.pth
    │       effnet_roop.pth
    │       effnet_segmind.pth
    │
    └───feature_based
            catboost_lbp.cbm
            catboost_tf_sf.cbm
            meta_model_ensemble.pkl
            random_forest_ef.pkl
            random_forest_fl.pkl
```
</p>

<h3 align="left">Datasets</h3>
<p align="left">

Datasets for testing are available here: https://disk.yandex.ru/d/9oX2v0c2E4zeFw.
</p>

<h3 align="left">Mirror</h3>
<p align="left">

The mirror of the repository is available at GitLab: https://gitlab.com/levshun/fake_detector.
</p>

<h3 align="left">Contacts</h3>
<p align="left">

You can contact us, using the following email: ```dmitry.levshun@gmail.com```.
</p>

<h3 align="left">Acknowledgement</h3>
<p align="left">

This library was funded by the FASIE project agreement No. 50GYCodeAIS13-D7/94529.
</p>
