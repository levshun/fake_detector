[![ru](https://img.shields.io/badge/lang-ru-red.svg)](https://github.com/levshun/fake_detector/blob/master/README.md)

<h1 align="center">FAKE DETECTOR LIBRARY</h1>

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

<h3 align="left">Applied Use Cases</h3>
<p align="left">

The developed library can be applied in the following areas:

1. <b>Forensic and court practice</b> — initial analysis of digital images to detect signs of synthetic generation or modification.
2. <b>Law enforcement</b> — analysis of cases related to fake visual content, information influence incidents, and checking the reliability of materials from open sources and operational channels.
3. <b>Corporate reputation protection</b> — verification of images used in public communications, support of reputation incident investigations, and reducing the risks of publishing or spreading unreliable visual materials.
4. <b>Content moderation in social networks</b> — a supporting tool for platforms with user-generated content, for example to highlight suspicious items for review by moderators or experts.
5. <b>Education</b> — courses, labs, and practical trainings in digital forensics, digital traces analysis, and trusted/safe use of AI.
6. <b>Research</b> — using the library and datasets as an open base for reproducible experiments, algorithm comparison, and preparation of methodological materials.
</p>

<h3 align="left">Minimum Technical Requirements</h3>
<p align="left">

Recommended minimum compute platform: CPU - a modern multi-core processor with at least 6-8 physical cores (Intel Core i5/i7 or AMD Ryzen 5/7 of a similar generation), or Apple Silicon at Apple M4 level or newer; for x86 platforms, CPU frequency is typically around 3.0 GHz or higher. RAM - at least 16 GB. GPU - optional.

To meet the target performance (processing time for one image for one fake type - no more than 5 seconds), we recommend a platform not lower than: Intel Core i9-12900KF (3.20 GHz) or equivalent, RAM 32 GB (DDR5), NVIDIA GeForce RTX 3090 Ti (24 GB VRAM recommended). On weaker systems (lower-class CPU, less RAM, no GPU, or a GPU with smaller VRAM), processing time may increase and in some modes can exceed 5 seconds.

Supported operating systems: Windows 11, Debian 12, and macOS 26.
</p>

<h3 align="left">Installation</h3>
<p align="left">

This library has been tested for Python programming language version 3.11.

Installation process:

Step 1: Create a project in Python IDE by copying it from VCS using the link 
[```https://github.com/levshun/fake_detector/```](https://github.com/levshun/fake_detector/). 
You can also do this from the console by running the command:

```git clone https://github.com/levshun/fake_detector/```

The file and folder structure should look like this:

```
│   .gitignore
│   LICENSE
│   README.md
│   requirements.txt
├───detect_ai
├───interface
├───tests
│   │   generating.py
│   │   modifying.py
│   │   swapping.py
│   │   swapping_performance.py
│   ├───generating_data
│   ├───swapping_data
└───tutorials
```

Step 2: Set up the Python IDE interpreter using Python 3.11 as the base. 
Use the ```.venv``` virtual environment to install dependencies.

Step 3. The required versions of third-party libraries (dependencies) are presented in the ```requirements.txt``` file. 
If the libraries were not installed when creating the project, you can do this manually using the following command:

```pip install -r requirements.txt```

Step 4. Install the library using TEST PyPi.

```pip install -i https://test.pypi.org/simple/ detect-ai```

Step 5. Download the archive with pre-trained models from the link: 
[```https://disk.yandex.ru/d/FUpmkBHhr7cacA```](https://disk.yandex.ru/d/FUpmkBHhr7cacA). 
Copy the archive to the project root and unzip it. 
The project root should now have a directory and file structure similar to the following:

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
│   ├───multiclass
│   │   ├───pilgram
│   │   │   └───eff_net_b3
│   │   │           class.json
│   │   │           eff_net_b3.keras
│   │   │
│   │   └───tool
│   │       └───eff_net_b3
│   │               class.json
│   │               eff_net_b3.keras
│   │
│   └───ensembles
│           beautification_detection.json
│
├───midas
│       dpt_large_384.pt
│       hubconf.py
│       └───midas/
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

Alternatively, the models can be downloaded from Hugging Face: [```https://huggingface.co/heidel-von-hymack/models```](https://huggingface.co/heidel-von-hymack/models).

Step 6. Download the archive with the testing data from the link:
[```https://disk.yandex.ru/d/YHuOkp-tSEX_Kg```](https://disk.yandex.ru/d/YHuOkp-tSEX_Kg). 
Copy the archive to the project root and unzip it. 
The project root should now have a directory and file structure similar to the following:

```
datasets
├── generating
├── swapping
    ├── fake
    ├── orig
└── modifying
    ├── bald_gan
    │   ├── modification
    │   └── original
    ├── beauty_gan
    │   ├── modification
    │   └── original
    ├── pilgram
    │   ├── modification
    │   ├── modification_multi
    │   │   ├── blending
    │   │   ├── css
    │   │   └── instagram
    │   └── original
    └── qwen
        ├── modification
        └── original
```

Similarly, data can be downloaded from Hugging Face: [```https://huggingface.co/heidel-von-hymack/datasets```](https://huggingface.co/heidel-von-hymack/datasets).

Step 7: Use the interactive notebooks from the ```tutorials``` catalog to explore the library.
</p>

<h3 align="left">Models</h3>
<p align="left">

The archive with pre-trained models is available at the following link: 
[```https://disk.yandex.ru/d/FUpmkBHhr7cacA```](https://disk.yandex.ru/d/FUpmkBHhr7cacA). 

Alternatively, the models can be downloaded from Hugging Face: [```https://huggingface.co/heidel-von-hymack/models```](https://huggingface.co/heidel-von-hymack/models).
</p>

<h3 align="left">Datasets</h3>
<p align="left">

The archive with test datasets is available at the following link: 
[```https://disk.yandex.ru/d/YHuOkp-tSEX_Kg```](https://disk.yandex.ru/d/YHuOkp-tSEX_Kg).

Similarly, the data can be downloaded from Hugging Face: [```https://huggingface.co/heidel-von-hymack/datasets```](https://huggingface.co/heidel-von-hymack/datasets).
</p>

<h3 align="left">Mirror</h3>
<p align="left">

The mirror of the repository is available at GitLab: 
[```https://gitlab.com/levshun/fake_detector```](https://gitlab.com/levshun/fake_detector).
</p>

<h3 align="left">Contacts</h3>
<p align="left">

You can contact us, using the following email: ```dmitry.levshun@gmail.com```.
</p>

<h3 align="left">Acknowledgement</h3>
<p align="left">

This library was funded by the FASIE project agreement No. 50GYCodeAIS13-D7/94529.
</p>
