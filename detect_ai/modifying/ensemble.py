import os

mod_ens_dict = \
    {
        "model_paths": [
            os.path.join(
                '..', 'models', 'modifying', 'binary',
                'makeup_wild', 'eff_net_b3', 'eff_net_b3.keras'
            ),
            os.path.join(
                '..', 'models', 'modifying', 'binary',
                'beauty_gan', 'eff_net_b3', 'eff_net_b3.keras'
            ),
            os.path.join(
                '..', 'models', 'modifying', 'binary',
                'b_lfw', 'effnetv2s', 'effnetv2s.keras'
            ),
        ],
        "class_indices": {
            "0": "no beautification",
            "1": "beautification"
        }
    }
