# nnUNetTrainerMyArchA.py

import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# Import your architecture — path is relative to wherever you defined it.
# Two reasonable options:
#   (a) define the nn.Module directly in this file (simple architectures)
#   (b) import from a separate file you add somewhere in the repo or your package
from nnunetv2.training.nnUNetTrainer.custom.architectures import TokenUNet, SwinUNETR

__all__ = [
    "nnUNet_NDSnnUNetTrainer_100epochs",
    "nnUNet_NoTokenUNetTrainer_100epochs",
    "nnUNet_8TokenUNetTrainer_100epochs",
    "nnUNet_8AttnTokenUNetTrainer_100epochs",
    "nnUNet_8MLPTokenUNetTrainer_100epochs",
    "nnUNet_32TokenUNetTrainer_100epochs",
    "nnUNet_32AttnTokenUNetTrainer_100epochs",
    "nnUNet_32AttnLongTokenUNetTrainer_100epochs",
    "nnUNet_32MLPTokenUNetTrainer_100epochs",
    "nnUNet_SwinUNETRTrainer_100epochs",
]       



class nnUNet_NDSnnUNetTrainer_100epochs(nnUNetTrainer):
    """
    Plans-defined architecture (no custom build_network_architecture),
    deep supervision disabled, fixed at 100 epochs.

    NDS = No Deep Supervision
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.enable_deep_supervision = False
        self.batch_size = 2

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        Called by nnUNetTrainer.initialize() to toggle DS on the network.
        We always keep it off regardless of what the base class requests.
        """
        self.enable_deep_supervision = False
        if hasattr(self.network, "deep_supervision"):
            self.network.deep_supervision = False

class nnUNet_NoTokenUNetTrainer_100epochs(nnUNetTrainer):
    """
    Trainer for MyArchA.

    What this class is responsible for:
      - Building the network (build_network_architecture)
      - Fixing training hyperparameters for fair comparison
      - Handling deep supervision (disable if your arch returns a single tensor)

    What you do NOT need to override:
      - Data loading (nnUNet handles it)
      - Augmentation pipeline (nnUNet handles it)
      - Loss function (inherited, unless you need a custom one)
      - Optimizer / LR schedule (inherited SGD + poly decay)
      - Checkpointing, logging, validation loop (all inherited)
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Fix these identically across ALL your trainers ---
        #self.initial_lr    = 1e-4       # or whatever you fix for all models
        #self.weight_decay  = 3e-5
        self.num_epochs    = 100

        # Batch size: either fix it here for all architectures (fair comparison)
        # or leave it to the plan (nnUNet default, tuned for standard U-Net).
        # For fair low-resource benchmarking: fix it explicitly.
        self.batch_size = 2

    def build_network_architecture(
        self,
        architecture_class_name,       # ignore — we impose our own class
        arch_init_kwargs,              # ignore — we impose our own kwargs
        arch_init_kwargs_req_import,   # ignore
        num_input_channels,            # READ THIS — comes from the plan
        num_output_channels,           # READ THIS — number of segmentation classes
        enable_deep_supervision,       # READ THIS — tells you what nnUNet expects
    ):

        # If your architecture does not produce multi-scale outputs,
        # you MUST disable deep supervision here.
        # If it does (e.g. you added aux heads), leave it True and return a list.
        if enable_deep_supervision:
            enable_deep_supervision = False
            self.enable_deep_supervision = False

        model = TokenUNet(
            in_channels=4,
            enc_stage_channels=[16,32,64,128],
            kernel_size=3,
            blocks_per_stage=[1,1,1,1],
            num_classes=self.label_manager.num_segmentation_heads,
            n_tokens=0,
            token_dim=None,
            tokenize=False,
            process_tokens=False,
            bias=True
        )

        return model

class nnUNet_8TokenUNetTrainer_100epochs(nnUNetTrainer):
    """
    Trainer for MyArchA.

    What this class is responsible for:
      - Building the network (build_network_architecture)
      - Fixing training hyperparameters for fair comparison
      - Handling deep supervision (disable if your arch returns a single tensor)

    What you do NOT need to override:
      - Data loading (nnUNet handles it)
      - Augmentation pipeline (nnUNet handles it)
      - Loss function (inherited, unless you need a custom one)
      - Optimizer / LR schedule (inherited SGD + poly decay)
      - Checkpointing, logging, validation loop (all inherited)
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Fix these identically across ALL your trainers ---
        #self.initial_lr    = 1e-4       # or whatever you fix for all models
        #self.weight_decay  = 3e-5
        self.num_epochs    = 100

        # Batch size: either fix it here for all architectures (fair comparison)
        # or leave it to the plan (nnUNet default, tuned for standard U-Net).
        # For fair low-resource benchmarking: fix it explicitly.
        self.batch_size = 2

    def build_network_architecture(
        self,
        architecture_class_name,       # ignore — we impose our own class
        arch_init_kwargs,              # ignore — we impose our own kwargs
        arch_init_kwargs_req_import,   # ignore
        num_input_channels,            # READ THIS — comes from the plan
        num_output_channels,           # READ THIS — number of segmentation classes
        enable_deep_supervision,       # READ THIS — tells you what nnUNet expects
    ):

        # If your architecture does not produce multi-scale outputs,
        # you MUST disable deep supervision here.
        # If it does (e.g. you added aux heads), leave it True and return a list.
        if enable_deep_supervision:
            enable_deep_supervision = False
            self.enable_deep_supervision = False

        model = TokenUNet(
            in_channels=4,
            enc_stage_channels=[16,32,64,128],
            kernel_size=3,
            blocks_per_stage=[1,1,1,1],
            num_classes=self.label_manager.num_segmentation_heads,
            n_tokens=8,
            token_dim=None,
            tokenize=True,
            process_tokens=False,
            bias=True
        )

        return model

class nnUNet_8AttnTokenUNetTrainer_100epochs(nnUNetTrainer):
    """
    Trainer for MyArchA.

    What this class is responsible for:
      - Building the network (build_network_architecture)
      - Fixing training hyperparameters for fair comparison
      - Handling deep supervision (disable if your arch returns a single tensor)

    What you do NOT need to override:
      - Data loading (nnUNet handles it)
      - Augmentation pipeline (nnUNet handles it)
      - Loss function (inherited, unless you need a custom one)
      - Optimizer / LR schedule (inherited SGD + poly decay)
      - Checkpointing, logging, validation loop (all inherited)
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Fix these identically across ALL your trainers ---
        #self.initial_lr    = 1e-4       # or whatever you fix for all models
        #self.weight_decay  = 3e-5
        self.num_epochs    = 100

        # Batch size: either fix it here for all architectures (fair comparison)
        # or leave it to the plan (nnUNet default, tuned for standard U-Net).
        # For fair low-resource benchmarking: fix it explicitly.
        self.batch_size = 2

    def build_network_architecture(
        self,
        architecture_class_name,       # ignore — we impose our own class
        arch_init_kwargs,              # ignore — we impose our own kwargs
        arch_init_kwargs_req_import,   # ignore
        num_input_channels,            # READ THIS — comes from the plan
        num_output_channels,           # READ THIS — number of segmentation classes
        enable_deep_supervision,       # READ THIS — tells you what nnUNet expects
    ):

        # If your architecture does not produce multi-scale outputs,
        # you MUST disable deep supervision here.
        # If it does (e.g. you added aux heads), leave it True and return a list.
        if enable_deep_supervision:
            enable_deep_supervision = False
            self.enable_deep_supervision = False

        model = TokenUNet(
            in_channels=4,
            enc_stage_channels=[16,32,64,128],
            kernel_size=3,
            blocks_per_stage=[1,1,1,1],
            num_classes=self.label_manager.num_segmentation_heads,
            n_tokens=8,
            token_dim=None,
            tokenize=True,
            process_tokens=True,
            attention=True,
            bias=True
        )

        return model

class nnUNet_8MLPTokenUNetTrainer_100epochs(nnUNetTrainer):
    """
    Trainer for MyArchA.

    What this class is responsible for:
      - Building the network (build_network_architecture)
      - Fixing training hyperparameters for fair comparison
      - Handling deep supervision (disable if your arch returns a single tensor)

    What you do NOT need to override:
      - Data loading (nnUNet handles it)
      - Augmentation pipeline (nnUNet handles it)
      - Loss function (inherited, unless you need a custom one)
      - Optimizer / LR schedule (inherited SGD + poly decay)
      - Checkpointing, logging, validation loop (all inherited)
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Fix these identically across ALL your trainers ---
        #self.initial_lr    = 1e-4       # or whatever you fix for all models
        #self.weight_decay  = 3e-5
        self.num_epochs    = 100

        # Batch size: either fix it here for all architectures (fair comparison)
        # or leave it to the plan (nnUNet default, tuned for standard U-Net).
        # For fair low-resource benchmarking: fix it explicitly.
        self.batch_size = 2

    def build_network_architecture(
        self,
        architecture_class_name,       # ignore — we impose our own class
        arch_init_kwargs,              # ignore — we impose our own kwargs
        arch_init_kwargs_req_import,   # ignore
        num_input_channels,            # READ THIS — comes from the plan
        num_output_channels,           # READ THIS — number of segmentation classes
        enable_deep_supervision,       # READ THIS — tells you what nnUNet expects
    ):

        # If your architecture does not produce multi-scale outputs,
        # you MUST disable deep supervision here.
        # If it does (e.g. you added aux heads), leave it True and return a list.
        if enable_deep_supervision:
            enable_deep_supervision = False
            self.enable_deep_supervision = False

        model = TokenUNet(
            in_channels=4,
            enc_stage_channels=[16,32,64,128],
            kernel_size=3,
            blocks_per_stage=[1,1,1,1],
            num_classes=self.label_manager.num_segmentation_heads,
            n_tokens=8,
            token_dim=None,
            tokenize=True,
            process_tokens=True,
            attention=False,
            bias=True
        )

        return model

class nnUNet_32TokenUNetTrainer_100epochs(nnUNetTrainer):
    """
    Trainer for MyArchA.

    What this class is responsible for:
      - Building the network (build_network_architecture)
      - Fixing training hyperparameters for fair comparison
      - Handling deep supervision (disable if your arch returns a single tensor)

    What you do NOT need to override:
      - Data loading (nnUNet handles it)
      - Augmentation pipeline (nnUNet handles it)
      - Loss function (inherited, unless you need a custom one)
      - Optimizer / LR schedule (inherited SGD + poly decay)
      - Checkpointing, logging, validation loop (all inherited)
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Fix these identically across ALL your trainers ---
        #self.initial_lr    = 1e-4       # or whatever you fix for all models
        #self.weight_decay  = 3e-5
        self.num_epochs    = 100

        # Batch size: either fix it here for all architectures (fair comparison)
        # or leave it to the plan (nnUNet default, tuned for standard U-Net).
        # For fair low-resource benchmarking: fix it explicitly.
        self.batch_size = 2

    def build_network_architecture(
        self,
        architecture_class_name,       # ignore — we impose our own class
        arch_init_kwargs,              # ignore — we impose our own kwargs
        arch_init_kwargs_req_import,   # ignore
        num_input_channels,            # READ THIS — comes from the plan
        num_output_channels,           # READ THIS — number of segmentation classes
        enable_deep_supervision,       # READ THIS — tells you what nnUNet expects
    ):

        # If your architecture does not produce multi-scale outputs,
        # you MUST disable deep supervision here.
        # If it does (e.g. you added aux heads), leave it True and return a list.
        if enable_deep_supervision:
            enable_deep_supervision = False
            self.enable_deep_supervision = False

        model = TokenUNet(
            in_channels=4,
            enc_stage_channels=[16,32,64,128],
            kernel_size=3,
            blocks_per_stage=[1,1,1,1],
            num_classes=self.label_manager.num_segmentation_heads,
            n_tokens=32,
            token_dim=None,
            tokenize=True,
            process_tokens=False,
            bias=True
        )

        return model

class nnUNet_32AttnTokenUNetTrainer_100epochs(nnUNetTrainer):
    """
    Trainer for MyArchA.

    What this class is responsible for:
      - Building the network (build_network_architecture)
      - Fixing training hyperparameters for fair comparison
      - Handling deep supervision (disable if your arch returns a single tensor)

    What you do NOT need to override:
      - Data loading (nnUNet handles it)
      - Augmentation pipeline (nnUNet handles it)
      - Loss function (inherited, unless you need a custom one)
      - Optimizer / LR schedule (inherited SGD + poly decay)
      - Checkpointing, logging, validation loop (all inherited)
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Fix these identically across ALL your trainers ---
        #self.initial_lr    = 1e-4       # or whatever you fix for all models
        #self.weight_decay  = 3e-5
        self.num_epochs    = 100

        # Batch size: either fix it here for all architectures (fair comparison)
        # or leave it to the plan (nnUNet default, tuned for standard U-Net).
        # For fair low-resource benchmarking: fix it explicitly.
        self.batch_size = 2

    def build_network_architecture(
        self,
        architecture_class_name,       # ignore — we impose our own class
        arch_init_kwargs,              # ignore — we impose our own kwargs
        arch_init_kwargs_req_import,   # ignore
        num_input_channels,            # READ THIS — comes from the plan
        num_output_channels,           # READ THIS — number of segmentation classes
        enable_deep_supervision,       # READ THIS — tells you what nnUNet expects
    ):

        # If your architecture does not produce multi-scale outputs,
        # you MUST disable deep supervision here.
        # If it does (e.g. you added aux heads), leave it True and return a list.
        if enable_deep_supervision:
            enable_deep_supervision = False
            self.enable_deep_supervision = False

        model = TokenUNet(
            in_channels=4,
            enc_stage_channels=[16,32,64,128],
            kernel_size=3,
            blocks_per_stage=[1,1,1,1],
            num_classes=self.label_manager.num_segmentation_heads,
            n_tokens=32,
            token_dim=None,
            tokenize=True,
            process_tokens=True,
            attention=True,
            bias=True
        )

        return model

class nnUNet_32AttnLongTokenUNetTrainer_100epochs(nnUNetTrainer):
    """
    Trainer for MyArchA.

    What this class is responsible for:
      - Building the network (build_network_architecture)
      - Fixing training hyperparameters for fair comparison
      - Handling deep supervision (disable if your arch returns a single tensor)

    What you do NOT need to override:
      - Data loading (nnUNet handles it)
      - Augmentation pipeline (nnUNet handles it)
      - Loss function (inherited, unless you need a custom one)
      - Optimizer / LR schedule (inherited SGD + poly decay)
      - Checkpointing, logging, validation loop (all inherited)
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Fix these identically across ALL your trainers ---
        #self.initial_lr    = 1e-4       # or whatever you fix for all models
        #self.weight_decay  = 3e-5
        self.num_epochs    = 100

        # Batch size: either fix it here for all architectures (fair comparison)
        # or leave it to the plan (nnUNet default, tuned for standard U-Net).
        # For fair low-resource benchmarking: fix it explicitly.
        self.batch_size = 2

    def build_network_architecture(
        self,
        architecture_class_name,       # ignore — we impose our own class
        arch_init_kwargs,              # ignore — we impose our own kwargs
        arch_init_kwargs_req_import,   # ignore
        num_input_channels,            # READ THIS — comes from the plan
        num_output_channels,           # READ THIS — number of segmentation classes
        enable_deep_supervision,       # READ THIS — tells you what nnUNet expects
    ):

        # If your architecture does not produce multi-scale outputs,
        # you MUST disable deep supervision here.
        # If it does (e.g. you added aux heads), leave it True and return a list.
        if enable_deep_supervision:
            enable_deep_supervision = False
            self.enable_deep_supervision = False

        model = TokenUNet(
            in_channels=4,
            enc_stage_channels=[16,32,64,128],
            kernel_size=3,
            blocks_per_stage=[1,1,1,1],
            num_classes=self.label_manager.num_segmentation_heads,
            n_tokens=32,
            token_dim=None,
            tokenize=True,
            process_tokens=True,
            token_blocks=8,
            attention=True,
            bias=True
        )

        return model

class nnUNet_32MLPTokenUNetTrainer_100epochs(nnUNetTrainer):
    """
    Trainer for MyArchA.

    What this class is responsible for:
      - Building the network (build_network_architecture)
      - Fixing training hyperparameters for fair comparison
      - Handling deep supervision (disable if your arch returns a single tensor)

    What you do NOT need to override:
      - Data loading (nnUNet handles it)
      - Augmentation pipeline (nnUNet handles it)
      - Loss function (inherited, unless you need a custom one)
      - Optimizer / LR schedule (inherited SGD + poly decay)
      - Checkpointing, logging, validation loop (all inherited)
    """

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Fix these identically across ALL your trainers ---
        #self.initial_lr    = 1e-4       # or whatever you fix for all models
        #self.weight_decay  = 3e-5
        self.num_epochs    = 100

        # Batch size: either fix it here for all architectures (fair comparison)
        # or leave it to the plan (nnUNet default, tuned for standard U-Net).
        # For fair low-resource benchmarking: fix it explicitly.
        self.batch_size = 2

    def build_network_architecture(
        self,
        architecture_class_name,       # ignore — we impose our own class
        arch_init_kwargs,              # ignore — we impose our own kwargs
        arch_init_kwargs_req_import,   # ignore
        num_input_channels,            # READ THIS — comes from the plan
        num_output_channels,           # READ THIS — number of segmentation classes
        enable_deep_supervision,       # READ THIS — tells you what nnUNet expects
    ):

        # If your architecture does not produce multi-scale outputs,
        # you MUST disable deep supervision here.
        # If it does (e.g. you added aux heads), leave it True and return a list.
        if enable_deep_supervision:
            enable_deep_supervision = False
            self.enable_deep_supervision = False

        model = TokenUNet(
            in_channels=4,
            enc_stage_channels=[16,32,64,128],
            kernel_size=3,
            blocks_per_stage=[1,1,1,1],
            num_classes=self.label_manager.num_segmentation_heads,
            n_tokens=32,
            token_dim=None,
            tokenize=True,
            process_tokens=True,
            attention=False,
            bias=True
        )

        return model

class nnUNet_SwinUNETRTrainer_100epochs(nnUNetTrainer):
    """..."""

    def __init__(self, plans, configuration, fold, dataset_json, device=torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.batch_size = 1

    def set_deep_supervision_enabled(self, enabled: bool):
        # SwinUNETR does not have a decoder.deep_supervision attribute
        # (it uses decoder1, decoder2, etc.) and does not support
        # deep supervision — safely skip.
        self.enable_deep_supervision = False

    def validation_step(self, batch: dict):
        with torch.no_grad():
            return super().validation_step(batch)

    def build_network_architecture(
        self,
        architecture_class_name,
        arch_init_kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        enable_deep_supervision,
    ):
        enable_deep_supervision = False
        self.enable_deep_supervision = False

        model = SwinUNETR(
            img_size=(128, 128, 128),
            in_channels=4,
            out_channels=self.label_manager.num_segmentation_heads,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            feature_size=24,
            norm_name='instance',
            use_checkpoint=True
        )
        return model
