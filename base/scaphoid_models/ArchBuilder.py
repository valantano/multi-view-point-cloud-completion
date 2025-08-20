
from easydict import EasyDict as edict
from base.scaphoid_utils.logger import print_log

from base.scaphoid_models.ScaphoidModules import *
class ArchConfigBuilder:

    def __init__(self, config: edict, pretrained: bool = False):
        self.config = config
        self.pretrained = pretrained
        self.model_name = config.model.NAME

        self.available_types = ['FE', 'SAM', 'SG', 'PG', 'OP', 'TNet', 'SCF', 'RE', 'ASSIGN', 'RG', 'RERG', 'PE', 
                                'PoseG', 'ALIGNER', 'PoseEst']
        self.available_modes = ['pointattn', 'affil', 'SG-', 'TNet', 'concat-dim-1', 'concat-dim-2', 'affil', 
                                'seeds_to_sparse_pc', 'fps', 'mlp', 'affine', 'assign', 'pose', 'aligner_ssm', 
                                'aligner_anchor', 'static']

        self.arch: str = config.model.arch      # e.g. default_arch_volar
        self.available_archs: dict[list] = config.model.available_archs     # { default_arch_volar: [feature_extractor: {in: volar, out: shape_code_v}, seed_generator_to_sparse: {in: volar, shape_code: shape_code_v, out: sparse_pc}, point_generator: {seeds: sparse_pc, shape_code: shape_code_v, out: points}], ... } 
        self.building_blocks: dict = config.model.available_building_blocks       # [feature_extractor: {type: 'FE', keys: [in, out], mode: 'pointattn'}, ...]

        if self.arch not in self.available_archs:
            raise ValueError(f"Architecture {self.arch} is not available. Available architectures are: \
                             {self.available_archs}")
        
    def get_blocks(self):
        used_arch: list[dict] = self.available_archs[self.arch]       # default_arch_volar: [feature_extractor: {in: volar, out: shape_code_v}, seed_generator_to_sparse: {in: volar, shape_code: shape_code_v, out: sparse_pc}, point_generator: {seeds: sparse_pc, shape_code: shape_code_v, out: points}]
        building_blocks: dict = self.building_blocks

        net_blocks = []
        sg_mode = ''
        for block_config in used_arch:
            
            block_name = list(block_config.keys())[0]         # e.g. feature_extractor
            block_params = block_config[block_name]     # e.g. {in: volar, out: shape_code_v}
            template = building_blocks[block_name]      # e.g. [feature_extractor: {type: 'FE', keys: [in, out], mode: 'pointattn'}]
            type = template['type']
            keys = template['keys']
            mode = template['mode']

            if block_name not in building_blocks:
                raise ValueError(f"Building block {block_config} is not available. Available building blocks are: \
                                 {building_blocks}")
            
            ########################################### Ensure consistency with the template ##########################
            if type not in self.available_types:
                raise ValueError(f"Type of template {type} is not available. Available types are: {self.available_types}")
            # check if block_params only has the keys of the template
            for key in block_params.keys():
                if key not in keys:
                    raise ValueError(f"Key {key} is not available in template {template} but was defined in building \
                                     block: {block_config}. Available keys are: {keys}")
            # check if block_params has all the keys of the template
            for key in keys:
                if key not in block_params:
                    raise ValueError(f"Key {key} is not defined in building block {block_config} but was defined in \
                                     template: {template}. Available keys are: {block_params.keys()}")
            ###########################################################################################################
            block_config = edict()
            for key, value in block_params.items():
                if key == 'in':
                    block_config.input = value
                elif key == 'out':
                    block_config.output = value
                else:
                    block_config[key] = value

            block_config['type'] = type
            block_config['mode'] = mode     # now block_params can contain the following keys: in=input, out=output, type, mode, shape_code, seeds
            
            
            if type == 'FE':
                block = ScaphoidFeatureExtractor(block_config)
                fe_mode = block_config.mode
            elif type == 'SAM':
                block = SeedAttnMatcher(block_config)
            elif type == 'SG':
                block = ScaphoidSeedGenerator(block_config)
                sg_block = block
                sg_mode = block_config.mode
            elif type == 'PG':
                if self.pretrained:
                    steps = [4, 8]
                elif sg_mode == 'SG-':
                    steps = [4, 8]
                elif sg_mode == 'pointattn':
                    steps = [4,8]   # needs to be changed to [4,4] if no pretrained weights were used during training
                else:
                    raise ValueError(f"Point generator mode {sg_mode} is not available. Available modes are: \
                                     {['SG-', 'pointattn']}")
                
                input_dim = 3
                if fe_mode == 'affil' and sg_mode != 'SG-':
                    if sg_block.input == 'volar' or sg_block.input == 'dorsal':
                        input_dim = 3
                    elif sg_block.input == 'xyz_affil':
                        input_dim = 4
                else:
                    input_dim = 3

                if block_config.mode == 'static':
                    steps = [4, 8]

                print_log(f"Building PointGenerator with steps: {steps} and input_dim: {input_dim}")
                print_log(f"Building PointGenerator with mode: {block_config.mode}")

                block = ScaphoidPointGenerator(block_config, steps=steps, input_channels=input_dim)
            elif type == 'TNet':
                block = TNet(block_config)
            elif type == 'OP':
                block = OP(block_config)
            elif type == 'SCF':
                block = ShapeCodeFuser(block_config)
            elif type == 'RE':
                block = ScaphoidRotationExtractor(block_config)
            elif type == 'RG':
                block = ScaphoidRotationGenerator(block_config)
            elif type == 'ASSIGN':
                block = ASSIGN(block_config)
            elif type == 'RERG':
                block = RERGenerator(block_config)
            elif type == 'PE':
                block = ScaphoidPoseExtractor(block_config)
            elif type == 'PoseG':
                block = ScaphoidPoseGenerator(block_config)
            elif type == 'PoseEst':
                block = BothPoseEstimator(block_config)
            elif type == 'ALIGNER':
                block = Aligner(block_config)
            else:
                raise ValueError(f"Building block {block_config} is not available. Available building blocks are: \
                                 {building_blocks}")

            net_blocks.append(block)
        
        self.ensure_consistency(net_blocks)

        return net_blocks


    def ensure_consistency(self, net_blocks):
        """
        Checks if the used inputs of each block have been made available before the block is used.
        @param net_blocks: list of blocks e.g. [ScaphoidFeatureExtractor, ScaphoidSeedGenerator, ScaphoidPointGenerator]
        """
        in_out = ['volar', 'dorsal', 'pre_points_v', 'pre_points_d']

        def check_available(name, block, kind="Input"):
            if name not in in_out:
                raise ValueError(f"{block}: {kind} '{name}' not available yet. Currently available: {in_out}")

        def check_not_set(name, block):
            if name in in_out:
                raise ValueError(f"{block}: Output '{name}' already set. Currently available: {in_out}")

        for block in net_blocks:
            t = block.type

            if t == 'OP':  # keys: input (list), output
                for inp in block.input:
                    check_available(inp, block, "Input")
                check_not_set(block.output, block)

            elif t == 'FE':  # keys: input, output
                check_available(block.input, block, "Input")
                check_not_set(block.output, block)

            elif t == 'SAM':  # keys: input, seeds, output
                check_available(block.input, block, "Input")
                check_available(block.seeds, block, "Seeds")
                check_not_set(block.output, block)

            elif t == 'SG':  # keys: input, shape_code, output
                if getattr(block, "mode", None) == "pointattn":
                    check_available(block.input, block, "Input")
                check_available(block.shape_code, block, "Shape code")
                check_not_set(block.output, block)

            elif t == 'PG':  # keys: seeds, shape_code, output
                check_available(block.seeds, block, "Seeds")
                check_available(block.shape_code, block, "Shape code")
                check_not_set(block.output, block)

            elif t == 'TNet':
                check_available(block.input, block, "Input")
                check_not_set(block.output, block)

            elif t == 'ASSIGN':
                # Might introduce outputs later? If so, handle here.
                continue

            elif t == 'RERG':
                check_available(block.input, block, "Input")
                check_not_set(block.output, block)

            elif t == 'ALIGNER':  # keys: src, tgt, src_pose, tgt_pose, output
                check_available(block.src, block, "Source")
                check_available(block.tgt, block, "Target")
                check_available(block.src_pose, block, "Source pose")
                check_available(block.tgt_pose, block, "Target pose")
                check_not_set(block.output, block)

                # Special case: introduce aligned outputs
                in_out.extend(['volar_aligned', 'dorsal_aligned'])
                continue

            elif t == 'PoseEst':
                # No consistency check needed
                continue

            # Finally, register the block output
            in_out.append(block.output)
