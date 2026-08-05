from ast import literal_eval


TEST_SINGLE = [
    'cartpole-swingup',
    ]

TEST_MULTI = [
    'cartpole-swingup',
    'cartpole-swingup',
    ]

TEST_MULTI_VARYING = [
    'cheetah-run',
    'cartpole-swingup',
    ]

DMC_DOGS = [
    'dog-stand', 
    'dog-walk', 
    'dog-trot', 
    'dog-run'
    ]

DMC_HUMANOIDS = [
    'humanoid-stand', 
    'humanoid-walk', 
    'humanoid-run'
    ]

METAWORLD_ALL = [
    'assembly-v2-goal-observable',
    'basketball-v2-goal-observable',
    'bin-picking-v2-goal-observable',
    'box-close-v2-goal-observable',
    'button-press-topdown-v2-goal-observable',
    'button-press-topdown-wall-v2-goal-observable',
    'button-press-v2-goal-observable',
    'button-press-wall-v2-goal-observable',
    'coffee-button-v2-goal-observable',
    'coffee-pull-v2-goal-observable',
    'coffee-push-v2-goal-observable',
    'dial-turn-v2-goal-observable',
    'disassemble-v2-goal-observable',
    'door-close-v2-goal-observable',
    'door-lock-v2-goal-observable',
    'door-open-v2-goal-observable',
    'door-unlock-v2-goal-observable',
    'hand-insert-v2-goal-observable',
    'drawer-close-v2-goal-observable',
    'drawer-open-v2-goal-observable',
    'faucet-open-v2-goal-observable',
    'faucet-close-v2-goal-observable',
    'hammer-v2-goal-observable',
    'handle-press-side-v2-goal-observable',
    'handle-press-v2-goal-observable',
    'handle-pull-side-v2-goal-observable',
    'handle-pull-v2-goal-observable',
    'lever-pull-v2-goal-observable',
    'pick-place-wall-v2-goal-observable',
    'pick-out-of-hole-v2-goal-observable',
    'pick-place-v2-goal-observable',
    'plate-slide-v2-goal-observable',
    'plate-slide-side-v2-goal-observable',
    'plate-slide-back-v2-goal-observable',
    'plate-slide-back-side-v2-goal-observable',
    'peg-insert-side-v2-goal-observable',
    'peg-unplug-side-v2-goal-observable',
    'soccer-v2-goal-observable',
    'stick-push-v2-goal-observable',
    'stick-pull-v2-goal-observable',
    'push-v2-goal-observable',
    'push-wall-v2-goal-observable',
    'push-back-v2-goal-observable',
    'reach-v2-goal-observable',
    'reach-wall-v2-goal-observable',
    'shelf-place-v2-goal-observable',
    'sweep-into-v2-goal-observable',
    'sweep-v2-goal-observable',
    'window-open-v2-goal-observable',
    'window-close-v2-goal-observable'
    ]

METAWORLD_DMC = [
    'walker-stand', 
    'walker-walk', 
    'walker-run', 
    'cheetah-run', 
    'reacher-easy',
	'reacher-hard', 
    'acrobot-swingup', 
    'pendulum-swingup', 
    'cartpole-balance', 
    'cartpole-balance_sparse',
    'cartpole-swingup', 
    'cartpole-swingup_sparse', 
    'ball_in_cup-catch', 
    'finger-spin', 
    'finger-turn_easy',
	'finger-turn_hard', 
    'fish-swim', 
    'hopper-stand', 
    'hopper-hop',
	'cheetah-run_backwards', 
    'cheetah-run_front', 
    'cheetah-run_back',
	'cheetah-jump', 
    'walker-walk_backwards', 
    'walker-run_backwards', 
    'hopper-hop_backwards', 
    'reacher-three_easy', 
    'reacher-three_hard', 
    'ball_in_cup-spin',
	'pendulum-spin', 
    'assembly-v2-goal-observable',
    'basketball-v2-goal-observable',
    'bin-picking-v2-goal-observable',
    'box-close-v2-goal-observable',
    'button-press-topdown-v2-goal-observable',
    'button-press-topdown-wall-v2-goal-observable',
    'button-press-v2-goal-observable',
    'button-press-wall-v2-goal-observable',
    'coffee-button-v2-goal-observable',
    'coffee-pull-v2-goal-observable',
    'coffee-push-v2-goal-observable',
    'dial-turn-v2-goal-observable',
    'disassemble-v2-goal-observable',
    'door-close-v2-goal-observable',
    'door-lock-v2-goal-observable',
    'door-open-v2-goal-observable',
    'door-unlock-v2-goal-observable',
    'hand-insert-v2-goal-observable',
    'drawer-close-v2-goal-observable',
    'drawer-open-v2-goal-observable',
    'faucet-open-v2-goal-observable',
    'faucet-close-v2-goal-observable',
    'hammer-v2-goal-observable',
    'handle-press-side-v2-goal-observable',
    'handle-press-v2-goal-observable',
    'handle-pull-side-v2-goal-observable',
    'handle-pull-v2-goal-observable',
    'lever-pull-v2-goal-observable',
    'pick-place-wall-v2-goal-observable',
    'pick-out-of-hole-v2-goal-observable',
    'pick-place-v2-goal-observable',
    'plate-slide-v2-goal-observable',
    'plate-slide-side-v2-goal-observable',
    'plate-slide-back-v2-goal-observable',
    'plate-slide-back-side-v2-goal-observable',
    'peg-insert-side-v2-goal-observable',
    'peg-unplug-side-v2-goal-observable',
    'soccer-v2-goal-observable',
    'stick-push-v2-goal-observable',
    'stick-pull-v2-goal-observable',
    'push-v2-goal-observable',
    'push-wall-v2-goal-observable',
    'push-back-v2-goal-observable',
    'reach-v2-goal-observable',
    'reach-wall-v2-goal-observable',
    'shelf-place-v2-goal-observable',
    'sweep-into-v2-goal-observable',
    'sweep-v2-goal-observable',
    'window-open-v2-goal-observable',
    'window-close-v2-goal-observable'
    ]

HB_NOHANDS = [   
    'h1-walk-v0', 
    'h1-stand-v0',
    'h1-run-v0', 
    'h1-stair-v0',
    'h1-crawl-v0',
    'h1-pole-v0', 
    'h1-slide-v0', 
    'h1-hurdle-v0', 
    'h1-maze-v0'
    ]

HB_BASIC = [
    'h1-walk-v0',
    'h1-stand-v0',
    'h1-pole-v0',
    ]

HB_EASY = [
    'h1-walk-v0',
    'h1-stand-v0',
    'h1-run-v0',
    'h1-pole-v0',
    'h1-slide-v0',
    'h1-maze-v0'
    ]

HB_HANDS = [   
    'h1hand-walk-v0', 
    'h1hand-stand-v0',
    'h1hand-run-v0', 
    'h1hand-stair-v0', 
    'h1hand-crawl-v0', 
    'h1hand-pole-v0', 
    'h1hand-slide-v0', 
    'h1hand-hurdle-v0', 
    'h1hand-maze-v0', 
    'h1hand-sit_simple-v0',
    'h1hand-sit_hard-v0', 
    'h1hand-balance_simple-v0', 
    'h1hand-balance_hard-v0', 
    'h1hand-reach-v0', 
    'h1hand-spoon-v0', 
    'h1hand-window-v0', 
    'h1hand-insert_small-v0', 
    'h1hand-insert_normal-v0',
    'h1hand-bookshelf_simple-v0', 
    'h1hand-bookshelf_hard-v0', 
    ]

HB_HANDS_EASY = [
    'h1hand-insert_normal-v0',
    'h1hand-room-v0',
    'h1hand-door-v0',
    'h1hand-push-v0',
]

HB_BOOKSHELF = [
    'h1hand-bookshelf_simple-v0',
    'h1hand-bookshelf_hard-v0',
]
HB_INSERT = [
    'h1hand-insert_small-v0',
    'h1hand-insert_normal-v0',
]

SHADOWHAND_TRAIN = [
    'e_toy_airplane', 
    'knife', 
    'flat_screwdriver', 
    'elephant', 
    'apple', 
    'scissors', 
    'i_cups', 
    'cup', 
    'foam_brick', 
    'pudding_box', 
    'wristwatch', 
    'padlock', 
    'power_drill', 
    'binoculars', 
    'b_lego_duplo', 
    'ps_controller', 
    'mouse', 
    'hammer', 
    'f_lego_duplo', 
    'piggy_bank', 
    'can', 
    'extra_large_clamp', 
    'peach', 
    'a_lego_duplo', 
    'racquetball', 
    'tuna_fish_can', 
    'a_cups', 
    'pan', 
    'strawberry', 
    'd_toy_airplane', 
    'wood_block', 
    'small_marker', 
    'sugar_box', 
    'ball', 
    'torus', 
    'i_toy_airplane', 
    'chain', 
    'j_cups', 
    'c_toy_airplane', 
    'airplane', 
    'nine_hole_peg_test', 
    'water_bottle', 
    'c_cups', 
    'medium_clamp',
    'large_marker', 
    'h_cups', 
    'b_colored_wood_blocks', 
    'j_lego_duplo', 
    'f_toy_airplane', 
    'toothbrush', 
    'tennis_ball', 
    'mug', 
    'sponge', 
    'k_lego_duplo', 
    'phillips_screwdriver', 
    'f_cups', 
    'c_lego_duplo', 
    'd_marbles', 
    'd_cups', 
    'camera', 
    'd_lego_duplo', 
    'golf_ball', 
    'k_toy_airplane', 
    'b_cups', 
    'softball', 
    'wine_glass', 
    'chips_can', 
    'cube', 
    'master_chef_can', 
    'alarm_clock', 
    'gelatin_box', 
    'h_lego_duplo', 
    'baseball', 
    'light_bulb', 
    'banana', 
    'rubber_duck', 
    'headphones', 
    'i_lego_duplo', 
    'b_toy_airplane', 
    'pitcher_base', 
    'j_toy_airplane', 
    'g_lego_duplo', 
    'cracker_box', 
    'orange', 
    'e_cups'
    ]

SHADOWHAND_TEST = [
    'rubiks_cube', 
    'dice', 
    'bleach_cleanser', 
    'pear', 
    'e_lego_duplo', 
    'pyramid', 
    'stapler', 
    'flashlight', 
    'large_clamp', 
    'a_toy_airplane', 
    'tomato_soup_can', 
    'fork', 
    'cell_phone', 
    'm_lego_duplo',
    'toothpaste', 
    'flute', 
    'stanford_bunny', 
    'a_marbles', 
    'potted_meat_can', 
    'timer', 
    'lemon', 
    'utah_teapot', 
    'train', 
    'g_cups', 
    'l_lego_duplo', 
    'bowl', 
    'door_knob', 
    'mustard_bottle', 
    'plum'
    ]

HB_MANI_2=[
    'h1hand-insert_normal-v0',
    'h1hand-bookshelf_simple-v0',
]

HB_MANI_2_bookshelf=[
    'h1hand-bookshelf_simple-v0',
    'h1hand-bookshelf_hard-v0',
]

HB_MANI_2_door_window=[
    'h1hand-door-v0',
    'h1hand-window-v0',
]

HB_MANI_4=[
    'h1hand-insert_small-v0',
    'h1hand-insert_normal-v0',
    'h1hand-bookshelf_simple-v0',
    'h1hand-bookshelf_hard-v0',
]

HB_MANI_6=[
    'h1hand-insert_small-v0',
    'h1hand-insert_normal-v0',
    'h1hand-bookshelf_simple-v0',
    'h1hand-bookshelf_hard-v0',
    'h1hand-spoon-v0',
    'h1hand-push-v0',
]

HB_MANI_8=[
    'h1hand-insert_small-v0',
    'h1hand-insert_normal-v0',
    'h1hand-bookshelf_simple-v0',
    'h1hand-bookshelf_hard-v0',
    'h1hand-spoon-v0',
    'h1hand-push-v0',
    'h1hand-door-v0',
    'h1hand-window-v0',
]

HB_MANI_6_different=[
    'h1hand-insert_normal-v0',
    'h1hand-bookshelf_simple-v0',
    'h1hand-spoon-v0',
    'h1hand-push-v0',
    'h1hand-door-v0',
    'h1hand-window-v0',
]

HB_MANI_4_no_insert_shelf=[
    'h1hand-spoon-v0',
    'h1hand-push-v0',
    'h1hand-door-v0',
    'h1hand-window-v0',
]

HB_MANI_4_single_insert_shelf=[
    'h1hand-spoon-v0',
    'h1hand-door-v0',
    'h1hand-insert_normal-v0',
    'h1hand-bookshelf_simple-v0',
]

HB_MIX_2 = [
    'h1hand-stand-v0',
    'h1hand-insert_normal-v0',
]

HB_MIX_4 = [
    'h1hand-stand-v0',
    'h1hand-walk-v0',
    'h1hand-insert_normal-v0',
    'h1hand-bookshelf_simple-v0',
]

HB_MIX_6 = [
    'h1hand-stand-v0',
    'h1hand-walk-v0',
    'h1hand-run-v0',
    'h1hand-insert_normal-v0',
    'h1hand-bookshelf_simple-v0',
    'h1hand-push-v0',
]

HB_MIX_8 = [
    'h1hand-stand-v0',
    'h1hand-walk-v0',
    'h1hand-run-v0',
    'h1hand-insert_normal-v0',
    'h1hand-bookshelf_simple-v0',
    'h1hand-push-v0',
    'h1hand-room-v0',
    'h1hand-truck-v0',
]

HB_LOCO_2 = [
    'h1hand-stand-v0',
    'h1hand-walk-v0',
]

HB_LOCO_4 = [
    'h1hand-stand-v0',
    'h1hand-walk-v0',
    'h1hand-run-v0',
    'h1hand-crawl-v0',
]

HB_LOCO_6 = [
    'h1hand-stand-v0',
    'h1hand-walk-v0',
    'h1hand-run-v0',
    'h1hand-crawl-v0',
    'h1hand-slide-v0',
    'h1hand-pole-v0',
]

HB_INSERT_SMALL = [
    'h1hand-insert_small-v0', ]

HB_INSERT_NORMAL = [
    'h1hand-insert_normal-v0', ]

HB_TEST = {
    'h1-walk-v0',
    'h1-stand-v0',
}

DMC_WALKER = {
    'dm_control/walker-stand', 'dm_control/walker-walk', 'dm_control/walker-run'
}

DMC_WALKER_DOG = {
    'dm_control/walker-stand', 'dm_control/walker-walk', 'dm_control/walker-run',
    'dm_control/dog-stand', 'dm_control/dog-walk', 'dm_control/dog-run',
}

DMC_LOCO_3 = {
    'dm_control/walker-stand',
    'dm_control/dog-stand',
    'dm_control/humanoid-stand',
}

DMC_LOCO_6 = {
    'dm_control/walker-stand', 'dm_control/walker-walk',
    'dm_control/dog-stand', 'dm_control/dog-walk',
    'dm_control/humanoid-stand', 'dm_control/humanoid-walk',
}

DMC_LOCO_9 = {
    'dm_control/walker-stand', 'dm_control/walker-walk', 'dm_control/walker-run',
    'dm_control/dog-stand', 'dm_control/dog-walk', 'dm_control/dog-run',
    'dm_control/humanoid-stand', 'dm_control/humanoid-walk', 'dm_control/humanoid-run',
}

EnvironmentsDict = {
    'METAWORLD_ALL': METAWORLD_ALL,
    'METAWORLD_DMC': METAWORLD_DMC,
    'DMC_DOGS': DMC_DOGS,
    'DMC_HUMANOIDS': DMC_HUMANOIDS,
    'HB_NOHANDS': HB_NOHANDS,
    'HB_BASIC': HB_BASIC,
    'HB_EASY': HB_EASY,
    'TEST_MULTI_VARYING': TEST_MULTI_VARYING,
    'HB_HANDS': HB_HANDS,
    'HB_HANDS_EASY': HB_HANDS_EASY,
    'HB_BOOKSHELF': HB_BOOKSHELF,
    'HB_INSERT': HB_INSERT,
    'HB_TEST': HB_TEST,
    'HB_MANI_2': HB_MANI_2,
    'HB_MANI_4': HB_MANI_4,
    'HB_MANI_6': HB_MANI_6,
    'HB_MANI_8': HB_MANI_8,
    'HB_MIX_2': HB_MIX_2,
    'HB_MIX_4': HB_MIX_4,
    'HB_MIX_6': HB_MIX_6,
    'HB_MIX_8': HB_MIX_8,
    'HB_LOCO_2': HB_LOCO_2,
    'HB_LOCO_4': HB_LOCO_4,
    'HB_LOCO_6': HB_LOCO_6,
    'DMC_WALKER': DMC_WALKER,
    'DMC_WALKER_DOG': DMC_WALKER_DOG,
    'DMC_LOCO_3': DMC_LOCO_3,
    'DMC_LOCO_6': DMC_LOCO_6,
    'DMC_LOCO_9': DMC_LOCO_9,
    'HB_INSERT_SMALL': HB_INSERT_SMALL,
    'HB_INSERT_NORMAL': HB_INSERT_NORMAL,
    'HB_MANI_6_different': HB_MANI_6_different,
    'HB_MANI_2_bookshelf': HB_MANI_2_bookshelf,
    'HB_MANI_2_door_window': HB_MANI_2_door_window,
    'HB_MANI_4_single_insert_shelf': HB_MANI_4_single_insert_shelf,
    'HB_MANI_4_no_insert_shelf': HB_MANI_4_no_insert_shelf,
    }

try:
    from omegaconf import ListConfig
except ImportError:
    ListConfig = tuple()


def get_environment_list(env_names: str | list):
    if isinstance(env_names, str):
        env_names = env_names.strip()
        if env_names in EnvironmentsDict:
            return EnvironmentsDict[env_names]
        if env_names.startswith("[") and env_names.endswith("]"):
            parsed = literal_eval(env_names)
            if isinstance(parsed, list):
                resolved = []
                for name in parsed:
                    resolved.extend(get_environment_list(str(name).strip()))
                return resolved
        if "," in env_names:
            resolved = []
            for name in env_names.split(","):
                name = name.strip()
                if name:
                    resolved.extend(get_environment_list(name))
            return resolved
        return [env_names]
    elif isinstance(env_names, (list, tuple, ListConfig)):
        resolved = []
        for name in env_names:
            resolved.extend(get_environment_list(str(name).strip()))
        return resolved
    raise TypeError(f"Unsupported env_names type: {type(env_names)!r}")
