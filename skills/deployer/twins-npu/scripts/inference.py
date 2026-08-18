#!/usr/bin/env python3
"""Twins 系列模型 NPU 推理脚本（通用，支持所有 6 个模型）"""
import argparse
import time
import gc
import sys
import os
from pathlib import Path

import torch
import torch_npu
import timm
from timm.data import resolve_model_data_config, create_transform
import numpy as np
from PIL import Image
from io import BytesIO

from ms_loader import load_timm_model


def parse_args():
    parser = argparse.ArgumentParser(description="Twins 模型 NPU 推理")
    parser.add_argument("--model-name", type=str, required=True,
                        help="模型名称, e.g. twins_svt_small.in1k")
    parser.add_argument("--device", type=str, default="npu",
                        choices=["cpu", "npu"], help="推理设备")
    parser.add_argument("--image-url", type=str, default=None,
                        help="测试图片 URL（默认使用 ImageNet 示例图）")
    parser.add_argument("--image-path", type=str, default=None,
                        help="本地测试图片路径")
    parser.add_argument("--num-runs", type=int, default=5,
                        help="推理轮次（用于性能统计）")
    return parser.parse_args()


def load_image(url=None, path=None):
    """加载测试图片，当 URL 不可达时自动生成测试图片"""
    img = None
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB")
        print(f"[INFO] 加载本地图片: {path}")
    elif url:
        try:
            print(f"[INFO] 下载图片: {url}")
            resp = requests.get(url, timeout=15)
            img = Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"[WARN] 下载图片失败: {e}，使用本地生成图片")
    if img is None:
        # 生成一张测试图片（渐变彩色图）
        import numpy as np
        arr = np.zeros((224, 224, 3), dtype=np.uint8)
        for y in range(224):
            for x in range(224):
                arr[y, x, 0] = int(255 * y / 223)       # R 渐变
                arr[y, x, 1] = int(255 * x / 223)       # G 渐变
                arr[y, x, 2] = int(128 + 127 * (x+y) / 446)  # B
        img = Image.fromarray(arr, "RGB")
        print(f"[INFO] 使用本地生成的测试图片")
    print(f"[INFO] 图片大小: {img.size}")
    return img


@torch.no_grad()
def main():
    args = parse_args()
    model_name = args.model_name
    device = args.device

    print(f"{'='*60}")
    print(f"  模型: {model_name}")
    print(f"  设备: {device}")
    print(f"{'='*60}")

    # 1. 加载模型
    print(f"\n[INFO] 加载模型: {model_name}")
    t0 = time.time()
    model = load_timm_model(model_name)
    print(f"[INFO] 模型加载耗时: {time.time()-t0:.2f}s")

    # 2. 数据预处理
    data_config = resolve_model_data_config(model)
    transforms = create_transform(**data_config)
    print(f"[INFO] 数据配置: {data_config}")

    # 3. 加载图片
    img = load_image(args.image_url, args.image_path)
    input_tensor = transforms(img).unsqueeze(0)  # (1, 3, 224, 224)

    # 4. 移到目标设备
    if device == "npu":
        model = model.npu()
        input_tensor = input_tensor.npu()
    print(f"[INFO] 输入张量: {input_tensor.shape}, 设备: {input_tensor.device}")

    # 5. 预热
    print(f"[INFO] 预热推理...")
    _ = model(input_tensor)
    if device == "npu":
        torch.npu.synchronize()
    print(f"[INFO] 预热完成")

    # 6. 推理（多轮）
    print(f"[INFO] 运行 {args.num_runs} 轮推理...")
    latencies = []
    outputs = []
    for i in range(args.num_runs):
        t0 = time.time()
        out = model(input_tensor)
        if device == "npu":
            torch.npu.synchronize()
        t = time.time() - t0
        latencies.append(t)
        outputs.append(out.cpu() if device == "npu" else out)
        print(f"  第 {i+1} 轮: {t*1000:.2f}ms")

    # 7. 输出结果
    final_out = outputs[-1]
    probs = torch.nn.functional.softmax(final_out, dim=1)
    top5_val, top5_idx = torch.topk(probs, 5, dim=1)

    # ImageNet 类别标签（内嵌常用类别名）
    # 使用 raw.githubusercontent.com 不稳定，使用内置标签
    IMAGENET_LABELS = {
        0: 'tench', 1: 'goldfish', 2: 'great_white_shark', 3: 'tiger_shark', 4: 'hammerhead',
        5: 'electric_ray', 6: 'stingray', 7: 'cock', 8: 'hen', 9: 'ostrich',
        10: 'brambling', 11: 'goldfinch', 12: 'house_finch', 13: 'junco', 14: 'indigo_bunting',
        15: 'robin', 16: 'bulbul', 17: 'jay', 18: 'blue_jay', 19: 'magpie',
        20: 'chickadee', 21: 'water_ouzel', 22: 'kite', 23: 'bald_eagle', 24: 'vulture',
        25: 'great_grey_owl', 26: 'European_fire_salamander', 27: 'common_newt', 28: 'eft', 29: 'spotted_salamander',
        30: 'axolotl', 31: 'bullfrog', 32: 'tree_frog', 33: 'tailed_frog', 34: 'loggerhead',
        35: 'leatherback_turtle', 36: 'mud_turtle', 37: 'terrapin', 38: 'box_turtle', 39: 'banded_gecko',
        40: 'common_iguana', 41: 'American_chameleon', 42: 'whiptail', 43: 'agama', 44: 'frilled_lizard',
        45: 'alligator_lizard', 46: 'Gila_monster', 47: 'green_lizard', 48: 'African_chameleon', 49: 'Komodo_dragon',
        50: 'African_crocodile', 51: 'American_alligator', 52: 'triceratops', 53: 'thunder_snake', 54: 'ringneck_snake',
        55: 'hognose_snake', 56: 'green_snake', 57: 'king_snake', 58: 'garter_snake', 59: 'water_snake',
        60: 'vine_snake', 61: 'night_snake', 62: 'boa_constrictor', 63: 'rock_python', 64: 'Indian_cobra',
        65: 'green_mamba', 66: 'sea_snake', 67: 'horned_viper', 68: 'diamondback', 69: 'sidewinder',
        70: 'trilobite', 71: 'harvestman', 72: 'scorpion', 73: 'black_and_gold_garden_spider', 74: 'barn_spider',
        75: 'garden_spider', 76: 'black_widow', 77: 'tarantula', 78: 'wolf_spider', 79: 'tick',
        80: 'centipede', 81: 'black_grouse', 82: 'ptarmigan', 83: 'ruffed_grouse', 84: 'prairie_chicken',
        85: 'peacock', 86: 'quail', 87: 'partridge', 88: 'African_grey', 89: 'macaw',
        90: 'sulphur_crested_cockatoo', 91: 'lorikeet', 92: 'coucal', 93: 'bee_eater', 94: 'hornbill',
        95: 'hummingbird', 96: 'jacamar', 97: 'toucan', 98: 'drake', 99: 'red_breasted_merganser',
        100: 'goose', 101: 'black_swan', 102: 'tusker', 103: 'echidna', 104: 'platypus',
        105: 'wallaby', 106: 'koala', 107: 'wombat', 108: 'jellyfish', 109: 'sea_anemone',
        110: 'brain_coral', 111: 'flatworm', 112: 'nematode', 113: 'conch', 114: 'snail',
        115: 'slug', 116: 'sea_slug', 117: 'chiton', 118: 'chambered_nautilus', 119: 'Dungeness_crab',
        120: 'rock_crab', 121: 'fiddler_crab', 122: 'king_crab', 123: 'American_lobster', 124: 'spiny_lobster',
        125: 'crayfish', 126: 'hermit_crab', 127: 'isopod', 128: 'white_stork', 129: 'black_stork',
        130: 'spoonbill', 131: 'flamingo', 132: 'little_blue_heron', 133: 'American_egret', 134: 'bittern',
        135: 'crane_bird', 136: 'limpkin', 137: 'European_gallinule', 138: 'American_coot', 139: 'bustard',
        140: 'ruddy_turnstone', 141: 'red_backed_sandpiper', 142: 'redshank', 143: 'dowitcher', 144: 'oystercatcher',
        145: 'pelican', 146: 'king_penguin', 147: 'albatross', 148: 'grey_whale', 149: 'killer_whale',
        150: 'dugong', 151: 'sea_lion', 152: 'Chihuahua', 153: 'Japanese_spaniel', 154: 'Maltese_dog',
        155: 'Pekinese', 156: 'Shih_Tzu', 157: 'Blenheim_spaniel', 158: 'papillon', 159: 'toy_terrier',
        160: 'Rhodesian_ridgeback', 161: 'Afghan_hound', 162: 'basset', 163: 'beagle', 164: 'bloodhound',
        165: 'bluetick', 166: 'black_and_tan_coonhound', 167: 'Walker_hound', 168: 'English_foxhound', 169: 'redbone',
        170: 'borzoi', 171: 'Irish_wolfhound', 172: 'Italian_greyhound', 173: 'whippet', 174: 'Ibizan_hound',
        175: 'Norwegian_elkhound', 176: 'otterhound', 177: 'Saluki', 178: 'Scottish_deerhound', 179: 'Weimaraner',
        180: 'Staffordshire_bullterrier', 181: 'American_Staffordshire_terrier', 182: 'Bedlington_terrier', 183: 'Border_terrier', 184: 'Kerry_blue_terrier',
        185: 'Irish_terrier', 186: 'Norfolk_terrier', 187: 'Norwich_terrier', 188: 'Yorkshire_terrier', 189: 'wire_haired_fox_terrier',
        190: 'Lakeland_terrier', 191: 'Sealyham_terrier', 192: 'Airedale', 193: 'cairn', 194: 'Australian_terrier',
        195: 'Dandie_Dinmont', 196: 'Boston_bull', 197: 'miniature_schnauzer', 198: 'giant_schnauzer', 199: 'standard_schnauzer',
        200: 'Scotch_terrier', 201: 'Tibetan_terrier', 202: 'silky_terrier', 203: 'soft_coated_wheaten_terrier', 204: 'West_Highland_white_terrier',
        205: 'Lhasa', 206: 'flat_coated_retriever', 207: 'curly_coated_retriever', 208: 'golden_retriever', 209: 'Labrador_retriever',
        210: 'Chesapeake_Bay_retriever', 211: 'German_short_haired_pointer', 212: 'vizsla', 213: 'English_setter', 214: 'Irish_setter',
        215: 'Gordon_setter', 216: 'Brittany_spaniel', 217: 'clumber', 218: 'English_springer', 219: 'Welsh_springer_spaniel',
        220: 'cocker_spaniel', 221: 'Sussex_spaniel', 222: 'Irish_water_spaniel', 223: 'kuvasz', 224: 'schipperke',
        225: 'groenendael', 226: 'malinois', 227: 'briard', 228: 'kelpie', 229: 'komondor',
        230: 'Old_English_sheepdog', 231: 'Shetland_sheepdog', 232: 'collie', 233: 'Border_collie', 234: 'Bouvier_des_Flandres',
        235: 'Rottweiler', 236: 'German_shepherd', 237: 'Doberman', 238: 'miniature_pinscher', 239: 'Greater_Swiss_Mountain_dog',
        240: 'Bernese_mountain_dog', 241: 'Appenzeller', 242: 'EntleBucher', 243: 'boxer', 244: 'bull_mastiff',
        245: 'Tibetan_mastiff', 246: 'French_bulldog', 247: 'Great_Dane', 248: 'Saint_Bernard', 249: 'Eskimo_dog',
        250: 'malamute', 251: 'Siberian_husky', 252: 'dalmatian', 253: 'affenpinscher', 254: 'basenji',
        255: 'pug', 256: 'Leonberg', 257: 'Newfoundland', 258: 'Great_Pyrenees', 259: 'Samoyed',
        260: 'Pomeranian', 261: 'chow', 262: 'keeshond', 263: 'Brabancon_griffon', 264: 'Pembroke',
        265: 'Cardigan', 266: 'toy_poodle', 267: 'miniature_poodle', 268: 'standard_poodle', 269: 'Mexican_hairless',
        270: 'timber_wolf', 271: 'white_wolf', 272: 'red_wolf', 273: 'coyote', 274: 'dingo',
        275: 'dhole', 276: 'African_hunting_dog', 277: 'hyena', 278: 'red_fox', 279: 'kit_fox',
        280: 'Arctic_fox', 281: 'grey_fox', 282: 'tabby', 283: 'tiger_cat', 284: 'Persian_cat',
        285: 'Siamese_cat', 286: 'Egyptian_cat', 287: 'cougar', 288: 'lynx', 289: 'leopard',
        290: 'snow_leopard', 291: 'jaguar', 292: 'lion', 293: 'tiger', 294: 'cheetah',
        295: 'brown_bear', 296: 'American_black_bear', 297: 'ice_bear', 298: 'sloth_bear', 299: 'mongoose',
        300: 'meerkat', 301: 'tiger_beetle', 302: 'ladybug', 303: 'ground_beetle', 304: 'long_horned_beetle',
        305: 'leaf_beetle', 306: 'dung_beetle', 307: 'rhinoceros_beetle', 308: 'weevil', 309: 'fly',
        310: 'bee', 311: 'ant', 312: 'grasshopper', 313: 'cricket', 314: 'walking_stick',
        315: 'cockroach', 316: 'mantis', 317: 'cicada', 318: 'leafhopper', 319: 'lacewing',
        320: 'dragonfly', 321: 'damselfly', 322: 'admiral', 323: 'ringlet', 324: 'monarch',
        325: 'cabbage_butterfly', 326: 'sulphur_butterfly', 327: 'lycaenid', 328: 'starfish', 329: 'sea_urchin',
        330: 'sea_cucumber', 331: 'wood_rabbit', 332: 'hare', 333: 'Angora', 334: 'hamster',
        335: 'porcupine', 336: 'fox_squirrel', 337: 'marmot', 338: 'beaver', 339: 'guinea_pig',
        340: 'sorrel', 341: 'zebra', 342: 'hog', 343: 'wild_boar', 344: 'warthog',
        345: 'hippopotamus', 346: 'ox', 347: 'water_buffalo', 348: 'bison', 349: 'ram',
        350: 'bighorn', 351: 'ibex', 352: 'hartebeest', 353: 'impala', 354: 'gazelle',
        355: 'Arabian_camel', 356: 'llama', 357: 'weasel', 358: 'mink', 359: 'polecat',
        360: 'black_footed_ferret', 361: 'otter', 362: 'skunk', 363: 'badger', 364: 'armadillo',
        365: 'three_toed_sloth', 366: 'orangutan', 367: 'gorilla', 368: 'chimpanzee', 369: 'gibbon',
        370: 'siamang', 371: 'guenon', 372: 'patas', 373: 'baboon', 374: 'macaque',
        375: 'langur', 376: 'proboscis_monkey', 377: 'marmoset', 378: 'capuchin', 379: 'howler_monkey',
        380: 'titi', 381: 'spider_monkey', 382: 'squirrel_monkey', 383: 'Madagascar_cat', 384: 'indri',
        385: 'Indian_elephant', 386: 'African_elephant', 387: 'lesser_panda', 388: 'giant_panda', 389: 'barracouta',
        390: 'eel', 391: 'coho', 392: 'rock_beauty', 393: 'anemone_fish', 394: 'sturgeon',
        395: 'gar', 396: 'lionfish', 397: 'puffer', 398: 'abacus', 399: 'abaya',
        400: 'academic_gown', 401: 'accordion', 402: 'acoustic_guitar', 403: 'aircraft_carrier', 404: 'airliner',
        405: 'airship', 406: 'altar', 407: 'ambulance', 408: 'amphibian', 409: 'analog_clock',
        410: 'apiary', 411: 'apron', 412: 'ashcan', 413: 'assault_rifle', 414: 'backpack',
        415: 'bakery', 416: 'balance_beam', 417: 'balloon', 418: 'ballpoint', 419: 'Band_Aid',
        420: 'banjo', 421: 'bannister', 422: 'barbell', 423: 'barber_chair', 424: 'barbershop',
        425: 'barn', 426: 'barometer', 427: 'barrel', 428: 'barrow', 429: 'baseball',
        430: 'basketball', 431: 'bassinet', 432: 'bassoon', 433: 'bathing_cap', 434: 'bath_towel',
        435: 'bathtub', 436: 'beach_wagon', 437: 'beacon', 438: 'beaker', 439: 'bearskin',
        440: 'beer_bottle', 441: 'beer_glass', 442: 'bell_cote', 443: 'bib', 444: 'bicycle_built_for_two',
        445: 'bikini', 446: 'binder', 447: 'binoculars', 448: 'birdhouse', 449: 'boathouse',
        450: 'bobsled', 451: 'bolo_tie', 452: 'bonnet', 453: 'bookcase', 454: 'bookshop',
        455: 'bottlecap', 456: 'bow', 457: 'bow_tie', 458: 'brass', 459: 'brassiere',
        460: 'breakwater', 461: 'breastplate', 462: 'broom', 463: 'bucket', 464: 'buckle',
        465: 'bulletproof_vest', 466: 'bullet_train', 467: 'butcher_shop', 468: 'cab', 469: 'caldron',
        470: 'candle', 471: 'cannon', 472: 'canoe', 473: 'can_opener', 474: 'cardigan',
        475: 'car_mirror', 476: 'carousel', 477: "carpenter's_kit", 478: 'carton', 479: 'car_wheel',
        480: 'cash_machine', 481: 'cassette', 482: 'cassette_player', 483: 'castle', 484: 'catamaran',
        485: 'CD_player', 486: 'cello', 487: 'cellular_telephone', 488: 'chain', 489: 'chainlink_fence',
        490: 'chain_mail', 491: 'chain_saw', 492: 'chest', 493: 'chiffonier', 494: 'chime',
        495: 'china_cabinet', 496: 'Christmas_stocking', 497: 'church', 498: 'cinema', 499: 'cleaver',
        500: 'cliff_dwelling', 501: 'cloak', 502: 'clog', 503: 'cocktail_shaker', 504: 'coffee_mug',
        505: 'coffeepot', 506: 'coil', 507: 'combination_lock', 508: 'computer_keyboard', 509: 'confectionery',
        510: 'container_ship', 511: 'convertible', 512: 'corkscrew', 513: 'cornet', 514: 'cowboy_boot',
        515: 'cowboy_hat', 516: 'cradle', 517: 'crane', 518: 'crash_helmet', 519: 'crate',
        520: 'crib', 521: 'Crock_Pot', 522: 'croquet_ball', 523: 'crutch', 524: 'cuirass',
        525: 'dam', 526: 'desk', 527: 'desktop_computer', 528: 'dial_telephone', 529: 'diaper',
        530: 'digital_clock', 531: 'digital_watch', 532: 'dining_table', 533: 'dishrag', 534: 'dishwasher',
        535: 'disk_brake', 536: 'dock', 537: 'dogsled', 538: 'dome', 539: 'doormat',
        540: 'drilling_platform', 541: 'drum', 542: 'drumstick', 543: 'dumbbell', 544: 'Dutch_oven',
        545: 'electric_fan', 546: 'electric_guitar', 547: 'electric_locomotive', 548: 'entertainment_center', 549: 'envelope',
        550: 'espresso_maker', 551: 'face_powder', 552: 'feather_boa', 553: 'file', 554: 'fireboat',
        555: 'fire_engine', 556: 'fire_screen', 557: 'flagpole', 558: 'flute', 559: 'folding_chair',
        560: 'football_helmet', 561: 'forklift', 562: 'fountain', 563: 'fountain_pen', 564: 'four_poster',
        565: 'freight_car', 566: 'French_horn', 567: 'frying_pan', 568: 'fur_coat', 569: 'garbage_truck',
        570: 'gasmask', 571: 'gas_pump', 572: 'goblet', 573: 'go-kart', 574: 'golf_ball',
        575: 'golfcart', 576: 'gondola', 577: 'gong', 578: 'gown', 579: 'grand_piano',
        580: 'greenhouse', 581: 'grille', 582: 'grocery_store', 583: 'guillotine', 584: 'hair_slide',
        585: 'hair_spray', 586: 'half_track', 587: 'hammer', 588: 'hamper', 589: 'hand_blower',
        590: 'hand_held_computer', 591: 'handkerchief', 592: 'hard_disc', 593: 'harmonica', 594: 'harp',
        595: 'harvester', 596: 'hatchet', 597: 'holster', 598: 'home_theater', 599: 'honeycomb',
        600: 'hook', 601: 'hoopskirt', 602: 'horizontal_bar', 603: 'horse_cart', 604: 'hourglass',
        605: 'iPod', 606: 'iron', 607: "jack_o_lantern", 608: 'jean', 609: 'jeep',
        610: 'jersey', 611: 'jigsaw_puzzle', 612: 'jinrikisha', 613: 'joystick', 614: 'kimono',
        615: 'knee_pad', 616: 'knot', 617: 'lab_coat', 618: 'ladle', 619: 'lampshade',
        620: 'laptop', 621: 'lawn_mower', 622: 'lens_cap', 623: 'letter_opener', 624: 'library',
        625: 'lifeboat', 626: 'lighter', 627: 'limousine', 628: 'liner', 629: 'lipstick',
        630: 'Loafer', 631: 'lotion', 632: 'loudspeaker', 633: 'loupe', 634: 'lumbermill',
        635: 'magnetic_compass', 636: 'mailbag', 637: 'mailbox', 638: 'maillot', 639: 'maillot_tank_suit',
        640: 'manhole_cover', 641: 'maraca', 642: 'marimba', 643: 'mask', 644: 'matchstick',
        645: 'maypole', 646: 'maze', 647: 'measuring_cup', 648: 'medicine_chest', 649: 'megalith',
        650: 'microphone', 651: 'microwave', 652: 'military_uniform', 653: 'milk_can', 654: 'minibus',
        655: 'miniskirt', 656: 'minivan', 657: 'missile', 658: 'mitten', 659: 'mixing_bowl',
        660: 'mobile_home', 661: 'Model_T', 662: 'modem', 663: 'monastery', 664: 'monitor',
        665: 'moped', 666: 'mortar', 667: 'mortarboard', 668: 'mosque', 669: 'mosquito_net',
        670: 'motor_scooter', 671: 'mountain_bike', 672: 'mountain_tent', 673: 'mouse', 674: 'mousetrap',
        675: 'moving_van', 676: 'muzzle', 677: 'nail', 678: 'neck_brace', 679: 'necklace',
        680: 'nipple', 681: 'notebook', 682: 'obelisk', 683: 'oboe', 684: 'ocarina',
        685: 'odometer', 686: 'oil_filter', 687: 'organ', 688: 'oscilloscope', 689: 'overskirt',
        690: 'oxcart', 691: 'oxygen_mask', 692: 'packet', 693: 'paddle', 694: 'paddlewheel',
        695: 'padlock', 696: 'paintbrush', 697: 'pajama', 698: 'palace', 699: 'panpipe',
        700: 'paper_towel', 701: 'parachute', 702: 'parallel_bars', 703: 'park_bench', 704: 'parking_meter',
        705: 'passenger_car', 706: 'patio', 707: 'pay_phone', 708: 'pedestal', 709: 'pencil_box',
        710: 'pencil_sharpener', 711: 'perfume', 712: 'Petri_dish', 713: 'photocopier', 714: 'pick',
        715: 'pickelhaube', 716: 'picket_fence', 717: 'pickup', 718: 'pier', 719: 'piggy_bank',
        720: 'pill_bottle', 721: 'pillow', 722: 'ping_pong_ball', 723: 'pinwheel', 724: 'pirate',
        725: 'pitcher', 726: 'plane', 727: 'planetarium', 728: 'plastic_bag', 729: 'plate_rack',
        730: 'plow', 731: 'plunger', 732: 'Polaroid_camera', 733: 'pole', 734: 'police_van',
        735: 'poncho', 736: 'pool_table', 737: 'pop_bottle', 738: 'pot', 739: "potter's_wheel",
        740: 'power_drill', 741: 'prayer_rug', 742: 'printer', 743: 'prison', 744: 'projectile',
        745: 'projector', 746: 'puck', 747: 'punching_bag', 748: 'purse', 749: 'quill',
        750: 'quilt', 751: 'racer', 752: 'racket', 753: 'radiator', 754: 'radio',
        755: 'radio_telescope', 756: 'rain_barrel', 757: 'recreational_vehicle', 758: 'reel', 759: 'reflex_camera',
        760: 'refrigerator', 761: 'remote_control', 762: 'restaurant', 763: 'revolver', 764: 'rifle',
        765: 'rocking_chair', 766: 'rotisserie', 767: 'rubber_eraser', 768: 'rugby_ball', 769: 'rule',
        770: 'running_shoe', 771: 'safe', 772: 'safety_pin', 773: 'saltshaker', 774: 'sandal',
        775: 'sarong', 776: 'sax', 777: 'scabbard', 778: 'scale', 779: 'school_bus',
        780: 'schooner', 781: 'scoreboard', 782: 'CRT_screen', 783: 'screw', 784: 'screwdriver',
        785: 'seat_belt', 786: 'sewing_machine', 787: 'shield', 788: 'shoe_shop', 789: 'shoji',
        790: 'shopping_basket', 791: 'shopping_cart', 792: 'shovel', 793: 'shower_cap', 794: 'shower_curtain',
        795: 'ski', 796: 'ski_mask', 797: 'sleeping_bag', 798: 'slide_rule', 799: 'sliding_door',
        800: 'slot', 801: 'snorkel', 802: 'snowmobile', 803: 'snowplow', 804: 'soap_dispenser',
        805: 'soccer_ball', 806: 'sock', 807: 'solar_dish', 808: 'sombrero', 809: 'soup_bowl',
        810: 'space_bar', 811: 'space_heater', 812: 'space_suit', 813: 'spider_web', 814: 'spindle',
        815: 'sports_car', 816: 'spotlight', 817: 'stage', 818: 'steam_locomotive', 819: 'steel_arch_bridge',
        820: 'steel_drum', 821: 'stethoscope', 822: 'stole', 823: 'stone_wall', 824: 'stopwatch',
        825: 'stove', 826: 'strainer', 827: 'streetcar', 828: 'stretcher', 829: 'studio_couch',
        830: 'stupa', 831: 'submarine', 832: 'suit', 833: 'sundial', 834: 'sunglass',
        835: 'sunglasses', 836: 'sunscreen', 837: 'suspension_bridge', 838: 'swab', 839: 'sweatshirt',
        840: 'swimming_trunks', 841: 'swing', 842: 'switch', 843: 'syringe', 844: 'table_lamp',
        845: 'tank', 846: 'tape_player', 847: 'teapot', 848: 'teddy', 849: 'television',
        850: 'tennis_ball', 851: 'thatched_roof', 852: 'theater_curtain', 853: 'thimble', 854: 'thresher',
        855: 'throne', 856: 'tile_roof', 857: 'toaster', 858: 'tobacco_shop', 859: 'toilet_seat',
        860: 'torch', 861: 'totem_pole', 862: 'tow_truck', 863: 'toyshop', 864: 'tractor',
        865: 'trailer_truck', 866: 'tray', 867: 'trench_coat', 868: 'tricycle', 869: 'trimaran',
        870: 'tripod', 871: 'triumphal_arch', 872: 'trolleybus', 873: 'trombone', 874: 'tub',
        875: 'turnstile', 876: 'typewriter_keyboard', 877: 'umbrella', 878: 'unicycle', 879: 'upright',
        880: 'vacuum', 881: 'vase', 882: 'vault', 883: 'velvet', 884: 'vending_machine',
        885: 'vestment', 886: 'viaduct', 887: 'violin', 888: 'volleyball', 889: 'waffle_iron',
        890: 'wall_clock', 891: 'wallet', 892: 'wardrobe', 893: 'warplane', 894: 'washbasin',
        895: 'washer', 896: 'water_bottle', 897: 'water_jug', 898: 'water_tower', 899: 'whiskey_jug',
        900: 'whistle', 901: 'wig', 902: 'window_screen', 903: 'window_shade', 904: 'Windsor_tie',
        905: 'wine_bottle', 906: 'wing', 907: 'wok', 908: 'wooden_spoon', 909: 'wool',
        910: 'worm_fence', 911: 'wreck', 912: 'yawl', 913: 'yurt', 914: 'web_site',
        915: 'comic_book', 916: 'crossword_puzzle', 917: 'street_sign', 918: 'traffic_light', 919: 'book_jacket',
        920: 'menu', 921: 'plate', 922: 'guacamole', 923: 'consomme', 924: 'hot_pot',
        925: 'trifle', 926: 'ice_cream', 927: 'ice_lolly', 928: 'French_loaf', 929: 'bagel',
        930: 'pretzel', 931: 'cheeseburger', 932: 'hotdog', 933: 'mashed_potato', 934: 'head_cabbage',
        935: 'broccoli', 936: 'cauliflower', 937: 'zucchini', 938: 'spaghetti_squash', 939: 'acorn_squash',
        940: 'butternut_squash', 941: 'cucumber', 942: 'artichoke', 943: 'bell_pepper', 944: 'cardoon',
        945: 'mushroom', 946: 'Granny_Smith', 947: 'strawberry', 948: 'orange', 949: 'lemon',
        950: 'fig', 951: 'pineapple', 952: 'banana', 953: 'jackfruit', 954: 'custard_apple',
        955: 'pomegranate', 956: 'hay', 957: 'carbonara', 958: 'chocolate_sauce', 959: 'dough',
        960: 'meat_loaf', 961: 'pizza', 962: 'potpie', 963: 'burrito', 964: 'red_wine',
        965: 'espresso', 966: 'cup', 967: 'eggnog', 968: 'alp', 969: 'bubble',
        970: 'cliff', 971: 'coral_reef', 972: 'geyser', 973: 'lakeside', 974: 'promontory',
        975: 'sandbar', 976: 'seashore', 977: 'valley', 978: 'volcano', 979: 'ballplayer',
        980: 'groom', 981: 'scuba_diver', 982: 'bridegroom', 983: 'diver', 984: 'maori',
        985: 'bowler', 986: 'manhole', 987: 'rescue', 988: 'baby_bed', 989: 'bird',
        990: 'bit', 991: 'broom', 992: 'christening', 993: 'cross', 994: 'desert',
        995: 'electronic', 996: 'fairy', 997: 'fire_hydrant', 998: 'flag', 999: 'umbrella',
    }
    labels = IMAGENET_LABELS

    print(f"\n{'='*60}")
    print(f"  Top-5 预测结果")
    print(f"{'='*60}")
    for i in range(5):
        idx = top5_idx[0][i].item()
        score = top5_val[0][i].item()
        print(f"  {i+1}. {labels.get(idx, idx)} (score: {score:.4f})")

    # 8. 性能统计
    if args.num_runs > 1:
        avg_lat = sum(latencies) / len(latencies)
        min_lat = min(latencies)
        max_lat = max(latencies)
        print(f"\n{'='*60}")
        print(f"  性能统计 (共 {args.num_runs} 轮)")
        print(f"{'='*60}")
        print(f"  平均耗时: {avg_lat*1000:.2f}ms")
        print(f"  最小耗时: {min_lat*1000:.2f}ms")
        print(f"  最大耗时: {max_lat*1000:.2f}ms")
        print(f"  FPS: {1.0/avg_lat:.2f}")

    logits_np = final_out.cpu().numpy() if device == "npu" else final_out.numpy()
    np.save(f"{model_name.replace('.', '_')}_{device}_logits.npy", logits_np)
    print(f"\n[INFO] Logits 已保存: {model_name.replace('.', '_')}_{device}_logits.npy")

    # 释放资源
    del model, input_tensor, outputs
    gc.collect()
    if device == "npu":
        torch.npu.empty_cache()
    print(f"[INFO] 资源已释放")


if __name__ == "__main__":
    main()
