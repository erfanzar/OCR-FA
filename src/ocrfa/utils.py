from __future__ import print_function

from PIL import Image, JpegImagePlugin
from scipy import ndimage
import hashlib
import sys, os
from zipfile import ZipFile

if sys.version_info[0] == 2:
    from six.moves.urllib.request import urlretrieve
else:
    from urllib.request import urlretrieve
import torch.utils.data
import torchvision.transforms as transforms
from collections import OrderedDict
import importlib
from skimage import io
import os
import numpy as np
import cv2
import math
from scipy.ndimage import label

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import Vgg16BN, init_weights


class DoubleConv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class OCRFA(nn.Module):
    def __init__(self, pretrained=False, freeze=False):
        super(OCRFA, self).__init__()

        """ Base network """
        self.basenet = Vgg16BN(pretrained, freeze)

        """ U network """
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)

        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )

        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """ Base network """
        sources = self.basenet(x)

        """ U network """
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)

        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)

        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)

        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature


def warp_coord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])


def get_det_boxes_core(textmap, linkmap, text_threshold, link_threshold, low_text, estimate_num_chars=False):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    ret, text_score = cv2.threshold(textmap, low_text, 1, 0)
    ret, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8),
        connectivity=4
    )

    det = []
    mapper = []
    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold: continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255
        if estimate_num_chars:
            _, character_locs = cv2.threshold((textmap - linkmap) * segmap / 255., text_threshold, 1, 0)
            _, n_chars = label(character_locs)
            mapper.append(n_chars)
        else:
            mapper.append(k)
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0  # remove link area
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = x - niter, x + w + niter + 1, y - niter, y + h + niter + 1
        # boundary check
        if sx < 0: sx = 0
        if sy < 0: sy = 0
        if ex >= img_w: ex = img_w
        if ey >= img_h: ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_contours = np.roll(np.array(np.where(segmap != 0)), 1, axis=0).transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)

    return det, labels, mapper


def get_poly_core(boxes, labels, mapper, linkmap):
    num_cp = 5
    max_len_ratio = 0.7
    expand_ratio = 1.45
    max_r = 2.0
    step_r = 0.2

    polys = []
    for k, box in enumerate(boxes):
        # size filter for small instance
        w, h = int(np.linalg.norm(box[0] - box[1]) + 1), int(np.linalg.norm(box[1] - box[2]) + 1)
        if w < 10 or h < 10:
            polys.append(None)
            continue

        # warp image
        tar = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        M = cv2.getPerspectiveTransform(box, tar)
        word_label = cv2.warpPerspective(labels, M, (w, h), flags=cv2.INTER_NEAREST)
        try:
            Minv = np.linalg.inv(M)
        except:
            polys.append(None)
            continue

        cur_label = mapper[k]
        word_label[word_label != cur_label] = 0
        word_label[word_label > 0] = 1
        cp = []
        max_len = -1
        for i in range(w):
            region = np.where(word_label[:, i] != 0)[0]
            if len(region) < 2:
                continue
            cp.append((i, region[0], region[-1]))
            length = region[-1] - region[0] + 1
            if length > max_len:
                max_len = length

        # pass if max_len is similar to h
        if h * max_len_ratio < max_len:
            polys.append(None)
            continue

        # get pivot points with fixed length
        tot_seg = num_cp * 2 + 1
        seg_w = w / tot_seg  # segment width
        pp = [None] * num_cp  # init pivot points
        cp_section = [[0, 0]] * tot_seg
        seg_height = [0] * num_cp
        seg_num = 0
        num_sec = 0
        prev_h = -1
        for i in range(0, len(cp)):
            (x, sy, ey) = cp[i]
            if (seg_num + 1) * seg_w <= x and seg_num <= tot_seg:
                # average previous segment
                if num_sec == 0: break
                cp_section[seg_num] = [cp_section[seg_num][0] / num_sec,
                                       cp_section[seg_num][1] / num_sec]  # type:ignore
                num_sec = 0

                # reset variables
                seg_num += 1
                prev_h = -1

            # accumulate center points
            cy = (sy + ey) * 0.5
            cur_h = ey - sy + 1
            cp_section[seg_num] = [cp_section[seg_num][0] + x, cp_section[seg_num][1] + cy]
            num_sec += 1

            if seg_num % 2 == 0: continue  # No polygon area

            if prev_h < cur_h:
                pp[int((seg_num - 1) / 2)] = (x, cy)  # type:ignore
                seg_height[int((seg_num - 1) / 2)] = cur_h
                prev_h = cur_h

        # processing last segment
        if num_sec != 0:
            cp_section[-1] = [cp_section[-1][0] / num_sec, cp_section[-1][1] / num_sec]

        # pass if num of pivots is not sufficient or segment width is smaller than character height
        if None in pp or seg_w < np.max(seg_height) * 0.25:
            polys.append(None)
            continue

        # calc median maximum of pivot points
        half_char_h = np.median(seg_height) * expand_ratio / 2

        # calc gradiant and apply to make horizontal pivots
        new_pp = []
        for i, (x, cy) in enumerate(pp):
            dx = cp_section[i * 2 + 2][0] - cp_section[i * 2][0]
            dy = cp_section[i * 2 + 2][1] - cp_section[i * 2][1]
            if dx == 0:  # gradient if zero
                new_pp.append([x, cy - half_char_h, x, cy + half_char_h])
                continue
            rad = - math.atan2(dy, dx)
            c, s = half_char_h * math.cos(rad), half_char_h * math.sin(rad)
            new_pp.append([x - s, cy - c, x + s, cy + c])

        # get edge points to cover character heatmaps
        isSppFound, isEppFound = False, False
        grad_s = (pp[1][1] - pp[0][1]) / (pp[1][0] - pp[0][0]) + (pp[2][1] - pp[1][1]) / (  # type:ignore
                pp[2][0] - pp[1][0])  # type:ignore
        grad_e = (pp[-2][1] - pp[-1][1]) / (pp[-2][0] - pp[-1][0]) + (pp[-3][1] - pp[-2][1]) / (  # type:ignore
                pp[-3][0] - pp[-2][0])  # type:ignore
        for r in np.arange(0.5, max_r, step_r):
            dx = 2 * half_char_h * r
            if not isSppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_s * dx
                p = np.array(new_pp[0]) - np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)  # type:ignore
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    spp = p
                    isSppFound = True
            if not isEppFound:
                line_img = np.zeros(word_label.shape, dtype=np.uint8)
                dy = grad_e * dx
                p = np.array(new_pp[-1]) + np.array([dx, dy, dx, dy])
                cv2.line(line_img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 1, thickness=1)  # type:ignore
                if np.sum(np.logical_and(word_label, line_img)) == 0 or r + 2 * step_r >= max_r:
                    epp = p
                    isEppFound = True
            if isSppFound and isEppFound:
                break

        # pass if boundary of polygon is not found
        if not (isSppFound and isEppFound):
            polys.append(None)
            continue

        # make final polygon
        poly = [warp_coord(Minv, (spp[0], spp[1]))]
        for p in new_pp:
            poly.append(warp_coord(Minv, (p[0], p[1])))
        poly.append(warp_coord(Minv, (epp[0], epp[1])))
        poly.append(warp_coord(Minv, (epp[2], epp[3])))
        for p in reversed(new_pp):
            poly.append(warp_coord(Minv, (p[2], p[3])))
        poly.append(warp_coord(Minv, (spp[2], spp[3])))

        # add to final result
        polys.append(np.array(poly))

    return polys


def get_det_boxes(textmap, linkmap, text_threshold, link_threshold, low_text, poly=False, estimate_num_chars=False):
    if poly and estimate_num_chars:
        raise Exception("Estimating the number of characters not currently supported with poly.")
    boxes, labels, mapper = get_det_boxes_core(textmap, linkmap, text_threshold, link_threshold, low_text,
                                               estimate_num_chars)

    if poly:
        polys = get_poly_core(boxes, labels, mapper, linkmap)
    else:
        polys = [None] * len(boxes)

    return boxes, polys, mapper


def adjust_result_coordinates(polys, ratio_w, ratio_h, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys


os.environ["LRU_CACHE_CAPACITY"] = "1"

BASE_PATH = os.path.dirname(__file__)
MODULE_PATH = os.environ.get("OCR_FA_MODULE_PATH") or os.environ.get("MODULE_PATH") or os.path.expanduser(
    "~/.OCR-FA/"
)

# detector parameters
detection_models = {
    "craft": {
        "filename": "craft_mlt_25k.pth",
        "url": "https://github.com/erfanzar/OCR-FA/releases/download/pre-v1.1.6/craft_mlt_25k.zip",
        "md5sum": "2f8227d2def4037cdb3b34389dcf9ec1"
    },
    "dbnet18": {
        "filename": "pretrained_ic15_res18.pt",
        "url": "https://github.com/erfanzar/OCR-FA/releases/download/v1.6.0/pretrained_ic15_res18.zip",
        "md5sum": "aee04f8ffe5fc5bd5abea73223800425"
    },
    "dbnet50": {
        "filename": "pretrained_ic15_res50.pt",
        "url": "https://github.com/erfanzar/OCR-FA/releases/download/v1.6.0/pretrained_ic15_res50.zip",
        "md5sum": "a8e90144c131c2467d1eb7886c2e93a6"
    }
}
symbols = "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "

latin_lang_list = [
    "af", "az", "bs", "cs", "cy", "da", "de", "en", "es", "et", "fr", "ga",
    "hr", "hu", "id", "is", "it", "ku", "la", "lt", "lv", "mi", "ms", "mt",
    "nl", "no", "oc", "pi", "pl", "pt", "ro", "rs_latin", "sk", "sl", "sq",
    "sv", "sw", "tl", "tr", "uz", "vi"
]
arabic_lang_list = ["ar", "fa", "ug", "ur"]

all_lang_list = latin_lang_list + arabic_lang_list
imgH = 64
separator_list = {
    "th": ["\xa2", "\xa3"],
    "en": ["\xa4", "\xa5"]
}
separator_char = []
for lang, sep in separator_list.items():
    separator_char += sep

recognition_models = {
    'gen1': {
        'latin_g1': {
            'filename': 'latin.pth',
            'model_script': 'latin',
            'url': 'https://github.com/erfanzar/OCR-FA/releases/download/pre-v1.1.6/latin.zip',
            'md5sum': 'fb91b9abf65aeeac95a172291b4a6176',
            'characters': "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÍÎÑÒÓÔÕÖØÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿąęĮįıŁłŒœŠšųŽž",
            'symbols': "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
        },
        'japanese_g1': {
            'filename': 'japanese.pth',
            'model_script': 'japanese',
            'url': 'https://github.com/erfanzar/OCR-FA/releases/download/pre-v1.1.6/japanese.zip',
            'md5sum': '6d891a4aad9cb7f492809515e4e9fd2e',
            'symbols': "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ",
            'characters': '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろわゐをんゝゞァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶ・ー一丁七万丈三上下不与丑且世丘丙丞両並中串丸丹主乃久之乍乎乏乗乙九乞也乱乳乾亀了予争事二云互五井亘亙亜亡交亥亦亨享京亭亮人什仁仇今介仏仔仕他付仙代令以仮仰仲件任企伊伍伎伏伐休会伝伯伴伶伸伺似伽佃但位低住佐佑体何余作佳併佼使侃例侍供依侠価侮侯侵侶便係促俄俊俗保信俣修俳俵俸俺倉個倍倒倖候借倣値倦倫倭倶倹偉偏停健偲側偵偶偽傍傑傘備催傭債傷傾僅働像僑僕僚僧僻儀億儒償優儲允元兄充兆兇先光克免兎児党兜入全八公六共兵其具典兼内円冊再冒冗写冠冥冨冬冴冶冷凄准凋凌凍凝凡処凪凱凶凸凹出函刀刃分切刈刊刑列初判別利到制刷券刺刻剃則削前剖剛剣剤剥副剰割創劃劇劉力功加劣助努励労効劾勃勅勇勉動勘務勝募勢勤勧勲勾勿匁匂包化北匝匠匡匹区医匿十千升午半卑卒卓協南単博卜占卯印危即却卵卸卿厄厘厚原厨厩厭厳去参又叉及友双反収叔取受叙叛叡叢口古句叩只叫召可台叱史右叶号司吃各合吉吊同名后吏吐向君吟吠否含吸吹吻吾呂呆呈呉告呑周呪味呼命咋和咲咳咽哀品哉員哨哲唄唆唇唐唯唱唾商問啓善喉喋喚喜喝喧喪喫喬喰営嗣嘆嘉嘗嘘嘩嘱噂噌噛器噴嚇嚢囚四回因団困囲図固国圃圏園土圧在圭地坂均坊坐坑坦坪垂型垢垣埋城埜域埠埴執培基埼堀堂堅堆堕堤堪堰報場堵堺塀塁塊塑塔塗塘塚塞塩填塵塾境墓増墜墨墳墾壁壇壊壌壕士壬壮声壱売壷変夏夕外多夜夢大天太夫央失夷奄奇奈奉奏契奔套奥奨奪奮女奴好如妃妄妊妓妖妙妥妨妬妹妻妾姉始姐姑姓委姥姦姪姫姶姻姿威娃娘娠娩娯娼婁婆婚婦婿媒媛嫁嫉嫌嫡嬉嬢嬰子孔字存孜孝孟季孤学孫宅宇守安宋完宍宏宕宗官宙定宛宜宝実客宣室宥宮宰害宴宵家容宿寂寄寅密富寒寓寛寝察寡寧審寮寵寸寺対寿封専射将尉尊尋導小少尖尚尤尭就尺尻尼尽尾尿局居屈届屋屍屑展属屠層履屯山岐岡岩岬岱岳岸峠峡峨峯峰島峻崇崎崖崩嵐嵩嵯嶋嶺巌川州巡巣工左巧巨差己巳巴巷巻巽巾市布帆希帖帝帥師席帯帰帳常帽幅幌幕幡幣干平年幸幹幻幼幽幾庁広庄庇床序底店庚府度座庫庭庵庶康庸廃廉廊廓廟廠延廷建廻廼廿弁弄弊式弐弓弔引弘弛弟弥弦弧弱張強弼弾彊当形彦彩彪彫彬彰影役彼往征径待律後徐徒従得御復循微徳徴徹徽心必忌忍志忘忙応忠快念忽怒怖怜思怠急性怨怪怯恋恐恒恕恢恥恨恩恭息恰恵悉悌悔悟悠患悦悩悪悲悶悼情惇惑惚惜惟惣惨惰想惹愁愉意愚愛感慈態慌慎慕慢慣慧慨慮慰慶憂憎憐憤憧憩憲憶憾懇懐懲懸戊戎成我戒或戚戟戦戯戴戸戻房所扇扉手才打払托扮扱扶批承技抄把抑投抗折抜択披抱抵抹押抽担拍拐拒拓拘拙招拝拠拡括拭拳拶拷拾持指按挑挙挟挨挫振挺挽挿捉捌捕捗捜捧捨据捲捷捻掃授掌排掘掛掠採探接控推掩措掲掴掻揃描提揖揚換握揮援揺損搬搭携搾摂摘摩摸摺撃撒撤撫播撮撰撲撹擁操擢擦擬擾支改攻放政故敏救敗教敢散敦敬数整敵敷文斉斌斎斐斑斗料斜斡斤斥斧斬断斯新方於施旅旋族旗既日旦旧旨早旬旭旺昂昆昇昌明昏易昔星映春昧昨昭是昼時晃晋晒晦晩普景晴晶智暁暇暑暖暗暢暦暫暮暴曇曙曜曝曲曳更書曹曽曾替最月有朋服朔朕朗望朝期木未末本札朱朴机朽杉李杏材村杖杜束条来杭杯東杵松板析枕林枚果枝枠枢枯架柄柊柏某柑染柔柘柚柱柳柴柵査柿栂栃栄栓栖栗校栢株核根格栽桁桂桃案桐桑桓桔桜桝桟桧桶梁梅梓梗梢梧梨梯械梱梶梼棄棋棒棚棟森棲棺椀椅椋植椎椙検椿楊楓楕楚楠楢業楯極楼楽概榊榎榛構槌槍様槙槻槽樋標樟模権横樫樵樹樺樽橋橘機橡橿檀檎櫓櫛欄欠次欣欧欲欺欽款歌歓止正此武歩歪歯歳歴死殆殉殊残殖殴段殺殻殿毅母毎毒比毘毛氏民気水氷永氾汀汁求汎汐汗汚汝江池汰汲決汽沃沈沌沓沖沙没沢沫河沸油治沼沿況泉泊泌法泡波泣泥注泰泳洋洗洛洞津洩洪洲活派流浄浅浜浦浩浪浬浮浴海浸消涌涙涛涜涯液涼淀淑淘淡淫深淳淵混添清渇済渉渋渓渚減渠渡渥渦温測港湊湖湘湛湧湯湾湿満源準溜溝溢溶溺滅滋滑滝滞滴漁漂漆漉漏演漕漠漢漣漫漬漸潔潜潟潤潮潰澄澗澱激濁濃濠濡濫濯瀕瀞瀧瀬灘火灯灰灸灼災炉炊炎炭点為烈烏焔焚無焦然焼煉煎煙煤照煩煮煽熊熟熱燃燈燐燕燥燦燭爆爪爵父爺爽爾片版牌牒牙牛牝牟牡牢牧物牲特牽犀犠犬犯状狂狐狗狙狛狩独狭狸狼猛猟猪猫献猶猷猿獄獅獣獲玄率玉王玖玩玲珂珊珍珠珪班現球理琉琢琳琴琵琶瑚瑛瑞瑠瑳璃環璽瓜瓢瓦瓶甘甚甜生産甥用甫田由甲申男町画界畏畑畔留畜畝畠畢略番異畳畷畿疋疎疏疑疫疲疹疾病症痕痘痛痢痩痴療癌癒癖発登白百的皆皇皐皮皿盃盆盈益盗盛盟監盤目盲直相盾省眉看県真眠眺眼着睡督睦瞬瞭瞳矛矢知矩短矯石砂研砕砦砧砲破砺硝硫硬碁碍碑碓碕碗碧碩確磁磐磨磯礁礎示礼社祁祇祈祉祐祖祝神祢祥票祭祷禁禄禅禍禎福禰禽禾禿秀私秋科秒秘租秤秦秩称移稀程税稔稗稚稜種稲稼稽稿穀穂穆積穏穣穫穴究空穿突窃窄窒窓窟窪窮窯窺立竜章竣童竪端競竹竺竿笈笑笛笠符第笹筆筈等筋筑筒答策箇箕算管箪箭箱箸節範篇築篠篤篭簡簾簿籍米粉粋粒粕粗粘粛粟粥粧精糊糖糞糟糠糧糸系糾紀約紅紋納紐純紗紘紙級紛素紡索紫累細紳紹紺終組経結絞絡絢給統絵絶絹継続綜綬維綱網綴綻綾綿緊緋総緑緒線締編緩緯練縁縄縛縞縦縫縮績繁繊繋繍織繕繭繰纂纏缶罪置罰署罵罷羅羊美群羨義羽翁翌習翠翰翻翼耀老考者而耐耕耗耳耶耽聖聞聡聯聴職聾肇肉肋肌肖肘肝股肢肥肩肪肯育肴肺胃胆背胎胞胡胤胴胸能脂脅脆脇脈脊脚脱脳脹腎腐腔腕腫腰腸腹腺腿膏膚膜膝膨膳膿臆臓臣臥臨自臭至致臼興舌舎舗舘舛舜舞舟航般舵舶舷船艇艦良色艶芋芙芝芥芦芭芯花芳芸芹芽苅苑苔苗苛若苦苫英茂茄茅茎茜茨茶草荊荏荒荘荷荻莞莫莱菅菊菌菓菖菜菟菩華菰菱萄萌萎萩萱落葉葎著葛葡董葦葬葱葵葺蒋蒐蒔蒙蒜蒲蒸蒼蓄蓉蓋蓑蓬蓮蔑蔓蔚蔦蔭蔵蔽蕃蕉蕊蕗蕨蕩蕪薄薙薦薩薪薫薬薮藁藍藤藩藻蘇蘭虎虐虚虜虞虫虹虻蚊蚕蛇蛋蛍蛭蛮蛸蜂蜘蜜蝉蝋蝕蝦蝶融螺蟹蟻血衆行術街衛衝衡衣表衰衷衿袈袋袖被袴裁裂装裏裕補裟裡裳裸製裾複褐褒襖襟襲西要覆覇見規視覗覚覧親観角解触言訂計訊討訓託記訟訣訪設許訳訴診註証詐詔評詞詠詣試詩詫詮詰話該詳誇誉誌認誓誕誘語誠誤説読誰課誹誼調談請諌諏諒論諜諦諭諮諸諺諾謀謁謂謎謙講謝謡謬謹識譜警議譲護讃讐谷豆豊豚象豪豹貌貝貞負財貢貧貨販貫責貯貰貴買貸費貼貿賀賂賃賄資賊賑賓賛賜賞賠賢賦質賭購贈贋赤赦赫走赴起超越趣趨足距跡跨路跳践踊踏蹄蹟蹴躍身躯車軌軍軒軟転軸軽較載輔輝輩輪輯輸輿轄轍轟辛辞辰辱農辺辻込辿迂迄迅迎近返迦迫迭述迷追退送逃逆透逐逓途逗這通逝速造逢連逮週進逸逼遁遂遅遇遊運遍過道達違遜遠遡遣遥適遭遮遵遷選遺遼避還邑那邦邪邸郁郊郎郡部郭郵郷都鄭酉酋酌配酎酒酔酢酪酬酵酷酸醇醍醐醒醜醤醸采釈里重野量金釘釜針釣釧鈍鈴鈷鉄鉛鉢鉱鉾銀銃銅銑銘銚銭鋒鋪鋭鋲鋳鋸鋼錆錐錘錠錦錨錫錬錯録鍋鍔鍛鍬鍵鍾鎌鎖鎗鎚鎧鎮鏑鏡鐘鐸鑑鑓長門閃閉開閏閑間関閣閤閥閲闇闘阜阪防阻阿陀附降限陛院陣除陥陪陰陳陵陶陸険陽隅隆隈隊階随隔隙際障隠隣隷隻隼雀雁雄雅集雇雌雑雛離難雨雪雫雰雲零雷電需震霊霜霞霧露青靖静非面革靭靴鞄鞍鞘鞠鞭韓韮音韻響頁頂頃項順須預頑頒頓頗領頚頬頭頴頻頼題額顎顔顕願顛類顧風飛食飢飯飲飴飼飽飾餅養餌餐餓館饗首香馨馬馳馴駁駄駅駆駈駐駒駕駿騎騒験騨騰驚骨骸髄高髪髭鬼魁魂魅魔魚魯鮎鮒鮫鮭鮮鯉鯖鯨鰐鰭鰻鱒鱗鳥鳩鳳鳴鳶鴎鴛鴨鴻鵜鵠鵡鵬鶏鶴鷲鷹鷺鹸鹿麓麗麟麦麹麺麻麿黄黒黙黛鼎鼓鼠鼻齢龍'
        },
        'arabic_g1': {
            'filename': 'arabic.pth',
            'model_script': 'arabic',
            'url': 'https://github.com/erfanzar/OCR-FA/releases/download/pre-v1.1.6/arabic.zip',
            'md5sum': '993074555550e4e06a6077d55ff0449a',
            'symbols': '«»؟،؛٠١٢٣٤٥٦٧٨٩' + '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ ',
            'characters': '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ٠١٢٣٤٥٦٧٨٩«»؟،؛ءآأؤإئااًبةتثجحخدذرزسشصضطظعغفقكلمنهوىيًٌٍَُِّْٰٓٔٱٹپچڈڑژکڭگںھۀہۂۃۆۇۈۋیېےۓە'
        },

    },
    'gen2': {
        'english_g2': {
            'filename': 'english_g2.pth',
            'model_script': 'english',
            'url': 'https://github.com/erfanzar/OCR-FA/releases/download/v1.3/english_g2.zip',
            'md5sum': '5864788e1821be9e454ec108d61b887d',
            'symbols': "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €",
            'characters': "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ €ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
        },
        'latin_g2': {
            'filename': 'latin_g2.pth',
            'model_script': 'latin',
            'url': 'https://github.com/erfanzar/OCR-FA/releases/download/v1.3/latin_g2.zip',
            'md5sum': '469869130aad1a34e8f9086f4262bc59',
            'symbols': " !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`{|}~ €",
            'characters': " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ªÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿĀāĂăĄąĆćČčĎďĐđĒēĖėĘęĚěĞğĨĩĪīĮįİıĶķĹĺĻļĽľŁłŃńŅņŇňŒœŔŕŘřŚśŞşŠšŤťŨũŪūŮůŲųŸŹźŻżŽžƏƠơƯưȘșȚțə̇ḌḍḶḷṀṁṂṃṄṅṆṇṬṭẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ€"
        },

    }
}


def custom_mean(x):
    return x.prod() ** (2.0 / np.sqrt(len(x)))


def contrast_grey(img):
    high = np.percentile(img, 90)
    low = np.percentile(img, 10)
    return (high - low) / np.maximum(10, high + low), high, low


def adjust_contrast_grey(img, target=0.4):
    contrast, high, low = contrast_grey(img)
    if contrast < target:
        img = img.astype(int)
        ratio = 200. / np.maximum(10, high - low)
        img = (img - low + 25) * ratio
        img = np.maximum(np.full(img.shape, 0), np.minimum(np.full(img.shape, 255), img)).astype(np.uint8)
    return img


def loadImage(img_file):
    img = io.imread(img_file)
    if img.shape[0] == 2:
        img = img[0]
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.array(img)

    return img


def normalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    img = in_img.copy().astype(np.float32)

    img -= np.array([mean[0] * 255.0, mean[1] * 255.0, mean[2] * 255.0], dtype=np.float32)
    img /= np.array([variance[0] * 255.0, variance[1] * 255.0, variance[2] * 255.0], dtype=np.float32)
    return img


def denormalize_mean_variance(in_img, mean=(0.485, 0.456, 0.406), variance=(0.229, 0.224, 0.225)):
    # should be RGB order
    img = in_img.copy()
    img *= variance
    img += mean
    img *= 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def resize_aspect_ratio(img, square_size, interpolation, mag_ratio=1):
    height, width, channel = img.shape

    # magnify image size
    target_size = mag_ratio * max(height, width)

    # set original image size
    if target_size > square_size:
        target_size = square_size

    ratio = target_size / max(height, width)

    target_h, target_w = int(height * ratio), int(width * ratio)
    proc = cv2.resize(img, (target_w, target_h), interpolation=interpolation)

    # make canvas and paste image
    target_h32, target_w32 = target_h, target_w
    if target_h % 32 != 0:
        target_h32 = target_h + (32 - target_h % 32)
    if target_w % 32 != 0:
        target_w32 = target_w + (32 - target_w % 32)
    resized = np.zeros((target_h32, target_w32, channel), dtype=np.float32)
    resized[0:target_h, 0:target_w, :] = proc
    target_h, target_w = target_h32, target_w32

    size_heatmap = (int(target_w / 2), int(target_h / 2))

    return resized, ratio, size_heatmap


def cvt2_heatmap_img(img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    return img


class NormalizePAD:

    def __init__(self, max_size, pad_type="right"):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = pad_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img
        if self.max_size[2] != w:
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)

        return Pad_img


class ListDataset(torch.utils.data.Dataset):

    def __init__(self, image_list):
        self.image_list = image_list
        self.nSamples = len(image_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img = self.image_list[index]
        return Image.fromarray(img, "L")


class AlignCollate:

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False, adjust_contrast=0.):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad
        self.adjust_contrast = adjust_contrast

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        images = batch

        resized_max_w = self.imgW
        input_channel = 1
        transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

        resized_images = []
        for image in images:
            w, h = image.size
            #### augmentation here - change contrast
            if self.adjust_contrast > 0:
                image = np.array(image.convert("L"))
                image = adjust_contrast_grey(image, target=self.adjust_contrast)
                image = Image.fromarray(image, "L")

            ratio = w / float(h)
            if math.ceil(self.imgH * ratio) > self.imgW:
                resized_w = self.imgW
            else:
                resized_w = math.ceil(self.imgH * ratio)

            resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
            resized_images.append(transform(resized_image))

        image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)
        return image_tensors


def recognizer_predict(
        model,
        converter,
        test_loader,
        batch_max_length,
        ignore_idx,
        char_group_idx,
        decoder="greedy",
        beamWidth=5,
        device="cpu"
):
    model.eval()
    result = []
    with torch.no_grad():
        for image_tensors in test_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            length_for_pred = torch.IntTensor([batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)

            preds = model(image, text_for_pred)

            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds_prob = F.softmax(preds, dim=2)
            preds_prob = preds_prob.cpu().detach().numpy()
            preds_prob[:, :, ignore_idx] = 0.
            pred_norm = preds_prob.sum(axis=2)
            preds_prob = preds_prob / np.expand_dims(pred_norm, axis=-1)
            preds_prob = torch.from_numpy(preds_prob).float().to(device)

            if decoder == "greedy":
                # Select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds_prob.max(2)
                preds_index = preds_index.view(-1)
                preds_str = converter.decode_greedy(preds_index.data.cpu().detach().numpy(), preds_size.data)
            elif decoder == "beamsearch":
                k = preds_prob.cpu().detach().numpy()
                preds_str = converter.decode_beamsearch(k, beamWidth=beamWidth)
            elif decoder == "wordbeamsearch":
                k = preds_prob.cpu().detach().numpy()
                preds_str = converter.decode_wordbeamsearch(k, beamWidth=beamWidth)

            preds_prob = preds_prob.cpu().detach().numpy()
            values = preds_prob.max(axis=2)
            indices = preds_prob.argmax(axis=2)
            preds_max_prob = []
            for v, i in zip(values, indices):
                max_probs = v[i != 0]
                if len(max_probs) > 0:
                    preds_max_prob.append(max_probs)
                else:
                    preds_max_prob.append(np.array([0]))

            for pred, pred_max_prob in zip(preds_str, preds_max_prob):
                confidence_score = custom_mean(pred_max_prob)
                result.append([pred, confidence_score])

    return result


def get_recognizer(recog_network, network_params, character, \
                   separator_list, dict_list, model_path, \
                   device="cpu", quantize=True):
    converter = CTCLabelConverter(character, separator_list, dict_list)
    num_class = len(converter.character)

    if recog_network == "generation1":
        from .modules import Model as Model
    elif recog_network == "generation2":
        from .modules import VGGModel as Model
    else:
        Model = importlib.import_module(recog_network).Model
    model = Model(num_class=num_class, **network_params)

    if device == "cpu":
        state_dict = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            new_key = key[7:]
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict)
        if quantize:
            try:
                torch.quantization.quantize_dynamic(model, dtype=torch.qint8, inplace=True)
            except:
                pass
    else:
        model = torch.nn.DataParallel(model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

    return model, converter


def get_text(
        character,
        imgH,
        imgW,
        recognizer,
        converter, image_list,
        ignore_char='',
        decoder="greedy",
        beamWidth=5,
        batch_size=1,
        contrast_ths=0.1,
        adjust_contrast=0.5,
        filter_ths=0.003,
        workers=1,
        device="cpu"
):
    batch_max_length = int(imgW / 10)

    char_group_idx = {}
    ignore_idx = []
    for char in ignore_char:
        try:
            ignore_idx.append(character.index(char) + 1)
        except:
            pass

    coord = [item[0] for item in image_list]
    img_list = [item[1] for item in image_list]
    AlignCollate_normal = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True)
    test_data = ListDataset(img_list)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False,
        num_workers=int(workers), collate_fn=AlignCollate_normal, pin_memory=True)

    # predict first round
    result1 = recognizer_predict(recognizer, converter, test_loader, batch_max_length, \
                                 ignore_idx, char_group_idx, decoder, beamWidth, device=device)

    # predict second round
    low_confident_idx = [i for i, item in enumerate(result1) if (item[1] < contrast_ths)]
    if len(low_confident_idx) > 0:
        img_list2 = [img_list[i] for i in low_confident_idx]
        AlignCollate_contrast = AlignCollate(imgH=imgH, imgW=imgW, keep_ratio_with_pad=True,
                                             adjust_contrast=adjust_contrast)
        test_data = ListDataset(img_list2)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False,
            num_workers=int(workers), collate_fn=AlignCollate_contrast, pin_memory=True)
        result2 = recognizer_predict(
            recognizer,
            converter,
            test_loader,
            batch_max_length,
            ignore_idx,
            char_group_idx,
            decoder,
            beamWidth,
            device=device
        )

    result = []
    for i, zipped in enumerate(zip(coord, result1)):
        box, pred1 = zipped
        if i in low_confident_idx:
            pred2 = result2[low_confident_idx.index(i)]
            if pred1[1] > pred2[1]:
                result.append((box, pred1[0], pred1[1]))
            else:
                result.append((box, pred2[0], pred2[1]))
        else:
            result.append((box, pred1[0], pred1[1]))

    return result


def consecutive(data, mode="first", stepsize=1):
    result = None
    group = np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)
    group = [item for item in group if len(item) > 0]

    if mode == "first":
        result = [l[0] for l in group]
    elif mode == "last":
        result = [l[-1] for l in group]
    return result


def word_segmentation(mat, separator_idx=None, separator_idx_list=None):
    if separator_idx_list is None:
        separator_idx_list = [1, 2, 3, 4]
    if separator_idx is None:
        separator_idx = {"th": [1, 2], "en": [3, 4]}
    result = []
    sep_list = []
    start_idx = 0
    sep_lang = ''
    for sep_idx in separator_idx_list:
        if sep_idx % 2 == 0:
            mode = "first"
        else:
            mode = "last"
        a = consecutive(np.argwhere(mat == sep_idx).flatten(), mode)
        new_sep = [[item, sep_idx] for item in a]
        sep_list += new_sep
    sep_list = sorted(sep_list, key=lambda x: x[0])

    for sep in sep_list:
        for lang in separator_idx.keys():
            if sep[1] == separator_idx[lang][0]:  # start lang
                sep_lang = lang
                sep_start_idx = sep[0]
            elif sep[1] == separator_idx[lang][1]:  # end lang
                if sep_lang == lang:  # check if last entry if the same start lang
                    new_sep_pair = [lang, [sep_start_idx + 1, sep[0] - 1]]
                    if sep_start_idx > start_idx:
                        result.append(['', [start_idx, sep_start_idx - 1]])
                    start_idx = sep[0] + 1
                    result.append(new_sep_pair)
                sep_lang = ''  # reset

    if start_idx <= len(mat) - 1:
        result.append(['', [start_idx, len(mat) - 1]])
    return result


# code is based from https://github.com/githubharald/CTCDecoder/blob/master/src/BeamSearch.py
class BeamEntry:
    "information about one single beam at specific time-step"

    def __init__(self):
        self.prTotal = 0
        self.prNonBlank = 0
        self.prBlank = 0
        self.prText = 1
        self.lmApplied = False
        self.labeling = ()
        self.simplified = True


class BeamState:
    def __init__(self):
        self.entries = {}

    def norm(self):
        "length-normalise LM score"
        for (k, _) in self.entries.items():
            labelingLen = len(self.entries[k].labeling)
            self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

    def sort(self):
        "return beam-labelings, sorted by probability"
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal * x.prText)
        return [x.labeling for x in sortedBeams]

    def wordsearch(self, classes, ignore_idx, maxCandidate, dict_list):
        best_text = None
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal * x.prText)
        if len(sortedBeams) > maxCandidate: sortedBeams = sortedBeams[:maxCandidate]

        for j, candidate in enumerate(sortedBeams):
            idx_list = candidate.labeling
            text = ''
            for i, l in enumerate(idx_list):
                if l not in ignore_idx and (not (i > 0 and idx_list[i - 1] == idx_list[i])):
                    text += classes[l]

            if j == 0: best_text = text
            if text in dict_list:
                best_text = text
                break
            else:
                ...
        return best_text


def applyLM(parentBeam, childBeam, classes, lm):
    "calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
    if lm and not childBeam.lmApplied:
        c1 = classes[parentBeam.labeling[-1] if parentBeam.labeling else classes.index(" ")]
        c2 = classes[childBeam.labeling[-1]]
        lmFactor = 0.01
        bigramProb = lm.getCharBigram(c1, c2) ** lmFactor
        childBeam.prText = parentBeam.prText * bigramProb
        childBeam.lmApplied = True


def simplify_label(labeling, blankIdx=0):
    labeling = np.array(labeling)

    # collapse blank
    idx = np.where(~((np.roll(labeling, 1) == labeling) & (labeling == blankIdx)))[0]
    labeling = labeling[idx]

    # get rid of blank between different characters
    idx = np.where(~((np.roll(labeling, 1) != np.roll(labeling, -1)) & (labeling == blankIdx)))[0]

    if len(labeling) > 0:
        last_idx = len(labeling) - 1
        if last_idx not in idx: idx = np.append(idx, [last_idx])
    labeling = labeling[idx]

    return tuple(labeling)


def fast_simplify_label(labeling, c, blankIdx=0):
    # Adding BlankIDX after Non-Blank IDX
    if labeling and c == blankIdx and labeling[-1] != blankIdx:
        newLabeling = labeling + (c,)

    # Case when a nonBlankChar is added after BlankChar |len(char) - 1
    elif labeling and c != blankIdx and labeling[-1] == blankIdx:

        # If Blank between same character do nothing | As done by Simplify label
        if labeling[-2] == c:
            newLabeling = labeling + (c,)

        # if blank between different character, remove it | As done by Simplify Label
        else:
            newLabeling = labeling[:-1] + (c,)

    # if consecutive blanks : Keep the original label
    elif labeling and c == blankIdx and labeling[-1] == blankIdx:
        newLabeling = labeling

    # if empty beam & first index is blank
    elif not labeling and c == blankIdx:
        newLabeling = labeling

    # if empty beam & first index is non-blank
    elif not labeling and c != blankIdx:
        newLabeling = labeling + (c,)

    elif labeling and c != blankIdx:
        newLabeling = labeling + (c,)

    # Cases that might still require simplyfying
    else:
        newLabeling = labeling + (c,)
        newLabeling = simplify_label(newLabeling, blankIdx)

    return newLabeling


def addBeam(beamState, labeling):
    "add beam if it does not yet exist"
    if labeling not in beamState.entries:
        beamState.entries[labeling] = BeamEntry()


def ctcBeamSearch(mat, classes, ignore_idx, lm, beamWidth=25, dict_list=[]):
    blankIdx = 0
    maxT, maxC = mat.shape

    # initialise beam state
    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].prBlank = 1
    last.entries[labeling].prTotal = 1

    # go over all time-steps
    for t in range(maxT):
        curr = BeamState()
        # get beam-labelings of best beams
        bestLabelings = last.sort()[0:beamWidth]
        # go over best beams
        for labeling in bestLabelings:
            # probability of paths ending with a non-blank
            prNonBlank = 0
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

            # probability of paths ending with a blank
            prBlank = (last.entries[labeling].prTotal) * mat[t, blankIdx]

            # add beam at current time-step if needed
            prev_labeling = labeling
            if not last.entries[labeling].simplified:
                labeling = simplify_label(labeling, blankIdx)

            # labeling = simplify_label(labeling, blankIdx)
            addBeam(curr, labeling)

            # fill in data
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].prNonBlank += prNonBlank
            curr.entries[labeling].prBlank += prBlank
            curr.entries[labeling].prTotal += prBlank + prNonBlank
            curr.entries[labeling].prText = last.entries[prev_labeling].prText
            # beam-labeling not changed, therefore also LM score unchanged from

            # curr.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling

            # extend current beam-labeling
            # char_highscore = np.argpartition(mat[t, :], -5)[-5:] # run through 5 highest probability
            char_highscore = np.where(mat[t, :] >= 0.5 / maxC)[0]  # run through all probable characters
            for c in char_highscore:
                # for c in range(maxC - 1):
                # add new char to current beam-labeling
                # newLabeling = labeling + (c,)
                # newLabeling = simplify_label(newLabeling, blankIdx)
                newLabeling = fast_simplify_label(labeling, c, blankIdx)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    prNonBlank = mat[t, c] * last.entries[prev_labeling].prBlank
                else:
                    prNonBlank = mat[t, c] * last.entries[prev_labeling].prTotal

                # add beam at current time-step if needed
                addBeam(curr, newLabeling)

                # fill in data
                curr.entries[newLabeling].labeling = newLabeling
                curr.entries[newLabeling].prNonBlank += prNonBlank
                curr.entries[newLabeling].prTotal += prNonBlank

                # apply LM
                # applyLM(curr.entries[labeling], curr.entries[newLabeling], classes, lm)

        # set new beam state

        last = curr

    # normalise LM scores according to beam-labeling-length
    last.norm()

    if dict_list == []:
        bestLabeling = last.sort()[0]  # get most probable labeling
        res = ''
        for i, l in enumerate(bestLabeling):
            # removing repeated characters and blank.
            if l not in ignore_idx and (not (i > 0 and bestLabeling[i - 1] == bestLabeling[i])):
                res += classes[l]
    else:
        res = last.wordsearch(classes, ignore_idx, 20, dict_list)
    return res


class CTCLabelConverter:
    """ Convert between text-label and text-index """

    def __init__(self, character, separator_list=None, dict_pathlist=None):
        # character (str): set of the possible characters.
        if dict_pathlist is None:
            dict_pathlist = {}
        if separator_list is None:
            separator_list = {}
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        self.character = ["[blank]"] + dict_character  # dummy "[blank]" token for CTCLoss (index 0)

        self.separator_list = separator_list
        separator_char = []
        for lang, sep in separator_list.items():
            separator_char += sep
        self.ignore_idx = [0] + [i + 1 for i, item in enumerate(separator_char)]

        # TODO: latin dict Fix she.
        if len(separator_list) == 0:
            dict_list = []
            for lang, dict_path in dict_pathlist.items():
                try:
                    with open(dict_path, "r", encoding="utf-8-sig") as input_file:
                        word_count = input_file.read().splitlines()
                    dict_list += word_count
                except:
                    pass
        else:
            dict_list = {}
            for lang, dict_path in dict_pathlist.items():
                with open(dict_path, "r", encoding="utf-8-sig") as input_file:
                    word_count = input_file.read().splitlines()
                dict_list[lang] = word_count

        self.dict_list = dict_list

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]

        output:
            text: concatenated text index for CTCLoss.
                    [sum(text_lengths)] = [text_index_0 + text_index_1 + ... + text_index_(n - 1)]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]
        text = ''.join(text)
        text = [self.dict[char] for char in text]

        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode_greedy(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        for l in length:
            t = text_index[index:index + l]
            # Returns a boolean array where true is when the value is not repeated
            a = np.insert(~((t[1:] == t[:-1])), 0, True)
            # Returns a boolean array where true is when the value is not in the ignore_idx list
            b = ~np.isin(t, np.array(self.ignore_idx))
            # Combine the two boolean array
            c = a & b
            # Gets the corresponding character according to the saved indexes
            text = ''.join(np.array(self.character)[t[c.nonzero()]])
            texts.append(text)
            index += l
        return texts

    def decode_beamsearch(self, mat, beamWidth=5):
        texts = []
        for i in range(mat.shape[0]):
            t = ctcBeamSearch(mat[i], self.character, self.ignore_idx, None, beamWidth=beamWidth)
            texts.append(t)
        return texts

    def decode_wordbeamsearch(self, mat, beamWidth=5):
        texts = []
        argmax = np.argmax(mat, axis=2)

        for i in range(mat.shape[0]):
            string = ''

            if len(self.separator_list) == 0:
                space_idx = self.dict[" "]

                data = np.argwhere(argmax[i] != space_idx).flatten()
                group = np.split(data, np.where(np.diff(data) != 1)[0] + 1)
                group = [list(item) for item in group if len(item) > 0]

                for j, list_idx in enumerate(group):
                    matrix = mat[i, list_idx, :]
                    t = ctcBeamSearch(
                        matrix,
                        self.character,
                        self.ignore_idx,
                        None,
                        beamWidth=beamWidth,
                        dict_list=self.dict_list
                    )
                    if j == 0:
                        string += t
                    else:
                        string += " " + t

            # with separators
            else:
                words = word_segmentation(argmax[i])

                for word in words:
                    matrix = mat[i, word[1][0]:word[1][1] + 1, :]
                    if word[0] == '':
                        dict_list = []
                    else:
                        dict_list = self.dict_list[word[0]]
                    t = ctcBeamSearch(
                        matrix,
                        self.character,
                        self.ignore_idx,
                        None,
                        beamWidth=beamWidth,
                        dict_list=dict_list
                    )
                    string += t
            texts.append(string)
        return texts


def merge_to_free(merge_result, free_list):
    merge_result_buf, mr_buf = [], []

    if not free_list:
        return merge_result

    free_list_buf = merge_result[-len(free_list):]
    merge_result = merge_result[:-len(free_list)]

    for idx, r in enumerate(merge_result):
        if idx == len(merge_result) - 1:
            mr_buf.append(r)
            merge_result_buf.append(mr_buf)
            mr_buf = []
            continue

        if (mr_buf == []) or (mr_buf[-1][0] < r[0]):
            mr_buf.append(r)
        else:
            merge_result_buf.append(mr_buf)
            mr_buf = [r]

    for free_pos in free_list_buf:
        y_pos = len(merge_result_buf)
        x_pos = len(merge_result_buf[y_pos - 1])
        for i, result_pos in enumerate(merge_result_buf[1:]):
            if free_pos[0][0][1] < result_pos[0][0][0][1]:
                y_pos = i
                break

        for i, result_pos in enumerate(merge_result_buf[y_pos]):
            if free_pos[0][0][0] < result_pos[0][0][0]:
                x_pos = i
                break

        merge_result_buf[y_pos].insert(x_pos, free_pos)

    merge_result = []
    [merge_result.extend(r) for r in merge_result_buf]
    return merge_result


def four_point_transform(image, rect):
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_bs = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    mw = max(int(width_a), int(width_bs))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_bs = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    mh = max(int(height_a), int(height_bs))

    dst = np.array([[0, 0], [mw - 1, 0], [mw - 1, mh - 1], [0, mh - 1]], dtype="float32")

    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(rect, dst), (mw, mh))


# TODO: Inja Fix 
def group_text_box(
        polys,
        slope_ths=0.1,
        ycenter_ths=0.5,
        height_ths=0.5,
        width_ths=1.0,
        add_margin=0.05,
        sort_output=True
):
    horizontal_list, free_list, combined_list, merged_list = [], [], [], []

    for poly in polys:
        slope_up = (poly[3] - poly[1]) / np.maximum(10, (poly[2] - poly[0]))
        slope_down = (poly[5] - poly[7]) / np.maximum(10, (poly[4] - poly[6]))
        if max(abs(slope_up), abs(slope_down)) < slope_ths:
            x_max = max([poly[0], poly[2], poly[4], poly[6]])
            x_min = min([poly[0], poly[2], poly[4], poly[6]])
            y_max = max([poly[1], poly[3], poly[5], poly[7]])
            y_min = min([poly[1], poly[3], poly[5], poly[7]])
            horizontal_list.append([x_min, x_max, y_min, y_max, 0.5 * (y_min + y_max), y_max - y_min])
        else:
            height = np.linalg.norm([poly[6] - poly[0], poly[7] - poly[1]])
            width = np.linalg.norm([poly[2] - poly[0], poly[3] - poly[1]])

            margin = int(1.44 * add_margin * min(width, height))

            theta13 = abs(np.arctan((poly[1] - poly[5]) / np.maximum(10, (poly[0] - poly[4]))))
            theta24 = abs(np.arctan((poly[3] - poly[7]) / np.maximum(10, (poly[2] - poly[6]))))
            # do I need to clip minimum, maximum value here?
            x1 = poly[0] - np.cos(theta13) * margin
            y1 = poly[1] - np.sin(theta13) * margin
            x2 = poly[2] + np.cos(theta24) * margin
            y2 = poly[3] - np.sin(theta24) * margin
            x3 = poly[4] + np.cos(theta13) * margin
            y3 = poly[5] + np.sin(theta13) * margin
            x4 = poly[6] - np.cos(theta24) * margin
            y4 = poly[7] + np.sin(theta24) * margin

            free_list.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    if sort_output:
        horizontal_list = sorted(horizontal_list, key=lambda item: item[4])

    # combine box
    new_box = []
    for poly in horizontal_list:

        if len(new_box) == 0:
            b_height = [poly[5]]
            b_ycenter = [poly[4]]
            new_box.append(poly)
        else:
            # comparable height and comparable y_center level up to ths*height
            if abs(np.mean(b_ycenter) - poly[4]) < ycenter_ths * np.mean(b_height):
                b_height.append(poly[5])
                b_ycenter.append(poly[4])
                new_box.append(poly)
            else:
                b_height = [poly[5]]
                b_ycenter = [poly[4]]
                combined_list.append(new_box)
                new_box = [poly]
    combined_list.append(new_box)

    # merge list use sort again
    for boxes in combined_list:
        if len(boxes) == 1:  # one box per line
            box = boxes[0]
            margin = int(add_margin * min(box[1] - box[0], box[5]))
            merged_list.append([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin])
        else:  # multiple boxes per line
            boxes = sorted(boxes, key=lambda item: item[0])

            merged_box, new_box = [], []
            for box in boxes:
                if len(new_box) == 0:
                    b_height = [box[5]]
                    x_max = box[1]
                    new_box.append(box)
                else:
                    if (abs(np.mean(b_height) - box[5]) < height_ths * np.mean(b_height)) and (
                            (box[0] - x_max) < width_ths * (box[3] - box[2])):  # merge boxes
                        b_height.append(box[5])
                        x_max = box[1]
                        new_box.append(box)
                    else:
                        b_height = [box[5]]
                        x_max = box[1]
                        merged_box.append(new_box)
                        new_box = [box]
            if len(new_box) > 0: merged_box.append(new_box)

            for mbox in merged_box:
                if len(mbox) != 1:  # adjacent box in same line
                    # do I need to add margin here?
                    x_min = min(mbox, key=lambda x: x[0])[0]
                    x_max = max(mbox, key=lambda x: x[1])[1]
                    y_min = min(mbox, key=lambda x: x[2])[2]
                    y_max = max(mbox, key=lambda x: x[3])[3]

                    box_width = x_max - x_min
                    box_height = y_max - y_min
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append([x_min - margin, x_max + margin, y_min - margin, y_max + margin])
                else:  # non adjacent box in same line
                    box = mbox[0]

                    box_width = box[1] - box[0]
                    box_height = box[3] - box[2]
                    margin = int(add_margin * (min(box_width, box_height)))

                    merged_list.append([box[0] - margin, box[1] + margin, box[2] - margin, box[3] + margin])
    # may need to check if box is really in image
    return merged_list, free_list


def calculate_ratio(width, height):
    """
    Calculate aspect ratio for normal use case (w>h) and vertical text (h>w)
    """
    ratio = width / height
    if ratio < 1.0:
        ratio = 1. / ratio
    return ratio


def compute_ratio_and_resize(img, width, height, model_height):
    """
    Calculate ratio and resize correctly for both horizontal text
    and vertical case
    """
    ratio = width / height
    if ratio < 1.0:
        ratio = calculate_ratio(width, height)
        img = cv2.resize(img, (model_height, int(model_height * ratio)), interpolation=Image.Resampling.LANCZOS)
    else:
        img = cv2.resize(img, (int(model_height * ratio), model_height), interpolation=Image.Resampling.LANCZOS)
    return img, ratio


def get_image_list(horizontal_list, free_list, img, model_height=64, sort_output=True):
    image_list = []
    maximum_y, maximum_x = img.shape

    max_ratio_hori, max_ratio_free = 1, 1
    for box in free_list:
        rect = np.array(box, dtype="float32")
        transformed_img = four_point_transform(img, rect)
        ratio = calculate_ratio(transformed_img.shape[1], transformed_img.shape[0])
        new_width = int(model_height * ratio)
        if new_width == 0:
            pass
        else:
            crop_img, ratio = compute_ratio_and_resize(transformed_img, transformed_img.shape[1],
                                                       transformed_img.shape[0], model_height)
            image_list.append((box, crop_img))  # box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            max_ratio_free = max(ratio, max_ratio_free)

    max_ratio_free = math.ceil(max_ratio_free)

    for box in horizontal_list:
        x_min = max(0, box[0])
        x_max = min(box[1], maximum_x)
        y_min = max(0, box[2])
        y_max = min(box[3], maximum_y)
        crop_img = img[y_min: y_max, x_min:x_max]
        width = x_max - x_min
        height = y_max - y_min
        ratio = calculate_ratio(width, height)
        new_width = int(model_height * ratio)
        if new_width == 0:
            pass
        else:
            crop_img, ratio = compute_ratio_and_resize(crop_img, width, height, model_height)
            image_list.append(([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], crop_img))
            max_ratio_hori = max(ratio, max_ratio_hori)

    max_ratio_hori = math.ceil(max_ratio_hori)
    max_ratio = max(max_ratio_hori, max_ratio_free)
    max_width = math.ceil(max_ratio) * model_height

    if sort_output:
        image_list = sorted(image_list, key=lambda item: item[0][0][1])  # sort by vertical position
    return image_list, max_width


def download_and_unzip(url, filename, model_storage_directory, verbose=True):
    zip_path = os.path.join(model_storage_directory, "temp.zip")
    reporthook = print_custom(prefix="Progress:", suffix="Complete", length=50) if verbose else None
    urlretrieve(url, zip_path, reporthook=reporthook)
    with ZipFile(zip_path, "r") as zipObj:
        zipObj.extract(filename, model_storage_directory)
    os.remove(zip_path)


def calculate_md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def diff(input_list):
    return max(input_list) - min(input_list)


def get_paragraph(raw_result, x_ths=1, y_ths=0.5, mode="ltr"):
    # create basic attributes
    box_group = []
    for box in raw_result:
        all_x = [int(coord[0]) for coord in box[0]]
        all_y = [int(coord[1]) for coord in box[0]]
        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)
        height = max_y - min_y
        box_group.append(
            [box[1], min_x, max_x, min_y, max_y, height, 0.5 * (min_y + max_y), 0])  # last element indicates group
    # cluster boxes into paragraph
    current_group = 1
    while len([box for box in box_group if box[7] == 0]) > 0:
        box_group0 = [box for box in box_group if box[7] == 0]  # group0 = non-group
        # new group
        if len([box for box in box_group if box[7] == current_group]) == 0:
            box_group0[0][7] = current_group  # assign first box to form new group
        # try to add group
        else:
            current_box_group = [box for box in box_group if box[7] == current_group]
            mean_height = np.mean([box[5] for box in current_box_group])
            min_gx = min([box[1] for box in current_box_group]) - x_ths * mean_height
            max_gx = max([box[2] for box in current_box_group]) + x_ths * mean_height
            min_gy = min([box[3] for box in current_box_group]) - y_ths * mean_height
            max_gy = max([box[4] for box in current_box_group]) + y_ths * mean_height
            add_box = False
            for box in box_group0:
                same_horizontal_level = (min_gx <= box[1] <= max_gx) or (min_gx <= box[2] <= max_gx)
                same_vertical_level = (min_gy <= box[3] <= max_gy) or (min_gy <= box[4] <= max_gy)
                if same_horizontal_level and same_vertical_level:
                    box[7] = current_group
                    add_box = True
                    break
            # cannot add more box, go to next group
            if add_box == False:
                current_group += 1
    # arrage order in paragraph
    result = []
    for i in set(box[7] for box in box_group):
        current_box_group = [box for box in box_group if box[7] == i]
        mean_height = np.mean([box[5] for box in current_box_group])
        min_gx = min([box[1] for box in current_box_group])
        max_gx = max([box[2] for box in current_box_group])
        min_gy = min([box[3] for box in current_box_group])
        max_gy = max([box[4] for box in current_box_group])

        text = ''
        while len(current_box_group) > 0:
            highest = min([box[6] for box in current_box_group])
            candidates = [box for box in current_box_group if box[6] < highest + 0.4 * mean_height]
            # get the far left
            if mode == "ltr":
                most_left = min([box[1] for box in candidates])
                for box in candidates:
                    if box[1] == most_left: best_box = box
            elif mode == "rtl":
                most_right = max([box[2] for box in candidates])
                for box in candidates:
                    if box[2] == most_right: best_box = box
            text += " " + best_box[0]
            current_box_group.remove(best_box)

        result.append([[[min_gx, min_gy], [max_gx, min_gy], [max_gx, max_gy], [min_gx, max_gy]], text[1:]])

    return result


def print_custom(prefix='', suffix='', decimals=1, length=100, fill="#"):
    def progress_hook(count, blockSize, totalSize):
        progress = count * blockSize / totalSize
        state = progress * 100
        state = state if state < 100 else 100
        percent = ("{0:." + str(decimals) + "f}").format(state)
        fl = int(length * progress)
        bar = fill * fl + " " * (length - fl)
        print(f"\r{'TBAR'} |{bar}| {percent}% {suffix}", end='')

    return progress_hook


def reformat_input(image):
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            tmp, _ = urlretrieve(image, reporthook=print_custom(prefix="Progress:", suffix="Complete", length=50))
            img_cv_grey = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
            os.remove(tmp)
        else:
            img_cv_grey = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            image = os.path.expanduser(image)
        img = loadImage(image)  # can accept URL
        return img, img_cv_grey
    elif isinstance(image, bytes):
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, img_cv_grey
    elif isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            img_cv_grey = image
            img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            img_cv_grey = np.squeeze(image)
            img = cv2.cvtColor(img_cv_grey, cv2.COLOR_GRAY2BGR)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            img = image
            img_cv_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            img = image[:, :, :3]
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError()
        return img, img_cv_grey
    elif isinstance(image, JpegImagePlugin.JpegImageFile):
        image_array = np.array(image)
        img = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        img_cv_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, img_cv_grey
    else:
        raise ValueError("Invalid input type. Supporting format = string(file path or url), bytes, numpy array")


def reformat_input_batched(image, n_width=None, n_height=None):
    """
    reformats an image or list of images or a 4D numpy image array &
    returns a list of corresponding img, img_cv_grey nd.arrays
    image:
        [file path, numpy-array, byte stream object,
        list of file paths, list of numpy-array, 4D numpy array,
        list of byte stream objects]
    """
    if (isinstance(image, np.ndarray) and len(image.shape) == 4) or isinstance(image, list):
        # process image batches if image is list of image np arr, paths, bytes
        img, img_cv_grey = [], []
        for single_img in image:
            clr, gry = reformat_input(single_img)
            if n_width is not None and n_height is not None:
                clr = cv2.resize(clr, (n_width, n_height))
                gry = cv2.resize(gry, (n_width, n_height))
            img.append(clr)
            img_cv_grey.append(gry)
        img, img_cv_grey = np.array(img), np.array(img_cv_grey)
        # ragged tensors created when all input imgs are not of the same size
        if len(img.shape) == 1 and len(img_cv_grey.shape) == 1:
            raise ValueError(
                "The input image array contains images of different sizes. "
                "Please resize all images to same shape or pass n_width, n_height to auto-resize"
            )
    else:
        img, img_cv_grey = reformat_input(image)
    return img, img_cv_grey


def make_rotated_img_list(rotationInfo, img_list):
    result_img_list = img_list[:]

    # add rotated images to original image_list
    max_ratio = 1

    for angle in rotationInfo:
        for img_info in img_list:
            rotated = ndimage.rotate(img_info[1], angle, reshape=True)
            height, width = rotated.shape
            ratio = calculate_ratio(width, height)
            max_ratio = max(max_ratio, ratio)
            result_img_list.append((img_info[0], rotated))
    return result_img_list


def set_result_with_confidence(results):
    """ Select highest confidence augmentation for TTA
    Given a list of lists of results (outer list has one list per augmentation,
    inner lists index the images being recognized), choose the best result
    according to confidence level.
    Each "result" is of the form (box coords, text, confidence)
    A final_result is returned which contains one result for each image
    """
    final_result = []
    for col_ix in range(len(results[0])):
        # Take the row_ix associated with the max confidence
        best_row = max(
            [(row_ix, results[row_ix][col_ix][2]) for row_ix in range(len(results))],
            key=lambda x: x[1])[0]
        final_result.append(results[best_row][col_ix])

    return final_result
