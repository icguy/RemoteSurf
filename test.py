from ctypes.wintypes import _SMALL_RECT
import cv2
import numpy as np
import FeatureLoader
import MatchLoader
import Utils

def extract_full(graph, max_num):
    triplets = []
    for k in graph:
        neigh = list(graph[k])
        num_n = len(neigh)
        for i in range(num_n):
            for j in range(i + 1, num_n):
                n1 = neigh[i]
                n2 = neigh[j]
                if k in graph[n1] and k in graph[n2]:
                    triplets.append((k, n1, n2))
    num = 3
    cliques = triplets
    all_levels = [triplets]
    while num < max_num:
        cliques = add_level(graph, cliques)
        all_levels.append(cliques)
        num += 1
    return all_levels

def add_level(graph, cliques):
    newcliques = set()
    for c in cliques:
        elem = c[0]
        for n in graph[elem]:
            if connected_to_all(graph, n, c):
                newclique = list(c)
                newclique.append(n)
                newcliques.add(tuple(sorted(newclique)))
    return newcliques

def connected_to_all(graph, node, clique):
    for n in clique:
        if node not in graph[n]:
            return False
    return True


def main():
    print("hello")
    files = ["imgs/00%d.jpg" % (i) for i in range(5, 10)]
    num = len(files)

    cst = 100 * 1000
    imgs = [cv2.imread(f) for f in files]
    masks = [cv2.imread("imgs/00%d_mask.png" % i, 0) for i in range(5, 10)]
    fl = FeatureLoader.FeatureLoader()
    ml = MatchLoader.MatchLoader()
    kpts = [fl.getFeatures(f, "surf") for f in files]

#masking
    print("masking")
    print(len(kpts[0][1]))
    for i in range(num):
        kp, des = kpts[i]
        j = 0
        while j < len(kp):
            pt = kp[j].pt
            x = int(pt[0])
            y = int(pt[1])
            if masks[i][y, x] > 100:
                j += 1
            else:
                kp.pop(j)
                des.pop(j)
    print(len(kpts[0][1]))

#matching
    print("matching")
    matches = [[None] * num for i in range(num)]
    for i in range(num):
        for j in range(num):
            if i == j: continue
            matches[i][j] = ml.getMatches(
                files[i], files[j], kpts[i][1], kpts[j][1], "surf", MatchLoader.MATCHER_BF_CROSS, version="mask")
#graph
    print("graph")
    graph = {}
    for i in range(num):
        for j in range(num):
            if i == j: continue
            for m in matches[i][j]:
                id1 = i * cst + m.queryIdx
                id2 = j * cst + m.trainIdx
                if id1 not in graph:
                    graph[id1] = set()
                graph[id1].add(id2)

#both ways
    for k in graph.keys():
        for v in list(graph[k]):
            if v not in graph:
                graph[v] = set()
            graph[v].add(k)
    print("graph size: %d" % len(graph))

#connectivity print
    print("connectivity")
    total = 0
    i = 0
    while True:
        subg = [(k, graph[k]) for k in graph if len(graph[k]) == i]
        print(i, len(subg))
        total += len(subg)
        i += 1
        if total == len(graph):
            break

#extract cliques
    print("levels")
    all_levels = extract_full(graph, 5)
    for level in all_levels:
        print(len(level))

    print(" ")
    level5 = list(all_levels[-1])
    print(level5)
    print(list(all_levels[-2]))

#draw
    for m in all_levels[-2]:
        for i in range(1, len(m)):
            img_idx1 = m[0] / cst
            img_idx2 = m[i] / cst
            kpt_idx1 = m[0] % cst
            kpt_idx2 = m[i] % cst
            print(img_idx1, img_idx2, kpt_idx1, kpt_idx2)

            img1 = imgs[img_idx1]
            img2 = imgs[img_idx2]

            pt1 = kpts[img_idx1][0][kpt_idx1].pt
            pt2 = kpts[img_idx2][0][kpt_idx2].pt
            Utils.drawMatch(img1, img2, pt1, pt2, scale=4)
            cv2.waitKey()
        print(m)
        print("new")
        cv2.waitKey()



if __name__ == "__main__":

    all_levels = main()
    exit()

#region level5
    level5 = [(1614, 104606, 201790, 305778, 406258), (199, 100946, 200500, 300175, 400111), (981, 101756, 200946, 301508, 401044), (410, 101651, 200690, 301124, 400323), (38, 100052, 200067, 300022, 400014), (647, 100638, 201244, 300512, 400661), (541, 100794, 200779, 300373, 400426), (223, 100224, 200242, 300191, 400569), (1125, 101872, 201582, 301120, 401333), (1237, 102470, 203571, 302865, 400252), (571, 101297, 200804, 300611, 400827), (1078, 102228, 202659, 301390, 401564)]
#endregion
#region level4
    level4 = [(4012, 202435, 305033, 404029), (3868, 102902, 203642, 305550), (101583, 200965, 300763, 400504), (3413, 203570, 304069, 401479), (4140, 105405, 202530, 404265), (377, 200434, 300498, 400438), (200, 376, 200102, 400281), (65, 200150, 300083, 400064), (23, 100061, 300014, 400027), (261, 100997, 300335, 400379), (1078, 102228, 301390, 401564), (1061, 101211, 200824, 300899), (102228, 202659, 301390, 401564), (102363, 201491, 307340, 402765), (408, 100667, 300601, 400523), (3167, 103776, 202383, 405514), (3804, 105396, 203756, 307099), (2751, 102470, 302865, 400252), (279, 100910, 300625, 400199), (104040, 203712, 307124, 406400), (1852, 103569, 203516, 402175), (412, 100300, 300289, 400269), (1456, 104394, 203126, 403630), (1175, 100272, 200466, 306459), (1008, 102963, 300405, 401744), (3383, 104810, 202270, 404248), (225, 100437, 200261, 300900), (100021, 200100, 300034, 400047), (4051, 105499, 306355, 404933), (680, 100479, 202554, 400172), (13, 100028, 200006, 400037), (3868, 102902, 305550, 406463), (3319, 202794, 303464, 405559), (2885, 106131, 202582, 403867), (1763, 104019, 300051, 400894), (3474, 106151, 203613, 406094), (599, 100884, 201500, 400441), (3069, 201586, 303096, 405854), (1237, 102470, 203571, 400252), (2686, 200533, 302465, 402835), (3148, 103963, 201672, 402551), (1731, 102986, 302583, 401111), (1074, 203412, 301514, 402962), (174, 100752, 300089, 400287), (1748, 202194, 305753, 404221), (3136, 102476, 201510, 301879), (2530, 101030, 301012, 401976), (104180, 201679, 302964, 401701), (916, 200711, 301365, 400458), (1028, 101492, 301175, 401221), (100562, 200807, 301757, 401329), (1468, 104775, 201133, 305766), (571, 101297, 200804, 300611), (1093, 101255, 201329, 301279), (1632, 201413, 302605, 401298), (2950, 105858, 203367, 304022), (161, 100238, 300167, 400098), (4031, 105738, 304930, 403823), (105672, 202493, 304202, 402576), (3456, 104056, 203498, 302434), (225, 100437, 200261, 400209), (1237, 102470, 302865, 400252), (2936, 104864, 202926, 403966), (3115, 101839, 302191, 402298), (4027, 105346, 203893, 404451), (57, 100081, 300035, 400042), (1834, 201964, 304260, 401186), (514, 200489, 303032, 403137), (228, 100188, 300386, 400492), (966, 105827, 304571, 401199), (1417, 201358, 301536, 406232), (511, 201158, 301187, 400726), (4288, 202279, 305953, 405389), (892, 101501, 302602, 401069), (543, 100004, 200010, 400038), (1903, 203899, 307176, 402350), (854, 100116, 200241, 400338), (102387, 201849, 301612, 401416), (987, 105858, 203367, 304022), (1035, 201105, 301268, 400868), (199, 100946, 300175, 400111), (726, 101806, 200680, 400330), (102476, 201510, 301879, 402043), (410, 101651, 301124, 400323), (1078, 102228, 202659, 401564), (59, 200076, 300052, 400049), (70, 200100, 300034, 400047), (105730, 203944, 306792, 405056), (897, 101783, 200973, 401451), (1155, 101334, 200888, 400686), (578, 100301, 201108, 401388), (1143, 101928, 201406, 302091), (3804, 105396, 203756, 203763), (105666, 202435, 306000, 405015), (412, 100300, 200570, 300289), (3922, 102335, 305530, 403044), (981, 101756, 200946, 301508), (376, 200102, 300136, 400281), (103542, 203515, 302903, 404759), (4012, 103610, 305033, 404029), (101155, 200670, 301627, 400988), (3608, 203142, 301483, 401194), (723, 104615, 303710, 403648), (1803, 104334, 204074, 405284), (647, 100638, 300512, 400661), (2411, 103089, 306445, 406214), (104522, 203166, 304650, 405243), (1242, 103869, 305313, 401565), (38, 100052, 300022, 400014), (410, 101651, 200690, 301124), (101454, 201086, 301580, 403089), (106035, 204149, 305925, 403651), (101402, 200734, 300983, 400771), (3553, 103385, 202931, 403586), (278, 100098, 300020, 400202), (3776, 103151, 306046, 405591), (3673, 201427, 301971, 402270), (100002, 200007, 301180, 400022), (2691, 104765, 307225, 404479), (3845, 201064, 303023, 404259), (828, 201083, 301157, 400549), (4128, 103504, 201465, 302407), (100837, 200710, 304359, 400519), (4288, 104922, 202279, 305953), (3673, 201427, 300162, 301971), (2305, 102098, 203034, 304011), (4025, 202610, 304552, 405686), (201455, 303941, 401708, 402468), (4162, 102674, 201221, 301772), (1653, 102848, 202134, 404729), (223, 100224, 300191, 400569), (347, 100206, 200445, 301339), (3804, 105396, 203756, 404407), (158, 100293, 300176, 400145), (4077, 105955, 306281, 403200), (1614, 104606, 201790, 305778), (104579, 202068, 305109, 402016), (1400, 101654, 200976, 304659), (4027, 105346, 105848, 404451), (102876, 204058, 302252, 404686), (536, 100150, 200190, 300104), (4215, 106044, 302542, 402404), (3527, 103114, 201200, 300613), (888, 201483, 304043, 401180), (966, 103054, 202778, 401227), (1137, 202954, 302272, 402688), (2160, 101615, 202025, 301074), (1696, 105935, 202204, 302317), (103260, 203018, 304933, 406334), (108, 100102, 300007, 400062), (101501, 203408, 302602, 401069), (832, 300752, 400759, 403643), (726, 101806, 300813, 400330), (3307, 103479, 202480, 307248), (3524, 102616, 302974, 406259), (1912, 104878, 204129, 303219), (2836, 103265, 202058, 302036), (3325, 105653, 203393, 404257), (2885, 106131, 202582, 305284), (3960, 201880, 307214, 402767), (599, 100884, 300551, 400441), (105189, 202704, 304539, 406551), (1287, 102296, 201728, 301811), (535, 100043, 200170, 300053), (105109, 201273, 306832, 406438), (3440, 202569, 304911, 403523), (370, 200319, 300128, 400206), (1513, 103228, 201619, 304310), (1507, 100989, 301221, 401476), (868, 200495, 300902, 404065), (38, 100052, 200067, 300022), (151, 100090, 300028, 302185), (105894, 203225, 305003, 405098), (1635, 101825, 202751, 304065), (4172, 102594, 305678, 402999), (6, 100004, 300002, 400016), (1155, 101334, 200888, 301568), (106205, 204037, 305055, 401776), (445, 101061, 300704, 400495), (105653, 203393, 303423, 404257), (3905, 104846, 203335, 303620), (2721, 104864, 304822, 403966), (3440, 105939, 304911, 403523), (101070, 202841, 306453, 401960), (861, 104102, 200816, 400683), (422, 100712, 201954, 300520), (541, 100794, 200779, 300373), (1614, 201790, 305778, 406258), (101634, 201122, 303597, 402102), (2470, 104622, 306395, 402003), (2262, 101357, 301037, 401169), (3607, 105100, 305662, 403715), (100052, 200067, 300015, 400014), (571, 101297, 300611, 400827), (1355, 100825, 202731, 400854), (103103, 202927, 301609, 401538), (1214, 201207, 301874, 401399), (1256, 202984, 305245, 406050), (100116, 200241, 301518, 400338), (597, 200380, 300098, 400282), (102902, 203642, 305550, 406463), (230, 101520, 201251, 301495), (103794, 202122, 301512, 406102), (3967, 104953, 203427, 404646), (916, 101340, 200711, 301365), (100926, 200427, 301296, 400454), (1774, 105475, 302046, 401385), (100102, 200070, 300007, 400062), (2315, 104924, 304655, 402490), (105831, 203514, 304977, 402529), (103864, 203833, 302677, 404401), (3534, 200922, 302960, 402698), (1078, 202659, 301390, 401564), (649, 201334, 300153, 400114), (105923, 201271, 306748, 404833), (1155, 101334, 301533, 400986), (3259, 106137, 204050, 306997), (103772, 202488, 303159, 404362), (958, 101177, 200828, 400743), (4184, 105220, 202665, 404391), (104047, 201542, 301041, 403712), (949, 105614, 204106, 304840), (2783, 100317, 203380, 400887), (100438, 203985, 303859, 400362), (2459, 105140, 303148, 403411), (1246, 105081, 306753, 401297), (4094, 101909, 203025, 301924), (2462, 203528, 302715, 401308), (938, 101491, 301326, 406423), (3930, 105191, 203992, 304314), (102024, 201000, 303066, 306328), (2862, 104919, 306949, 405424), (3396, 101788, 201479, 402452), (3197, 105169, 202908, 402319), (647, 100638, 201244, 400661), (105354, 202610, 304552, 405686), (103373, 201599, 307399, 404182), (2469, 100581, 302213, 400830), (822, 102945, 200350, 301371), (102972, 202569, 304911, 403523), (440, 101086, 200628, 400106), (308, 100423, 200213, 300360), (649, 100749, 300153, 400114), (3492, 201545, 303087, 401293), (1748, 101799, 202880, 305753), (840, 100795, 200385, 301022), (250, 100670, 201203, 300364), (4182, 104396, 203681, 304324), (4271, 104498, 203788, 406409), (2748, 104623, 202437, 302755), (351, 100149, 300314, 400219), (519, 101487, 201182, 300953), (223, 200242, 300191, 400569), (101817, 101988, 201187, 301378), (1372, 101369, 301530, 400910), (687, 102005, 200636, 400558), (3979, 104627, 203153, 305281), (4282, 103704, 203599, 307320), (668, 202025, 301103, 400584), (1427, 201824, 301123, 403351), (105567, 201692, 305207, 402040), (106205, 204037, 401776, 406224), (3114, 105284, 202755, 306276), (1133, 202927, 301609, 401538), (2237, 104515, 301803, 405493), (1423, 100826, 300896, 400729), (3942, 101785, 301422, 404924), (3734, 203734, 303071, 405640), (693, 200966, 301349, 400365), (603, 101942, 300367, 400848), (119, 200550, 300337, 400117), (1579, 103314, 306707, 404589), (827, 103689, 301087, 400937), (3607, 202660, 305662, 403715), (217, 200358, 300135, 400156), (101334, 200888, 301568, 400686), (3673, 201427, 300162, 402270), (345, 200257, 300706, 400272), (410, 200690, 301124, 400323), (1168, 100328, 300664, 400489), (1301, 100889, 300684, 400066), (1168, 202955, 300664, 400489), (3033, 202131, 301487, 404446), (4279, 103202, 303984, 403783), (137, 200099, 301506, 400097), (102079, 203528, 302715, 401308), (3096, 101293, 202193, 402283), (97, 105288, 300026, 400059), (2064, 105826, 202892, 303864), (3765, 101517, 200777, 304169), (2533, 106121, 306306, 404444), (143, 103748, 300130, 400091), (2458, 102555, 307323, 402772), (2748, 104623, 302755, 405960), (4287, 103717, 201335, 402909), (63, 200075, 300037, 400043), (1456, 104394, 302075, 403630), (1046, 102525, 201104, 401419), (1030, 105015, 203603, 303848), (3834, 102088, 307200, 403395), (2963, 100181, 200572, 406171), (102243, 203694, 306308, 405824), (2449, 101901, 202482, 402236), (1812, 105933, 201228, 400732), (1720, 102235, 303534, 401155), (415, 200453, 301009, 400515), (12, 100069, 300012, 400013), (2483, 102472, 202633, 400979), (1298, 101873, 200914, 401025), (4212, 202030, 301774, 405787), (4282, 203599, 307320, 405211), (2733, 104185, 203270, 401316), (3319, 106089, 303464, 405559), (3996, 103145, 203792, 300814), (2160, 101615, 301074, 401471), (9, 100078, 200040, 400011), (2023, 106053, 300995, 402391), (3952, 105923, 201271, 306748), (4271, 203788, 304584, 406409), (1134, 101523, 301293, 401367), (70, 100021, 200100, 400047), (2717, 105974, 305168, 403730), (101340, 200711, 301365, 400458), (1093, 201329, 301279, 401542), (2062, 100428, 303694, 403745), (102280, 201188, 300690, 400998), (216, 100483, 200447, 300163), (3299, 101217, 203251, 404764), (2160, 101615, 202025, 401471), (873, 103392, 200771, 302162), (100339, 200363, 300738, 400151), (102290, 202331, 304953, 403178), (118, 100073, 200267, 300092), (3073, 202308, 305326, 405162), (1155, 101334, 200888, 301533), (174, 200477, 300089, 403791), (205, 200171, 300262, 400293), (539, 101339, 202393, 405339), (796, 101874, 201895, 300421), (4268, 103916, 202622, 305412), (102480, 201455, 303941, 402468), (1020, 201300, 301082, 400934), (101756, 200946, 301508, 401044), (103805, 203809, 303810, 403358), (244, 100247, 200209, 300158), (101310, 200286, 300736, 400715), (3129, 103737, 201020, 305547), (1323, 200268, 300115, 400176), (2028, 105296, 304897, 406262), (1663, 105044, 201998, 303457), (897, 200973, 300686, 401451), (3607, 105156, 202660, 406312), (2697, 105400, 202572, 404147), (2503, 101910, 201818, 304896), (3229, 105656, 304088, 406023), (434, 102058, 202479, 400308), (1767, 202226, 403446, 406231), (459, 100431, 200449, 300306), (308, 100423, 200213, 400265), (3052, 105189, 202704, 304539), (518, 100468, 200398, 300293), (4077, 204142, 306281, 405783), (2879, 101739, 101859, 405453), (202927, 301609, 401312, 401538), (3396, 101788, 201479, 305052), (3945, 106016, 302807, 404606), (354, 200289, 300343, 400288), (1237, 203571, 302865, 400252), (187, 101150, 300073, 400260), (3307, 103479, 307248, 405110), (879, 101407, 305519, 404726), (1217, 104218, 303319, 401232), (101416, 201105, 301268, 400868), (2897, 103933, 201174, 401244), (2943, 103462, 201718, 305296), (4256, 105998, 204046, 300352), (252, 100908, 300334, 400466), (4099, 103148, 203640, 403738), (3333, 101847, 201041, 400845), (4132, 104216, 203525, 404747), (100479, 202554, 300258, 400172), (3607, 202660, 403715, 406312), (408, 100667, 200383, 400523), (4182, 104396, 304324, 405148), (726, 200680, 301063, 400330), (3319, 106089, 202794, 303464), (3333, 101847, 201041, 306867), (1125, 101872, 201582, 301120), (2809, 103025, 303748, 402829), (3425, 203109, 303904, 404510), (179, 100870, 200291, 400169), (127, 100120, 300067, 400349), (3608, 201730, 301483, 401194), (3262, 202346, 300636, 404045), (2091, 102290, 304953, 403178), (4051, 306355, 403960, 404933), (2171, 102313, 201598, 306451), (102073, 201836, 302210, 401797), (5, 200040, 300009, 400020), (4141, 104471, 203157, 406129), (103130, 202148, 307205, 405236), (2831, 203232, 305573, 401325), (3259, 106137, 204050, 405550), (3446, 106194, 202900, 401723), (3978, 104531, 200983, 405064), (3725, 103704, 203599, 307320), (856, 101305, 300970, 400662), (3741, 201179, 302451, 402165), (1551, 201473, 307384, 401503), (106053, 201509, 300995, 402391), (101255, 201329, 301279, 401542), (1748, 101799, 305753, 404221), (1085, 4219, 305206, 405068), (835, 100765, 301224, 400799), (101088, 201330, 301338, 402495), (1635, 101825, 202751, 400820), (646, 100562, 200807, 301757), (2064, 105826, 202892, 403828), (1463, 201849, 301612, 401416), (1663, 105044, 303457, 402061), (3328, 203737, 304304, 401350), (422, 100712, 201954, 400402), (3587, 100219, 302119, 400656), (128, 100112, 200053, 400009), (482, 100616, 300201, 400237), (1701, 105109, 306832, 406438), (38, 100052, 300015, 400014), (199, 200500, 300175, 400111), (410, 101651, 200690, 400323), (3684, 103103, 200972, 401312), (21, 100027, 300010, 400012), (841, 101245, 300586, 400244), (1451, 101912, 200584, 301186), (3115, 101839, 201165, 402298), (1085, 103465, 305206, 405068), (101360, 201779, 300816, 401004), (2809, 103892, 303748, 402829), (601, 201116, 301363, 401645), (2671, 104704, 203615, 403307), (3895, 105653, 203393, 404257), (104498, 203788, 304584, 406409), (2359, 202685, 306723, 402965), (2424, 104929, 305520, 401033), (2934, 105968, 304466, 405325), (100083, 200080, 300056, 400086), (756, 101365, 300858, 400681), (105968, 203546, 304466, 405325), (2838, 204128, 303344, 404278), (104864, 202926, 304822, 403966), (2269, 100629, 201418, 303180), (3063, 102396, 200944, 303077), (3873, 103635, 202513, 404881), (4031, 203187, 304930, 403823), (3260, 101624, 200930, 401023), (1010, 200617, 301195, 400617), (2961, 104670, 202485, 305116), (4215, 202143, 302542, 402404), (104623, 202437, 302755, 405960), (57, 100081, 200055, 300035), (4135, 100861, 203789, 406558), (1994, 105895, 202721, 306622), (4016, 105056, 202409, 403921), (2610, 202666, 306055, 403277), (1069, 104019, 201993, 402025), (3550, 102876, 204058, 404686), (3735, 103160, 202036, 402214), (2048, 103373, 201599, 404182), (100224, 200242, 300191, 400569), (103492, 202706, 305573, 401325), (651, 104745, 200714, 400605), (482, 200400, 300201, 400237), (3768, 103500, 203102, 306300), (473, 101132, 300511, 400386), (101328, 203875, 305277, 403625), (203, 100172, 300091, 400195), (4135, 105466, 203789, 406558), (3034, 3967, 104953, 203427), (872, 201309, 301135, 400624), (102223, 200929, 301313, 401641), (1256, 101657, 305245, 406050), (118, 100073, 200267, 400079), (103560, 203530, 304214, 405734), (3355, 105918, 306679, 401502), (4128, 201465, 302407, 402939), (541, 100794, 200779, 400426), (3421, 105399, 305582, 406157), (2203, 200811, 301554, 302154), (370, 100421, 300128, 400206), (3318, 201290, 305228, 403444), (1035, 101416, 201105, 400868), (4132, 104216, 203525, 302012), (1456, 203126, 302075, 403630), (199, 100946, 200500, 300175), (103103, 202927, 301609, 401312), (1084, 200989, 301151, 400938), (4215, 106044, 202143, 305909), (88, 102564, 200124, 400061), (1551, 102094, 307384, 401503), (104602, 203978, 307166, 405814), (2767, 105859, 304989, 401314), (3952, 105923, 306748, 404833), (2879, 101859, 302893, 405453), (89, 100083, 300056, 400086), (327, 100472, 300642, 400305), (4173, 102066, 301838, 400159), (100004, 200029, 300002, 400016), (103234, 202997, 303439, 400914), (38, 200067, 300022, 400014), (1362, 101392, 302145, 402487), (3325, 104959, 203393, 303423), (329, 100534, 300358, 400304), (100854, 200902, 300842, 400960), (4249, 104527, 306446, 405308), (3122, 202336, 302066, 403974), (1804, 101723, 202400, 401008), (3362, 100865, 200526, 401165), (892, 101501, 203408, 302602), (1125, 101872, 301120, 401333), (2468, 101929, 300636, 405201), (482, 100616, 200400, 400237), (3694, 103687, 203193, 401558), (548, 100812, 300280, 400399), (105028, 202850, 301932, 406276), (4259, 105541, 202410, 405087), (100948, 201034, 201056, 401027), (3270, 105281, 301318, 403690), (1507, 100989, 202176, 300822), (414, 200554, 300370, 400381), (3453, 105656, 304088, 406023), (603, 101089, 300367, 400848), (103716, 201412, 302322, 402616), (4115, 202692, 302276, 405101), (4249, 104527, 201121, 306446), (3607, 202660, 305662, 406312), (100620, 200122, 300160, 400084), (4142, 203740, 306472, 405160), (3115, 101839, 201165, 302191), (103926, 203195, 303888, 403546), (243, 100673, 300144, 400108), (1049, 101322, 300976, 401011), (433, 200365, 301196, 400550), (3016, 105188, 202157, 406128), (3492, 103310, 303087, 401293), (434, 102058, 202479, 300302), (1614, 104606, 201790, 406258), (2462, 102079, 302715, 401308), (60, 100052, 200067, 300022), (101723, 202400, 306901, 401008), (1078, 102228, 202659, 301390), (2091, 102290, 202175, 403178), (100638, 201244, 300512, 400661), (101652, 200858, 301208, 400484), (102822, 202639, 302685, 401303), (106044, 202143, 305909, 402404), (101297, 200804, 300611, 400827), (3071, 100322, 202087, 401664), (100802, 200479, 300836, 400452), (4025, 105354, 304552, 405686), (100331, 200367, 300174, 400185), (100800, 201467, 300832, 400479), (104310, 201730, 301483, 401194), (2831, 202706, 305573, 401325), (483, 102280, 300690, 400998), (3262, 101387, 202346, 404045), (106205, 204037, 305055, 406224), (102876, 204058, 302252, 400694), (103704, 203599, 307320, 405295), (106072, 203962, 305712, 404185), (4161, 102935, 202810, 303316), (854, 100116, 200241, 301518), (3804, 105396, 203763, 307099), (3307, 202480, 203646, 307248), (840, 200385, 301022, 401217), (793, 200962, 301171, 400484), (3493, 105584, 201324, 303498), (201324, 303498, 304897, 406262), (3247, 104911, 203168, 406101), (1219, 201261, 301501, 401663), (787, 103295, 200976, 301278), (422, 100712, 300520, 400402), (205, 100225, 200171, 300262), (3916, 102260, 202375, 304079), (3034, 104953, 203427, 404646), (993, 100013, 200004, 300517), (1767, 102602, 202226, 406231), (3834, 102088, 203882, 307200), (1165, 101633, 301312, 400490), (649, 101790, 200791, 400636), (100637, 200704, 300356, 400343), (2064, 105826, 303853, 403828), (100865, 200526, 300913, 400518), (2892, 104339, 203980, 402844), (3994, 203263, 303500, 406227), (840, 100795, 301022, 401217), (103948, 202246, 307314, 404714), (647, 100638, 201244, 300512), (4077, 204142, 306281, 403200), (105440, 203207, 306205, 405527), (3930, 103630, 203992, 304314), (2889, 200886, 301060, 400927), (2814, 204193, 306766, 404569), (1007, 200913, 301159, 400921), (104299, 203995, 302234, 406438), (104516, 202733, 307334, 401387), (86, 100046, 200513, 400071), (103176, 203369, 307383, 404267), (101305, 200659, 300970, 400662), (1711, 103715, 203780, 401091), (541, 100794, 300373, 400426), (3052, 105189, 304934, 406551), (105894, 203225, 303644, 405098), (4, 100057, 200042, 301134), (1748, 106064, 202194, 404221), (4145, 202003, 303364, 403860), (100884, 200813, 300551, 400441), (1614, 104606, 305778, 406258), (101690, 200806, 300928, 400235), (1038, 100851, 201185, 400393), (883, 200676, 305004, 400805), (3587, 100219, 203607, 302119), (3922, 102335, 203583, 305530), (3333, 101847, 201041, 406254), (3723, 203890, 306677, 406304), (3333, 201041, 400845, 406254), (100, 100102, 300007, 400062), (796, 201895, 300421, 404400), (105831, 203514, 304977, 406392), (1878, 103176, 203369, 404267), (1184, 100529, 302193, 400817), (571, 101297, 200804, 400827), (100045, 200084, 300037, 400043), (250, 100670, 201203, 400909), (2659, 101631, 201491, 403035), (859, 101054, 300959, 400945), (3131, 103367, 203462, 405192), (100052, 300015, 300022, 400014), (2, 100499, 200039, 400007), (105226, 202142, 303746, 406110), (1935, 106108, 301902, 406467), (3834, 203882, 307200, 403395), (1112, 101613, 202736, 402397), (2820, 102875, 203128, 402739), (2507, 200935, 304238, 404887), (3333, 101847, 306867, 406254), (100946, 200500, 300175, 400111), (3674, 106072, 203962, 404185), (1262, 101412, 301908, 401296), (4291, 105297, 303788, 406397), (647, 201244, 300512, 400661), (1355, 202731, 302837, 400854), (2091, 102290, 102372, 403178), (722, 201640, 300799, 406370), (4077, 105955, 204142, 306281), (174, 100752, 200477, 300089), (2751, 102470, 203571, 302865), (101783, 200973, 300686, 401451), (2028, 201324, 304897, 406262), (1463, 102387, 301612, 401416), (2934, 203546, 304466, 405325), (3325, 203393, 303423, 404257), (3610, 105327, 203262, 402312), (3077, 102120, 201141, 306531), (4145, 102084, 202003, 403860), (4143, 204101, 301048, 403758), (100794, 200779, 300373, 400426), (816, 200394, 301682, 400715), (4037, 203869, 305939, 402101), (2468, 101929, 202346, 300636), (1047, 203572, 305537, 405421), (1212, 101148, 200232, 301860), (101874, 201895, 300421, 401107), (4082, 105822, 203706, 302644), (408, 100667, 200383, 300601), (1237, 102470, 203571, 302865), (1139, 105951, 202090, 403565), (5, 100037, 300621, 400784), (3684, 103103, 301609, 401312), (103504, 201465, 302407, 402939), (4051, 105499, 306355, 403960), (103809, 202337, 203179, 304074), (101696, 201601, 300675, 402420), (101790, 200791, 300569, 400636), (4152, 200341, 304561, 403673), (3493, 201324, 303498, 406262), (3853, 106124, 201663, 401086), (38, 100052, 200067, 400014), (168, 200117, 300172, 400225), (100, 100102, 200085, 300007), (103160, 202036, 306445, 402214), (856, 101305, 200659, 300970), (1878, 203369, 307383, 404267), (103628, 203401, 301677, 402301), (104334, 203557, 304199, 405284), (380, 100612, 300477, 400276), (2930, 203133, 306620, 402989), (955, 101593, 200798, 301493), (1424, 102409, 202001, 403288), (1653, 102848, 202134, 405554), (173, 200230, 300468, 400148), (2710, 105483, 203776, 304454), (101765, 202363, 305458, 406439), (1513, 103228, 201619, 403488), (3553, 103385, 304503, 403586), (103056, 301823, 304923, 401901), (217, 100160, 300135, 400156), (2006, 103665, 201341, 404184), (3527, 103114, 300613, 400695), (3008, 105543, 200865, 304404), (649, 101790, 200791, 300697), (673, 100995, 300620, 400646), (981, 200946, 301508, 401044), (719, 100992, 200832, 301105), (60, 100052, 300022, 400014), (279, 100910, 200473, 400199), (104045, 202444, 306871, 404279), (101279, 202663, 300792, 401693), (4038, 203651, 304521, 406042), (4219, 204051, 305206, 405068), (508, 200416, 300345, 400406), (100328, 202955, 300664, 400489), (3993, 103922, 203814, 400734), (3325, 104959, 203393, 404257), (4002, 105257, 204017, 406426), (3674, 106072, 203962, 405722), (104755, 202136, 300409, 404346), (476, 100032, 300017, 400024), (1256, 101657, 202984, 406050), (4199, 104516, 202733, 307334), (1823, 101310, 300736, 400715), (105173, 202458, 300185, 401959), (2091, 102372, 304953, 403178), (3853, 106124, 305639, 401086), (721, 200460, 300718, 404404), (100015, 200761, 301636, 400240), (101788, 201479, 305052, 402452), (2574, 104909, 202593, 304672), (3307, 103479, 203646, 307248), (3052, 202704, 304934, 406551), (3521, 105888, 201841, 404008), (1507, 100989, 202176, 401476), (108, 200070, 300007, 400062), (3091, 104540, 203410, 403166), (370, 100131, 200319, 400206), (100684, 200289, 300343, 400288), (3288, 104769, 301164, 400368), (1849, 105216, 301565, 403542), (3598, 203563, 305208, 405377), (1348, 101882, 200525, 400836), (103080, 203749, 300667, 401611), (102093, 201926, 306402, 406340), (106176, 201393, 304963, 402442), (4, 100057, 200042, 400030), (104951, 201714, 304773, 406474), (3804, 203756, 307099, 404407), (2556, 201205, 306090, 401176), (104218, 201023, 303319, 401232), (1165, 101740, 200895, 400907), (2510, 105816, 202848, 406519), (118, 100073, 300092, 400079), (1433, 203818, 303143, 401861), (413, 101522, 200557, 300689), (102058, 202479, 300302, 400308), (149, 200122, 300160, 400084), (1034, 102462, 301301, 400971), (105600, 203086, 305779, 401098), (3607, 105100, 202660, 403715), (2963, 100181, 201406, 406171), (1298, 101965, 200914, 401025), (89, 100083, 200080, 400086), (119, 100370, 300337, 400117), (1125, 101872, 201582, 401333), (101133, 200799, 301569, 404578), (2666, 103575, 305436, 401172), (571, 200804, 300611, 400827), (1748, 202880, 305753, 404221), (437, 101366, 200967, 401148), (1460, 202126, 304147, 403174), (101755, 201845, 301951, 405293), (241, 100336, 200207, 300316), (1094, 101356, 301077, 400952), (3338, 103260, 203018, 304933), (102767, 204168, 303214, 404585), (3508, 105072, 302239, 405869), (1054, 104102, 200816, 400683), (3456, 104187, 203498, 302434), (104761, 203970, 306305, 404102), (957, 101382, 201302, 401417), (122, 101034, 200344, 400100), (2370, 203829, 306365, 403674), (1969, 101569, 203170, 403866), (3712, 101914, 305448, 403234), (981, 101756, 301508, 401044), (886, 100854, 300842, 400960), (100910, 200473, 300625, 400199), (225, 200261, 300900, 400209), (329, 100534, 201751, 400304), (2691, 104765, 307225, 406359), (1584, 2870, 203655, 303438), (4219, 103465, 305206, 405068), (104993, 203685, 301079, 401411), (981, 101756, 200946, 401044), (105831, 203514, 402529, 406392), (725, 200174, 300602, 400289), (3113, 200366, 300574, 401018), (3521, 105888, 201841, 303063), (3665, 203607, 307031, 405019), (101872, 201582, 301120, 401333), (2298, 104678, 404875, 405165), (1803, 104334, 203557, 405284), (106131, 202582, 305284, 403867), (3608, 104310, 301483, 401194), (2458, 201796, 307323, 402772), (1707, 101447, 200720, 305651), (707, 101383, 200733, 400709), (4177, 203820, 305687, 405480), (2889, 101095, 301060, 400927), (102878, 201971, 301327, 402463), (1272, 103331, 202216, 302029), (1125, 201582, 301120, 401333), (3221, 103981, 204191, 306600), (1136, 101569, 203170, 403866), (101929, 202346, 300636, 404045), (125, 100620, 300160, 400084), (628, 101707, 300072, 401882), (2736, 202248, 302632, 403040), (1042, 104003, 201168, 401094), (3650, 101788, 201479, 305052), (756, 300554, 300858, 400681), (4042, 103560, 203530, 405734), (2830, 202514, 301690, 403883), (3028, 201926, 306402, 406340), (3814, 4082, 203706, 302644), (2329, 3922, 102335, 305530), (105772, 200841, 302665, 404845), (4143, 204101, 301246, 403758), (368, 100967, 200719, 300692), (4219, 103465, 204051, 405068), (4215, 106044, 202143, 402404), (521, 104920, 200685, 400279), (199, 100946, 200500, 400111), (541, 200779, 300373, 400426), (3299, 101217, 307242, 404764), (3475, 105223, 203161, 304076), (1618, 103766, 301581, 406498), (2037, 102575, 201852, 402596), (567, 100733, 300707, 400536), (2091, 102290, 202175, 405353), (3684, 103103, 202927, 401312), (63, 100045, 300037, 400043), (104606, 201790, 305778, 406258), (102290, 102372, 304953, 403178), (711, 102435, 300226, 400373), (3288, 104769, 203633, 301164), (2329, 102335, 305530, 403044), (3259, 204050, 306997, 405550), (223, 100224, 200242, 300191), (284, 101258, 200673, 403152), (2665, 200430, 300526, 402469), (4042, 203530, 304214, 405734), (440, 200628, 300469, 400106), (759, 202533, 300847, 400757), (2736, 101847, 202248, 403040), (101651, 200690, 301124, 400323), (549, 100338, 200549, 300399), (105000, 200123, 300059, 400053), (3660, 105493, 204064, 304029), (1193, 104537, 203120, 306137), (1791, 102190, 203959, 302021), (270, 100531, 300095, 400465), (1711, 203780, 301904, 401091), (103553, 203890, 306677, 406304), (3052, 105189, 202704, 406551), (3867, 102744, 202008, 305647), (4039, 201189, 301689, 406252), (104849, 202533, 300847, 400757), (1663, 201998, 303457, 402061), (2091, 102290, 202331, 403178), (3362, 100865, 200526, 400518), (2069, 201644, 301712, 405830), (3842, 201810, 306073, 402053), (368, 100967, 300692, 400680), (2650, 104849, 202533, 300847), (446, 100869, 200607, 400603), (1814, 102650, 302418, 402811), (101704, 200176, 300971, 400471), (102470, 203571, 302865, 400252), (223, 100224, 200242, 400569), (1804, 202400, 306901, 401008), (100052, 200067, 300022, 400014), (1410, 201455, 303941, 401708), (3273, 103387, 306430, 403686), (284, 101258, 200673, 300694), (1737, 201979, 303345, 406039), (839, 102450, 301208, 400658), (2784, 105070, 306458, 403260), (1719, 103066, 202061, 401637), (102848, 202134, 303917, 404729), (4041, 202615, 202918, 306530), (309, 100949, 200490, 300663), (4172, 203744, 305678, 402999), (2865, 102242, 102437, 301676), (4173, 203508, 301838, 400159), (2879, 101739, 302893, 405453), (745, 100582, 301305, 400749), (3905, 104846, 203335, 404962), (2685, 203169, 306316, 404773), (756, 101365, 300554, 400681), (2462, 102079, 203528, 401308), (2508, 105125, 203913, 306105), (104745, 200714, 300694, 400605), (1701, 105109, 306832, 406361), (370, 100421, 200319, 400206), (49, 100071, 200052, 300016), (3333, 201041, 306867, 406254), (3947, 106045, 200262, 404332), (1551, 102094, 201473, 401503), (119, 100370, 200550, 300337), (39, 100008, 200025, 400028), (4, 200042, 301134, 400030), (3355, 105918, 204095, 401502)]
#endregion

    files = ["imgs/00%d.jpg" % (i) for i in range(5, 10)]
    num = len(files)

    cst = 100 * 1000
    imgs = [cv2.imread(f) for f in files]
    fl = FeatureLoader.FeatureLoader()
    ml = MatchLoader.MatchLoader()
    kpts = [fl.getFeatures(f, "surf") for f in files]
    matches = [[None] * num for i in range(num)]
    for i in range(num):
        for j in range(num):
            if i == j: continue
            print(i,j)
            matches[i][j] = ml.getMatches(
                files[i], files[j], kpts[i][1], kpts[j][1], "surf", MatchLoader.MATCHER_BF_CROSS)

    for m in level4:
        for i in range(1, len(m)):
            img_idx1 = m[0] / cst
            img_idx2 = m[i] / cst
            kpt_idx1 = m[0] % cst
            kpt_idx2 = m[i] % cst
            print(img_idx1, img_idx2, kpt_idx1, kpt_idx2)

            img1 = imgs[img_idx1]
            img2 = imgs[img_idx2]

            pt1 = kpts[img_idx1][0][kpt_idx1].pt
            pt2 = kpts[img_idx2][0][kpt_idx2].pt
            Utils.drawMatch(img1, img2, pt1, pt2, scale=4)
            cv2.waitKey()
        print(m)
        print("new")
        cv2.waitKey()
