
SINGLE_HAND_GESTURE_NAMES = [
        ## right hand gestures
        '0000_neutral_relaxed', '0001_neutral_rigid', '0002_good_luck', '0003_fake_gun', '0004_star_trek',
        '0005_star_trek_extended_thumb', '0006_thumbup_relaxed', '0007_thumbup_normal', '0008_thumbup_rigid', '0009_thumbtucknormal',
        '0010_thumbtuckrigid', '0011_aokay', '0012_aokay_upright', '0013_surfer', '0014_rocker',
        '0014_rocker_frontside', '0015_rocker_backside', '0016_fist', '0017_fist_rigid', '0018_alligator_open',
        '0019_alligator_closed', '0020_indexpoint', '0023_one_count', '0024_two_count', '0025_three_count',
        '0026_four_count', '0027_five_count', '0029_indextip', '0030_middletip', '0031_ringtip',
        '0032_pinkytip', '0035_palmdown', '0036_palmdownwave', '0037_fingerspreadrelaxed', '0038_fingerspreadnormal',
        '0039_fingerspreadrigid', '0040_capisce', '0041_claws', '0042_pucker', '0043_peacock',
        '0044_cup', '0045_shakespearesyorick', '0048_index_point', '0051_dinosaur', '0058_middlefinger',
        
        ## left hand gestures
        '0100_neutral_relaxed', '0101_neutral_rigid', '0102_good_luck', '0103_fake_gun', '0104_star_trek',
        '0105_star_trek_extended_thumb', '0106_thumbup_relaxed', '0107_thumbup_normal', '0108_thumbup_rigid', '0109_thumbtucknormal',
        '0110_thumbtuckrigid', '0111_aokay', '0112_aokay_upright', '0113_surfer', '0114_rocker',
        '0114_rocker_frontside', '0115_rocker_backside', '0116_fist', '0117_fist_rigid', '0118_alligator_open',
        '0119_alligator_closed', '0120_indexpoint', '0123_one_count', '0124_two_count', '0125_three_count',
        '0126_four_count', '0127_five_count', '0129_indextip', '0130_middletip', '0131_ringtip',
        '0132_pinkytip', '0133_relaxedwave', '0135_palmdown', '0137_fingerspreadrelaxed', '0138_fingerspreadnormal',
        '0139_fingerspreadrigid', '0140_capisce', '0141_claws', '0142_pucker', '0143_peacock',
        '0144_cup', '0145_shakespearesyorick', '0148_index_point', '0151_dinosaur', '0158_middlefinger',
        
        ## Two-hand gestures
        # '0259_dh_rightclaspleft', '0260_dh_leftclaspright', '0261_dh_fingergun', '0262_dh_rightfistcoverleft', '0263_dh_leftfistcoverright',
        # '0264_dh_interlockedfingers', '0266_dh_pray', '0267_dh_rightfistoverleft', '0268_dh_leftfistoverright', '0269_dh_interlockedthumbtiddle',
        # '0270_dh_rightfingercountindexpoint', '0271_dh_leftfingercountindexpoint', '0272_dh_rightreceivethewafer', '0273_leftreceivethewafer', '0274_rightbabybird',
        # '0275_leftbabybird', '0276_interlockedfingerspread', '0277_itsybitsyspider', '0278_fingernoodle', '0279_knucklecrack',
        # '0280_golfclaplor', '0281_golfclaprol', '0282_sarcasticclap', '0283_palmerrub', '0284_fingersqueeze',
        # '0285_rockpaperscissors', '0286_handscratch', '0287_pointingtowardsfeatures', '0389_dh_nontouchROM', '0390_dh_touchROM',
    ]
                        


RIGHT_HAND_GESTURE_NAMES = SINGLE_HAND_GESTURE_NAMES[:len(SINGLE_HAND_GESTURE_NAMES) // 2]

LEFT_HAND_GESTURE_NAMES = SINGLE_HAND_GESTURE_NAMES[len(SINGLE_HAND_GESTURE_NAMES) // 2:]


TWO_HAND_GESTURE_CLASS_MAPPING = {
    "0000_neutral_relaxed":     {"main_id": 0, "sub_id": 0, "global_id": 0},
    "0037_fingerspreadrelaxed": {"main_id": 0, "sub_id": 1, "global_id": 1},
    "0001_neutral_rigid":       {"main_id": 0, "sub_id": 2, "global_id": 2},
    "0027_five_count":          {"main_id": 0, "sub_id": 2, "global_id": 2},
    "0038_fingerspreadnormal":  {"main_id": 0, "sub_id": 2, "global_id": 2},
    "0039_fingerspreadrigid":   {"main_id": 0, "sub_id": 2, "global_id": 2},
    
    "0002_good_luck":           {"main_id": 1, "sub_id": 0, "global_id": 28},
    "0003_fake_gun":            {"main_id": 2, "sub_id": 0, "global_id": 21},
    "0004_star_trek":           {"main_id": 3, "sub_id": 0, "global_id": 14},
    "0005_star_trek_extended_thumb": {"main_id": 4, "sub_id": 0, "global_id": 7},
    
    "0006_thumbup_relaxed":     {"main_id": 5, "sub_id": 0, "global_id": 31},
    "0007_thumbup_normal":      {"main_id": 5, "sub_id": 1, "global_id": 32},
    "0008_thumbup_rigid":       {"main_id": 5, "sub_id": 1, "global_id": 32},
    
    "0009_thumbtucknormal":     {"main_id": 6, "sub_id": 0, "global_id": 25},
    "0010_thumbtuckrigid":      {"main_id": 6, "sub_id": 0, "global_id": 25},
    
    "0011_aokay":               {"main_id": 7, "sub_id": 0, "global_id": 17},
    "0043_peacock":             {"main_id": 7, "sub_id": 0, "global_id": 17},  # This is a duplicate gesture in the original list
    "0012_aokay_upright":       {"main_id": 7, "sub_id": 1, "global_id": 18},
    
    "0013_surfer":              {"main_id": 8, "sub_id": 0, "global_id": 11},
    
    "0014_rocker":              {"main_id": 9, "sub_id": 0, "global_id": 3},
    "0014_rocker_frontside":    {"main_id": 9, "sub_id": 1, "global_id": 4},
    "0015_rocker_backside":     {"main_id": 9, "sub_id": 2, "global_id": 5},
    
    "0016_fist":                {"main_id": 10, "sub_id": 0, "global_id": 29},
    "0017_fist_rigid":          {"main_id": 10, "sub_id": 0, "global_id": 29},
    "0018_alligator_open":      {"main_id": 11, "sub_id": 0, "global_id": 22},
    "0019_alligator_closed":    {"main_id": 11, "sub_id": 1, "global_id": 23},
    
    "0020_indexpoint":          {"main_id": 12, "sub_id": 0, "global_id": 15},
    "0023_one_count":           {"main_id": 12, "sub_id": 0, "global_id": 15},
    "0048_index_point":         {"main_id": 12, "sub_id": 0, "global_id": 15},
    
    "0024_two_count":           {"main_id": 13, "sub_id": 0, "global_id": 8},
    "0025_three_count":         {"main_id": 14, "sub_id": 0, "global_id": 33},
    "0026_four_count":          {"main_id": 15, "sub_id": 0, "global_id": 26},
    "0029_indextip":            {"main_id": 16, "sub_id": 0, "global_id": 19},
    "0030_middletip":           {"main_id": 17, "sub_id": 0, "global_id": 12},
    "0031_ringtip":             {"main_id": 18, "sub_id": 0, "global_id": 6},
    "0032_pinkytip":            {"main_id": 19, "sub_id": 0, "global_id": 30},
    
    "0035_palmdown":            {"main_id": 20, "sub_id": 0, "global_id": 24},
    "0036_palmdownwave":        {"main_id": 20, "sub_id": 0, "global_id": 24},
    
    "0040_capisce":             {"main_id": 21, "sub_id": 0, "global_id": 16},
    "0041_claws":               {"main_id": 22, "sub_id": 0, "global_id": 9},
    "0051_dinosaur":            {"main_id": 22, "sub_id": 1, "global_id": 10},
    
    "0042_pucker":              {"main_id": 23, "sub_id": 0, "global_id": 34},
    "0044_cup":                 {"main_id": 24, "sub_id": 0, "global_id": 27},
    "0045_shakespearesyorick":  {"main_id": 25, "sub_id": 0, "global_id": 20},
    "0058_middlefinger":        {"main_id": 26, "sub_id": 0, "global_id": 13},
    
    "0100_neutral_relaxed": {"main_id": 27,"sub_id": 0,"global_id": 35},
    "0137_fingerspreadrelaxed": {"main_id": 27,"sub_id": 1,"global_id": 36},
    "0101_neutral_rigid": {"main_id": 27,"sub_id": 2,"global_id": 37},
    "0127_five_count": {"main_id": 27,"sub_id": 2,"global_id": 37},
    "0138_fingerspreadnormal": {"main_id": 27,"sub_id": 2,"global_id": 37},
    "0139_fingerspreadrigid": {"main_id": 27,"sub_id": 2,"global_id": 37},
    "0102_good_luck": {"main_id": 28,"sub_id": 0,"global_id": 63},
    "0103_fake_gun": {"main_id": 29,"sub_id": 0,"global_id": 56},
    "0104_star_trek": {"main_id": 30,"sub_id": 0,"global_id": 49},
    "0105_star_trek_extended_thumb": {"main_id": 31,"sub_id": 0,"global_id": 42},
    "0106_thumbup_relaxed": {"main_id": 32,"sub_id": 0,"global_id": 66},
    "0107_thumbup_normal": {"main_id": 32,"sub_id": 1,"global_id": 67},
    "0108_thumbup_rigid": {"main_id": 32,"sub_id": 1,"global_id": 67},
    "0109_thumbtucknormal": {"main_id": 33,"sub_id": 0,"global_id": 60},
    "0110_thumbtuckrigid": {"main_id": 33,"sub_id": 0,"global_id": 60},
    "0111_aokay": {"main_id": 34,"sub_id": 0,"global_id": 52},
    "0143_peacock": {"main_id": 34,"sub_id": 0,"global_id": 52},
    "0112_aokay_upright": {"main_id": 34,"sub_id": 1,"global_id": 53},
    "0113_surfer": {"main_id": 35,"sub_id": 0,"global_id": 46},
    "0114_rocker": {"main_id": 36,"sub_id": 0,"global_id": 38},
    "0114_rocker_frontside": {"main_id": 36,"sub_id": 1,"global_id": 39},
    "0115_rocker_backside": {"main_id": 36,"sub_id": 2,"global_id": 40},
    "0116_fist": {"main_id": 37,"sub_id": 0,"global_id": 64},
    "0117_fist_rigid": {"main_id": 37,"sub_id": 0,"global_id": 64},
    "0118_alligator_open": {"main_id": 38,"sub_id": 0,"global_id": 57},
    "0119_alligator_closed": {"main_id": 38,"sub_id": 1,"global_id": 58},
    "0120_indexpoint": {"main_id": 39,"sub_id": 0,"global_id": 50},
    "0123_one_count": {"main_id": 39,"sub_id": 0,"global_id": 50},
    "0148_index_point": {"main_id": 39,"sub_id": 0,"global_id": 50},
    "0124_two_count": {"main_id": 40,"sub_id": 0,"global_id": 43},
    "0125_three_count": {"main_id": 41,"sub_id": 0,"global_id": 68},
    "0126_four_count": {"main_id": 42,"sub_id": 0,"global_id": 61},
    "0129_indextip": {"main_id": 43,"sub_id": 0,"global_id": 54},
    "0130_middletip": {"main_id": 44,"sub_id": 0,"global_id": 47},
    "0131_ringtip": {"main_id": 45,"sub_id": 0,"global_id": 41},
    "0132_pinkytip": {"main_id": 46,"sub_id": 0,"global_id": 65},
    "0133_relaxedwave": {"main_id": 47,"sub_id": 0,"global_id": 59},
    "0135_palmdown": {"main_id": 47,"sub_id": 0,"global_id": 59},

    "0140_capisce": {"main_id": 48,"sub_id": 0,"global_id": 51},
    "0141_claws": {"main_id": 49,"sub_id": 0,"global_id": 44},
    "0151_dinosaur": {"main_id": 49,"sub_id": 1,"global_id": 45},
    "0142_pucker": {"main_id": 50,"sub_id": 0,"global_id": 69},
    "0144_cup": {"main_id": 51,"sub_id": 0,"global_id": 62},
    "0145_shakespearesyorick": {"main_id": 52,"sub_id": 0,"global_id": 55},
    "0158_middlefinger": {"main_id": 53,"sub_id": 0,"global_id": 48}

}

SINGLE_HAND_GESTURE_CLASS_MAPPING = {
    "neutral_relaxed": {"main_id": 0,"sub_id": 0,"global_id": 0},
    "fingerspreadrelaxed": {"main_id": 0,"sub_id": 1,"global_id": 1},
    "neutral_rigid": {"main_id": 0,"sub_id": 2,"global_id": 2},
    "five_count": {"main_id": 0,"sub_id": 2,"global_id": 2},
    "fingerspreadnormal": {"main_id": 0,"sub_id": 2,"global_id": 2},
    "fingerspreadrigid": {"main_id": 0,"sub_id": 2,"global_id": 2},
    "good_luck": {"main_id": 1,"sub_id": 0,"global_id": 28},
    "fake_gun": {"main_id": 2,"sub_id": 0,"global_id": 21},
    "star_trek": {"main_id": 3,"sub_id": 0,"global_id": 14},
    "star_trek_extended_thumb": {"main_id": 4,"sub_id": 0,"global_id": 7},
    "thumbup_relaxed": {"main_id": 5,"sub_id": 0,"global_id": 31},
    "thumbup_normal": {"main_id": 5,"sub_id": 1,"global_id": 32},
    "thumbup_rigid": {"main_id": 5,"sub_id": 1,"global_id": 32},
    "thumbtucknormal": {"main_id": 6,"sub_id": 0,"global_id": 25},
    "thumbtuckrigid": {"main_id": 6,"sub_id": 0,"global_id": 25},
    "aokay": {"main_id": 7,"sub_id": 0,"global_id": 17},
    "peacock": {"main_id": 7,"sub_id": 0,"global_id": 17},
    "aokay_upright": {"main_id": 7,"sub_id": 1,"global_id": 18},
    "surfer": {"main_id": 8,"sub_id": 0,"global_id": 11},
    "rocker": {"main_id": 9,"sub_id": 0,"global_id": 3},
    "rocker_frontside": {"main_id": 9,"sub_id": 1,"global_id": 4},
    "rocker_backside": {"main_id": 9,"sub_id": 2,"global_id": 5},
    "fist": {"main_id": 10,"sub_id": 0,"global_id": 29},
    "fist_rigid": {"main_id": 10,"sub_id": 0,"global_id": 29},
    "alligator_open": {"main_id": 11,"sub_id": 0,"global_id": 22},
    "alligator_closed": {"main_id": 11,"sub_id": 1,"global_id": 23},
    "indexpoint": {"main_id": 12,"sub_id": 0,"global_id": 15},
    "one_count": {"main_id": 12,"sub_id": 0,"global_id": 15},
    "index_point": {"main_id": 12,"sub_id": 0,"global_id": 15},
    "two_count": {"main_id": 13,"sub_id": 0,"global_id": 8},
    "three_count": {"main_id": 14,"sub_id": 0,"global_id": 33},
    "four_count": {"main_id": 15,"sub_id": 0,"global_id": 26},
    "indextip": {"main_id": 16,"sub_id": 0,"global_id": 19},
    "middletip": {"main_id": 17,"sub_id": 0,"global_id": 12},
    "ringtip": {"main_id": 18,"sub_id": 0,"global_id": 6},
    "pinkytip": {"main_id": 19,"sub_id": 0,"global_id": 30},
    
    "palmdown": {"main_id": 20,"sub_id": 0,"global_id": 24},
    "palmdownwave": {"main_id": 20,"sub_id": 0,"global_id": 24},
    "relaxedwave": {"main_id": 20,"sub_id": 0,"global_id": 24},  # This is a duplicate gesture in the original list
    
    "capisce": {"main_id": 21,"sub_id": 0,"global_id": 16},
    "claws": {"main_id": 22,"sub_id": 0,"global_id": 9},
    "dinosaur": {"main_id": 22,"sub_id": 1,"global_id": 10},
    "pucker": {"main_id": 23,"sub_id": 0,"global_id": 34},
    "cup": {"main_id": 24,"sub_id": 0,"global_id": 27},
    "shakespearesyorick": {"main_id": 25,"sub_id": 0,"global_id": 20},
    "middlefinger": {"main_id": 26,"sub_id": 0,"global_id": 13}
}



main_ids = []
global_sub_ids = []
for class_name, class_info in SINGLE_HAND_GESTURE_CLASS_MAPPING.items():
    main_ids.append(class_info['main_id'])
    global_sub_ids.append(class_info['global_id'])

main_ids = set(main_ids)
global_sub_ids = set(global_sub_ids)
NUM_SINGLE_HAND_GESTURE_MAIN_CLASS = len(main_ids)
NUM_SINGLE_HAND_GESTURE_GLOBAL_SUB_CLASSES = len(global_sub_ids)



main_ids = []
global_sub_ids = []
for class_name, class_info in TWO_HAND_GESTURE_CLASS_MAPPING.items():
    main_ids.append(class_info['main_id'])
    global_sub_ids.append(class_info['global_id'])
main_ids = set(main_ids)
global_sub_ids = set(global_sub_ids)
NUM_TWO_HAND_GESTURE_MAIN_CLASS = len(main_ids)
NUM_TWO_HAND_GESTURE_GLOBAL_SUB_CLASSES = len(global_sub_ids)



if __name__ == "__main__":
    print("Right Hand Gestures:")
    print(len(SINGLE_HAND_GESTURE_NAMES))
    
    print("Left Hand Gestures:", LEFT_HAND_GESTURE_NAMES)
    print("Right Hand Gestures:", RIGHT_HAND_GESTURE_NAMES)
        
