

#image_folder = '/opt/kaggle/skin/all_images'
image_folder = 'gs://skin-image-221423-vcm/all_images'


image_label_file = '/Users/rajib/progs/skin/data/HAM10000_metadata.csv'

# Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
# basal cell carcinoma (bcc),
# benign keratosis-like lesions (solar lentigines /
# seborrheic keratoses and lichen-planus like keratoses, bkl),
# dermatofibroma (df),
# melanoma (mel),
# melanocytic nevi (nv) and
# vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc).

label_dict = dict()
label_dict['akiec'] = 0
label_dict['bcc'] = 1
label_dict['bkl'] = 2
label_dict['mel'] = 3
label_dict['nv'] = 4
label_dict['vasc'] = 5
label_dict['df'] = 6

# histopathology (histo),
# follow-up examination (follow_up),
# expert consensus (consensus),
# confirmation by in-vivo confocal microscopy (confocal)
dx_type = dict()
dx_type['histo'] = 0
dx_type['follow_up'] = 1
dx_type['consensus'] = 2
dx_type['confocal'] = 3

# abdomen, acral, back, chest, ear, face, foot, genital
# hand, lower extremity, neck, scalp, trunk, unknown, upper extremity
localization = dict()
localization['abdomen'] = 0
localization['acral'] = 1
localization['back'] = 2
localization['chest'] = 3
localization['ear'] = 4
localization['face'] = 5
localization['foot'] = 6
localization['genital'] = 7
localization['hand'] = 8
localization['lower extremity'] = 9
localization['neck'] = 10
localization['scalp'] = 11
localization['trunk'] = 12
localization['upper extremity'] = 13
localization['abdomen'] = 14


train_part = 0.75
validation_part = 0.10


