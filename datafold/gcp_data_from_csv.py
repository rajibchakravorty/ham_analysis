import pickle
from os.path import join
from pandas import read_csv
from random import shuffle

from constants import (image_folder, label_dict, dx_type, localization,
                       image_label_file, train_part, validation_part)


def _convert_lesion_id(string_id):
    return string_id


def _convert_image_name(string_image):
    image_file = join(image_folder, '{0}.jpg'.format(string_image))
    return image_file


def _convert_label(string_label):
    if string_label in label_dict:
        return label_dict[string_label]
    else:
        return -1


def _convert_dx_type(string_dx_type):
    if string_dx_type in dx_type:
        return dx_type[string_dx_type]
    else:
        return -1


def _convert_age(string_age):
    try:
        return float(string_age)
    except:
        return -1.0


def _convert_gender(string_gender):
    if string_gender == 'male':
        return 0
    elif string_gender == 'female':
        return 1
    else:
        return -1


def _convert_localization(string_localization):
    if string_localization in localization:
        return localization[string_localization]
    else:
        return -1


def _cancer_benign_label(disease_label):

    if disease_label == 1 or disease_label == 3:
        return 1
    else:
        return 0


def _read_csv(filename):

    with open(filename, 'rb') as f:
        csv_data = read_csv(f, header=0,
                            converters = {0: _convert_lesion_id,
                                          1: _convert_image_name,
                                          3: _convert_dx_type,
                                          4: _convert_age,
                                          5: _convert_gender,
                                          6: _convert_localization})
    return csv_data


def collect_fold_data(fold_data):

    data_list = list()

    for index, row in fold_data.iterrows():
        image_file_name = row['image_id']
        disease_label = row['dx']
        dx_type = row['dx_type']
        age = row['age']
        sex = row['sex']
        localization = row['localization']

        if disease_label == -1:
            continue

        cancer = _cancer_benign_label(disease_label)

        data_list.append((image_file_name,
                          disease_label,
                          cancer,
                          dx_type,
                          age,
                          sex,
                          localization))

    return data_list


if __name__ == '__main__':
    csv_data = _read_csv(image_label_file)

    unique_labels = list(set(csv_data.lesion_id))

    shuffle(unique_labels)

    train_number = int(len(unique_labels) * train_part)
    validation_number = train_number + int(len(unique_labels)*validation_part)

    train_lesions = unique_labels[0:train_number]
    validation_lesions = unique_labels[train_number:validation_number]
    test_lesions = unique_labels[validation_number:]

    print('{0}/{1}/{2} selected for train/validation/test in {3}'.format(len(train_lesions),
                                                                         len(validation_lesions),
                                                                         len(test_lesions),
                                                                         (len(train_lesions)+
                                                                          len(validation_lesions)+
                                                                          len(test_lesions))))

    train_data = csv_data.loc[csv_data['lesion_id'].isin(train_lesions)]
    validation_data = csv_data.loc[csv_data['lesion_id'].isin(validation_lesions)]
    test_data = csv_data.loc[csv_data['lesion_id'].isin(test_lesions)]

    print('{0}/{1}/{2} data found selected for train/validation/test in {3}'.format(len(train_data.index),
                                                                                    len(validation_data.index),
                                                                                    len(test_data.index),
                                                                                    (len(train_data.index)+
                                                                                    len(validation_data.index)+
                                                                                    len(test_data.index))))
    train_list = collect_fold_data(train_data)
    valid_list = collect_fold_data(validation_data)
    test_list = collect_fold_data(test_data)

    print('{0}/{1}/{2} data found selected for train/validation/test in {3}'.format(len(train_list),
                                                                                    len(valid_list),
                                                                                    len(test_list),
                                                                                    (len(train_list) +
                                                                                     len(valid_list) +
                                                                                     len(test_list))))

    #with open('fold_data.pkl', 'wb') as f:
    #    pickle.dump((train_list, valid_list, test_list), f)

    with open('gcp_data_cancer.csv', 'wt') as f:

        for data in train_list:

            f.write('TRAIN,{0},{1}\n'.format(data[0], data[2]))

        for data in valid_list:
            f.write('VALIDATION,{0},{1}\n'.format(data[0], data[2]))

        for data in test_list:
            f.write('TEST,{0},{1}\n'.format(data[0], data[2]))

