from evaluation.Image_Captioning.utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='./data/Images/MSCOCO/karpathy_splits/dataset_coco.json',
                       image_folder='../data/Images/MSCOCO/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='./evaluation/Image_Captioning/output',
                       max_len=50)