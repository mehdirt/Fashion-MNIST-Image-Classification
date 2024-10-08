import os
import glob
import shutil
import random
from tqdm import tqdm


def split_images_by_class(directory: str, *classes) -> None:
    """
    Take a directory of images from different classes,
    split them into folders corresponding to each class.
    """
    # Create a directory for each class and map lowercase class names to directories
    class_dirs = {class_name.lower(): os.path.join(directory, class_name) for class_name in classes}

    for class_dir in class_dirs.values():
        os.makedirs(class_dir, exist_ok=True)
        print(f"Created directory: {class_dir}")

    # Iterate through the classes and use glob to find matching files
    for class_name in class_dirs:
        pattern = os.path.join(directory, f"*{class_name.lower()}*.jpg")
        print(f"Looking for files matching: {pattern}")
        
        for filepath in glob.glob(pattern):
            filename = os.path.basename(filepath)
            dest_path = os.path.join(class_dirs[class_name], filename)
            shutil.move(filepath, dest_path)
        print(f"{class_name} images successfully moved.")

def create_validation_data(trn_dir: str, val_dir: str, split: float = 0.1) -> None:
    """
    """
    if not os.path.exists(val_dir):
        os.makedirs(val_dir, exist_ok=True)

    pattern =  trn_dir + '/*/*.jpg'
    train_ds = glob.glob(pattern)
    print(f"Number of total images: {len(train_ds)}")
    
    valid_sz = min(len(train_ds), int(split * len(train_ds)) if split < 1.0 else split)
    
    valid_ds = random.sample(train_ds, valid_sz)
    print(f"Number of validation images: {len(valid_ds)}")
    
    for fname in tqdm(valid_ds):
        basename = os.path.basename(fname)
        label = fname.split(os.path.sep)[-2]
        src_folder = os.path.join(trn_dir, label)
        tgt_folder = os.path.join(val_dir, label)
        if not os.path.exists(tgt_folder):
            os.mkdir(tgt_folder)
        shutil.move(os.path.join(src_folder, basename), os.path.join(tgt_folder, basename))
		

def pseudo_label(probs, tst_dir, test_dl, class_names, threshold=0.99999):
    num_data = len(test_dl.dataset)
    preds = np.argmax(probs, axis=1)
    candidate_idxs = np.arange(num_data)[probs.max(axis=1) >= threshold]
    
    fnames = [f[0].split('\\')[-1] for f in test_dl.dataset.imgs]
    imgs = [fnames[i] for i in candidate_idxs]
    labels = [class_names[preds[i]] for i in candidate_idxs]
    
    dest_folder = os.path.join(DATA_DIR, 'pseudo', 'train')
#     for name in class_names:
#         folder = os.path.join(dest_folder, name)
#         if not os.path.exists(folder):
#             os.mkdir(folder)
        
    for _, (img, label) in tqdm(enumerate(zip(imgs, labels))):
        src = os.path.join(tst_dir, 'unk', img)
        dst = os.path.join(dest_folder, label, img)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
