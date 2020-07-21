def img_resize(imgs, img_rows, img_cols, equalize=True):

    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist( img, clip_limit=0.05 )
            # img = clahe.apply(cv2.convertScaleAbs(img))

        new_imgs[mm] = cv2.resize( img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST )

    return new_imgs

def data_to_array(img_rows, img_cols):

    clahe = cv2.createCLAHE(clipLimit=0.05, tileGridSize=(int(img_rows/8),int(img_cols/8)) )

    fileList =  os.listdir('../data/train/')
    fileList = filter(lambda x: '.mhd' in x, fileList)
    fileList.sort()

    val_list = [5,15,25,35,45]
    train_list = list( set(range(50)) - set(val_list) )
    count = 0
    for the_list in [train_list,  val_list]:
        images = []
        masks = []

        filtered = filter(lambda x: any(str(ff).zfill(2) in x for ff in the_list), fileList)

        for filename in filtered:

            itkimage = sitk.ReadImage('../data/train/'+filename)
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs= img_resize(imgs, img_rows, img_cols, equalize=False)
                masks.append( imgs )

            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
                images.append(imgs )

        images = np.concatenate( images , axis=0 ).reshape(-1, img_rows, img_cols, 1)
        masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols, 1)
        masks = masks.astype(int)

        #Smooth images using CurvatureFlow
        images = smooth_images(images)

        if count==0:
            mu = np.mean(images)
            sigma = np.std(images)
            images = (images - mu)/sigma

            np.save('../data/X_train.npy', images)
            np.save('../data/y_train.npy', masks)
        elif count==1:
            images = (images - mu)/sigma

            np.save('../data/X_val.npy', images)
            np.save('../data/y_val.npy', masks)
        count+=1

    fileList =  os.listdir('../data/test/')
    fileList = filter(lambda x: '.mhd' in x, fileList)
    fileList.sort()
    n_imgs=[]
    images=[]
    for filename in fileList:
        itkimage = sitk.ReadImage('../data/test/'+filename)
        imgs = sitk.GetArrayFromImage(itkimage)
        imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
        images.append(imgs)
        n_imgs.append( len(imgs) )

    images = np.concatenate( images , axis=0 ).reshape(-1, img_rows, img_cols, 1)
    images = smooth_images(images)
    images = (images - mu)/sigma
    np.save('../data/X_test.npy', images)
    np.save('../data/test_n_imgs.npy', np.array(n_imgs) )


def load_data():

    X_train = np.load('../data/X_train.npy')
    y_train = np.load('../data/y_train.npy')
    X_val = np.load('../data/X_val.npy')
    y_val = np.load('../data/y_val.npy')


    return X_train, y_train, X_val, y_val
