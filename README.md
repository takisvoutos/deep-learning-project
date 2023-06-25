# deep-learning-project

MURA Dataset:<br> 
https://stanfordmlgroup.github.io/competitions/mura/![image](https://github.com/takisvoutos/deep-learning-project/assets/17747120/20982647-d08a-4cef-bd4c-033fa45d20f5)

**Model training & evaluation** ->
mura-project.py

**Model testing & example** -> 
test_model.py

**Saved model** -> model.h5

## File paths

**mura-project.py:**

train_csv = "**your-file-path**/MURA-v1.1/train_image_paths.csv" <br>
valid_csv = "**your-file-path**/MURA-v1.1/valid_image_paths.csv"

return os.path.join('**your-file-path**',path)

os.environ['REQUESTS_CA_BUNDLE'] = '**your-file-path**/cacert.pem'

**test_model.py:**

model = tf.keras.models.load_model('**your-file-path**/model.h5')

image = cv2.imread('**your-file-path**/positive.png')<br>
image = cv2.imread('**your-file-path**/negative.png')
