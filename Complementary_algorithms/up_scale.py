import cv2
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np

class resizePicture:
        # jpg形式ファイルの画像サイズを変更する
        def upScaleImage(self, inputImagePath, outputImagePath, upScaleRatio, save=True):

                # 元画像読み込み
                img = cv2.imread(inputImagePath)
                # サイズ計測
                height = img.shape[0]
                width = img.shape[1]
                # リサイズ
                # img2 = cv2.resize(img , (128, 96), interpolation=cv2.INTER_AREA)
                img2 = cv2.resize(img , (int(width*upScaleRatio), int(height*upScaleRatio)), interpolation=cv2.INTER_LANCZOS4)
                # imagePil.show()
                img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                plt.imshow(img2)
                plt.show()
                if save:
                        # 画像の保存
                        cv2.imwrite(outputImagePath , img2)

def main():
        resizer =  resizePicture()
        input_path = "../test_images/pillow/low_img03.jpg"
        output_path = "../test_images/upscale/256to512_Nearest_Neighbor.png"
        upScaleRatio = 2
        save = False
        resizer.upScaleImage(input_path, output_path, upScaleRatio, save)

if __name__ == "__main__":
        main()