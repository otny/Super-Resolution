# 画像処理モジュール "Pillow" で画像をリサイズする。
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np


class resizePicture:
        # jpg形式ファイルの画像サイズを変更する
        def resizeImage(self, inputImage, outputImage, height, width, save=True):

                # 元画像読み込み
                imagePil = Image.open(inputImage)
                # リサイズ
                imagePil.thumbnail((width, height),resample=Image.BICUBIC)
                # imagePil.show()
                img = np.asarray(imagePil)
                plt.imshow(img)
                plt.show()
                if save:
                        # 画像の保存
                        imagePil.save(outputImage, quality=85)

        # pngファイルの画像サイズを変更し、jpg形式で保存する。
        def resizeImageWithConvert(self, inputImage, outputImage, height, width):

                # 元画像読み込み
                imagePil = Image.open(inputImage)
                # リサイズ
                imagePil.thumbnail((width, height),resample=Image.BICUBIC)
                # jpgに変換して保存
                convertPil = imagePil.convert('RGB')
                convertPil.save(outputImage, quality=85)


def main():
        resizer =  resizePicture()
        input_path = "../test_images/original.jpg"
        output_path = "../test_images/low_img03.jpg"
        height = 128
        width = 128
        save = False
        resizer.resizeImage(input_path, output_path, height, width, save)

if __name__ == "__main__":
        main()