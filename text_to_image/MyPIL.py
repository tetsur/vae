# -*- coding: utf-8 -*-
import os
import cv2
from PIL import Image, ImageFilter

def main():
    data_dir_path = u"./001.Black_footed_Albatross_img"
    savedata_dir_path = u"./001.Black_footed_Albatross_img"
    file_list = os.listdir(
        r'./001.Black_footed_Albatross_img')  # os.listdirで中身の確認
    

    for file_name in file_list:
        root, ext = os.path.splitext(file_name) #ファイル名を名前と拡張子のところで分ける
        if ext == u'.jpg':
            abs_name = data_dir_path + '/' + file_name
            new_name = savedata_dir_path + '/' + file_name
            print(abs_name)
            img = Image.open(abs_name)
            copy_img = img.copy()
            img_resize = copy_img.resize((128, 128), Image.LANCZOS)
            img_gray = img_resize.convert("L")
            img_gray.save(new_name, quality=95, optimize=True)

            # 結果を出力
          

if __name__ == '__main__':
    main()
