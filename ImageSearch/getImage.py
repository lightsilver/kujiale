import numpy as np
import tensorflow as tf
import sklearn as sk
import MySQLdb
import urllib
import os
import os.path
import Image
import AutoEncoder
from PIL import ImageFile
import modelTest


def getImgInDB():
    db = MySQLdb.connect(
        host="rr-bp100r3q888rgqa21o.mysql.rds.aliyuncs.com",
        port=3306,
        user="dtsqunhero02",
        passwd="Q862sxSM24Tuer",
        db="fenshua123",
        charset="utf8")

    cursor = db.cursor()
    print("db connected")
    sql = "SELECT * FROM fenshua123.brandgood where brandgoodid >70000;"
    cursor.execute(sql)
    print ("executing sql")
    result = cursor.fetchall()
    print("data fetched")
    data_file = open('img_profile.txt', 'w+')
    for row in result:
        preview_url = row[26]
        brandgoodid = row[0]
        itemid = row[1]
        categoryid = row[20]

        print(brandgoodid, preview_url)

        if preview_url:
            write_str = ' '.join([str(brandgoodid), str(itemid), str(categoryid), '\n'])
            data_file.write(write_str)
            #     urllib.urlretrieve(preview_url, 'img/%s.jpg' % brandgoodid)
    db.close()
    data_file.close()


def resizeImage():
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    dir = "/home/wuxie/Desktop/train"
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            im = Image.open(os.path.join(dir, filename))
            if cmp(im.size, (800, 800)):
                print (im.size)
            im = im.resize((28, 28), Image.ANTIALIAS)
            im.save(os.path.join(dir, filename), "jpeg")
            # im.show()
            # return 0
    return 0


def main():
    # getImgInDB()
    # resizeImage()
    # AutoEncoder.AutoEncoder()
    modelTest.model_test()
    return 0


if __name__ == '__main__':
    main()
