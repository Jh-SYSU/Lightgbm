# -*- coding: utf-8 -*-

import os,shutil

def copy_file(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)    #分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)                #创建路径
        shutil.copyfile(srcfile,dstfile)      #复制文件
        # print ("copy %s -> %s"%(srcfile, dstfile))

# srcfile='/Users/xxx/git/project1/test.sh'
# dstfile='/Users/xxx/tmp/tmp/1/test.sh'
# copy_file(srcfile, dstfile)

dstfile = r"C:\Users\Jh\Desktop\智慧中国杯\pro_mol_data\feature_pro"
pro_path = r"C:\Users\Jh\Desktop\智慧中国杯\pro_mol_data\feature_pro_cluster\pro_spd3_cluster_csv_train_centering"

for foldername in os.listdir(pro_path):
    folder_path = os.path.join(pro_path, foldername)
    if os.listdir(folder_path):
	    for filename in os.listdir(folder_path):
	    	fullname = os.path.join(folder_path, filename)
	    	fpath,fname = os.path.split(fullname)    #分离文件名和路径
	    	dstfilename = os.path.join(dstfile, fname)
	    	print (fullname)
    		copy_file(fullname, dstfilename)