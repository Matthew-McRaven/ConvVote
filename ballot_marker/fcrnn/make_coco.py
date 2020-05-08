import os, sys
from PIL import Image, ImageDraw
import argparse, json

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--input", required=True) 		##	input file
	parser.add_argument("--directory", required=True) 	##	directory with image files of ballot tmp____.jpg
														##	one for each page
	parser.add_argument("--dest1", required=False) 		##	where to save contest images
	parser.add_argument("--dest2", required=False)
	parser.add_argument("--coco1", required=False) 		##	name of coco file
	parser.add_argument("--coco2", required=False)
	parser.add_argument("--show", required=False) 		##	show an image?
	args = parser.parse_args()

	img_files = {}
	subs=[]
	for f in os.listdir(args.directory):
		if os.path.isdir(args.directory+"/"+f):
			subs.append(f)
	subs.sort()
	print("subs ",subs)
	g = input("pause")
	for f in subs:
		print("directory: ",f)
		print(os.listdir(args.directory+"/"+f))
		tmplist=[]
		for g in os.listdir(args.directory+"/"+f) :
			if "jpg" in g:
				tmplist.append(g)
		tmplist.sort()
		img_files[f]=tmplist
	print("image files in directories",img_files)
	# img_files.sort() # images of ballot pages

	print("reading from",args.input)

	rawtext= open(args.input,"r")
	lines = rawtext.readlines()
	print("number of lines",len(lines))

	contests=[]
	options=[]
	for line in lines:
		temp_dict={}
		temp_str = line.split(",")
		temp_str[len(temp_str)-1] = temp_str[len(temp_str)-1].strip()
		if temp_str[0]=="C":
			temp_dict['id']=int(temp_str[1])
			temp_dict['x1']=float(temp_str[3])
			temp_dict['y1']=float(temp_str[4])
			temp_dict['x2']=float(temp_str[5])
			temp_dict['y2']=float(temp_str[6])
			temp_dict['page']=int(temp_str[7])
			temp_dict['page-width']=int(temp_str[8])
			temp_dict['page-height']=int(temp_str[9])
			temp_dict['options']=[]
			temp_dict['ballot']=int(temp_str[10])
			contests.append(temp_dict)
		else :
			temp_dict['id']=int(temp_str[1])
			temp_dict['x1']=float(temp_str[3])
			temp_dict['y1']=float(temp_str[4])
			temp_dict['x2']=float(temp_str[5])
			temp_dict['y2']=float(temp_str[6])
			temp_dict['contest_id']=int(temp_str[7])
			options.append(temp_dict)

	for op in options:

		for con in contests:
			if op['contest_id']==con['id']:
				con['options'].append(op)
				break

	# make contest images
	# img_files[x] is the file for any contest with contest[page]==x+1
	# print(contests)

	coco_c={}
	coco_c['images']=[]
	coco_c['annotations']=[]
	coco_c['categories']=[]
	coco_o={}
	coco_o['images']=[]
	coco_o['annotations']=[]
	coco_o['categories']=[]

	imgs=[]
	page_names={}
	for sub in subs:
		tmp=[]
		print(f"images in subdir {sub}: {img_files[sub]}")
		for imgf in img_files[sub]:
			img = Image.open(args.directory+"/"+sub+"/"+imgf)
			tmp.append((img,imgf))
		imgs.append(tmp)
	contest_imgs={}
	# print(f"there are {sum(len(x) for x in imgs)} images")
	# i = input("pause")

	idx=0
	img_names={}
	for contest in contests:
		# print(f"contest{contest['id']}({contest['page']}) --> {img_files[contest['page']-1]}")
		shape  = (int(contest['page-width']*contest['x1']),int(contest['page-height']*contest['y1']),int(contest['page-width']*contest['x2']),int(contest['page-height']*contest['y2']))
		contest_x1,contest_y1,_,_ = shape
		print(f"shape of contest{contest['id']} {shape}")
		print(f"contest{contest['id']} page {contest['page']}")
		# save cropped image
		# print(f"img = imgs[{contest['ballot']-1}][{contest['page']-1}].crop({shape})")
		img, img_name = imgs[contest['ballot']-1][contest['page']-1]
		contest_img = img.crop(shape)

		new_name=f"contest{idx}.jpg"
		if args.dest2:
			contest_img.save(args.dest2+"/"+new_name)
		contest_imgs[contest['id']]=contest_img
		idx=idx+1

		# make COCO library for Images for the CONTESTS
		if img_name not in img_names.keys():
			num = len(img_names.keys())
			if args.dest1:
				img.save(f"{args.dest1}/page{num}.jpg")
			w,h=img.size
			img_names[img_name] = len(img_names.keys())+1
			tmp_dict = {}
			tmp_dict['id']=len(img_names.keys())
			tmp_dict['width']=w
			tmp_dict['height']=h
			tmp_dict['file_name']=f"page{num}.jpg"
			tmp_dict['license']=""
			tmp_dict['date_captured']=""
			# print("IMAGE ", contest['id'],img_name)
			coco_c['images'].append(tmp_dict)

		# contest data COCO annotations
		tmp_dict={}
		w,h=img.size
		new_x1,new_y1,new_x2,new_y2 = (int(contest['page-width']*contest['x1']),int(contest['page-height']*contest['y1']),int(contest['page-width']*contest['x2']),int(contest['page-height']*contest['y2']))
		cw=new_x2-new_x1
		ch=new_y2-new_y1

		tmp_dict['id']=contest['id']
		tmp_dict['image_id']=img_names[img_name]
		tmp_dict['category_id']=1
		tmp_dict['segmentation']=[]
		tmp_dict['area']=cw*ch
		tmp_dict['bbox']=[new_x1,new_y1,cw,ch]
		tmp_dict['iscrowd']=0

		# print("options: ",tmp_dict)
		coco_c['annotations'].append(tmp_dict)

		

		# make COCO Images for the option data
		tmp_dict_contest = {}
		tmp_dict_contest['id']=contest['id']
		w,h=contest_img.size
		tmp_dict_contest['width']=w
		tmp_dict_contest['height']=h
		tmp_dict_contest['file_name']=new_name
		tmp_dict_contest['license']=""
		tmp_dict_contest['date_captured']=""

		coco_o['images'].append(tmp_dict_contest)
	
		# make COCO annotations for each option
		for option in contest['options']:
			tmp_dict_option={}
			new_x1,new_y1,new_x2,new_y2 = (int(contest['page-width']*option['x1']),int(contest['page-height']*option['y1']),int(contest['page-width']*option['x2']),int(contest['page-height']*option['y2']))
			
			if option['id'] == 57:
				print(f"{option['id']} {new_x1} {new_y1} {new_x2} {new_y2} , contest x and y: {contest_x1} {contest_y1}")

			op_w=new_x2-new_x1
			op_h=new_y2-new_y1


			new_x1=new_x1-contest_x1
			new_y1=new_y1-contest_y1
			new_x2=new_x2-contest_x1
			new_y2=new_y2-contest_y1
			if new_x1 < 0:
				print(f"{option['id']} {new_x1} {new_y1} {new_x2} {new_y2} , contest x and y: {contest_x1} {contest_y1}")
				j = input("asdfas")

			tmp_dict_option['id']=option['id']
			tmp_dict_option['image_id']=option['contest_id']
			tmp_dict_option['category_id']=1
			tmp_dict_option['segmentation']=[]
			tmp_dict_option['area']=op_w*op_h
			tmp_dict_option['bbox']=[new_x1,new_y1,op_w,op_h]
			tmp_dict_option['iscrowd']=0

			# print("options: ",tmp_dict_option)
			coco_o['annotations'].append(tmp_dict_option)

	coco_o['categories'].append({
		"id":1,
		"name":"option",
		"supercategory":"none"
		})
	coco_c['categories'].append({
		"id":1,
		"name":"contest",
		"supercategory":"none"
		})
	# print(coco_c)

	# make COCO categories
	if args.coco2:
		with open(args.coco2, "w+") as outfile: 
			json.dump(coco_o, outfile) 
	if args.coco1:
		with open(args.coco1, "w+") as outfile: 
			json.dump(coco_c, outfile) 


	if args.show :
		my_img = contest_imgs[1]
		imgdrw = ImageDraw.Draw(my_img)
		for ann in coco_o['annotations'] :
				
			if ann['image_id']==1:
				
				imgdrw.rectangle([ann['bbox'][0],ann['bbox'][1],ann['bbox'][0]+ann['bbox'][2],ann['bbox'][1]+ann['bbox'][3]],outline="red",fill=None)
				# my_img.show()

		my_img.show()
