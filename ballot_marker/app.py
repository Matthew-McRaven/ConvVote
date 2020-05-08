import sys, os
from flask import Flask, render_template, url_for, redirect, request
from flask_sqlalchemy import SQLAlchemy
from PIL import Image
from pdf2image import convert_from_path, convert_from_bytes
import argparse
from pdf2image.exceptions import (
    PDFInfoNotInstalledError,
    PDFPageCountError,
    PDFSyntaxError
)

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///contest.db'
app.config['SQLALCHEMY_BINDS'] = {'option' : 'sqlite:///option.db'}
db = SQLAlchemy(app)
global filepaths
filepaths=[]

class ContestData(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	leftX = db.Column(db.Integer,default=0)
	leftY = db.Column(db.Integer,default=0)
	rightX =  db.Column(db.Integer,default=0)
	rightY =  db.Column(db.Integer,default=0)
	ballot = db.Column(db.Integer,default=1)
	page = db.Column(db.Integer,default=1)
	name = db.Column(db.String,default=None)

	def __repr__(self):
		return '<Contest %r>' % self.id

class OptionData(db.Model):
	__bind_key__ = 'option'
	id = db.Column(db.Integer, primary_key=True)
	contest = db.Column(db.Integer, default=0) # id of the contest this option resides in
	leftX = db.Column(db.Integer,default=0)
	leftY = db.Column(db.Integer,default=0)
	rightX =  db.Column(db.Integer,default=0)
	rightY =  db.Column(db.Integer,default=0)
	name = db.Column(db.String,default=None)

	def __repr__(self):
		return '<Option %r>' % self.id


@app.route('/', methods=['POST','GET'])
def index():
	# return 
	if request.method=='POST':
		rows = db.session.query(ContestData).all()
		file = open(request.form['file-name'],"w+")
		image_dir= "."+url_for('static',filename="images")
		page_dims=[]
		dirs=[]
		for path in os.listdir(image_dir):
			if "tmp" in path:
				dirs.append(path)
		dirs.sort()
		for path in dirs:
			img = Image.open("static/images/"+path)
			page_dims.append(img.size)

		for row in rows:
			w,h = page_dims[(row.page-1)]
			if row.name:
				file.write(f"C,{row.id},{row.name},{row.leftX},{row.leftY},{row.rightX},{row.rightY},{row.page},{w},{h},{row.ballot}\n")
			else:
				file.write(f"C,{row.id},Contest{row.id},{row.leftX},{row.leftY},{row.rightX},{row.rightY},{row.page},{w},{h},{row.ballot}\n")
		rows = db.session.query(OptionData).all()
		# file = open(request.form['file-name'],"w+")
		for row in rows:
			if row.name:
				file.write(f"O,{row.id},{row.name},{row.leftX},{row.leftY},{row.rightX},{row.rightY},{row.contest}\n")
			else:
				file.write(f"O,{row.id},Option{row.id},{row.leftX},{row.leftY},{row.rightX},{row.rightY},{row.contest}\n")
		file.close()
	# redirect to homepage

	contests = ContestData.query.order_by(ContestData.id).all()
	options = OptionData.query.order_by(OptionData.id).all()


	return render_template("home.html", contests=contests, options=options)

@app.route('/contest/delete/<int:id>')
def deleteContest(id):
	contest_to_delete = ContestData.query.get_or_404(id)

	try :
		db.session.delete(contest_to_delete)
		db.session.commit()
		return redirect('/')

	except:
		return "There was an error"

@app.route('/option/delete/<int:id>')
def deleteOption(id):
	option_to_delete = OptionData.query.get_or_404(id)

	try :
		db.session.delete(option_to_delete)
		db.session.commit()
		return redirect('/')

	except:
		return "There was an error"

@app.route('/contest/delete_all')
def deleteAllContests():
	try :
		rows_deleted = db.session.query(ContestData).delete()
		db.session.commit()
		return redirect('/')

	except:
		return "There was an error"

@app.route('/option/delete_all')
def deleteAllOptions():
	try :
		rows_deleted = db.session.query(OptionData).delete()
		db.session.commit()
		return redirect('/')

	except:
		return "There was an error"

@app.route('/contest', methods=['GET','POST'])
def contest():
	
	if request.method == 'POST' :
		data = request.form
		new_contest = ContestData(leftX=data['x-click-l'], leftY = data['y-click-l'], rightX = data['x-click-r'], rightY = data['y-click-r'],page=data['pageNumber'],name=data['contest-name'],ballot=data['ballotNumber'])

		try :
			db.session.add(new_contest)
			db.session.commit()
			# return redirect('/')
			paths=[]
			temp = os.listdir(data['image-dir'])
			for p in temp:
				if "tmp" in p:
					paths.append(f"{data['image-dir']}/{p}")
			paths.sort()
			return render_template('markup.html', paths=paths, dir=data['image-dir'])
		except:
			return "there was an error"

	else :
		# print('get method markup')
		# print("---->filepaths",filepaths)
		image_dir= "."+url_for('static',filename="images")
		paths=[]

		if not request.args.get('use-existing-files'):

			if not request.args.get('imagefile'):
				return redirect('/')

			image_file = request.args.get('imagefile')
			image_file = url_for('static',filename=f"images/{image_file}")
			images = convert_from_path("."+image_file, output_folder=image_dir,output_file="tmp",fmt='pdf')
		
			temp = os.listdir(image_dir)
			for p in temp:
				if "tmp" in p:
					i = Image.open("."+url_for('static',filename=f"images/{p}"))
					os.remove("."+url_for('static',filename=f"images/{p}"))
					p_i = p[:-3]+"jpg"
					i.save("./static/images/"+p_i)
					paths.append("."+url_for('static',filename=f"images/{p_i}"))
			
		else :
			temp = os.listdir(image_dir)
			for p in temp:
				if "tmp" in p:
					paths.append("."+url_for('static',filename=f"images/{p}"))

		paths.sort()
		return render_template('markup.html', paths=paths,dir=image_dir)

@app.route('/option', methods=['GET','POST'])
def option():

	if request.method=='POST':
		data = request.form
		new_option = OptionData(leftX=data['x-click-l'], leftY = data['y-click-l'], rightX = data['x-click-r'], rightY = data['y-click-r'],contest=data['contest-id'],name=data['option-name'])

		try :
			db.session.add(new_option)
			db.session.commit()
			# return redirect('/')
			paths=[]
			temp = os.listdir(data['image-dir'])
			for p in temp:
				if "tmp" in p:
					paths.append(f"{data['image-dir']}/{p}")
			paths.sort()
			contests = ContestData.query.order_by(ContestData.id).all()
			return render_template('option.html', paths=paths, dir=data['image-dir'],contests=contests)
		except:
			return "there was an error"

	else :
		# get contests and list them; as well as images to click on
		image_dir= "."+url_for('static',filename="images")
		paths=[]
		temp = os.listdir(image_dir)
		for p in temp:
			if "tmp" in p:
				paths.append("."+url_for('static',filename=f"images/{p}"))
		paths.sort()
		contests = ContestData.query.order_by(ContestData.id).all()
		return render_template('option.html', paths=paths, contests=contests, dir=image_dir)



if __name__ == "__main__":
	app.run(debug=True)