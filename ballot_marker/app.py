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
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
db = SQLAlchemy(app)
global filepaths
filepaths=[]

class TemplateData(db.Model):
	id = db.Column(db.Integer, primary_key=True)
	leftX = db.Column(db.Integer,default=0)
	leftY = db.Column(db.Integer,default=0)
	rightX =  db.Column(db.Integer,default=0)
	rightY =  db.Column(db.Integer,default=0)
	page = db.Column(db.Integer,default=1)
	name = db.Column(db.String,default=None)

	def __repr__(self):
		return '<Contest %r>' % self.id

@app.route('/', methods=['POST','GET'])
def index():
	# return 
	if request.method=='POST':
		rows = db.session.query(TemplateData).all()
		file = open(request.form['file-name'],"w+")
		for row in rows:
			if row.name:
				file.write(f"{row.name} {row.leftX},{row.leftY} {row.rightX},{row.rightY} {row.page}\n")
			else:
				file.write(f"Contest{row.id} {row.leftX},{row.leftY} {row.rightX},{row.rightY} {row.page}\n")
		file.close()
	# redirect to homepage

	contests = TemplateData.query.order_by(TemplateData.id).all()

	return render_template("home.html", contests=contests)

@app.route('/delete/<int:id>')
def delete(id):
	contest_to_delete = TemplateData.query.get_or_404(id)

	try :
		db.session.delete(contest_to_delete)
		db.session.commit()
		return redirect('/')

	except:
		return "There was an error"

@app.route('/delete_all')
def deleteAll():
	try :
		rows_deleted = db.session.query(TemplateData).delete()
		db.session.commit()
		return redirect('/')

	except:
		return "There was an error"

@app.route('/markup', methods=['GET','POST'])
def markup():
	
	if request.method == 'POST' :
		data = request.form
		new_contest = TemplateData(leftX=data['x-click-l'], leftY = data['y-click-l'], rightX = data['x-click-r'], rightY = data['y-click-r'],page=data['pageNumber'],name=data['contest-name'])

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


if __name__ == "__main__":
	app.run(debug=True)