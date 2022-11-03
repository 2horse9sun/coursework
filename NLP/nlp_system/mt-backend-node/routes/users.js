var express = require('express');
const uuid = require('uuid')
let mysql = require('../lib/mysql/people')
const schedule = require('node-schedule');
const fs = require('fs');
const path = require("path");
const multiparty = require('multiparty')
const child_process = require("child_process");
const axios = require('axios')

exports.uploadwenzi = async function (req, res) {
	let name = req.body.name;
	let type = req.body.type;
	let openid = req.body.openid;
	let content = req.body.txt;

	try {
		await mysql.addwenzi(name, type, openid, content)
		res.json({
			key: 0
		})
	}
	catch (e) {
		console.log(e)
		res.json({
			key: 1
		})
	}
}

exports.emoji_predict = async function (req, res) {
	let input_str = req.query.input_str
	try{
		axios.get("http://127.0.0.1:5000/emoji_predict", {params:{
				input_str
			}}).then((data) => {
			data = data.data

			if(data.status == 0){
				let predict_res = data.data
				res.json({
					data: predict_res,
					"status": 0
				})
			}else{
				res.json({
					"status": -1
				})
			}

		})
	} catch (e) {
		console.log(e);
		res.json({
			"status": -1
		})
	}
}

exports.ner = async function (req, res) {
	let input_str = req.query.input_str
	try{
		axios.get("http://127.0.0.1:5000/ner", {params:{
				input_str
			}}).then((data) => {
			data = data.data

			if(data.status == 0){
				let ner_res = data.data
				res.json({
					data: ner_res,
					"status": 0
				})
			}else{
				res.json({
					"status": -1
				})
			}

		})
	} catch (e) {
		console.log(e);
		res.json({
			"status": -1
		})
	}
}

exports.translate = async function (req, res) {
	let input_str = req.query.input_str
	let opt = req.query.opt
	console.log(input_str)
	try{
		axios.get("http://127.0.0.1:5000/translate", {params:{
				input_str,
				opt
			}}).then((data) => {
				data = data.data

				if(data.status == 0){
					let output_str = data.data.toString()
					res.json({
						data: output_str,
						"status": 0
					})
				}else{
					res.json({
						"status": -1
					})
				}

		})
	} catch (e) {
		console.log(e);
		res.json({
			"status": -1
		})
	}
}

exports.search = async function (req, res) {
	let key_words = req.query.key_words
	let offset = req.query.offset
	let page_size = req.query.page_size
	console.log(key_words)
	try{
		axios.get("http://127.0.0.1:5000/search", {params:{
				key_words,
				offset,
				page_size
			}}).then((data) => {
			data = data.data

			if(data.status == 0){
				let search_res = data.data
				res.json({
					data: search_res,
					"status": 0
				})
			}else{
				res.json({
					"status": -1
				})
			}

		})
	} catch (e) {
		console.log(e);
		res.json({
			"status": -1
		})
	}
}

exports.get_random_sentence = async function (req, res) {
	try{

		let random_sentence = await mysql.get_random_sentence()
		res.json({
			data: random_sentence[0]['en'],
			"status": 0
		})

		// axios.get("http://127.0.0.1:5000/get_random_sentence")
		// 	.then((data) => {
		// 	data = data.data
		// 	if(data.status == 0){
		// 		let output_str = data.data.toString()
		// 		res.json({
		// 			data: output_str,
		// 			"status": 0
		// 		})
		// 	}else{
		// 		res.json({
		// 			"status": -1
		// 		})
		// 	}
		// })
	} catch (e) {
		console.log(e);
		res.json({
			"status": -1
		})
	}
}

exports.uploadmusicalnotes = async function (req, res) {
	let name = req.body.name;
	let type = req.body.type;
	let openid = req.body.openid;
	let content = req.body.txt;
	let noteList = content.split(',');
	let exec = require('child_process').exec;
	let cmd = "python3 ./xrhythm/test.py";
	for (let i = 0; i < noteList.length; i++) {
		cmd += " " + noteList[i];
	}
	let cmd_trans = "timidity "

	try {
		exec(cmd, async function (error, stdout, stderr) {
			if (error) {
				console.error('error: ' + error);
				return;
			}
			stdout = stdout.replace(/[\r\n]/g, "") + "_0";
			stdout = stdout.substr(3)
			console.log(stdout)
			let fileName = stdout + ".mid";
			// let cmd_trans = "timidity " + "./xrhythm/outputs/" + fileName + " -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 256k " + "./xrhythm/outputs/" + stdout + ".mp3";
			let cmd_trans = "timidity " + "./public/miniapp/" + fileName + " -Ow -o - | ffmpeg -i - -acodec libmp3lame -ab 256k " + "./public/miniapp/" + stdout + ".mp3";
			let fileName_mp3 = stdout + ".mp3";
			console.log(fileName_mp3)
			await mysql.addaudio(name, type, openid, fileName_mp3)
			exec(cmd_trans, async function (error, stdout, stderr) {
				console.log(error, stdout)
				res.json({
					"status": 0
				})
			})
		});
	} catch (e) {
		console.log(e);
		res.json({
			"status": -1
		})
	}

}




exports.uploadimg = async function (req, res) {
	const form = new multiparty.Form();
	//设置单文件大小限制 2M 
	form.maxFieldsSize = 20 * 1024 * 1024;
	form.uploadDir = 'upload'


	form.parse(req, async function (err, flields, files) {
		console.log(files, " :files")
		console.log(flields, " :flields")
		var filepath = './upload/images'

		//拿到扩展名
		const extname = path.extname(files.file[0].originalFilename);
		//uuid生成 图片名称
		const nameID = (uuid.v1()).replace(/\-/g, '');
		const oldpath = path.normalize(files.file[0].path);
		//新的路径
		let newfilename = nameID + extname;
		var newpath = './upload/images/' + newfilename;
		console.log(oldpath, newpath)


		fs.exists(filepath, async function (exists) {
			if (!exists) {
				console.log("文件不存在")
				fs.mkdirSync(filepath, { recursive: true })
			}
			if (exists) {
				console.log("文件存在")
				//改名
				fs.rename(oldpath, newpath, async function (err) {
					if (err) {
						console.log(err)
						res.json({
							key: 1
						})
					} else {
						try {
							let url = '/upload/images/' + newfilename;
							let key = await mysql.uploadimg(flields.type, flields.name, url, flields.openid, flields.txt);
							console.log(key, "asdhsaga57687980!@#!@#!@#")
							res.json({
								key: 0
							})
						}
						catch (e) {
							console.log(e)
							res.json({
								key: 1
							})
						}
					}
				})
			};
		})
	})
}



exports.uploadvideo = async function (req, res) {
	const form = new multiparty.Form();
	//设置单文件大小限制 2M 
	form.maxFieldsSize = 20 * 1024 * 1024;
	form.uploadDir = 'upload'


	form.parse(req, async function (err, flields, files) {
		console.log(files, " :files")
		console.log(flields, " :flields")
		var filepath = './upload/video'

		//拿到扩展名
		const extname = path.extname(files.files[0].originalFilename);
		//uuid生成 图片名称
		const nameID = (uuid.v1()).replace(/\-/g, '');
		const oldpath = path.normalize(files.files[0].path);
		//新的路径
		let newfilename = nameID + extname;
		var newpath = './upload/video/' + newfilename;
		console.log(oldpath, newpath)


		fs.exists(filepath, async function (exists) {
			if (!exists) {
				console.log("文件不存在")
				fs.mkdirSync(filepath, { recursive: true })
			}
			if (exists) {
				console.log("文件存在")
				//改名
				fs.rename(oldpath, newpath, async function (err) {
					if (err) {
						console.log(err)
						res.json({
							key: 1
						})
					} else {
						try {
							let url = '/upload/video/' + newfilename;
							let key = await mysql.uploadvideo(flields.type, flields.name, url, flields.openid, flields.txt);
							console.log(key, "asdhsaga57687980!@#!@#!@#")
							res.json({
								key: 0
							})

						}
						catch (e) {
							console.log(e)
							res.json({
								key: 1
							})
						}
					}
				})
			};
		})
	})
}


exports.getmydynamic = async function (req, res) {
	let openid = req.body.openid;
	let type = req.body.type;
	try {
		let dynamiclist = await mysql.getmydynamic(openid, type)
		//console.log(dynamiclist)
		res.json({
			key: 0,
			dynamiclist: dynamiclist
		})
	}
	catch (e) {
		console.log(e)
		res.json({
			key: 1
		})
	}

}

exports.deletemydynamic = async function (req, res) {
	let openid = req.body.openid;
	let type = req.body.type;
	let dy_id = req.body.dy_id;
	try {
		let ans = await mysql.deletemydynamic(openid, type, dy_id)
		res.json({
			key: 0,
			status: 0
		})
	}
	catch (e) {
		console.log(e)
		res.json({
			key: 1
		})
	}

}

exports.deletemyaudio = async function (req, res) {
	let openid = req.body.openid;
	let fileName = req.body.fileName.split('.')[0];

	let exec = require('child_process').exec;
	// let cmd = "python3 ./xrhythm/test.py";
	let cmd = "rm -rf ./public/miniapp/" + fileName + ".mp3" + " ./public/miniapp/" + fileName + ".mid";

	try {
		exec(cmd, async function (error, stdout, stderr) {
			if (error) {
				console.error('error: ' + error);
				return;
			}
			res.json({
				"status": 0
			})
		});
	} catch (e) {
		console.log(e);
		res.json({
			"status": -1
		})
	}

}


exports.gettimelinedynamic = async function (req, res) {
	try {
		let dynamiclist = await mysql.gettimelinedynamic(req.body.openid)
		//console.log(dynamiclist)
		res.json({
			key: 0,
			dynamiclist: dynamiclist
		})
	}
	catch (e) {
		console.log(e)
		res.json({
			key: 1
		})
	}
}


exports.dy_zan = async function (req, res) {
	let type = req.body.type;
	let openid = req.body.openid;
	let dy_id = req.body.dy_id;
	console.log(type)
	if (type == 1) //type == 1 加赞
	{
		console.log('add')
		await mysql.add_zan(openid, dy_id)
	}
	if (type == 0) {
		console.log('del')
		await mysql.del_zan(openid, dy_id)
	}
	res.end();
}

exports.getmyzan = async function (req, res) {
	let type = req.body.type;
	let openid = req.body.openid;
	console.log(type)
	if (type == 1) //type == 1 别人赞我
	{
		console.log('add')
		let zan_list = await mysql.zan_my(openid)
		res.json({
			zan_list: zan_list
		});
	}
	if (type == 0) {	//type == 0 我赞别人
		console.log('del')
		let zan_list = await mysql.my_zan(openid)
		res.json({
			zan_list: zan_list
		});
	}

}

