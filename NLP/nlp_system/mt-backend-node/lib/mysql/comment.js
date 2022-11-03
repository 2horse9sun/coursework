var moment = require('moment')
let query = require('./connection');


exports.isuser = async function(openId) {
    console.log('isuser')
    let sql = "SELECT * FROM `account` where  openId = ?"
    let reault = await query(sql,[openId])
    return reault.length;
}
exports.updateuser = async function(openId,session_key,skey,update_time){
    let sql = "UPDATE account set skey = ? , session_key = ? , update_time = ? where openId = ?"
    await query(sql,[skey,session_key,update_time,openId])
}

exports.insertuser = async function(openId,session_key,skey,nickName,gender,language,city,province,country,avatarUrl,create_time,update_time){
    let sql = "insert into `account` (openId,session_key,skey,nickName,gender,language,city,province,country,avatarUrl,create_time,update_time) values (?,?,?,?,?,?,?,?,?,?,?,?)"
    await query(sql,[openId,session_key,skey,nickName,gender,language,city,province,country,avatarUrl,create_time,update_time])
}

exports.getuserinfo = async function(openId){
	let sql = "SELECT `openid`,`nickName`,`city`,`province`,`country`,`avatarUrl`,`create_time`,`update_time` FROM `account` where openId = ?"
	return reault = await query(sql,[openId])
}