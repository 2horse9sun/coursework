var express = require('express');
var router = express.Router();
var fs = require('fs');
var readline = require('readline');
let mysql = require('../lib/mysql/people')

    var url = require('url');
    var util = require('util');
    var fs = require('fs');
    var callfile = require('child_process');
    var request = require('request');

var http = require('axios')
const { decryptByAES, encryptSha1 } = require('./util');

// const appid = 'wx7df37d6b2ff755e8';
const appid = "wx3da5445de748579e";
const appSecret = 'ea4a5221f16057d8e7443f988d57941f'

/* GET home page. */
// router.get('/', function(req, res, next) {
//   res.render('index.handlebars', { title: 'Express' });
// });



exports.login = async function (req, res) {
    let iv = req.body.iv;
    let encryptedData = req.body.encryptedData;
    let code = req.body.code;
  console.log(iv,encryptedData,code)
  console.log('-------------data---------------');

    let data = await getSessionKey(code, appid, appSecret)
        .then(resData => {
            // 选择加密算法生成自己的登录态标识
            console.log(resData)
            const { session_key } = resData;
            console.log(session_key)
            const skey = encryptSha1(session_key);
            console.log('-------------skey---------------');
            console.log(skey)

            let decryptedData = JSON.parse(decryptByAES(encryptedData, session_key, iv));
            console.log('-------------decryptedData---------------');
            console.log(decryptedData);
            console.log('-------------decryptedData---------------');
            
            return mysql.saveUserInfo(decryptedData,session_key,skey)
            // 存入用户数据表中
            
                        
        })
        .catch(err => {
          console.log(err)
            return {
                result: -3,
                errmsg: '返回数据字段不ss完整'
            }
        })

    res.json({
      data
    })
}

function getSessionKey (code, appid, appSecret) {
    
    const opt = {
        method: 'GET',
        url: 'https://api.weixin.qq.com/sns/jscode2session',
        params: {
            appid: appid,
            secret: appSecret,
            js_code: code,
            grant_type: 'authorization_code'
        }
    };
   
    return http(opt).then(function (response) {
        const data = response.data;
        console.log(data,"ceshi")
        if(!data.openid || !data.session_key || data.errcode) {
            return {
                result: -2,
                errmsg: data.errmsg || '返回数据字段不完整'
            }
        }
        else {
            return data
        }
        
    });
}


exports.getrank = async function(req,res){
  let openid = req.body.openid;
  try{
    // let wenzilist = await mysql.getranktype(openid,0);
    // let tupianlist = await mysql.getranktype(openid,1);
    // let shipinglist = await mysql.getranktype(openid,2);
    let yinyuelist = await mysql.getranktype(openid,3);
    // console.log(wenzilist,tupianlist,shipinglist,yinyuelist)
    res.json({
      key: 0,
      // wenzilist: wenzilist,
      // tupianlist: tupianlist,
      // shipinglist: shipinglist,
      yinyuelist: yinyuelist
    })
  }
  catch(e){
    console.log(e)
    res.json({
      key: 1
    })
  }
  
}

